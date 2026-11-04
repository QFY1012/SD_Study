import sys, types, importlib.util, json
import torch

# ---- stub the ComfyUI modules the node pack imports
def mod(name, **kw):
    m = types.ModuleType(name); m.__dict__.update(kw); sys.modules[name] = m; return m
comfy = mod("comfy")
comfy.samplers = mod("comfy.samplers", KSampler=types.SimpleNamespace(SAMPLERS=["euler"], SCHEDULERS=["simple"]))
comfy.sd = mod("comfy.sd"); comfy.utils = mod("comfy.utils")
mod("folder_paths", get_filename_list=lambda k: ["m.safetensors"], get_folder_paths=lambda k: [],
    get_full_path_or_raise=lambda k, n: "/nonexistent/" + n)
calls = []
def fake_ksampler(model, seed, steps, cfg, sampler, sched, pos, neg, lat, denoise=1.0):
    calls.append({"shape": tuple(lat["samples"].shape), "mask": lat.get("noise_mask"), "denoise": denoise})
    s = lat["samples"].clone(); s[:] = 1.0   # "model output": latent of ones
    return ({"samples": s},)
mod("nodes", common_ksampler=fake_ksampler)

class FakeVAE:   # encode: downscale x8 -> 4ch ; decode: constant red image upscaled x8
    def encode(self, pix): return torch.zeros(pix.shape[0], 4, pix.shape[1] // 8, pix.shape[2] // 8)
    def decode(self, lat):
        out = torch.zeros(lat.shape[0], lat.shape[2] * 8, lat.shape[3] * 8, 3); out[..., 0] = 1.0; return out
    encode_tiled = encode; decode_tiled = decode

spec = importlib.util.spec_from_file_location("segs", "../../../../colab/custom_nodes/ComfyUI-SEGSInpaint/__init__.py")
segs_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(segs_mod)
segs_mod._MODEL_CACHE["m.safetensors"] = object()           # skip real checkpoint loading
segs_mod.apply_loras = lambda m, j: m

# ---- geometry
from geometry import boxes_from_input, target_size
b = boxes_from_input([[{"x": 8, "y": 255, "width": 192, "height": 202}, {"x": 499, "y": 349, "width": 269, "height": 349}]], 768, 1344)
assert b == [(8, 255, 200, 457), (499, 349, 768, 698)], b
assert boxes_from_input([[{"x": -50, "y": 0, "width": 10, "height": 10}]], 100, 100) == []   # fully outside -> dropped
print("target_size(192x202 box, 450x470 crop):", target_size(192, 202, 450, 470, 512, 1024))

# ---- SEGS conversion
img = torch.full((1, 1344, 768, 3), 0.5)
bboxes = [[{"x": 8, "y": 255, "width": 192, "height": 202}, {"x": 499, "y": 349, "width": 269, "height": 349}]]
segs, = segs_mod.SAM3BoxesToSEGS().convert(bboxes, 2.0, 16, img)
assert len(segs["segs"]) == 2
for s in segs["segs"]:
    x1, y1, x2, y2 = s["crop_region"]; print("bbox", s["bbox"], "crop", s["crop_region"], "mask", tuple(s["mask"].shape), "sum", int(s["mask"].sum()))
    assert s["mask"].shape == (y2 - y1, x2 - x1)

# ---- detailer end-to-end with fake model
det = segs_mod.RemoteSEGSDetailer()
out, = det.detail(512.0, True, 1024.0, 1, 15, 3.0, "euler", "simple", 0.4, 30, True, True, 1, 30, False, False,
                  "m.safetensors", "[]", img, segs, FakeVAE(), None, None)
assert out.shape == img.shape
changed = (out - img).abs().sum(-1) > 1e-4
ys, xs = torch.where(changed[0])
print("changed pixel bbox:", int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
# box centre must be fully replaced by red (1,0,0); far away must be untouched
cx, cy = (8 + 200) // 2, (255 + 457) // 2
assert torch.allclose(out[0, cy, cx], torch.tensor([1.0, 0, 0]), atol=1e-3), out[0, cy, cx]
assert torch.allclose(out[0, 900, 400], torch.tensor([0.5, 0.5, 0.5])), "far-away pixel modified"
assert torch.allclose(out[0, 5, 5], torch.tensor([0.5, 0.5, 0.5])), "corner modified"
print("sampler calls:", [(c["shape"], tuple(c["mask"].shape) if c["mask"] is not None else None) for c in calls])
# feather: transition pixels at mask border are blended (neither 0.5 nor pure red)
edge = out[0, cy, 8 - 16 + 16]   # at x = bbox left - padding + padding => near the padded-box edge region
print("pixel just inside padded edge:", edge.tolist())
# no boxes -> no-op
empty, = segs_mod.SAM3BoxesToSEGS().convert([], 2.0, 16, img)
same, = det.detail(512.0, True, 1024.0, 1, 15, 3.0, "euler", "simple", 0.4, 30, True, True, 1, 30, False, False,
                   "m.safetensors", "[]", img, empty, FakeVAE(), None, None)
assert torch.equal(same, img)
# force_inpaint False + big box -> skipped
big, = segs_mod.SAM3BoxesToSEGS().convert([[{"x": 0, "y": 0, "width": 700, "height": 700}]], 1.0, 0, img)
n = len(calls)
det.detail(512.0, True, 1024.0, 1, 15, 3.0, "euler", "simple", 0.4, 30, True, False, 1, 30, False, False,
           "m.safetensors", "[]", img, big, FakeVAE(), None, None)
assert len(calls) == n, "should skip when upscale<=1 and not force_inpaint"
print("ALL OK")

# ---- SegsToCombinedMask
cm, = segs_mod.SegsToCombinedMask().combine(segs)
assert cm.shape == (1, 1344, 768) and cm.dtype == torch.float32
assert cm[0, 350, 100] == 1 and cm[0, 900, 400] == 0 and cm[0, 500, 600] == 1
print("combined mask sum:", int(cm.sum()), "== union of seg masks")

# ---- batch > 1: N candidates from one pass, original untouched, all items repaired
det = segs_mod.RemoteSEGSDetailer()
outb, = det.detail(512.0, True, 1024.0, 1, 15, 3.0, "euler", "simple", 0.4, 30, True, True, 1, 30, False, False,
                   "m.safetensors", "[]", img, segs, FakeVAE(), None, None, batch=4)
assert outb.shape == (4, 1344, 768, 3), outb.shape
assert torch.allclose(outb[3, cy, cx], torch.tensor([1.0, 0, 0]), atol=1e-3)
assert torch.allclose(outb[2, 900, 400], torch.tensor([0.5, 0.5, 0.5]))
print("batch=4 ok, sampler latent batch seen:", calls[-1]["shape"])
