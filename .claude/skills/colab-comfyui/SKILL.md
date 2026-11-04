---
name: colab-comfyui
description: Drive a Google Colab GPU from the terminal with the `colab` CLI to run ComfyUI against the user's Drive models - Anima+LoRA txt2img for training data, and local (box) repaint of wrongly drawn hands/feet with the SDXL checkpoint. Use when the user wants images generated or repaired on their Colab, or wants a Colab runtime started/stopped from here.
---

# colab-comfyui

Runs everything on a Colab runtime that **you create and name yourself** (e.g. `claude-test`). Models stay on the user's Drive (`MyDrive/ComfyUI/models`), outputs are written to the VM and downloaded with `colab download`.

## 0. Ground rules
- Only touch sessions you created. `colab sessions` may list the user's own runtimes (marked `[?]`) - never exec/stop those.
- Stop your session when done: `colab stop -s <name>` (it burns compute units while alive, ~1.15/h on T4).
- The OAuth token (`~/.config/colab-cli/token.json`) carries `cloud-platform` + `colaboratory` + `drive.file`. Tell the user to revoke at https://myaccount.google.com/permissions when finished.
- Do not write into the user's dataset folders. New outputs go to a new folder (e.g. `MyDrive/ComfyUI/output/<project>/`) and only after the user okayed the results.

## 1. Connect (once per container)
```bash
uv tool install google-colab-cli      # Linux/macOS only, needs Python 3.13 (uv fetches it)
mkfifo in && (sleep 3000 > in &)       # keep stdin open for the interactive login
nohup colab --auth oauth2 usage < in > out.log 2>&1 &
grep -o 'https://accounts.google.com[^ ]*' out.log   # give this URL to the user
# user approves in a browser and sends back the authorization code; then:
echo '<code>' > in
```
Then `colab new -s claude-test --gpu T4`.

## 2. Mount Drive (needs a second consent per runtime)
`colab drivemount -s claude-test` prints another accounts.google.com URL and waits for Enter. Run it with a fifo like above, give the URL to the user, and send an empty line to the fifo after they say "done". Running it with `</dev/null` fails with a 400.

## 3. Bring up ComfyUI on the VM
```bash
S=.claude/skills/colab-comfyui/scripts; N=colab/custom_nodes/ComfyUI-SEGSInpaint
colab exec -s claude-test -f $S/vm_install.py          # clone + pip in background
colab exec -s claude-test -f $S/vm_status.py           # poll until "install: done"
colab upload -s claude-test $N/__init__.py /content/ComfyUI/custom_nodes/ComfyUI-SEGSInpaint/__init__.py
colab upload -s claude-test $N/geometry.py /content/ComfyUI/custom_nodes/ComfyUI-SEGSInpaint/geometry.py
colab exec -s claude-test -f $S/vm_start.py            # starts ComfyUI, checks nodes registered
```
Custom nodes are only loaded at ComfyUI start: after uploading new node code, kill ComfyUI (`pkill -f "main.py --listen"`) and run `vm_start.py` again - not while a job is running.

## 4. Operating the `colab` CLI - lessons learned
- `colab exec` gives up after roughly a minute (`TimeoutError: Timeout waiting for reply`) but the VM code keeps running. Never put long work in an exec: detach with `subprocess.Popen(..., start_new_session=True)` and poll.
- The notebook kernel is single-threaded: a long exec **blocks every later exec** until it finishes. Don't launch heavy Drive reads while you need the kernel; run them detached.
- Commands over ~2 min get auto-backgrounded by the shell harness; use `run_in_background` + a polling loop.
- First read of a multi-GB model from the Drive mount takes minutes (the 6.5 GB SDXL checkpoint took ~3.5 min; Anima DiT similar). Subsequent jobs reuse the loaded model.
- Files to the VM: `colab upload LOCAL REMOTE`; from it: `colab download REMOTE LOCAL`; `colab ls -s name PATH`.

## 5. ComfyUI API pitfalls
- A list in an API input means "link to a node". List-valued **widget** values (e.g. `CreateBoundingBoxes.editor_state`, `last_incoming`) must be sent as `{"__value__": [...]}` or validation fails with `unhashable type: 'dict'`.
- Workflows embedded in the user's result PNGs (`Image.open(f).info['prompt']`) are API format and reference `DualCLIPLoader`/`VAELoader` files that only exist on the user's local machine. `scripts/convert_inpaint_workflow.py` swaps them for a `CheckpointLoaderSimple` and wraps list widgets.
- `RemoteSEGSDetailer`, `SAM3BoxesToSEGS`, `SegsToCombinedMask` come from `colab/custom_nodes/ComfyUI-SEGSInpaint` (drop-in for the user's local plugin / Impact Pack; `url`/`token` ignored).

## 6. Anima + LoRA generation (`scripts/vm_generate.py`)
Base is **waiANIMA_v10Base10** (`diffusion_models/waiANIMA_v10Base10.safetensors`, text encoder `text_encoders/waiANIMA_v10Base10_txt.safetensors`, VAE `vae/qwenImage_qwenImageVAE.safetensors`), LoRA `loras/cheesev1.safetensors` (trained unet-only -> `LoraLoaderModelOnly`). Graph: UNETLoader -> LoraLoaderModelOnly -> KSampler(er_sde, simple, 30 steps, cfg 4.5), CLIPLoader type `stable_diffusion`, `EmptySD3LatentImage` 768x1344. Write `/content/gen_jobs.json` (`{"defaults":{...},"jobs":[{"name","positive","seed"}]}`), run `vm_generate.py`, poll `vm_status.py`, download from `/content/test_out/<prefix>/`.

The character **cheese is a cheetah** (blue fur, pale chest/belly, dark markings, red star on the belly, spiky yellow hair, yellow eyes, curled striped tail, yellow claws/pads).
Dataset captions look like: `@cheese, anime coloring, fluffy, full body, standing, looking at viewer, pawpads, nude, <background>, <pose tags>, <expression>, yellow eyes, curled tail, ..., <one natural-language sentence>`. The LoRA's baseline is the nude tag, so clothed images need `nude, nsfw` in the negative prompt. Anima prefers lowercase tags with spaces, artist-style `@` prefix, quality prefix `masterpiece, best quality, score_9, score_8, score_7`.

## 7. Repair hands/feet (`scripts/vm_repair.py`)
The LoRA draws paw pads stiff/ugly, so hands/feet are repainted with a *different* model: SDXL `xaviernocturne_v3.safetensors` through `RemoteSEGSDetailer` (guide 512, max 1024, 15 steps, cfg 3, euler/simple, denoise 0.4, feather 30) - same settings and prompts as the user's `repair_regions.py`.
Reference anatomy: **hands = 5 fingers with yellow paw pads and claws; feet = 4 toes with claws, yellow pads/claws.**
Stages chain (hands then feet); `kind` is `hands`, `feet` (script's derived prompt) or `feet_strong` (refined prompt from the two-stage PNG workflows). Optional `restore_original` resets the boxed area to the pre-repaint image first. Boxes are `[x, y, w, h]` in source-image pixels - look at the image first to choose them.

### Repair lessons (hands/feet)
- **Check palm vs back first** (zoom in; brighten dark images). Forcing pads on the back of a hand grows brown blobs on the knuckles, and `hands_back` on a palm redraws the hand reversed. Wrong guesses cost a whole batch.
- **Pad structure = feline/cheetah**: one large trilobed palm pad + one small separate oval pad per finger. Not a pile of balls. Suffix that works as a starting point:
  `positive_suffix`: `feline paw pads, cheetah paw pads, one large trilobed palm pad, four small separate oval toe beans, one pad per finger, light yellow paw pads, pale yellow, flat soft shading`
  `negative_suffix`: `cluster of round balls, bubbly palm pad, multiple blobs, lumpy pad, many bumps, orange paw pads, brown paw pads, gray paw pads, dark pads`
- denoise 0.4 barely changes structure; 0.5-0.55 is the useful range; 0.7 melts hands into glowing blobs.
- Pad **shape** is what the user's reference image gets right (it is the target): one connected, slightly elongated palm pad plus one small separate oval pad per fingertip, hands paler than the arms, small pale claws. Colour is only roughly pale yellow - the user does not want a numeric/pixel standard, so do not chase brightness values and do not recolour pixels by hand; repaint with the model (more seeds / pad prompt weight) if a batch looks off.
- Judge candidates at crop level (2x crops of the box, before/after). Thumbnails hide finger/toe counts - one "3 toes" flag turned out to be 4 toes.
- Odd dataset ids are mirrors of the even id before them: repair the even one, then mirror.

## 8. Status
Verified on a real T4: login, session creation, Drive mount, ComfyUI install/start, custom nodes registered, SDXL box-repaint workflow end to end (217 s cold, converted PNG workflow).
Not yet confirmed at the time of writing: Anima+cheesev1 generation output quality, `vm_repair.py` end to end (incl. `restore_original`).
