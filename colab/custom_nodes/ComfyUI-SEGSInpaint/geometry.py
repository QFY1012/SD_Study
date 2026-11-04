"""Pure-python box/crop math (no torch, easy to unit-test)."""


def boxes_from_input(bboxes, img_w, img_h):
    """Accepts CreateBoundingBoxes output ([[{x,y,width,height}, ...]]) or a flat list of dicts.
    Returns a list of integer (x1, y1, x2, y2) clipped to the image; empty boxes are dropped."""
    flat = []

    def walk(item):
        if isinstance(item, dict):
            flat.append(item)
        elif isinstance(item, (list, tuple)):
            for sub in item:
                walk(sub)

    walk(bboxes)
    out = []
    for b in flat:
        if not all(k in b for k in ("x", "y", "width", "height")):
            continue
        x1 = max(0, min(img_w, int(round(b["x"]))))
        y1 = max(0, min(img_h, int(round(b["y"]))))
        x2 = max(0, min(img_w, int(round(b["x"] + b["width"]))))
        y2 = max(0, min(img_h, int(round(b["y"] + b["height"]))))
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            out.append((x1, y1, x2, y2))
    return out


def expand_box(box, padding, img_w, img_h):
    x1, y1, x2, y2 = box
    return (max(0, x1 - padding), max(0, y1 - padding), min(img_w, x2 + padding), min(img_h, y2 + padding))


def crop_region_for(box, crop_factor, img_w, img_h, min_size=64):
    """Scale `box` about its centre by crop_factor and clip to the image."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w = max((x2 - x1) * crop_factor, min_size)
    h = max((y2 - y1) * crop_factor, min_size)
    rx1 = max(0, int(round(cx - w / 2.0)))
    ry1 = max(0, int(round(cy - h / 2.0)))
    rx2 = min(img_w, int(round(cx + w / 2.0)))
    ry2 = min(img_h, int(round(cy + h / 2.0)))
    return (rx1, ry1, rx2, ry2)


def target_size(ref_w, ref_h, crop_w, crop_h, guide_size, max_size, multiple=8):
    """Impact-Pack style: scale so the shorter reference side reaches guide_size, capped by max_size.
    Returns (new_w, new_h, upscale)."""
    upscale = guide_size / max(1, min(ref_w, ref_h))
    new_w, new_h = crop_w * upscale, crop_h * upscale
    longest = max(new_w, new_h)
    if longest > max_size:
        upscale *= max_size / longest
        new_w, new_h = crop_w * upscale, crop_h * upscale
    snap = lambda v: max(multiple, int(round(v / multiple)) * multiple)
    return snap(new_w), snap(new_h), upscale
