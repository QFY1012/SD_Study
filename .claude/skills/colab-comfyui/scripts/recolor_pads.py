#!/usr/bin/env python3
"""Lift dark/brown/olive paw-pad pixels to the dataset's standard pale yellow, inside given boxes only.

Standard (measured on the user's good images 168/170/190): median value ~0.97, hue 38-51 deg,
saturation 0.48-0.63, mean RGB ~ (244, 200-226, 96-131). Structure is untouched: only pixels that look like
pad colour (hue 25-65, sat > 0.28, value > 0.40) are moved, with a soft mask, keeping their internal shading.

usage: recolor_pads.py IN.png OUT.png --box x y w h [--box ...] [--strength 1.0] [--hue 44] [--sat 0.55]
"""
import argparse
import numpy as np
from PIL import Image, ImageFilter


def recolor(im, boxes, strength=1.0, hue_t=44.0, sat_t=0.55, val_t=0.97):
    a = np.asarray(im.convert('RGB'), dtype=np.float32) / 255
    out = a.copy()
    H, W, _ = a.shape
    for x, y, w, h in boxes:
        x2, y2 = min(W, x + w), min(H, y + h)
        sub = a[y:y2, x:x2]
        mx, mn = sub.max(-1), sub.min(-1)
        d = mx - mn + 1e-6
        r, g, b = sub[..., 0], sub[..., 1], sub[..., 2]
        hue = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
        sat = d / (mx + 1e-6)
        hard = (hue > 25) & (hue < 65) & (sat > 0.28) & (mx > 0.40)
        if hard.sum() < 30:
            print('box', (x, y, w, h), ': no pad-coloured pixels found'); continue
        soft = np.asarray(Image.fromarray((hard * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2)), dtype=np.float32) / 255
        soft = (soft * strength)[..., None]
        v_med = np.median(mx[hard])
        # keep the pad's own light/shade variation, re-centred on the standard brightness
        v_new = np.clip(val_t + (mx - v_med) * 0.8, 0.0, 1.0)
        s_new = np.clip(sat_t + (sat - np.median(sat[hard])) * 0.8, 0.15, 0.9)
        h_new = np.full_like(hue, hue_t) + (hue - np.median(hue[hard])) * 0.3
        c = v_new * s_new; hp = (h_new % 360) / 60; xx = c * (1 - np.abs(hp % 2 - 1)); m0 = v_new - c
        z = np.zeros_like(c)
        k = np.floor(hp).astype(int) % 6
        rr = np.choose(k, [c, xx, z, z, xx, c]); gg = np.choose(k, [xx, c, c, xx, z, z]); bb = np.choose(k, [z, z, xx, c, c, xx])
        new = np.stack([rr + m0, gg + m0, bb + m0], -1)
        sw = soft[..., :1]
        out[y:y2, x:x2] = sub * (1 - sw) + new * sw
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('src'); ap.add_argument('dst')
    ap.add_argument('--box', nargs=4, type=int, action='append', required=True)
    ap.add_argument('--strength', type=float, default=1.0)
    ap.add_argument('--hue', type=float, default=44.0)
    ap.add_argument('--sat', type=float, default=0.55)
    a = ap.parse_args()
    recolor(Image.open(a.src), a.box, a.strength, a.hue, a.sat).save(a.dst)
    print('saved', a.dst)
