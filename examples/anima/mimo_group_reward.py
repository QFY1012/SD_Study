"""MiMo-style two-stage visual verifier: binary single-image gates + survivors-only group pick.

Adapts XiaomiMiMo/verl (recipes/design) to character/anatomy RL:
  1. Stage 1 (`audit_single`): Parallel 5-gate binary audit across all images
     (limbs L, spatial digits D, paw pads P, @cheese identity/nude I, broad prompt
     Q). Any gate failure is an outright 0.000 veto.
  2. Stage 2 (`pick_round`): Only groups with >= 2 [OK] survivors enter a
     balanced Williams latin-square visual pick strictly among those survivors
     (`pick_norm` with three-stage denoising), mapping clean rewards into
     [0.60, 1.00] (neutral/sole survivor = 0.800) so passing images always keep
     positive contrast above 0.000 vetoes.
  3. Fault isolation: errored/missing samples receive the mean of valid rewards
     in their group so their GRPO advantage is strictly 0.0.

Reads API key from environment variables (VERIFIER_API_KEY / NEW_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY); never hardcodes secrets.
"""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
import os
from pathlib import Path
import random
import re
import threading
import time
from typing import Any
import urllib.error
import urllib.request

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - optional Pillow dependency
    Image = None
    ImageDraw = None
    ImageFont = None

_IMG_CACHE: dict[tuple[str, int, int], dict] = {}
_IMG_LOCK = threading.Lock()


class SharedCooldownGate:
    """Process-wide pause gate so HTTP 429/503 backs off all worker threads together."""

    def __init__(self, max_pause_s: float = 30.0):
        self.max_pause_s = max_pause_s
        self._until = 0.0
        self._lock = threading.Lock()

    def wait_if_paused(self, deadline: float | None = None) -> bool:
        with self._lock:
            remaining = self._until - time.time()
        if remaining <= 0:
            return True
        if deadline is not None and time.time() + remaining >= deadline:
            return False
        time.sleep(remaining)
        return True

    def trigger(self, seconds: float) -> None:
        pause = max(0.5, min(self.max_pause_s, seconds))
        with self._lock:
            self._until = max(self._until, time.time() + pause)


_PAUSE_GATE = SharedCooldownGate()


def b64_image_block(image_path: str | Path, max_dim: int = 768, quality: int = 88) -> dict:
    """Encode and cache a resized JPEG (or raw fallback) as an OpenAI image_url block."""
    p = Path(image_path)
    key = (str(p.resolve()), max_dim, quality)
    with _IMG_LOCK:
        cached = _IMG_CACHE.get(key)
    if cached is not None:
        return cached

    if Image is not None and max_dim > 0:
        with Image.open(p) as im:
            im = im.convert("RGB")
            w, h = im.size
            if max(w, h) > max_dim:
                scale = max_dim / float(max(w, h))
                im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=quality)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            block = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
    else:
        ext = p.suffix.lower().lstrip(".") or "png"
        mime = "jpeg" if ext == "jpg" else ext
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        block = {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}}

    with _IMG_LOCK:
        _IMG_CACHE[key] = block
    return block


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first valid JSON object from model output, handling markdown fences."""
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text or "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_single_audit(text: str) -> dict[str, Any] | None:
    """Parse structured binary single-image gate JSON verdict."""
    d = extract_json_object(text)
    if d is None or not isinstance(d.get("prompt_ok"), bool):
        return None
    return {
        "prompt_ok": bool(d["prompt_ok"]),
        "severed_or_extra_limbs": bool(d.get("severed_or_extra_limbs", False)),
        "digit_error": bool(d.get("digit_error", False)),
        "paw_pad_error": bool(d.get("paw_pad_error", False)),
        "identity_or_clothes_error": bool(d.get("identity_or_clothes_error", False)),
        "reason": str(d.get("reason", ""))[:240],
        "observed_digits": str(d.get("hands_and_feet") or d.get("observed_digits", ""))[:200],
        "observed_pads": str(d.get("pads") or d.get("observed_pads", ""))[:200],
    }


def parse_pick(text: str, n: int | None = None) -> dict[str, Any] | None:
    """Parse group pick JSON; filter out-of-range indices rather than discarding valid info."""
    d = extract_json_object(text)
    if d is None:
        return None

    def _ints(raw_list: Any) -> list[int]:
        out: list[int] = []
        if not isinstance(raw_list, list):
            return out
        for item in raw_list:
            if isinstance(item, bool):
                continue
            try:
                val = int(item)
            except (TypeError, ValueError):
                continue
            if n is None or 0 <= val < n:
                out.append(val)
        return out

    return {
        "good": _ints(d.get("good")),
        "bad": _ints(d.get("bad")),
        "no_clear_difference": bool(d.get("no_clear_difference", False)),
        "reason": str(d.get("reason", ""))[:240],
    }


def williams_orders(n: int, rounds: int | None = None, seed: int = 0) -> list[list[int]]:
    """Construct balanced permutation orders for n candidates.

    When n is even and rounds == n, uses a row-complete Williams latin square so
    every candidate appears at each position once and every ordered pair once.
    Otherwise builds cyclical Latin shifts with an interleaved random shuffle.
    """
    if n <= 0:
        return []
    r_count = n if rounds is None else max(1, rounds)
    perm = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(perm)

    if n % 2 == 0 and r_count == n:
        base = [0]
        lo, hi = 1, n - 1
        while len(base) < n:
            base.append(hi)
            hi -= 1
            if len(base) < n:
                base.append(lo)
                lo += 1
        return [[perm[(x + r) % n] for x in base] for r in range(n)]

    orders: list[list[int]] = []
    for r in range(r_count):
        if r < n:
            orders.append([perm[(i + r) % n] for i in range(n)])
        else:
            orders.append(rng.sample(range(n), n))
    return orders


def combine_votes(
    votes: list[list[str]],
    n: int,
    *,
    min_ok: int | None = None,
    ncd_zero_min: int | None = None,
    net_deadzone: int = 1,
) -> dict[str, Any]:
    """Aggregate G/B/-/N vote rows with MiMo three-stage denoising."""
    r_ok = len(votes)
    req_ok = max(2, math.ceil(n * 0.625)) if min_ok is None else min_ok
    req_ncd = (r_ok // 2 + 1) if ncd_zero_min is None else ncd_zero_min
    ok = r_ok >= req_ok
    if r_ok == 0:
        return {
            "ok": ok,
            "pick_sum": [0] * n,
            "pick_norm": [0.0] * n,
            "rounds": 0,
            "ncd_zeroed": False,
            "n_ncd": 0,
            "noise_zeroed": [],
        }

    n_ncd = sum(1 for v in votes if v and v[0] == "N")
    ncd_zeroed = req_ncd > 0 and n_ncd >= req_ncd
    pick_sum = [sum(1 if v[i] == "G" else -1 if v[i] == "B" else 0 for v in votes) for i in range(n)]
    noise_zeroed = [i for i in range(n) if pick_sum[i] != 0 and abs(pick_sum[i]) <= net_deadzone]
    eff = [0 if (ncd_zeroed or i in noise_zeroed) else pick_sum[i] for i in range(n)]
    pick_norm = [s / float(r_ok) for s in eff] if ok else [0.0] * n
    return {
        "ok": ok,
        "pick_sum": pick_sum,
        "pick_norm": pick_norm,
        "rounds": r_ok,
        "ncd_zeroed": ncd_zeroed,
        "n_ncd": n_ncd,
        "noise_zeroed": noise_zeroed,
    }


def is_audit_ok(audit: dict[str, Any] | None) -> tuple[bool, str | None]:
    """Return (passed_all_gates, veto_reason_or_none)."""
    if audit is None:
        return False, "missing_audit"
    for flag in ("severed_or_extra_limbs", "digit_error", "paw_pad_error", "identity_or_clothes_error"):
        if audit.get(flag):
            return False, f"veto:{flag}"
    if not audit.get("prompt_ok", False):
        return False, "veto:prompt_mismatch"
    return True, None


def compute_sample_reward(pick_norm: float | None, audit: dict[str, Any] | None) -> tuple[float | None, str | None]:
    """Five-gate binary veto (0.0 on any failure); clean survivors map pick_norm in [-1, 1] -> [0.60, 1.00]."""
    if audit is None or pick_norm is None:
        return None, "missing_pick_or_audit"
    ok, veto_reason = is_audit_ok(audit)
    if not ok:
        return 0.0, veto_reason
    clamped = max(-1.0, min(1.0, float(pick_norm)))
    reward = round(0.80 + 0.20 * clamped, 6)
    return reward, None


def fill_invalid_with_group_mean(rewards: list[float | None], fallback: float = 0.5) -> list[float]:
    """Replace None rewards with valid group mean so invalid rows get zero GRPO gradient."""
    valid = [r for r in rewards if r is not None]
    mean = sum(valid) / len(valid) if valid else fallback
    return [mean if r is None else float(r) for r in rewards]


def post_chat_json(
    api_key: str,
    base_url: str,
    body: dict[str, Any],
    *,
    timeout: float = 120.0,
    max_attempts: int = 3,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Call OpenAI-compatible chat endpoint with shared 429/503 pause gate & deadline."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = json.dumps(body).encode("utf-8")
    last_err: Exception | None = None

    for attempt in range(max_attempts):
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError("Group reward budget expired before API attempt")
        if not _PAUSE_GATE.wait_if_paused(deadline=deadline):
            raise TimeoutError("Group reward budget expired during rate-limit pause")

        req_timeout = timeout if deadline is None else max(5.0, min(timeout, deadline - time.time()))
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "OpenAI/Python",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=req_timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as err:
            last_err = err
            if err.code in (429, 500, 502, 503, 504):
                _PAUSE_GATE.trigger(2.0 * (attempt + 1) + random.uniform(0.2, 1.0))
            elif 400 <= err.code < 500:
                raise
        except Exception as err:  # noqa: BLE001
            last_err = err
            if attempt < max_attempts - 1:
                time.sleep(min(4.0, 1.0 * (attempt + 1)))

    assert last_err is not None
    raise last_err


class MiMoAnimaGroupGrader:
    """Two-track parallel VLM grader modeled on XiaomiMiMo/verl `design_grader`."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-6-astra",
        base_url: str = "https://openoneapi.com",
        reference_images: list[str] | None = None,
        single_thinking: str = "enabled",
        pick_thinking: str = "enabled",
        reasoning_effort: str = "high",
        pick_reasoning_effort: str = "medium",
        api_style: str = "auto",
        skip_pick: bool = False,
        max_dim: int = 1024,
        pick_max_dim: int = 896,
        rounds: int | None = None,
        seed: int = 42,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.reference_images = [str(p) for p in (reference_images or [])]
        self.single_thinking = single_thinking
        self.pick_thinking = pick_thinking
        self.reasoning_effort = reasoning_effort
        self.pick_reasoning_effort = pick_reasoning_effort
        if api_style == "auto":
            is_ds = "deepseek" in model.lower() or "deepseek.com" in base_url.lower()
            self.api_style = "deepseek" if is_ds else "openai"
        else:
            self.api_style = api_style
        self.skip_pick = skip_pick
        self.max_dim = max_dim
        self.pick_max_dim = pick_max_dim
        self.rounds = rounds
        self.seed = seed

    def _ref_blocks(self) -> list[dict]:
        blocks: list[dict] = []
        for idx, rp in enumerate(self.reference_images, 1):
            stem = Path(rp).stem.lower()
            if "hand_open" in stem:
                label = "CLOSE-UP canonical open 5-finger palm: cohesive 3-lobe inverted-heart central pad + 5 oval fingertip pads"
            elif "hand_cupped" in stem:
                label = "CLOSE-UP naturally cupped 5-finger palm: 5 fingers & 3-lobe central pad intact under perspective curl"
            elif "sole" in stem or "foot" in stem:
                label = "CLOSE-UP canonical 4-toe feline paw sole: 4 yellow toe pads + clean 3-lobe sole pad, no extra heel blobs"
            else:
                label = "FULL-BODY canonical @cheese identity: blue striped fur, blonde tuft, white chest fluff, red 4-point belly star"
            blocks.extend(
                [
                    {"type": "text", "text": f"Reference {idx} ({label}):"},
                    b64_image_block(rp, max_dim=512),
                ]
            )
        return blocks

    def _thinking_kwargs(self, mode: str, effort: str | None = None) -> dict[str, Any]:
        eff = effort or self.reasoning_effort
        if self.api_style == "deepseek":
            if mode == "enabled":
                return {"thinking": {"type": "enabled"}, "reasoning_effort": eff}
            return {"thinking": {"type": "disabled"}}
        if mode == "enabled":
            return {"reasoning_effort": eff}
        return {"reasoning_effort": "low"}

    def audit_single(self, prompt_text: str, image_path: str, *, deadline: float | None = None) -> dict[str, Any]:
        ref_note = (
            "Compare Candidate strictly against the canonical Reference image(s) for paw-pad shape and @cheese appearance.\n"
            if self.reference_images
            else ""
        )
        instruction = (
            "You are a strict anime & furry character anatomy verifier inspecting ONE image of @cheese.\n"
            f"{ref_note}"
            f"Prompt: {prompt_text}\n\n"
            "Inspect EVERY visible hand, foot, paw pad, limb connection, and character detail carefully before judging:\n"
            "- `severed_or_extra_limbs`: Extra, severed, detached, or dislocated limbs/paws, an arm melting into a stump without a hand, or stray hands intruding into a legs/feet close-up.\n"
            "- `digit_error`: Wrong digit counts or fused/mangled digits (visible unobstructed hands must have 5 fingers, feet 4 toes; allow a digit only when clearly hidden behind a solid held object, never excuse an open 4-finger hand or melted claws; on side-angled or overlapping feet, flag 5 toes only when 5 distinct separated toe tips/claws are clearly visible rather than counting an outer contour shadow).\n"
            "- `paw_pad_error`: Compared with the canonical Reference close-ups, inspect central pad shape carefully and flag true for: (1) a central pad distorted into twin parallel bean/peanut halves, a pinched butterfly/X-shape with a deep center split, a 4-lobe clover, or a boxy/ragged slab cut by strings (allow a connected 3-lobe inverted-heart pad mildly elongated or curved by palm perspective, with normal shading lines inside the pad); (2) detached extra heel/wrist yellow blobs; (3) yellow spots on the back/knuckles of a hand or toe (do not confuse underside tip pads peeking beneath claws pressing a surface); or (4) straight, flat, camera-facing palm/sole digits clearly missing yellow tip pads (do not flag curled fingers whose pad face turns inward).\n"
            "- `identity_or_clothes_error`: Wrong fur/hair colors, missing stripes/cheek marks, unwanted clothes when nude, or missing red 4-point star when the lower-right abdomen itself is unobstructed and clearly visible (never mistake @cheese's small pink upper-chest nipples for a malformed belly star; allow white belly fur width variation from chest-only fluff to a full white torso stripe).\n"
            "- `prompt_ok`: True if the core scene and main action match broadly; minor framing/pose variations (1 hand vs 2, 1 sole vs 2, domed dough) do not fail `prompt_ok`.\n\n"
            "Respond with JSON ONLY in this exact field order (no markdown):\n"
            "{\n"
            '  "hands_and_feet": "brief factual finger/toe count & condition of each visible hand and foot",\n'
            '  "pads": "central pad overall shape (connected 3-lobe heart vs split twin-peanut/butterfly/4-lobe/boxy slab), tip pads, dorsal spots or heel blobs",\n'
            '  "severed_or_extra_limbs": false,\n'
            '  "digit_error": false,\n'
            '  "paw_pad_error": false,\n'
            '  "identity_or_clothes_error": false,\n'
            '  "prompt_ok": true,\n'
            '  "reason": "concise verdict under 180 chars"\n'
            "}"
        )
        content = [{"type": "text", "text": instruction}] + self._ref_blocks() + [
            {"type": "text", "text": "Candidate:"},
            b64_image_block(image_path, max_dim=self.max_dim),
        ]
        temp = 1.0 if (self.api_style == "deepseek" and self.single_thinking == "enabled") else 0.0
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": temp,
            **self._thinking_kwargs(self.single_thinking),
        }
        if self.single_thinking == "disabled":
            body["max_tokens"] = 1024
        last_err = "unparseable single audit JSON"
        for _ in range(2):
            try:
                resp = post_chat_json(self.api_key, self.base_url, body, deadline=deadline)
                parsed = parse_single_audit(resp["choices"][0]["message"].get("content") or "")
                if parsed is not None:
                    return parsed
            except Exception as err:  # noqa: BLE001
                last_err = str(err)[:160]
        return {"error": last_err}

    def pick_round(
        self,
        prompt_text: str,
        image_paths: list[str],
        order: list[int],
        *,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        n = len(image_paths)
        rubric = (
            f"You are comparing {n} sibling Candidate images (0 to {n - 1}, shuffled order) generated for the same prompt.\n"
            f"Task Prompt: {prompt_text}\n\n"
            "Pick ONLY images that are CLEARLY better or CLEARLY worse at a glance; do NOT force a full ranking when images look similar.\n"
            "Visual criteria (compare against canonical Reference paw-pad/character style when provided):\n"
            "- Clearly WORSE (`bad`): severed/extra limbs, wrong finger/toe counts (target 5 fingers / 4 toes), fused digits, "
            "malformed/missing fingertip pads or bloated/floating central paw pad, wrong character features/clothes, or missed pose/camera.\n"
            "- Clearly BETTER (`good`): clean anatomy, accurate 5-finger hands & 4-toe soles, clean inverted-heart central pad + oval tip pads on visible digits, strong @cheese fidelity & pose.\n"
            "- If no meaningful visual/anatomical gap separates candidates, set `no_clear_difference` to true with empty `good` and `bad`.\n\n"
            "Respond with JSON ONLY (no markdown):\n"
            "{\n"
            '  "reason": "short comparison <= 180 chars",\n'
            '  "good": [],\n'
            '  "bad": [],\n'
            '  "no_clear_difference": false\n'
            "}"
        )
        content: list[dict] = self._ref_blocks()
        for disp_idx, orig_idx in enumerate(order):
            content.append({"type": "text", "text": f"[Candidate {disp_idx}]"})
            content.append(b64_image_block(image_paths[orig_idx], max_dim=self.pick_max_dim))
        content.append({"type": "text", "text": rubric})

        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.4 if self.pick_thinking == "disabled" else 1.0,
            **self._thinking_kwargs(self.pick_thinking, self.pick_reasoning_effort),
        }
        if self.pick_thinking == "disabled":
            body["max_tokens"] = 1024
        try:
            resp = post_chat_json(self.api_key, self.base_url, body, deadline=deadline)
            parsed = parse_pick(resp["choices"][0]["message"].get("content") or "", n=n)
            if parsed is None:
                return {"good": [], "bad": [], "no_clear_difference": False, "error": "unparseable pick JSON", "order": order}
        except Exception as err:  # noqa: BLE001
            return {"good": [], "bad": [], "no_clear_difference": False, "error": str(err)[:160], "order": order}

        good = sorted({order[k] for k in parsed["good"]})
        bad = [i for i in sorted({order[k] for k in parsed["bad"]}) if i not in good]
        ncd = bool(parsed["no_clear_difference"] and not good and not bad)
        return {"good": good, "bad": bad, "no_clear_difference": ncd, "reason": parsed["reason"], "order": order}

    def grade_groups(
        self,
        samples: list[dict[str, Any]],
        *,
        max_workers: int = 24,
        budget_s: float = 420.0,
    ) -> dict[str, Any]:
        """Stage 1: parallel binary single-image gates; Stage 2: pick strictly among >=2 [OK] survivors."""
        grouped: dict[str, list[int]] = defaultdict(list)
        for idx, s in enumerate(samples):
            gid = str(s.get("group_id") or s["prompt"])
            grouped[gid].append(idx)

        deadline = time.time() + budget_s if budget_s > 0 else None
        single_jobs = [(i, s["prompt"], s["image"]) for i, s in enumerate(samples)]
        audits: dict[int, dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(single_jobs)))) as pool:
            single_futs = {
                pool.submit(self.audit_single, p_txt, img_p, deadline=deadline): s_idx
                for s_idx, p_txt, img_p in single_jobs
            }
            for fut, s_idx in single_futs.items():
                try:
                    audits[s_idx] = fut.result()
                except Exception as err:  # noqa: BLE001
                    audits[s_idx] = {"error": str(err)[:160]}

        survivors_by_group: dict[str, list[int]] = {}
        pick_plans: dict[str, list[list[int]]] = {}
        pick_jobs: list[tuple[str, int, str, list[str], list[int]]] = []

        for gid, idxs in grouped.items():
            survivors = [
                local_i
                for local_i, global_i in enumerate(idxs)
                if "error" not in audits[global_i] and is_audit_ok(audits[global_i])[0]
            ]
            survivors_by_group[gid] = survivors
            k = len(survivors)
            if not self.skip_pick and k >= 2:
                r_cnt = k if self.rounds is None else self.rounds
                seed0 = self.seed + int(samples[idxs[0]].get("seed", 0) or 0)
                orders = williams_orders(k, r_cnt, seed=seed0)
                pick_plans[gid] = orders
                s_paths = [samples[idxs[li]]["image"] for li in survivors]
                prompt_text = samples[idxs[0]]["prompt"]
                for r_idx, order in enumerate(orders):
                    pick_jobs.append((gid, r_idx, prompt_text, s_paths, order))
            else:
                pick_plans[gid] = []

        picks_by_group: dict[str, list[dict[str, Any]]] = {gid: [{} for _ in orders] for gid, orders in pick_plans.items()}
        if pick_jobs:
            with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(pick_jobs)))) as pool:
                pick_futs = {
                    pool.submit(self.pick_round, p_txt, s_paths, order, deadline=deadline): (gid, r_idx)
                    for gid, r_idx, p_txt, s_paths, order in pick_jobs
                }
                for fut, (gid, r_idx) in pick_futs.items():
                    try:
                        picks_by_group[gid][r_idx] = fut.result()
                    except Exception as err:  # noqa: BLE001
                        picks_by_group[gid][r_idx] = {"error": str(err)[:160]}

        raw_rewards: list[float | None] = [None] * len(samples)
        notes: dict[int, str | None] = {}
        group_traces: dict[str, Any] = {}

        for gid, idxs in grouped.items():
            n = len(idxs)
            survivors = survivors_by_group[gid]
            k = len(survivors)
            full_pick_sum = [0] * n
            full_pick_norm = [0.0] * n
            mapped_picks: list[dict[str, Any]] = []

            if pick_plans[gid]:
                votes: list[list[str]] = []
                for p_res in picks_by_group[gid]:
                    if p_res.get("error"):
                        mapped_picks.append(p_res)
                        continue
                    g_orig = [survivors[si] for si in p_res.get("good", [])]
                    b_orig = [survivors[si] for si in p_res.get("bad", [])]
                    mapped_picks.append(
                        {
                            "good": g_orig,
                            "bad": b_orig,
                            "no_clear_difference": p_res.get("no_clear_difference", False),
                            "reason": p_res.get("reason", ""),
                            "order": [survivors[si] for si in p_res.get("order", [])],
                        }
                    )
                    if p_res.get("no_clear_difference"):
                        votes.append(["N"] * k)
                        continue
                    row = ["-"] * k
                    for si in p_res.get("good", []):
                        row[si] = "G"
                    for si in p_res.get("bad", []):
                        row[si] = "B"
                    votes.append(row)

                sub_agg = combine_votes(votes, k, min_ok=max(2, math.ceil(len(pick_plans[gid]) * 0.625)))
                for sub_i, orig_li in enumerate(survivors):
                    full_pick_sum[orig_li] = sub_agg["pick_sum"][sub_i]
                    full_pick_norm[orig_li] = sub_agg["pick_norm"][sub_i]
                agg = {
                    **sub_agg,
                    "pick_sum": full_pick_sum,
                    "pick_norm": full_pick_norm,
                    "survivor_indices": survivors,
                }
            else:
                agg = {
                    "ok": True,
                    "pick_sum": full_pick_sum,
                    "pick_norm": full_pick_norm,
                    "rounds": 0,
                    "ncd_zeroed": False,
                    "n_ncd": 0,
                    "noise_zeroed": [],
                    "survivor_indices": survivors,
                }

            group_raw: list[float | None] = []
            for local_i, global_i in enumerate(idxs):
                audit = None if "error" in audits[global_i] else audits[global_i]
                rw, note = compute_sample_reward(full_pick_norm[local_i], audit)
                group_raw.append(rw)
                notes[global_i] = note

            filled = fill_invalid_with_group_mean(group_raw)
            for local_i, global_i in enumerate(idxs):
                raw_rewards[global_i] = filled[local_i]

            group_traces[gid] = {
                "sample_ids": [samples[i]["id"] for i in idxs],
                "agg": agg,
                "picks": mapped_picks,
                "audits": {samples[i]["id"]: audits[i] for i in idxs},
                "reward_notes": {samples[i]["id"]: notes[i] for i in idxs},
                "rewards": {samples[i]["id"]: raw_rewards[i] for i in idxs},
            }

        scores = [
            {"id": samples[i]["id"], "reward": round(float(raw_rewards[i] if raw_rewards[i] is not None else 0.5), 6)}
            for i in range(len(samples))
        ]
        return {"scores": scores, "groups": group_traces}


def _load_report_font(candidates: list[str], size: int) -> Any:
    """Load TrueType font across Windows & Linux Colab with clean fallback."""
    if ImageFont is None:
        return None
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def render_group_report(
    samples: list[dict[str, Any]],
    group_traces: dict[str, dict[str, Any]],
    out_path: str | Path,
    *,
    title: str | None = None,
) -> Path | None:
    """Render composite contact-sheet report sorted descending within each prompt group."""
    if Image is None or ImageDraw is None:
        return None

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_order: list[str] = []
    for s in samples:
        gid = str(s.get("group_id") or s["id"])
        if gid not in by_group:
            group_order.append(gid)
        by_group[gid].append(s)

    if not group_order:
        return None

    counts = {k: 0 for k in ("digit_error", "paw_pad_error", "identity_or_clothes_error", "prompt_mismatch", "severed_or_extra_limbs")}
    ok_cnt = 0
    for gid in group_order:
        audits = group_traces.get(gid, {}).get("audits", {})
        for a in audits.values():
            bad = False
            for k in ("severed_or_extra_limbs", "digit_error", "paw_pad_error", "identity_or_clothes_error"):
                if a.get(k):
                    counts[k] += 1
                    bad = True
            if not a.get("prompt_ok", True):
                counts["prompt_mismatch"] += 1
                bad = True
            if not bad:
                ok_cnt += 1

    font_title = _load_report_font(["msyh.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "arialbd.ttf"], 21)
    font_sub = _load_report_font(["msyh.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "arial.ttf"], 14)
    font_bold = _load_report_font(["arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"], 14)
    font_small = _load_report_font(["arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"], 12)

    w, h, hdr, top_bar = 270, 360, 48, 74
    max_cols = max(len(by_group[g]) for g in group_order)
    canvas = Image.new("RGB", (max_cols * w, top_bar + len(group_order) * (h + hdr)), (248, 249, 251))
    draw = ImageDraw.Draw(canvas)

    banner_title = title or f"MiMo GRPO Summary ({len(group_order)} groups, {len(samples)} rollouts; single-gate + survivors pick)"
    draw.rectangle([0, 0, canvas.width, top_bar], fill=(28, 34, 46))
    draw.text((16, 12), banner_title, fill=(255, 255, 255), font=font_title)
    summary_line = (
        f"OK: {ok_cnt}/{len(samples)}   |   "
        f"D(digits): {counts['digit_error']}   "
        f"P(pads): {counts['paw_pad_error']}   "
        f"I(identity): {counts['identity_or_clothes_error']}   "
        f"Q(prompt): {counts['prompt_mismatch']}   "
        f"L(limbs): {counts['severed_or_extra_limbs']}"
    )
    draw.text((16, 44), summary_line, fill=(206, 224, 248), font=font_sub)

    resampling_enum = getattr(Image, "Resampling", None)
    resample = resampling_enum.LANCZOS if resampling_enum is not None else Image.LANCZOS
    for row_idx, gid in enumerate(group_order):
        trace = group_traces.get(gid, {})
        pick_sums = trace.get("agg", {}).get("pick_sum", [])
        rewards = trace.get("rewards", {})
        audits = trace.get("audits", {})
        items = by_group[gid]
        ranked = sorted(
            [
                {
                    "local_i": k,
                    "sample": items[k],
                    "score": float(rewards.get(items[k]["id"], 0.0) or 0.0),
                    "pick_sum": int(pick_sums[k]) if k < len(pick_sums) else 0,
                    "audit": audits.get(items[k]["id"], {}),
                }
                for k in range(len(items))
            ],
            key=lambda x: (x["score"], x["pick_sum"]),
            reverse=True,
        )
        y0 = top_bar + row_idx * (h + hdr)
        for col_idx, cand in enumerate(ranked):
            x0 = col_idx * w
            img_p = cand["sample"].get("image", "")
            if img_p and Path(img_p).is_file():
                im = Image.open(img_p).convert("RGB").resize((w, h), resample)
                canvas.paste(im, (x0, y0 + hdr))
            a = cand["audit"]
            if "error" in a:
                flags = "ERR"
            else:
                flags = "".join(
                    ch
                    for flg, ch in [
                        (a.get("severed_or_extra_limbs"), "L"),
                        (a.get("digit_error"), "D"),
                        (a.get("paw_pad_error"), "P"),
                        (a.get("identity_or_clothes_error"), "I"),
                        (not a.get("prompt_ok", True), "Q"),
                    ]
                    if flg
                ) or "OK"
            is_ok = flags == "OK"
            bg = (205, 242, 214) if is_ok else (249, 224, 226)
            draw.rectangle([x0, y0, x0 + w, y0 + hdr], fill=bg)
            draw.rectangle([x0, y0, x0 + w - 1, y0 + h + hdr - 1], outline=(210, 214, 220), width=1)
            short_gid = gid[-6:] if len(gid) > 6 else gid
            line1 = f"{short_gid} #{col_idx + 1} r{cand['local_i']}  {cand['score']:.3f}  v={cand['pick_sum']:+d} [{flags}]"
            reason = str(a.get("reason") or a.get("error") or "").replace("\n", " ")[:40]
            draw.text((x0 + 6, y0 + 6), line1, fill=(18, 22, 28), font=font_bold)
            draw.text((x0 + 6, y0 + 27), reason, fill=(70, 75, 85), font=font_small)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def resolve_api_key(base_url: str, key_env: str | None = None) -> str:
    """Resolve API key from explicit env var or endpoint-matched fallback env vars."""
    if key_env:
        val = os.environ.get(key_env)
        if not val:
            raise RuntimeError(f"{key_env} environment variable is required")
        return val
    candidates = (
        ("DEEPSEEK_API_KEY", "VERIFIER_API_KEY")
        if "deepseek.com" in base_url.lower()
        else ("VERIFIER_API_KEY", "NEW_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY")
    )
    for name in candidates:
        val = os.environ.get(name)
        if val:
            return val
    raise RuntimeError(f"API key environment variable required (checked: {', '.join(candidates)})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request_json")
    parser.add_argument("response_json")
    parser.add_argument("--rounds", type=int, default=None, help="Pick rounds per group (default = group size).")
    parser.add_argument("--max-workers", type=int, default=24)
    parser.add_argument("--budget-s", type=float, default=420.0)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--base-url", default="https://openoneapi.com")
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--api-style", choices=["auto", "openai", "deepseek"], default="auto")
    parser.add_argument("--skip-pick", action="store_true", help="Run only single-image binary gates without group comparisons.")
    parser.add_argument("--single-thinking", choices=["disabled", "enabled"], default="enabled")
    parser.add_argument("--pick-thinking", choices=["disabled", "enabled"], default="enabled")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--pick-reasoning-effort", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--reference-images", nargs="*", default=[])
    parser.add_argument("--ref", action="append", default=[], help="Append a single canonical reference image path safely before positional args.")
    parser.add_argument("--report-out", default=None, help="Optional path for composite PNG report (default: <response_dir>/mimo_report.png).")
    parser.add_argument("--no-report", action="store_true", help="Disable writing the composite PNG report.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    api_key = resolve_api_key(args.base_url, args.api_key_env)
    refs = [*args.reference_images, *args.ref]

    samples = json.loads(Path(args.request_json).read_text(encoding="utf-8"))["samples"]
    grader = MiMoAnimaGroupGrader(
        api_key=api_key,
        model=args.model,
        base_url=args.base_url,
        reference_images=refs,
        single_thinking=args.single_thinking,
        pick_thinking=args.pick_thinking,
        reasoning_effort=args.reasoning_effort,
        pick_reasoning_effort=args.pick_reasoning_effort,
        api_style=args.api_style,
        skip_pick=args.skip_pick,
        rounds=args.rounds,
        seed=args.seed,
    )
    out = grader.grade_groups(samples, max_workers=args.max_workers, budget_s=args.budget_s)
    resp_path = Path(args.response_json)
    resp_path.write_text(json.dumps({"scores": out["scores"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    resp_path.with_name("mimo_trace.json").write_text(
        json.dumps(out["groups"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not args.no_report:
        report_target = Path(args.report_out) if args.report_out else resp_path.with_name("mimo_report.png")
        render_group_report(samples, out["groups"], report_target, title=f"MiMo GRPO Report - {resp_path.parent.name}")


if __name__ == "__main__":
    main()
