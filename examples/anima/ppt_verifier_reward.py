"""Probabilistic Pivot Tournament (PPT) + two-stage VLM verifier reward script.

Implements Algorithm 1, Eq. 3.1, Eq. 3.2, and Appendix B.6 from
"LLM-as-a-Verifier: A General-Purpose Verification Framework" (arXiv:2607.05391)
for use as `reward_command` in `anima_grpo.train`.

Reads DEEPSEEK_API_KEY from environment variables; never hardcodes secrets.
"""
from __future__ import annotations

import argparse
import base64
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
from typing import Callable, Iterable
import urllib.request

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional fast JPEG resize fallback
    Image = None

_IMG_CACHE: dict[tuple[str, int], dict] = {}
_IMG_LOCK = threading.Lock()

LETTERS = [chr(ord("A") + i) for i in range(20)]  # A=1 .. T=20
LETTER_VAL = {ch: (i / 19.0) for i, ch in enumerate(LETTERS)}

CRITERIA_SETS: dict[str, list[tuple[str, str]]] = {
    "combined": [
        (
            "overall",
            "Evaluate (1) anatomy correctness (no extra/missing/severed limbs, coherent joints, "
            "5 fingers per visible hand and 4 toes per visible foot without fused digits), "
            "(2) yellow paw pad shape & @cheese identity (each visible palm/sole has one compact rounded/three-lobed "
            "central pad plus clean oval pads under fingertips/toes; strictly penalize detached floating crescents/arches, "
            "bloated shapeless yellow blobs, smeared random yellow spots, wrong fur, or unwanted clothes), "
            "and (3) faithful adherence to the prompt pose, camera angle, and leg/foot visibility.",
        )
    ],
    "decomposed": [
        (
            "anatomy",
            "Focus strictly on anatomy & paw-pad shape: exact 2 arms and 2 legs unless cropped, no extra disembodied limbs, "
            "5 distinct fingers per visible hand, 4 distinct toes per visible foot, no fused digits, and clean "
            "paw-pad anatomy (one compact central pad + clean oval digit pads; penalize bizarre floating/bloated/merged yellow pads).",
        ),
        (
            "identity_style",
            "Focus strictly on @cheese identity and art style: blue fur with dark stripes, spiky blond hair, "
            "red four-point star on abdomen when visible, striped blue tail, clean yellow paw pads, crisp cel shading.",
        ),
        (
            "prompt_adherence",
            "Focus strictly on prompt adherence: matching requested action, viewpoint, required visible palms/soles, "
            "and avoiding unnatural animal-hock distortion unless requested.",
        ),
    ],
}


def b64_image_block(image_path: str | Path, max_dim: int = 768) -> dict:
    """Encode and cache a resized JPEG (or fallback raw) image as an OpenAI image_url block."""
    p = Path(image_path)
    cache_key = (str(p.resolve()), max_dim)
    with _IMG_LOCK:
        cached = _IMG_CACHE.get(cache_key)
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
            im.save(buf, format="JPEG", quality=90)
            block = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"}}
    else:
        ext = p.suffix.lower().lstrip(".") or "png"
        mime = "jpeg" if ext == "jpg" else ext
        data = base64.b64encode(p.read_bytes()).decode("ascii")
        block = {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{data}"}}

    with _IMG_LOCK:
        _IMG_CACHE[cache_key] = block
    return block


def extract_letter_scores(choice: dict) -> dict[str, dict | None]:
    """Compute Eq. 3.1 continuous expectations over A..T top_logprobs inside <score_A>/<score_B>."""
    lp_items = (choice.get("logprobs") or {}).get("content") or []
    joined = ""
    spans: list[tuple[int, int, dict]] = []
    for item in lp_items:
        start = len(joined)
        joined += item["token"]
        spans.append((start, len(joined), item))

    if not joined:
        joined = choice.get("message", {}).get("content") or ""

    out: dict[str, dict | None] = {}
    for tag in ("score_A", "score_B"):
        open_tag, close_tag = f"<{tag}>", f"</{tag}>"
        p1 = joined.find(open_tag)
        p2 = joined.find(close_tag, p1 + len(open_tag)) if p1 != -1 else -1
        if p1 == -1 or p2 == -1:
            out[tag] = None
            continue
        inner_s, inner_e = p1 + len(open_tag), p2
        cand_item = next(
            (item for s, e, item in spans if s >= inner_s and e <= inner_e and item["token"].strip() in LETTER_VAL),
            None,
        )
        if cand_item is None:
            raw_tok = joined[inner_s:inner_e].strip()
            if raw_tok in LETTER_VAL:
                out[tag] = {"token": raw_tok, "exp": LETTER_VAL[raw_tok], "dist": {raw_tok: 1.0}}
            else:
                out[tag] = None
            continue

        probs: dict[str, float] = {}
        for top in cand_item.get("top_logprobs") or []:
            tok = top["token"].strip()
            if tok in LETTER_VAL and tok not in probs and top["logprob"] > -100:
                probs[tok] = math.exp(top["logprob"])
        total_p = sum(probs.values())
        tok_str = cand_item["token"].strip()
        if total_p <= 0:
            out[tag] = {"token": tok_str, "exp": LETTER_VAL[tok_str], "dist": {tok_str: 1.0}}
        else:
            norm_p = {k: v / total_p for k, v in probs.items()}
            exp_val = sum(p * LETTER_VAL[k] for k, p in norm_p.items())
            out[tag] = {"token": tok_str, "exp": exp_val, "dist": norm_p}
    return out


def bradley_terry_prob(r_i: float, r_j: float) -> float:
    """Eq. 3.2 soft pairwise preference probability sigma(R_i - R_j)."""
    return 1.0 / (1.0 + math.exp(-(r_i - r_j)))


def probabilistic_pivot_tournament(
    num_candidates: int,
    pivots: int,
    pair_scorer: Callable[[int, int], tuple[float, float]],
    seed: int = 42,
    max_workers: int = 4,
) -> dict:
    """Run Algorithm 1 (Ring Pass -> Top-k Pivot Selection -> Pivot Rounds).

    Returns normalized mean win-mass scores w_i / c_i plus trace metadata.
    """
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive")
    if num_candidates == 1:
        return {"scores": [0.5], "pivots": [0], "ring_edges": [], "pivot_edges": []}

    rng = random.Random(seed)
    w = [0.0] * num_candidates
    c = [0] * num_candidates

    perm = rng.sample(range(num_candidates), num_candidates)
    if num_candidates == 2:
        # Both slot orders neutralize positional bias when N == 2.
        ring_edges = [(perm[0], perm[1]), (perm[1], perm[0])]
    else:
        ring_edges = [(perm[t], perm[(t + 1) % num_candidates]) for t in range(num_candidates)]

    def eval_edges(edges: Iterable[tuple[int, int]]) -> list[tuple[int, int, float, float, float]]:
        edge_list = list(edges)
        if not edge_list:
            return []
        workers = min(max_workers, len(edge_list))
        if workers <= 1:
            raw = [pair_scorer(i, j) for i, j in edge_list]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                raw = list(pool.map(lambda ij: pair_scorer(ij[0], ij[1]), edge_list))
        out = []
        for (i, j), (ri, rj) in zip(edge_list, raw):
            p = bradley_terry_prob(ri, rj)
            out.append((i, j, ri, rj, p))
        return out

    ring_results = eval_edges(ring_edges)
    for i, j, _, _, p in ring_results:
        w[i] += p
        w[j] += 1.0 - p
        c[i] += 1
        c[j] += 1

    ring_means = [w[idx] / c[idx] for idx in range(num_candidates)]
    k = max(1, min(pivots, num_candidates))
    pivot_set = sorted(range(num_candidates), key=lambda idx: (-ring_means[idx], idx))[:k]
    pivot_lookup = set(pivot_set)

    seen_unordered = {frozenset((i, j)) for i, j in ring_edges}
    raw_piv_edges: list[tuple[int, int]] = []
    for i in range(num_candidates):
        if i not in pivot_lookup:
            for p_idx in pivot_set:
                if frozenset((i, p_idx)) not in seen_unordered:
                    raw_piv_edges.append((i, p_idx))
                    seen_unordered.add(frozenset((i, p_idx)))
    for a_idx in range(len(pivot_set)):
        for b_idx in range(a_idx + 1, len(pivot_set)):
            p1, p2 = pivot_set[a_idx], pivot_set[b_idx]
            if frozenset((p1, p2)) not in seen_unordered:
                raw_piv_edges.append((p1, p2))
                seen_unordered.add(frozenset((p1, p2)))

    # Alternate slot placement across pivot-round pairs to avoid pivot slot bias.
    piv_edges = [(j, i) if idx % 2 else (i, j) for idx, (i, j) in enumerate(raw_piv_edges)]
    piv_results = eval_edges(piv_edges)
    for i, j, _, _, p in piv_results:
        w[i] += p
        w[j] += 1.0 - p
        c[i] += 1
        c[j] += 1

    scores = [w[idx] / c[idx] for idx in range(num_candidates)]
    return {
        "scores": scores,
        "pivots": pivot_set,
        "ring_edges": ring_results,
        "pivot_edges": piv_results,
    }


def _post_chat(api_key: str, base_url: str, body: dict, timeout: int = 60) -> dict:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps(body).encode("utf-8")
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("Unreachable")


class DeepSeekTwoStagePairVerifier:
    """Appendix B.6 two-stage verifier on DeepSeek multimodal API."""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-flash",
        base_url: str = "https://api.deepseek.com",
        criteria: list[tuple[str, str]] | None = None,
        repeats: int = 1,
        reasoning_effort: str = "medium",
        skip_stage1: bool = False,
        stage2_with_images: bool = True,
        reference_images: list[str] | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.criteria = criteria or CRITERIA_SETS["combined"]
        self.repeats = max(1, repeats)
        self.reasoning_effort = reasoning_effort
        self.skip_stage1 = skip_stage1
        self.stage2_with_images = stage2_with_images
        self.reference_images = [str(p) for p in (reference_images or [])]
        self._analysis_cache: dict[tuple[str, str, str], str] = {}
        self._analysis_lock = threading.Lock()

    def _ref_blocks(self) -> list[dict]:
        blocks: list[dict] = []
        for idx, rp in enumerate(self.reference_images, 1):
            blocks.extend([{"type": "text", "text": f"Reference {idx} (canonical @cheese anatomy & paw pads):"}, b64_image_block(rp, max_dim=512)])
        return blocks

    def _inspect_single_image(self, prompt_text: str, img_path: str, crit_name: str, crit_desc: str) -> str:
        """Run cached single-image Stage 1 feline anatomy & paw-pad audit (O(N) vs O(N*k) pairwise)."""
        key = (prompt_text, crit_name, str(Path(img_path).resolve()))
        with self._analysis_lock:
            cached = self._analysis_cache.get(key)
        if cached is not None:
            return cached

        ref_note = (
            "Compare the Candidate strictly against the canonical @cheese Reference image(s) for paw-pad shape and character appearance.\n"
            if self.reference_images
            else ""
        )
        s1_prompt = (
            "You are an expert anime character and feline anatomy verifier auditing ONE Candidate image.\n"
            f"{ref_note}"
            f"Task Prompt: {prompt_text}\n"
            f"Criterion ({crit_name}): {crit_desc}\n"
            "Write a concise factual audit in <reasoning>...</reasoning> (under 100 words):\n"
            "1) Digits: count exact fingers on each visible hand (target=5) and toes on each visible foot/sole (target=4); flag extra/missing/fused digits or severed limbs.\n"
            "2) Feline Paw Pads: @cheese palms & soles require 1 compact inverted-heart trilobed central pad (1 upper peak toward digits, 2 lower lobes toward wrist/heel, ~1/3-1/2 palm area) + 1 clean oval yellow pad on EVERY visible fingertip/toe (all 5 tips on open palms, all 4 on soles). Flag bare palm-facing fingers missing tip pads, bloated/cloud-like multi-lobe central pads, detached floating heel crescents, or dorsal smudges.\n"
            "3) Identity & Pose: blue fur/stripes, blond hair, nude (no unwanted clothes), required palms/soles facing camera cleanly.\n"
            "Do NOT output letter scores."
        )
        content = [{"type": "text", "text": s1_prompt}] + self._ref_blocks() + [{"type": "text", "text": "Candidate:"}, b64_image_block(img_path)]
        r1 = _post_chat(
            self.api_key,
            self.base_url,
            {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 1.0,
                "reasoning_effort": self.reasoning_effort,
                "thinking": {"type": "enabled"},
            },
        )
        analysis = (r1["choices"][0]["message"].get("content") or "").strip()
        with self._analysis_lock:
            self._analysis_cache[key] = analysis
        return analysis

    def precompute_analyses(self, prompt_text: str, image_paths: Iterable[str], max_workers: int = 16) -> None:
        """Pre-audit all unique candidate images in one parallel wave before tournament pairs."""
        if self.skip_stage1:
            return
        unique_paths = list(dict.fromkeys(str(p) for p in image_paths))
        jobs = [(p, cname, cdesc) for p in unique_paths for cname, cdesc in self.criteria]
        if not jobs:
            return
        workers = max(1, min(max_workers, len(jobs)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda j: self._inspect_single_image(prompt_text, j[0], j[1], j[2]), jobs))

    def _score_criterion(self, prompt_text: str, img_a: str, img_b: str, crit_name: str, crit_desc: str) -> tuple[float, float]:
        blk_a = b64_image_block(img_a)
        blk_b = b64_image_block(img_b)

        analysis = ""
        if not self.skip_stage1:
            # Reuse O(N) cached single-image audits (compute both concurrently on cache miss).
            key_a = (prompt_text, crit_name, str(Path(img_a).resolve()))
            key_b = (prompt_text, crit_name, str(Path(img_b).resolve()))
            with self._analysis_lock:
                have_both = key_a in self._analysis_cache and key_b in self._analysis_cache
            if have_both:
                ana_a = self._analysis_cache[key_a]
                ana_b = self._analysis_cache[key_b]
            else:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    fa = pool.submit(self._inspect_single_image, prompt_text, img_a, crit_name, crit_desc)
                    fb = pool.submit(self._inspect_single_image, prompt_text, img_b, crit_name, crit_desc)
                    ana_a, ana_b = fa.result(), fb.result()
            analysis = f"[Image A Audit]\n{ana_a}\n\n[Image B Audit]\n{ana_b}"

        s2_header = (
            f"Task Prompt: {prompt_text}\n"
            f"Evaluation Criterion ({crit_name}): {crit_desc}\n"
        )
        if analysis:
            s2_header += f"Visual Inspection Analysis:\n{analysis}\n\n"
        s2_header += (
            "Rate Image A and Image B independently using ONE capital letter from A to T on a 1-20 scale:\n"
            "- A-E (1-5): severe anatomy or pad deformation (extra/severed limbs, human skin, wrong fur, detached/floating/bloated deformed yellow paw pads, broken pose)\n"
            "- F-K (6-11): clear digit count error (3 toes vs 4, 4/6 fingers vs 5, fused digits), hand/foot pad defects (bare palm-facing fingers missing fingertip pads, bloated/misshapen palm pad), or missing required visible palm/sole\n"
            "- L-N (12-14): passable digits, pads, and pose with minor overlap/framing flaws\n"
            "- O-T (15-20): clean 5-finger hands with pads on all 5 fingertips + compact 3-lobed palm pad, clean 4-toe soles, accurate @cheese identity, faithful pose.\n"
            "Output ONLY:\n<score_A>LETTER</score_A>\n<score_B>LETTER</score_B>"
        )

        s2_content: list[dict] = [{"type": "text", "text": s2_header}]
        if self.stage2_with_images or self.skip_stage1:
            if self.reference_images:
                s2_content.extend(self._ref_blocks())
            s2_content.extend([{"type": "text", "text": "Image A:"}, blk_a, {"type": "text", "text": "Image B:"}, blk_b])

        r2 = _post_chat(
            self.api_key,
            self.base_url,
            {
                "model": self.model,
                "messages": [{"role": "user", "content": s2_content}],
                "max_tokens": 32,
                "temperature": 1.0,
                "logprobs": True,
                "top_logprobs": 20,
                "thinking": {"type": "disabled"},
            },
        )
        parsed = extract_letter_scores(r2["choices"][0])
        if not parsed.get("score_A") or not parsed.get("score_B"):
            raw = r2["choices"][0]["message"].get("content") or ""
            m_a = re.search(r"<score_A>\s*([A-T])\s*</score_A>", raw)
            m_b = re.search(r"<score_B>\s*([A-T])\s*</score_B>", raw)
            if m_a and m_b:
                return LETTER_VAL[m_a.group(1)], LETTER_VAL[m_b.group(1)]
            return 0.5, 0.5
        return float(parsed["score_A"]["exp"]), float(parsed["score_B"]["exp"])

    def score_pair(self, prompt_text: str, img_a: str, img_b: str) -> tuple[float, float]:
        tot_a, tot_b = 0.0, 0.0
        n = len(self.criteria) * self.repeats
        for crit_name, crit_desc in self.criteria:
            for _ in range(self.repeats):
                ra, rb = self._score_criterion(prompt_text, img_a, img_b, crit_name, crit_desc)
                tot_a += ra
                tot_b += rb
        return tot_a / n, tot_b / n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request_json")
    parser.add_argument("response_json")
    parser.add_argument("--pivots", type=int, default=2)
    parser.add_argument("--criteria", choices=sorted(CRITERIA_SETS), default="combined")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--no-stage2-images", action="store_true")
    parser.add_argument("--reference-images", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY environment variable is required")

    req_path = Path(args.request_json)
    resp_path = Path(args.response_json)
    samples = json.loads(req_path.read_text(encoding="utf-8"))["samples"]

    verifier = DeepSeekTwoStagePairVerifier(
        api_key=api_key,
        model=args.model,
        base_url=args.base_url,
        criteria=CRITERIA_SETS[args.criteria],
        repeats=args.repeats,
        reasoning_effort=args.reasoning_effort,
        skip_stage1=args.skip_stage1,
        stage2_with_images=not args.no_stage2_images,
        reference_images=args.reference_images,
    )

    if not args.skip_stage1:
        verifier.precompute_analyses(samples[0]["prompt"], [s["image"] for s in samples], max_workers=args.max_workers)

    group_seed = args.seed + int(samples[0].get("seed", 0) or 0)
    ppt = probabilistic_pivot_tournament(
        num_candidates=len(samples),
        pivots=args.pivots,
        pair_scorer=lambda i, j: verifier.score_pair(samples[i]["prompt"], samples[i]["image"], samples[j]["image"]),
        seed=group_seed,
        max_workers=args.max_workers,
    )

    scores = [{"id": samples[idx]["id"], "reward": round(float(ppt["scores"][idx]), 6)} for idx in range(len(samples))]
    resp_path.write_text(json.dumps({"scores": scores}, ensure_ascii=False, indent=2), encoding="utf-8")

    trace_path = resp_path.with_name("ppt_trace.json")
    trace_path.write_text(
        json.dumps(
            {
                "sample_ids": [s["id"] for s in samples],
                "pivots": [samples[i]["id"] for i in ppt["pivots"]],
                "ring_edges": [(samples[i]["id"], samples[j]["id"], ri, rj, p) for i, j, ri, rj, p in ppt["ring_edges"]],
                "pivot_edges": [(samples[i]["id"], samples[j]["id"], ri, rj, p) for i, j, ri, rj, p in ppt["pivot_edges"]],
                "scores": scores,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
