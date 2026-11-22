"""Build a self-contained HTML pair picker for curating offline Anima Flow-DPO (win, lose) pairs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any


def _gate_flags(audit: dict[str, Any] | None) -> str:
    if not audit:
        return ""
    if "error" in audit:
        return "ERR"
    flags = "".join(
        ch
        for flag, ch in (
            ("severed_or_extra_limbs", "L"),
            ("digit_error", "D"),
            ("paw_pad_error", "P"),
            ("identity_or_clothes_error", "I"),
        )
        if audit.get(flag)
    )
    if not audit.get("prompt_ok", True):
        flags += "Q"
    return flags or "OK"


def collect_groups_from_requests(paths: list[Path], *, prompts_jsonl: Path | None = None) -> list[dict[str, Any]]:
    """Collect prompt-grouped image candidates from req/manifest files or rollout directories."""
    prompt_lookup: dict[str, str] = {}
    if prompts_jsonl and prompts_jsonl.is_file():
        for line in prompts_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                prompt_lookup[str(row["id"])] = str(row["prompt"])

    groups_map: dict[str, dict[str, Any]] = {}
    seen_images: set[str] = set()

    def _add_candidate(group_key: str, prompt: str, img_path: Path, *, sid: str = "", hint: str = "", reason: str = "") -> None:
        resolved = str(img_path.resolve())
        if resolved in seen_images or not img_path.is_file():
            return
        seen_images.add(resolved)
        grp = groups_map.setdefault(
            group_key,
            {"group_id": group_key, "prompt": prompt, "candidates": []},
        )
        grp["candidates"].append(
            {
                "id": sid or img_path.stem,
                "path": resolved,
                "uri": img_path.resolve().as_uri(),
                "hint": hint,
                "reason": reason,
            }
        )

    for src in paths:
        if src.is_file() and src.suffix.lower() == ".json":
            data = json.loads(src.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "samples" in data:
                trace_path = src.with_name("mimo_trace.json")
                trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.is_file() else {}
                for s in data["samples"]:
                    gid = str(s.get("group_id") or s["id"])
                    sid = str(s["id"])
                    audit = trace.get(gid, {}).get("audits", {}).get(sid)
                    _add_candidate(
                        f"{src.parent.name}:{gid}",
                        str(s["prompt"]),
                        Path(s["image"]),
                        sid=sid,
                        hint=_gate_flags(audit),
                        reason=str((audit or {}).get("reason", ""))[:140],
                    )
            elif isinstance(data, list):
                for item in data:
                    tid = str(item.get("id", ""))
                    if not tid or "file" not in item:
                        continue
                    _add_candidate(
                        f"{src.parent.name}:{tid}",
                        prompt_lookup.get(tid, tid),
                        src.parent / str(item["file"]),
                        sid=f"{tid}_r{item.get('rollout', '')}",
                    )
        elif src.is_dir():
            iter_dirs = sorted(src.glob("iteration_*")) if any(src.glob("iteration_*")) else [src]
            for folder in iter_dirs:
                req = folder / "reward_request.json"
                if req.is_file():
                    sub = collect_groups_from_requests([req], prompts_jsonl=prompts_jsonl)
                    for grp in sub:
                        for c in grp["candidates"]:
                            _add_candidate(grp["group_id"], grp["prompt"], Path(c["path"]), sid=c["id"], hint=c["hint"], reason=c["reason"])
                    continue
                man = folder / "manifest.json"
                if man.is_file():
                    sub = collect_groups_from_requests([man], prompts_jsonl=prompts_jsonl)
                    for grp in sub:
                        for c in grp["candidates"]:
                            _add_candidate(grp["group_id"], grp["prompt"], Path(c["path"]), sid=c["id"], hint=c["hint"], reason=c["reason"])
                    continue
                by_prefix: dict[str, list[Path]] = defaultdict(list)
                for img in sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg")):
                    stem = re.sub(r"_r\d+$", "", img.stem)
                    by_prefix[stem].append(img)
                for prefix, imgs in by_prefix.items():
                    if len(imgs) >= 2:
                        for img in imgs:
                            _add_candidate(f"{folder.name}:{prefix}", prompt_lookup.get(prefix, prefix), img, sid=img.stem)

    return [g for g in groups_map.values() if len(g["candidates"]) >= 2]


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Anima DPO 人工选图配对器</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif; background: #11161d; color: #e6edf3; }
  .topbar { position: sticky; top: 0; z-index: 20; background: #161d27; border-bottom: 1px solid #2b3646; padding: 12px 20px; display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }
  .stats { font-size: 14px; font-weight: 600; color: #9ecbff; margin-right: auto; }
  button { border: 1px solid #38465a; background: #212b38; color: #e6edf3; padding: 7px 13px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 600; transition: all .15s; }
  button:hover { background: #2c394b; }
  button.save-btn { background: #1f6feb; border-color: #388bfd; color: #fff; }
  button.save-btn:hover { background: #388bfd; }
  button.primary { background: #238636; border-color: #2ea043; color: #fff; }
  button.primary:hover { background: #2ea043; }
  .save-toast { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; font-weight: 600; color: #7ee787; background: #132d1e; border: 1px solid #238636; border-radius: 6px; padding: 6px 12px; opacity: 0; transition: opacity .25s ease-in-out; pointer-events: none; }
  .save-toast.show { opacity: 1; }
  .help { width: 100%; font-size: 12px; color: #8b9bb0; margin-top: 2px; }
  .container { padding: 16px 20px 60px; display: flex; flex-direction: column; gap: 20px; }
  .group { background: #18202b; border: 1px solid #2a3444; border-radius: 10px; padding: 14px; }
  .group.active { border-color: #58a6ff; box-shadow: 0 0 0 1px #58a6ff44; }
  .group-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin-bottom: 10px; }
  .group-id { font-weight: 700; color: #79c0ff; white-space: nowrap; }
  .prompt { font-size: 13px; color: #b7c3d4; flex: 1; line-height: 1.4; }
  .pair-badge { font-size: 12px; font-weight: 700; padding: 3px 8px; border-radius: 999px; background: #253040; color: #9fb0c7; white-space: nowrap; }
  .pair-badge.ready { background: #1c4428; color: #7ee787; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 12px; }
  .card { background: #11161d; border: 3px solid #2b3646; border-radius: 8px; overflow: hidden; display: flex; flex-direction: column; transition: border-color .12s; }
  .card.win { border-color: #2ea043; background: #102417; }
  .card.lose { border-color: #f85149; background: #291316; }
  .card img { width: 100%; height: 320px; object-fit: contain; background: #0b0f14; cursor: pointer; display: block; }
  .meta { padding: 8px 10px; display: flex; flex-direction: column; gap: 6px; font-size: 12px; }
  .meta-top { display: flex; justify-content: space-between; align-items: center; gap: 6px; }
  .hint { font-size: 11px; padding: 1px 6px; border-radius: 4px; background: #273142; color: #aebccb; }
  .hint.ok { background: #173a22; color: #7ee787; }
  .reason { color: #8b9bb0; font-size: 11px; height: 28px; overflow: hidden; line-height: 1.25; }
  .btns { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px; }
  .btns button { padding: 5px 0; font-size: 12px; }
  .btns button.w.on { background: #238636; border-color: #3fb950; color: #fff; }
  .btns button.l.on { background: #da3633; border-color: #f85149; color: #fff; }
</style>
</head>
<body>
<div class="topbar">
  <div class="stats" id="stats">正在加载题组...</div>
  <button class="save-btn" id="manualSaveBtn" onclick="manualSave()">💾 保存标注 (Ctrl+S)</button>
  <button onclick="exportDraft()">📥 导出草稿备份</button>
  <button onclick="document.getElementById('importFile').click()">📤 导入草稿</button>
  <input type="file" id="importFile" style="display:none" accept=".json" onchange="importDraft(event)">
  <button onclick="toggleFilter()" id="filterBtn">筛选: 全部</button>
  <button onclick="applyVlmHints()">按 VLM 提示预填</button>
  <button onclick="clearAll()">清空选择</button>
  <button class="primary" onclick="exportPairs()">导出 dpo_pairs.json</button>
  <div id="saveToast" class="save-toast">✓ 已成功保存到本地缓存！</div>
  <div class="help">操作说明：直接点击图片循环切换 [未选 ➔ WIN(好/绿框) ➔ LOSE(坏/红框)]，或点击下方按钮；快捷键：数字 1~8 标记当前组图片，J/K 上下跳组，Ctrl+S 保存。实时自动保存已开启。</div>
</div>
<div class="container" id="app"></div>
<script>
const GROUPS = __GROUPS_JSON__;
const STORAGE_KEY = __STORAGE_KEY__;
let state = {};
try { state = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); } catch (e) { state = {}; }
let activeIdx = 0;
let filterMode = "all"; // all | ready | pending
let toastTimer = null;

function saveState(silent = true) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    if (!silent) {
      showToast("✓ 标注进度已成功保存！");
    }
  } catch (e) {
    alert("保存失败：" + e.message);
  }
}

function manualSave() {
  saveState(false);
  const btn = document.getElementById("manualSaveBtn");
  const origText = btn.textContent;
  btn.textContent = "✓ 已保存！";
  btn.style.background = "#238636";
  setTimeout(() => {
    btn.textContent = origText;
    btn.style.background = "";
  }, 1200);
}

function showToast(msg) {
  const toast = document.getElementById("saveToast");
  toast.textContent = msg;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toast.classList.remove("show");
  }, 2200);
}

function exportDraft() {
  saveState();
  const labeledCount = Object.keys(state).length;
  const draftData = {
    updated_at: new Date().toISOString(),
    labeled_items: labeledCount,
    state: state
  };
  const blob = new Blob([JSON.stringify(draftData, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "dpo_picker_draft_backup.json";
  a.click();
  showToast(`✓ 已导出草稿文件 (${labeledCount} 项已标记)`);
}

function importDraft(evt) {
  const file = evt.target.files && evt.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(e) {
    try {
      const data = JSON.parse(e.target.result);
      const importedState = data.state || data;
      if (typeof importedState === "object" && importedState !== null) {
        state = Object.assign({}, state, importedState);
        saveState();
        render();
        showToast("✓ 成功导入草稿！");
      } else {
        alert("无效的草稿文件格式！");
      }
    } catch (err) {
      alert("导入草稿失败：" + err.message);
    }
    evt.target.value = "";
  };
  reader.readAsText(file);
}

function pairCount(g) {
  let w = 0, l = 0;
  for (const c of g.candidates) {
    const v = state[c.path];
    if (v === "win") w++;
    else if (v === "lose") l++;
  }
  return [w, l, w * l];
}

function setLabel(path, val, gIdx) {
  if (!val) delete state[path];
  else state[path] = val;
  activeIdx = gIdx;
  saveState(true);
  render();
}

function cycleLabel(path, gIdx) {
  const cur = state[path] || "";
  const nxt = cur === "" ? "win" : cur === "win" ? "lose" : "";
  setLabel(path, nxt, gIdx);
}

function applyVlmHints() {
  let applied = 0;
  for (const g of GROUPS) {
    for (const c of g.candidates) {
      if (state[c.path]) continue;
      if (c.hint === "OK") { state[c.path] = "win"; applied++; }
      else if (c.hint && (c.hint.includes("D") || c.hint.includes("P") || c.hint.includes("L"))) { state[c.path] = "lose"; applied++; }
    }
  }
  saveState(true);
  render();
  showToast(`✓ 已应用 VLM 预填 ${applied} 张`);
}

function clearAll() {
  if (!confirm("确定要清空当前的全部标注记录吗？")) return;
  state = {};
  saveState(true);
  render();
  showToast("已清空标注记录");
}

function toggleFilter() {
  filterMode = filterMode === "all" ? "ready" : filterMode === "ready" ? "pending" : "all";
  document.getElementById("filterBtn").textContent = "筛选: " + (filterMode === "all" ? "全部" : filterMode === "ready" ? "已配对" : "未配对");
  render();
}

function exportPairs() {
  const pairs = [];
  for (const g of GROUPS) {
    const wins = g.candidates.filter(c => state[c.path] === "win");
    const loses = g.candidates.filter(c => state[c.path] === "lose");
    for (const w of wins) {
      for (const l of loses) {
        pairs.push({ group_id: g.group_id, prompt: g.prompt, win_image: w.path, lose_image: l.path, win_id: w.id, lose_id: l.id });
      }
    }
  }
  if (!pairs.length) { alert("还没有形成有效配对（每组至少需选 1 张 WIN 和 1 张 LOSE）"); return; }
  const blob = new Blob([JSON.stringify({ count: pairs.length, pairs }, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "dpo_pairs.json";
  a.click();
}

function render() {
  let readyGroups = 0, totalPairs = 0;
  for (const g of GROUPS) {
    const [, , p] = pairCount(g);
    if (p > 0) readyGroups++;
    totalPairs += p;
  }
  document.getElementById("stats").textContent = `总组数: ${GROUPS.length} | 已配对组: ${readyGroups} | 产生 (WIN, LOSE) 对数: ${totalPairs}`;

  const app = document.getElementById("app");
  app.innerHTML = "";
  GROUPS.forEach((g, gIdx) => {
    const [w, l, p] = pairCount(g);
    if (filterMode === "ready" && p === 0) return;
    if (filterMode === "pending" && p > 0) return;
    const box = document.createElement("div");
    box.className = "group" + (gIdx === activeIdx ? " active" : "");
    box.id = "grp_" + gIdx;
    box.onclick = () => { activeIdx = gIdx; };

    const head = document.createElement("div");
    head.className = "group-head";
    head.innerHTML = `<span class="group-id">${g.group_id}</span><span class="prompt">${g.prompt}</span><span class="pair-badge ${p > 0 ? "ready" : ""}">W:${w} / L:${l} ➔ ${p} 对</span>`;
    box.appendChild(head);

    const grid = document.createElement("div");
    grid.className = "grid";
    g.candidates.forEach((c, cIdx) => {
      const lbl = state[c.path] || "";
      const card = document.createElement("div");
      card.className = "card " + lbl;

      const img = document.createElement("img");
      img.src = c.uri;
      img.loading = "lazy";
      img.onclick = (e) => { e.stopPropagation(); cycleLabel(c.path, gIdx); };
      card.appendChild(img);

      const meta = document.createElement("div");
      meta.className = "meta";
      const hintBadge = c.hint ? `<span class="hint ${c.hint === "OK" ? "ok" : ""}">Astra: ${c.hint}</span>` : "";
      meta.innerHTML = `<div class="meta-top"><b>#${cIdx + 1} ${c.id}</b>${hintBadge}</div><div class="reason">${c.reason || ""}</div>`;

      const btns = document.createElement("div");
      btns.className = "btns";
      const bw = document.createElement("button"); bw.className = "w" + (lbl === "win" ? " on" : ""); bw.textContent = "WIN (好)";
      bw.onclick = (e) => { e.stopPropagation(); setLabel(c.path, lbl === "win" ? "" : "win", gIdx); };
      const bl = document.createElement("button"); bl.className = "l" + (lbl === "lose" ? " on" : ""); bl.textContent = "LOSE (坏)";
      bl.onclick = (e) => { e.stopPropagation(); setLabel(c.path, lbl === "lose" ? "" : "lose", gIdx); };
      const bc = document.createElement("button"); bc.textContent = "清除";
      bc.onclick = (e) => { e.stopPropagation(); setLabel(c.path, "", gIdx); };
      btns.append(bw, bl, bc);
      meta.appendChild(btns);
      card.appendChild(meta);
      grid.appendChild(card);
    });
    box.appendChild(grid);
    app.appendChild(box);
  });
}

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  // Ctrl+S / Cmd+S manual save
  if ((e.ctrlKey || e.metaKey) && (e.key === "s" || e.key === "S")) {
    e.preventDefault();
    manualSave();
    return;
  }
  if (e.key >= "1" && e.key <= "8") {
    const idx = parseInt(e.key, 10) - 1;
    const g = GROUPS[activeIdx];
    if (g && g.candidates[idx]) cycleLabel(g.candidates[idx].path, activeIdx);
  } else if (e.key === "j" || e.key === "J" || e.key === "ArrowDown") {
    activeIdx = Math.min(GROUPS.length - 1, activeIdx + 1);
    render();
    document.getElementById("grp_" + activeIdx)?.scrollIntoView({ behavior: "smooth", block: "center" });
  } else if (e.key === "k" || e.key === "K" || e.key === "ArrowUp") {
    activeIdx = Math.max(0, activeIdx - 1);
    render();
    document.getElementById("grp_" + activeIdx)?.scrollIntoView({ behavior: "smooth", block: "center" });
  }
});
render();
</script>
</body>
</html>
"""


def build_picker_html(groups: list[dict[str, Any]], out_html: Path) -> Path:
    """Write standalone HTML visual pair picker and return output path."""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    html = (
        HTML_TEMPLATE.replace("__GROUPS_JSON__", json.dumps(groups, ensure_ascii=False))
        .replace("__STORAGE_KEY__", json.dumps(f"anima_dpo_picker::{out_html.resolve()}"))
    )
    out_html.write_text(html, encoding="utf-8")
    return out_html


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", required=True, help="Rollout folders, iteration_* parents, manifest.json, or reward_request.json paths.")
    parser.add_argument("--prompts-jsonl", default=None, help="Optional tasks_all.jsonl to resolve full prompts for manifest folders.")
    parser.add_argument("--out", default="dpo_picker.html", help="Output HTML file path.")
    args = parser.parse_args()

    groups = collect_groups_from_requests(
        [Path(s) for s in args.sources],
        prompts_jsonl=Path(args.prompts_jsonl) if args.prompts_jsonl else None,
    )
    if not groups:
        raise SystemExit("No multi-candidate prompt groups found across given --sources")
    out = build_picker_html(groups, Path(args.out))
    total_imgs = sum(len(g["candidates"]) for g in groups)
    print(f"BUILT_PICKER: {out} (groups={len(groups)}, images={total_imgs})")


if __name__ == "__main__":
    main()
