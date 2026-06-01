"""Load an error-rich benchmark into MCItem form. The HF download is the ONLY
network action and is user-gated (allow_network). After first fetch, access is
offline. The pre-screen gate (Gate 1) decides whether the benchmark is suitable
(>=30% local-model error) BEFORE the full experiment is allowed to report a verdict.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.crc.eval.prescreen import prescreen_error_rate


class NetworkApprovalRequired(RuntimeError):
    pass


@dataclass(frozen=True)
class MCItem:
    qid: str
    question: str          # stem + rendered "A) .. B) .." options
    options: List[str]
    answer_letter: str
    category: str


BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "mmlu_pro": {"hf": "TIGER-Lab/MMLU-Pro", "split": "test", "gated": False},
}


def _render(stem: str, options: List[str]) -> str:
    lines = [stem.strip(), ""]
    for i, opt in enumerate(options):
        lines.append(f"{chr(ord('A') + i)}) {opt}")
    lines.append("\nReply with only the letter of the correct option.")
    return "\n".join(lines)


def load_items(name: str = "mmlu_pro", allow_network: bool = False,
               limit: Optional[int] = None, cache_dir: Optional[str] = None) -> List[MCItem]:
    """Load MCItems from the local HF cache (offline). If not cached, requires
    allow_network=True (raises NetworkApprovalRequired otherwise)."""
    spec = BENCHMARKS[name]
    try:
        from datasets import load_dataset
    except Exception as e:  # datasets not installed
        if not allow_network:
            raise NetworkApprovalRequired(
                "`datasets` is not installed and no benchmark is cached. Re-run with "
                "network approval to `pip install datasets` and fetch MMLU-Pro.") from e
        raise

    if not allow_network:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        ds = load_dataset(spec["hf"], split=spec["split"], cache_dir=cache_dir)
    except Exception as e:
        if not allow_network:
            raise NetworkApprovalRequired(
                f"{name} not in local cache; re-run with network approval to fetch.") from e
        raise

    items: List[MCItem] = []
    for row in ds:
        opts = list(row.get("options") or [])
        if not opts:
            continue
        ans = row.get("answer")  # MMLU-Pro: answer letter or index
        if isinstance(ans, int):
            letter = chr(ord("A") + ans)
        else:
            letter = str(ans).strip().upper()[:1]
        stem = str(row.get("question", "")).strip()
        qid = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:12]
        items.append(MCItem(qid=qid, question=_render(stem, opts), options=opts,
                            answer_letter=letter, category=str(row.get("category", "?"))))
        if limit and len(items) >= limit:
            break
    return items


def to_answer_key(items: List[MCItem]) -> Dict[str, str]:
    return {it.question: it.answer_letter for it in items}


def prescreen(panel, items: List[MCItem], n: int = 40, seed: int = 7, max_tokens: int = 256) -> Dict[str, Any]:
    """Run a stratified sample through the real panel, score with MC extraction,
    and apply Gate 1. Returns the pre-screen result (gate_pass decides the full run)."""
    import numpy as np
    from src.crc.eval.mc_extract import score_mc
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(items))[: min(n, len(items))]
    sample = [items[i] for i in idx]
    ensemble_wrong, unextractable = [], 0
    for it in sample:
        answers = panel.answer(it.question)  # {family: text}
        fam_correct = 0
        for txt in answers.values():
            ok, extracted = score_mc(txt, it.options, it.answer_letter)
            if not extracted:
                unextractable += 1
            fam_correct += 1 if ok else 0
        ensemble_wrong.append(0.0 if fam_correct * 2 > len(answers) else 1.0)
    pre = prescreen_error_rate(ensemble_wrong)
    pre["unextractable_rate"] = round(unextractable / max(1, len(sample) * len(panel.families)), 3)
    pre["sampled"] = len(sample)
    return pre
