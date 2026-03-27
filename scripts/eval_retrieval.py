#!/usr/bin/env python3
"""
离线检索评测脚本

数据集格式（JSONL）每行示例：
{"kb_id":"xxxx","question":"...","gold_chunk_ids":["c1","c2"]}

运行示例：
python scripts/eval_retrieval.py --dataset storages/eval/retrieval_eval.jsonl --mode all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_dataset(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            kb_id = item.get("kb_id")
            question = (item.get("question") or "").strip()
            gold = item.get("gold_chunk_ids") or []
            if not kb_id or not question or not gold:
                raise ValueError(f"第 {i} 行缺少 kb_id/question/gold_chunk_ids")
            rows.append(
                {
                    "kb_id": str(kb_id),
                    "question": question,
                    "gold_chunk_ids": [str(x) for x in gold],
                }
            )
    return rows


def docs_to_sources(docs):
    sources = []
    for doc in docs or []:
        meta = doc.metadata or {}
        sources.append(
            {
                "chunk_id": meta.get("chunk_id") or meta.get("id"),
                "doc_id": meta.get("doc_id"),
                "doc_name": meta.get("doc_name"),
                "content": doc.page_content or "",
            }
        )
    return sources


def retrieve(mode: str, kb_id: str, question: str, retrieval_service):
    collection_name = f"kb_{kb_id}"
    if mode == "vector":
        return retrieval_service.vector_search(collection_name, question, rerank=True)
    if mode == "keyword":
        return retrieval_service.keyword_search(collection_name, question, rerank=True)
    return retrieval_service.hybrid_search(collection_name, question)


def mean_metrics(metric_list: list[dict]):
    if not metric_list:
        return {}
    keys = sorted({k for m in metric_list for k in m.keys()})
    out = {}
    for k in keys:
        vals = [float(m.get(k, 0.0)) for m in metric_list]
        out[k] = sum(vals) / max(1, len(vals))
    return out


def run_eval(dataset, modes: list[str], k_values: list[int], retrieval_service, evaluation_service):
    report = {"samples": len(dataset), "modes": {}}
    for mode in modes:
        per_sample = []
        for row in dataset:
            docs = retrieve(mode, row["kb_id"], row["question"], retrieval_service)
            sources = docs_to_sources(docs)
            metrics = evaluation_service.evaluate_retrieval(
                sources=sources,
                gold_chunk_ids=row["gold_chunk_ids"],
                k_values=k_values,
            )
            per_sample.append(metrics)
        report["modes"][mode] = {
            "avg_metrics": mean_metrics(per_sample),
            "evaluated_samples": len(per_sample),
        }
    return report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="JSONL 数据集路径")
    p.add_argument(
        "--mode",
        default="all",
        choices=["vector", "keyword", "hybrid", "all"],
        help="评测模式",
    )
    p.add_argument("--k-values", default="1,3,5", help="例如 1,3,5")
    p.add_argument("--output", default="", help="可选，输出 JSON 文件路径")
    return p.parse_args()


def main():
    args = parse_args()
    dataset = load_dataset(Path(args.dataset))
    modes = ["vector", "keyword", "hybrid"] if args.mode == "all" else [args.mode]
    k_values = [int(x) for x in args.k_values.split(",") if x.strip()]

    from app import create_app
    from app.services.evaluation_service import evaluation_service
    from app.services.retrieval_service import retrieval_service

    # 初始化应用，确保数据库与配置可用
    create_app()
    report = run_eval(dataset, modes, k_values, retrieval_service, evaluation_service)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
