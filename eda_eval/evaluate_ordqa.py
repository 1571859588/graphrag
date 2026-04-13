"""
evaluate_ordqa.py  –  使用 GraphRAG 对 ORD-QA 基准进行完整评估。

评估指标（与 ORD-QA 原论文保持一致）：
  - BLEU-1/2/3/4     N-gram 精确匹配
  - ROUGE-L          最长公共子序列 F1
  - BERTScore F1     语义级别相似度（DeBERTa）
  - Recall@K (Hit Rate)  检索文档 ID 命中率（local 模式专属）

使用方法（在 baselines/graphrag 根目录）：
    uv run python eda_eval/evaluate_ordqa.py \\
        --benchmark  ../../benchmarks/ORD-QA/benchmark/ORD-QA.jsonl \\
        --workspace  eda_eval/workspace \\
        --config     eda_eval/config_ordqa.yml \\
        --query_type local \\
        --output     eda_eval/results_graphrag_local.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

# 加载 .env（如果存在）
load_dotenv()

# ── 指标计算 ──────────────────────────────────────────────────────────────────
import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

_smoother = SmoothingFunction().method1
_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def compute_bleu(reference: str, hypothesis: str) -> dict[str, float]:
    ref_tok = nltk.word_tokenize(reference.lower())
    hyp_tok = nltk.word_tokenize(hypothesis.lower())
    return {
        f"bleu-{n}": sentence_bleu(
            [ref_tok],
            hyp_tok,
            weights=tuple([1 / n] * n + [0] * (4 - n)),
            smoothing_function=_smoother,
        )
        for n in range(1, 5)
    }


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    return _rouge.score(reference, hypothesis)["rougeL"].fmeasure


def compute_bert_score(references: list[str], hypotheses: list[str]) -> list[float]:
    import bert_score as bs
    _, _, F = bs.score(hypotheses, references, lang="en", verbose=False)  # noqa: N806
    return F.tolist()


def compute_hit_rate(
    predicted_chunk_ids: list[str],
    gold_references: list[str],
) -> float:
    """检索命中率：gold references 中至少有一个被预测到则为命中。"""
    if not gold_references:
        return 0.0
    predicted_set = set(predicted_chunk_ids)
    hits = sum(1 for ref in gold_references if ref in predicted_set)
    return hits / len(gold_references)


# ── GraphRAG 查询封装 ─────────────────────────────────────────────────────────

def load_graphrag_config(workspace: str, config_override_path: str) -> dict:
    """读取 config_ordqa.yml 配置，展开环境变量。"""
    with open(config_override_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # 简单的环境变量展开 ${VAR} / ${VAR:default}
    def expand(val: str) -> str:
        import re
        def replacer(m: re.Match) -> str:
            parts = m.group(1).split(":", 1)
            env_val = os.environ.get(parts[0], parts[1] if len(parts) > 1 else "")
            return env_val
        return re.sub(r"\$\{([^}]+)\}", replacer, val) if isinstance(val, str) else val

    def deep_expand(obj):
        if isinstance(obj, dict):
            return {k: deep_expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [deep_expand(i) for i in obj]
        if isinstance(obj, str):
            return expand(obj)
        return obj

    return deep_expand(raw)


async def query_graphrag(
    question: str,
    workspace: str,
    cfg: dict,
    query_type: str = "local",
) -> tuple[str, list[str]]:
    """调用 GraphRAG 的 local 或 global 查询，返回 (answer, retrieved_chunk_ids)。

    retrieved_chunk_ids 仅在 local 模式下有意义（从文本单元文件名中提取）。
    """
    try:
        # GraphRAG ≥ 1.x 的统一查询 API
        from graphrag.query.api import local_search, global_search  # type: ignore[import]

        output_dir = Path(workspace) / "output"
        query_cfg = cfg.get("query_llm", cfg.get("llm", {}))

        if query_type == "local":
            result = await local_search(
                config_filepath=Path(workspace) / "settings.yaml",
                data_dir=str(output_dir),
                root_dir=workspace,
                community_level=2,
                response_type="QA",
                query=question,
            )
        else:
            result = await global_search(
                config_filepath=Path(workspace) / "settings.yaml",
                data_dir=str(output_dir),
                root_dir=workspace,
                community_level=2,
                response_type="QA",
                query=question,
            )

        answer = result.response if hasattr(result, "response") else str(result)
        # 尝试从 context_data 中提取检索到的 chunk id
        chunk_ids: list[str] = []
        if hasattr(result, "context_data"):
            for item in (result.context_data or []):
                # chunk id 格式为 "CHUNK_ID: global_routing_12"
                title = str(item.get("id", item.get("title", "")))
                chunk_ids.append(title.replace("CHUNK_ID: ", "").strip())
        return answer, chunk_ids

    except ImportError:
        # 降级：通过命令行调用并解析 stdout
        return await _query_via_cli(question, workspace, query_type)


async def _query_via_cli(
    question: str, workspace: str, query_type: str
) -> tuple[str, list[str]]:
    """通过 subprocess 调用 graphrag query CLI 并捕获输出。"""
    import subprocess  # noqa: PLC0415
    cmd = [
        sys.executable, "-m", "graphrag", "query",
        "--root", workspace,
        "--method", query_type,
        "--query", question,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    output = stdout.decode(errors="ignore")

    # 提取 "SUCCESS:" 后的答案文本
    answer = ""
    for line in output.splitlines():
        if line.startswith("SUCCESS:"):
            answer = line[len("SUCCESS:"):].strip()
            break
    if not answer:
        answer = output.strip()

    return answer, []   # CLI 模式下无法提取 chunk id


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_benchmark(benchmark_path: str) -> list[dict]:
    samples = []
    with open(benchmark_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def filter_samples(
    samples: list[dict],
    question_types: list[str] | None,
    max_samples: int,
) -> list[dict]:
    if question_types:
        # ORD-QA 类型字段值示例："functionality", "vlsi_flow", "gui & installation & test"
        type_set = {t.strip().lower() for t in question_types}
        samples = [
            s for s in samples
            if s.get("type", "").lower().replace(" ", "_") in type_set
            or s.get("type", "").lower() in type_set
        ]
    if max_samples > 0:
        samples = samples[:max_samples]
    return samples


# ── 主评估循环 ────────────────────────────────────────────────────────────────

async def evaluate(args: argparse.Namespace) -> None:
    print(f"[evaluate] 加载配置：{args.config}")
    cfg = load_graphrag_config(args.workspace, args.config)

    print(f"[evaluate] 加载基准测试集：{args.benchmark}")
    samples = load_benchmark(args.benchmark)

    question_types = args.question_types.split(",") if args.question_types else None
    samples = filter_samples(samples, question_types, args.max_samples)
    print(f"[evaluate] 共 {len(samples)} 条样本（类型过滤：{question_types}）")

    detailed: list[dict] = []
    predictions: list[str] = []
    references: list[str] = []

    for sample in tqdm(samples, desc=f"GraphRAG {args.query_type} 查询中"):
        question = sample.get("question", "").strip()
        gold_answer = sample.get("answer", "").strip()
        gold_refs = sample.get("reference", [])  # chunk id 列表

        # 调用 GraphRAG
        try:
            pred_answer, pred_chunk_ids = await query_graphrag(
                question, args.workspace, cfg, args.query_type
            )
        except Exception as e:  # noqa: BLE001
            print(f"\n  [警告] 样本 id={sample.get('id')} 查询失败：{e}")
            pred_answer = ""
            pred_chunk_ids = []

        # 计算逐条指标
        bleu_scores = compute_bleu(gold_answer, pred_answer)
        rouge_l = compute_rouge_l(gold_answer, pred_answer)
        hit_rate = compute_hit_rate(pred_chunk_ids, gold_refs) if args.query_type == "local" else None

        record: dict = {
            "id": sample.get("id"),
            "type": sample.get("type"),
            "question": question,
            "gold": gold_answer,
            "pred": pred_answer,
            "gold_refs": gold_refs,
            "pred_chunk_ids": pred_chunk_ids,
            **bleu_scores,
            "rouge_l": rouge_l,
        }
        if hit_rate is not None:
            record["hit_rate"] = hit_rate

        detailed.append(record)
        predictions.append(pred_answer)
        references.append(gold_answer)

    # BERTScore（批量计算更高效）
    print("[evaluate] 计算 BERTScore（批量）…")
    bert_f1_list = compute_bert_score(references, predictions)
    for item, bf in zip(detailed, bert_f1_list):
        item["bert_score_f1"] = bf

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    n = len(detailed)
    scalar_keys = ["bleu-1", "bleu-2", "bleu-3", "bleu-4", "rouge_l", "bert_score_f1"]
    if args.query_type == "local":
        scalar_keys.append("hit_rate")

    aggregate = {
        k: round(sum(d.get(k, 0.0) for d in detailed) / n, 4)
        for k in scalar_keys
    }

    # 按问题类型分层汇总
    by_type: dict[str, dict] = {}
    for qtype in {"functionality", "vlsi_flow", "gui & installation & test"}:
        subset = [d for d in detailed if d.get("type", "") == qtype]
        if subset:
            by_type[qtype] = {
                k: round(sum(d.get(k, 0.0) for d in subset) / len(subset), 4)
                for k in scalar_keys
            }
            by_type[qtype]["count"] = len(subset)

    # ── 打印报告 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  GraphRAG ({args.query_type}) × ORD-QA 评估结果")
    print("=" * 60)
    print(f"  样本总数: {n}")
    print(f"  {'指标':<22} {'值':>8}")
    print("  " + "-" * 32)
    for k, v in aggregate.items():
        print(f"  {k:<22} {v:>8.4f}")
    print()
    if by_type:
        print("  按问题类型细分：")
        for qtype, metrics in by_type.items():
            cnt = metrics.pop("count")
            print(f"    [{qtype}]  n={cnt}")
            for mk, mv in metrics.items():
                print(f"      {mk:<20} {mv:.4f}")
            metrics["count"] = cnt  # 恢复
    print("=" * 60)

    # ── 写出 JSON ─────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "config": {
            "query_type": args.query_type,
            "benchmark": args.benchmark,
            "workspace": args.workspace,
            "total_samples": n,
            "question_types": question_types,
        },
        "aggregate": aggregate,
        "by_type": by_type,
        "details": detailed,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[evaluate] 完整结果已写入：{output_path}")


# ── CLI 入口 ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GraphRAG × ORD-QA 评估脚本"
    )
    parser.add_argument("--benchmark",      required=True,
                        help="ORD-QA.jsonl 文件路径")
    parser.add_argument("--workspace",      default="eda_eval/workspace",
                        help="GraphRAG 工作空间目录")
    parser.add_argument("--config",         default="eda_eval/config_ordqa.yml",
                        help="EDA 评估配置文件路径")
    parser.add_argument("--query_type",     choices=["local", "global"], default="local",
                        help="GraphRAG 查询模式：local（实体级）或 global（社区级）")
    parser.add_argument("--output",         default="eda_eval/results_graphrag.json",
                        help="结果 JSON 输出路径")
    parser.add_argument("--question_types", default=None,
                        help="只评估指定类型（逗号分隔），如 'functionality,vlsi_flow'")
    parser.add_argument("--max_samples",    type=int, default=-1,
                        help="最大样本数（-1 表示全部，调试时可设为 10）")
    args = parser.parse_args()

    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
