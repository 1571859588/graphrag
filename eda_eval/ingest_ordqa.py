"""
ingest_ordqa.py  –  将 ORD-QA 的文档库写入 GraphRAG 工作空间并触发索引。

使用方法（在 baselines/graphrag 根目录）：
    uv run python eda_eval/ingest_ordqa.py \\
        --doc_json  ../../benchmarks/ORD-QA/benchmark/openroad_documentation.json \\
        --workspace eda_eval/workspace

原理：
    1. 读取 openroad_documentation.json（ORD-QA 提供的已切好的 chunk 列表）。
    2. 将每个 chunk 分别写成单独的 .txt 文件到 <workspace>/input/ 目录。
    3. 调用 `graphrag index --root <workspace>` 触发完整的 GraphRAG 索引流程
       （实体抽取 → 关系抽取 → Leiden 社区检测 → 社区摘要 → 嵌入入库）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def load_documentation(doc_json: str) -> list[dict]:
    """读取 openroad_documentation.json，返回 chunk 列表。

    支持两种格式：
      - 列表格式：[{"id": "...", "content": "..."}, ...]
      - 字典格式：{"id": "content", ...}
    """
    with open(doc_json, encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        # 兼容 key→value 结构
        return [{"id": k, "content": v} for k, v in raw.items()]
    msg = f"Unexpected format in {doc_json}"
    raise ValueError(msg)


def write_input_files(chunks: list[dict], input_dir: Path) -> None:
    """将每个 chunk 写成一个独立的 .txt 文件，文件名为 chunk id。"""
    input_dir.mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        chunk_id = str(chunk.get("id", "chunk"))
        # 内容字段名适配（content / text / body）
        content = chunk.get("content") or chunk.get("text") or chunk.get("body", "")
        # 写文件时在首行注入 chunk id，供后续检索命中率统计
        filepath = input_dir / f"{chunk_id}.txt"
        filepath.write_text(f"[CHUNK_ID: {chunk_id}]\n\n{content}", encoding="utf-8")

    print(f"[ingest] 已将 {len(chunks)} 个文档 chunk 写入 '{input_dir}'")


def run_graphrag_index(workspace: str) -> None:
    """调用 graphrag index 触发完整 GraphRAG 索引流程。"""
    cmd = [sys.executable, "-m", "graphrag", "index", "--root", workspace]
    print(f"[ingest] 开始索引，命令: {' '.join(cmd)}")
    print("[ingest] 注意：索引过程可能需要数分钟到数十分钟，取决于文档量与 API 速率。")
    result = subprocess.run(cmd, check=False)  # noqa: S603
    if result.returncode != 0:
        print("[ingest] ⚠️  graphrag index 返回非零退出码，请检查输出日志。")
        sys.exit(result.returncode)
    print("[ingest] ✅ 索引完成！")


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 ORD-QA 文档库写入 GraphRAG 工作空间并触发索引。"
    )
    parser.add_argument(
        "--doc_json",
        required=True,
        help="openroad_documentation.json 的路径（ORD-QA benchmark 目录下）",
    )
    parser.add_argument(
        "--workspace",
        default="eda_eval/workspace",
        help="GraphRAG 工作空间根目录（需已运行 graphrag init）",
    )
    parser.add_argument(
        "--skip_index",
        action="store_true",
        help="只写入文件不执行索引（用于调试文件写入逻辑）",
    )
    args = parser.parse_args()

    workspace = Path(args.workspace)
    input_dir = workspace / "input"

    # 1. 读取文档
    print(f"[ingest] 读取文档：{args.doc_json}")
    chunks = load_documentation(args.doc_json)
    print(f"[ingest] 共读取到 {len(chunks)} 个 chunk。")

    # 2. 写入 input 目录
    write_input_files(chunks, input_dir)

    # 3. 触发索引
    if args.skip_index:
        print("[ingest] --skip_index 已设置，跳过索引阶段。")
    else:
        run_graphrag_index(str(workspace))


if __name__ == "__main__":
    main()
