# GraphRAG × ORD-QA 一键评估集成指引

本目录下的所有文件均运行于 `EDAAgentMemory` 分支，专门用于在 **ORD-QA 基准测试** 上对 Microsoft GraphRAG 进行评估，以作为与多Agent认知进化方法的对比基线。

---

## 目录结构

```
eda_eval/                    ← 本评估套件根目录
├── config_ordqa.yml         ← GraphRAG 的 EDA 场景配置文件（覆盖默认参数）
├── ingest_ordqa.py          ← 将 ORD-QA 文档切块并索引到 GraphRAG 知识图谱
├── evaluate_ordqa.py        ← 主评估脚本（调用 GraphRAG 查询 + 计算所有指标）
├── requirements_eval.txt    ← 仅评估所需的额外依赖
└── USAGE.md                 ← 本文件
```

---

## 快速开始

### 0. 环境要求

- Python ≥ 3.11
- 已安装 `uv`（本项目使用 uv 管理依赖）
- OpenAI 兼容的 LLM API（需能调用 chat completion 接口）

### 1. 安装依赖

在 `baselines/graphrag` 的根目录执行：

```bash
# 安装 GraphRAG 本体（monorepo 方式）
uv sync --all-packages

# 安装评估额外依赖
uv pip install -r eda_eval/requirements_eval.txt
```

### 2. 配置 API Key

将你的 LLM API Key 写入环境变量，或新建 `.env` 文件放在 `baselines/graphrag` 根目录：

```ini
GRAPHRAG_API_KEY=你的API密钥
GRAPHRAG_API_BASE=https://api.openai.com/v1    # 如使用其他兼容端点请修改
```

如需修改模型或其他参数，编辑 `eda_eval/config_ordqa.yml`。

### 3. 初始化 GraphRAG 工作空间

```bash
# 在 baselines/graphrag 根目录下
uv run graphrag init --root eda_eval/workspace
```

这会在 `eda_eval/workspace/` 生成 `settings.yaml`。我们的 `config_ordqa.yml` 会在运行时覆盖关键参数。

### 4. 索引 ORD-QA 文档库

```bash
uv run python eda_eval/ingest_ordqa.py \
    --doc_json ../../benchmarks/ORD-QA/benchmark/openroad_documentation.json \
    --workspace eda_eval/workspace
```

索引完成后，`eda_eval/workspace/output/` 下会生成 GraphRAG 的 Parquet 知识图谱文件。

> ⚠️ **注意**：索引阶段会调用大量 LLM API（用于抽取实体和社区摘要），请确认 API 余额充足。对于 OpenROAD 文档（~400 个 chunk），大约需要 500~2000 次调用，视模型而定。

### 5. 运行评估

```bash
uv run python eda_eval/evaluate_ordqa.py \
    --benchmark   ../../benchmarks/ORD-QA/benchmark/ORD-QA.jsonl \
    --workspace   eda_eval/workspace \
    --config      eda_eval/config_ordqa.yml \
    --query_type  local \
    --output      eda_eval/results_graphrag_local.json
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--benchmark` | 必填 | ORD-QA.jsonl 的路径 |
| `--workspace` | `eda_eval/workspace` | GraphRAG 工作空间目录 |
| `--config` | `eda_eval/config_ordqa.yml` | 配置覆盖文件 |
| `--query_type` | `local` | `local`（实体级）或 `global`（社区级） |
| `--output` | `results.json` | 结果输出路径 |
| `--question_types` | 全部 | 用逗号分隔，如 `functionality,vlsi_flow` |
| `--max_samples` | -1（全部）| 快速调试时限制样本数，如 `--max_samples 10` |

### 6. 解读评估结果

评估完成后，终端会打印汇总表格，同时完整结果写入 `--output` 指定的 JSON 文件。

**指标说明：**

| 指标 | 说明 |
|---|---|
| `bleu-1/2/3/4` | N-gram 精确匹配，与 ORD-QA 原论文一致 |
| `rouge_l` | 最长公共子序列 F1 |
| `bert_score_f1` | 语义相似度（DeBERTa-xlarge-mnli） |
| `recall@k` | 检索到的文档 ID 命中率（仅限 local 模式） |

结果 JSON 的顶层结构：
```json
{
  "config": { ... },
  "aggregate": { "bleu-1": 0.xx, "rouge_l": 0.xx, ... },
  "by_type": {
    "functionality": { ... },
    "vlsi_flow": { ... },
    "gui_installation_test": { ... }
  },
  "details": [ { "id": 1, "question": "...", "pred": "...", "gold": "...", ... } ]
}
```

---

## 常见问题

**Q: 索引太慢 / API 费用太高怎么办？**
> 可以在 `config_ordqa.yml` 中把 `entity_extraction.max_gleanings` 设为 `0`，或者使用价格更低的模型（如 `gpt-4o-mini`）进行索引，再用更强的模型进行查询。

**Q: 如何复现 ORD-QA 原论文的 Naive RAG baseline？**
> 详见同目录下 `../naive_rag/` 的 `USAGE.md`。

**Q: `local` 和 `global` 查询模式有何区别？**
> `local` 以实体关系图为核心进行向量检索，适合精准的功能性问题；`global` 基于社区摘要作宏观回答，适合开放性、跨主题的问题。
