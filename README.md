# llm-model-bench

本地 / 云端大模型快速评测工具。零依赖（仅 Python 标准库），支持任何 OpenAI 兼容端点（LM Studio、vLLM、ollama、one-api / new-api 网关等），自动兼容推理模型（输出 `reasoning_content` 思维链的模型，如 DeepSeek-R1、Qwen thinking 系列、gpt-oss 等）。

## 评测内容（21 项，全部机器判分）

| 维度 | 项数 | 内容与判分方式 |
|---|---|---|
| A 速度 | 3 | TTFT 首字延迟、流式生成 tok/s、长上下文 prefill tok/s（流式精确计时） |
| B Code | 5 | 写代码 ×3（隔离子进程单测自动验证）、找 bug ×1（bug 点识别 + 修复代码跑单测）、代码推理 ×1（与真实执行输出比对） |
| C Agent | 3 | function calling 两轮工具调用（工具名/参数/回填推理）、json_schema 严格结构化输出、指令遵循 |
| D 准确率 | 10 | 数学方程、百分比陷阱、逻辑推理、抽屉原理、数字比较、字母计数、数列规律、技术常识，唯一答案逐题 0/1 |

采样统一 `temperature=0`（贪心解码），结果可复现。

## 快速开始

本地 LM Studio（默认 `http://localhost:1234`）：

```bash
python3 lm_model_bench.py --model <模型ID> --max-tokens-override 8192
```

任意 OpenAI 兼容网关（key 走环境变量，不落命令行/日志）：

```bash
export BENCH_API_KEY=sk-xxx
python3 lm_model_bench.py --base-url https://gw.example.com/v1 \
    --model <模型ID> --skip speed
```

只想测能力不测速度时加 `--skip speed`；全部分项可用 `--skip speed,code,agent,qa` 组合。

## 参数说明

| 参数 | 说明 |
|---|---|
| `--model` | 模型 ID（必填） |
| `--base-url` | OpenAI 兼容端点，默认 `http://localhost:1234`；以 `/v1` 结尾自动拼接 `/chat/completions`，也支持直接传完整 `/chat/completions` 地址 |
| `--max-tokens-override` | 覆盖能力题的 max_tokens 预算（推理模型建议 4096-8192） |
| `--skip` | 逗号分隔跳过某维度：`speed,code,agent,qa` |
| `--timeout` | 单请求超时秒数，默认 420 |
| `--api-key` / `BENCH_API_KEY` | 鉴权 key，推荐用环境变量 |
| `--out` | 输出目录，默认 `.temp/bench` |
| `--quant` / `--ctx` / `--machine` | 报告展示用的环境描述字段 |

输出：`<out>/result.json`（原始成绩）+ `<out>/report.md`（Markdown 报告，含逐项明细）。

## 推理模型实测经验（写在前面的坑）

1. **思维链计入 max_tokens**（LM Studio 等本地引擎）：预算要按正文 5-10 倍给，否则正文会被思维链吃光，`content` 为空。云端官方 API（如 DeepSeek）思维链通常不占 max_tokens，行为不同。
2. 判分文本正文优先，为空时回退思维链；Code 题对回复中**所有代码块逐个验证取最优**，思维链中的草稿代码也能被正确计分。
3. 个别 "thinking" 模型在本地引擎下思维链会**无限膨胀（重复退化）**：给多大预算吃多大、写重复内容、`content` 长期为空。遇到这种情况建议换量化版本或关闭 thinking，这不是脚本能修的。
4. 速度测试对并发敏感：后台有其他请求时测出的 tok/s 会明显失真，测速请保证无负载。

## 自定义题库

题库直接内置于脚本，按需修改：

- `CODE_TASKS`：写代码题用 `cases` 提供单测用例（表达式 + 期望 repr），脚本在隔离子进程中执行验证，超时 15s/用例；`bug_keywords` 用于找 bug 题的要点识别。
- `QA_TASKS`：每题一个 `check` 闭包做判分；新增题目时**先人工复核参考答案**——机器判分的参考答案本身也可能错，本项目开发中真实发生过一道平均数题参考答案笔误、导致四个模型被集体误判的案例。
- `TOOLS_DEF`：function calling 测试的工具定义。

## 环境要求

- Python 3.8+，无第三方依赖
- 被测端点支持 OpenAI `/chat/completions` 协议；function calling 与 json_schema 测试要求端点支持 `tools` / `response_format`，不支持时会记录 error 而不是误判

> AI生成
