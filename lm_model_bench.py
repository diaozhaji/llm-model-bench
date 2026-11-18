#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Model Bench —— 本地/云端大模型快速评测工具（零依赖：仅 Python 标准库）

用法:
    # 本地 LM Studio / 任意 OpenAI 兼容端点
    python3 lm_model_bench.py --model <模型ID> --max-tokens-override 8192

    # 远程网关（key 走环境变量，不落日志）
    export BENCH_API_KEY=sk-xxx
    python3 lm_model_bench.py --base-url https://gw.example.com/v1 \
        --model <模型ID> --skip speed

输出:
    <out>/result.json   原始成绩
    <out>/report.md     Markdown 报告

四个维度（21 项，全部机器判分）:
    A 速度:  TTFT / 流式生成速度 / 长上下文预填充速度
    B Code:  写代码(本地单测自动验证) / 找bug / 代码推理 / 日志解析
    C Agent: function calling 两轮 / json_schema 结构化输出 / 指令遵循
    D 准确率: 10 道有唯一答案的数学/逻辑/陷阱/常识题
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

# ============================================================ API 层

def api_chat(base_url, model, messages, max_tokens=512, temperature=0.0,
             tools=None, response_format=None, timeout=180, stream=False, api_key=None):
    """调用 OpenAI 兼容 /v1/chat/completions。stream=True 时返回 (text, ttft, usage)。"""
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer %s" % api_key
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if response_format:
        body["response_format"] = response_format
    data = json.dumps(body).encode("utf-8")
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        url = base
    elif base.endswith("/v1"):
        url = base + "/chat/completions"
    else:
        url = base + "/v1/chat/completions"
    req = urllib.request.Request(url, data=data, headers=headers)
    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=timeout)
    if not stream:
        obj = json.loads(resp.read().decode("utf-8", "ignore"))
        usage = obj.get("usage") or {}
        msg = (obj.get("choices") or [{}])[0].get("message", {})
        return {
            "text": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or "",
            "tool_calls": msg.get("tool_calls"),
            "usage": usage,
            "elapsed": time.perf_counter() - t0,
        }
    # 流式: 统计 TTFT 与生成 token（兼容推理模型的 reasoning_content）
    ttft, ttft_c, parts, rparts, usage, nchunks = None, None, [], [], None, 0
    for raw in resp:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if obj.get("usage"):
            usage = obj["usage"]
        for ch in obj.get("choices", []):
            delta = ch.get("delta", {}) or {}
            rc = delta.get("reasoning_content")
            c = delta.get("content")
            if rc:
                if ttft is None:
                    ttft = time.perf_counter() - t0  # 首个 delta（含思维链）
                rparts.append(rc)
            if c:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                if ttft_c is None:
                    ttft_c = time.perf_counter() - t0  # 正文首个 token
                parts.append(c)
        nchunks += 1
    text = "".join(parts)
    elapsed = time.perf_counter() - t0
    gen_toks = None
    if usage and usage.get("completion_tokens"):
        gen_toks = usage["completion_tokens"]
    elif nchunks:
        gen_toks = nchunks  # 近似: 每 chunk 约 1 token
    return {"text": text, "reasoning": "".join(rparts), "ttft": ttft,
            "ttft_content": ttft_c, "usage": usage, "elapsed": elapsed,
            "gen_tokens": gen_toks}


def _is_infra(err):
    """判断是否为基础设施错误(值得重试, 且不该记成模型失败)。"""
    e = err or ""
    m = re.match(r"HTTP (\d{3})", e)
    if m:
        return int(m.group(1)) >= 500 or int(m.group(1)) == 429
    return any(k in e for k in ("URLError", "timed out", "timeout", "TimeoutError",
                                "Connection reset", "ConnectionReset", "RemoteDisconnected",
                                "Connection refused", "ConnectionRefused", "SSL", "ssl"))


def safe_call(fn, *args, **kwargs):
    """带重试的调用: 基础设施错误最多试 3 次(1s/2s 退避), 仍失败则打 INFRA 标。"""
    last = None
    for attempt in range(3):
        try:
            return fn(*args, **kwargs), None
        except urllib.error.HTTPError as e:
            last = "HTTP %d: %s" % (e.code, e.read().decode("utf-8", "ignore")[:300])
        except Exception as e:
            last = "%s: %s" % (type(e).__name__, e)
        if not _is_infra(last) or attempt == 2:
            break
        time.sleep(2 ** attempt)
    return None, ("INFRA: " + last) if _is_infra(last) else last

# ============================================================ 工具: 代码提取与本地验证

def answer_text(r):
    """判分用文本: 正文优先, 为空则用思维链。"""
    t = (r.get("text") or "").strip()
    if t:
        return t
    return r.get("reasoning") or ""


def run_tests(code, cases, timeout=15):
    """在隔离子进程中执行 code 并逐用例验证。cases: [(表达式字符串, 期望repr)]。
    返回 (通过数, 总数, 失败详情列表)"""
    results, fails = 0, []
    for expr, expected in cases:
        prog = (
            "import json\n"
            + code + "\n"
            + "actual = eval(%r)\n" % expr
            + "print(json.dumps({'ok': actual == eval(%r), 'actual': repr(actual)})[:2000])\n" % expected
        )
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                             encoding="utf-8") as f:
                f.write(prog)
                path = f.name
            r = subprocess.run([sys.executable, path], capture_output=True,
                               text=True, timeout=timeout)
            os.unlink(path)
            out = r.stdout.strip()
            if r.returncode == 0 and out:
                obj = json.loads(out.splitlines()[-1])
                if obj.get("ok"):
                    results += 1
                else:
                    fails.append("用例 %r 实际=%s" % (expr, obj.get("actual")))
            else:
                err = (r.stderr or r.stdout).strip().splitlines()
                fails.append("用例 %r 执行出错: %s" % (expr, err[-1][:160] if err else "?"))
        except subprocess.TimeoutExpired:
            fails.append("用例 %r 超时" % expr)
        except Exception as e:
            fails.append("用例 %r 评测器异常: %s" % (expr, e))
    return results, len(cases), fails

# ============================================================ 答案提取

ANSWER_PAT = re.compile(r"(?:最终答案|答案|结论|综上|答)\s*[:：]\s*(.+)")


def extract_answer(text):
    """提取答案行: 支持 答案/答/结论/最终答案/综上 五种前缀, 去掉 markdown 修饰。
    找不到前缀时回退最后一行。"""
    t = (text or "").strip()
    # 先去 markdown 修饰, 否则 "**最终答案**：9" 匹配不上前缀
    t = t.replace("**", "").replace("`", "")
    matches = ANSWER_PAT.findall(t)
    if matches:
        line = matches[-1].strip()
    else:
        lines = [l.strip() for l in t.splitlines() if l.strip()]
        line = lines[-1] if lines else t
    return line.strip("*`_ >").strip()


def _pair_check(ans, name_a, val_a, name_b, val_b):
    """名值配对校验: 「苹果 6 斤，香蕉 14 斤」通过, 「苹果 14 斤，香蕉 6 斤」不通过。"""
    def grab(name):
        m = re.search(name + r"[^0-9]{0,10}(\d+(?:\.\d+)?)", ans)
        return float(m.group(1)) if m else None
    a, b = grab(name_a), grab(name_b)
    return (a is not None and b is not None
            and abs(a - val_a) < 1e-9 and abs(b - val_b) < 1e-9)


def find_num(text, n):
    """在答案文本中寻找数字 n（允许 6 / 6.0 / 六? 不支持中文数字）。"""
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    for x in nums:
        try:
            if abs(float(x) - n) < 1e-9:
                return True
        except ValueError:
            pass
    return False


# ============================================================ 多语言执行器 (hard 套件)

def _run_cmd(cmd, workdir, timeout):
    try:
        r = subprocess.run(cmd, cwd=workdir, capture_output=True,
                           text=True, timeout=timeout)
        return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()
    except subprocess.TimeoutExpired:
        return 124, "", "TIMEOUT"
    except Exception as e:
        return 125, "", str(e)


def run_java_tests(code, cases, timeout=30):
    """Java 判分: 提取 Java 源码, 包一层 Main+断言 harness 编译运行。
    code 需含一个类; cases: [(调用语句模板, 期望输出行)]。
    模板里 {expr} 会被替换。"""
    results, fails = 0, []
    d = tempfile.mkdtemp(prefix="jbench_")
    try:
        # 每个用例独立 try/catch: 某个用例抛异常不再抹掉前面已 PASS 的用例
        blocks = []
        for i, c in enumerate(cases):
            blocks.append(
                "    {\n"
                "      try {\n"
                "        " + c + "\n"
                "      } catch (Throwable t) {\n"
                '        System.out.println("CASE FAIL c' + str(i) + ' " + t.getClass().getSimpleName());\n'
                "      }\n"
                "    }")
        harness = ("public class Harness {\n"
                   "  public static void main(String[] a) throws Exception {\n"
                   + "\n".join(blocks) +
                   "\n  }\n}\n")
        src = code + "\n" + harness
        src_path = os.path.join(d, "Harness.java")
        # 去掉可能的 package 声明
        src = re.sub(r"^\s*package\s+[\w.]+\s*;", "", src, flags=re.M)
        # 模型常写 public class X → 文件名约束冲突, 降级为非 public(不影响行为)
        src = re.sub(r"\bpublic\s+(class|interface|enum)\b", r"\1", src)
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(src)
        rc, out, err = _run_cmd(["javac", "-encoding", "UTF-8", "Harness.java"], d, timeout)
        if rc != 0:
            return 0, len(cases), ["javac 编译失败: %s" % err.splitlines()[-1][:180] if err else "?"]
        rc, out, err = _run_cmd(["java", "-cp", ".", "Harness"], d, timeout)
        reported = sum(1 for l in out.splitlines() if l.startswith("CASE "))
        # 一个用例结果都没输出(启动即崩/超时)才算整体运行失败
        if rc != 0 and reported == 0:
            tail = (err or out).strip().splitlines()
            return 0, len(cases), ["运行失败: %s" % (tail[-1][:180] if tail else "?")]
        for line in out.splitlines():
            if line.startswith("CASE PASS"):
                results += 1
            elif line.startswith("CASE FAIL"):
                fails.append(line[:180])
        # 缺失的用例按失败计(超时/进程被杀时可能只剩前几条)
        if reported < len(cases):
            fails.append("仅输出了 %d/%d 个用例结果" % (reported, len(cases)))
        return results, len(cases), fails
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def run_js_tests(code, cases, timeout=15):
    """Node 判分: 追加 console.log 断言尾码, node 运行比对。
    cases: [(JS 表达式或语句, 期望 console.log 输出 repr)]"""
    results, fails = 0, []
    tail = "\n;(async()=>{\n"
    for i, (expr, expected) in enumerate(cases):
        tail += (
            "try{const v=await (eval(`(%s)`));"
            "console.log('R%d:'+JSON.stringify(v))}catch(e){console.log('R%d:E:'+e.message)}\n"
            % (expr, i, i)
        )
    tail += "})();\n"
    prog = code + "\n" + tail
    d = tempfile.mkdtemp(prefix="jsbench_")
    try:
        p = os.path.join(d, "t.js")
        with open(p, "w", encoding="utf-8") as f:
            f.write(prog)
        rc, out, err = _run_cmd(["node", "t.js"], d, timeout)
        if rc != 0 and not out:
            return 0, len(cases), ["node 运行失败: %s" % err.splitlines()[-1][:160] if err else "?"]
        for i, (expr, expected) in enumerate(cases):
            # expected 统一为 JSON 字面量 (与 JSON.stringify 输出对称)
            hit = False
            want_obj = None
            try:
                want_obj = json.loads(expected)
            except Exception:
                want_obj = expected
            for line in out.splitlines():
                if line.startswith("R%d:" % i):
                    got = line.split(":", 1)[1]
                    try:
                        hit = json.loads(got) == want_obj
                    except Exception:
                        hit = got == expected
                    if not hit:
                        fails.append("用例 %d 实际=%s 期望=%s" % (i, got[:60], expected[:60]))
                    break
            else:
                fails.append("用例 %d 无输出" % i)
            if hit:
                results += 1
        return results, len(cases), fails
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def run_lang_tests(lang, code, cases, timeout=15):
    if lang == "java":
        return run_java_tests(code, cases, timeout=timeout)
    if lang == "node":
        return run_js_tests(code, cases, timeout=timeout)
    return run_tests(code, cases, timeout=timeout)

# ============================================================ 题库

CODE_TASKS = [
    {
        "id": "C1", "name": "写代码: 区间解析 parse_ranges", "max_tokens": 2048,
        "prompt": (
            "请用 Python 实现函数 parse_ranges(s: str) -> list，要求:\n"
            "- 输入如 \"1-3,5,8-10\"，含区间与单值，返回升序去重后的整数列表\n"
            "- 区间可能倒置（如 \"10-8\"），需归一化后再展开\n"
            "- 输入可能含空白与重复（如 \"5, 5, 2\"）\n"
            "- 只输出一个 ```python 代码块，不要任何解释"
        ),
        "cases": [
            ("parse_ranges('1-3,5,8-10')", "[1, 2, 3, 5, 8, 9, 10]"),
            ("parse_ranges('10-8, 3-1')", "[1, 2, 3, 8, 9, 10]"),
            ("parse_ranges('5,5,2')", "[2, 5]"),
        ],
    },
    {
        "id": "C2", "name": "写代码: 嵌套字典展平 flatten_dict", "max_tokens": 2048,
        "prompt": (
            "请用 Python 实现函数 flatten_dict(d: dict) -> dict，要求:\n"
            "- 将嵌套字典展平为一级字典，键用 '.' 连接路径\n"
            "- 空字典视为叶子值保留（flatten_dict({'a': {}}) == {'a': {}}）\n"
            "- 非字典值（如列表、数字）一律视为叶子\n"
            "- 只输出一个 ```python 代码块，不要任何解释"
        ),
        "cases": [
            ("flatten_dict({'a': {'b': {'c': 1}}, 'd': 2})", "{'a.b.c': 1, 'd': 2}"),
            ("flatten_dict({'a': {}})", "{'a': {}}"),
            ("flatten_dict({'x': {'y': [1, 2]}, 'z': 3})", "{'x.y': [1, 2], 'z': 3}"),
        ],
    },
    {
        "id": "C3", "name": "找 bug: 滑动平均（可变默认参数 + off-by-one）", "max_tokens": 1600,
        "prompt": (
            "下面这段 Python 代码有两个 bug，请指出这两个 bug 并给出修复后的完整代码。\n"
            "注意：修复后的代码必须满足 moving_average([1,2,3,4,5], window=3) 返回 [2.0, 3.0, 4.0]。\n"
            "只输出一个 ```python 代码块（内含修复后的完整代码），另用一行简述两个 bug。\n\n"
            "```python\n"
            "def moving_average(values, window=3, averages=[]):\n"
            "    for i in range(len(values) - window):\n"
            "        chunk = values[i:i+window]\n"
            "        averages.append(sum(chunk) / window)\n"
            "    return averages\n"
            "```"
        ),
        "cases": [
            ("moving_average([1, 2, 3, 4, 5], window=3)", "[2.0, 3.0, 4.0]"),
            ("moving_average([10, 20, 30], window=3)", "[20.0]"),
        ],
        "bug_keywords": [
            ["默认参数", "mutable", "可变", "共享", "每次调用"],
            ["+ 1", "+1", "len(values) - window + 1", "少一个", "漏", "off-by-one", "差一"],
        ],
    },
    {
        "id": "C4", "name": "代码推理: 引用语义与集合去重", "max_tokens": 1200,
        "prompt": (
            "以下 Python 代码的 print 会输出什么？只输出 print 的完整输出内容，"
            "不要解释、不要代码块。\n\n"
            "```python\n"
            "x = [1, 2, 3]\n"
            "y = x\n"
            "y += [4]\n"
            "z = x * 2\n"
            "print(x, z, len({tuple(x), tuple(y)}))\n"
            "```"
        ),
        "exact": "[1, 2, 3, 4] [1, 2, 3, 4, 1, 2, 3, 4] 1",
    },
    {
        "id": "C5", "name": "写代码: 日志行解析 parse_log", "max_tokens": 2048,
        "prompt": (
            "请用 Python 实现函数 parse_log(line: str) -> dict，要求:\n"
            "- 解析形如 \"2024-01-15 10:23:45 ERROR Disk full at 95%\" 的日志行\n"
            "- 返回 {'date': 'YYYY-MM-DD', 'time': 'HH:MM:SS', 'level': 大写单词, 'msg': 剩余文本}\n"
            "- 日期与时间之间可能有一个或多个空格，msg 首尾不要多余空格\n"
            "- 只输出一个 ```python 代码块，不要任何解释"
        ),
        "cases": [
            ("parse_log('2024-01-15 10:23:45 ERROR Disk full at 95%')",
             "{'date': '2024-01-15', 'time': '10:23:45', 'level': 'ERROR', 'msg': 'Disk full at 95%'}"),
            ("parse_log('2023-12-01 08:00:00 WARN   low memory')",
             "{'date': '2023-12-01', 'time': '08:00:00', 'level': 'WARN', 'msg': 'low memory'}"),
        ],
    },
]

AGENT_TASKS = [
    {
        "id": "A1", "name": "Function calling: 单工具调用 + 工具结果回填推理",
    },
    {
        "id": "A2", "name": "结构化输出: json_schema 严格抽取",
    },
    {
        "id": "A3", "name": "指令遵循: 句数约束 + 结尾标记",
    },
]

QA_TASKS = [
    {
        "id": "Q1", "topic": "数学-方程",
        "prompt": ("苹果每斤 3 元，香蕉每斤 5 元。用 88 元恰好买 20 斤水果，且香蕉不少于苹果。"
                   "求苹果和香蕉各多少斤。一步步推理，最后一行写：答案：苹果 X 斤，香蕉 Y 斤"),
        "check": lambda ans: _pair_check(ans, "苹果", 6, "香蕉", 14),
    },
    {
        "id": "Q2", "topic": "数学-百分比陷阱",
        "prompt": ("一件商品先涨价 25%，再降价 25%，最终价格比原价低百分之几？"
                   "一步步推理，最后一行写：答案：X%"),
        "check": lambda ans: find_num(ans, 6.25),
    },
    {
        "id": "Q3", "topic": "数学-平均数",
        "prompt": ("小明前 3 天平均每天读 20 页书，后 2 天平均每天读 30 页。这 5 天平均每天读多少页？"
                   "一步步推理，最后一行写：答案：X 页"),
        "check": lambda ans: find_num(ans, 24),
    },
    {
        "id": "Q4", "topic": "数学-工程合作",
        "prompt": ("一项工程甲单独做 6 天完成，乙单独做 12 天完成。两人合作需要几天完成？"
                   "一步步推理，最后一行写：答案：X 天"),
        "check": lambda ans: find_num(ans, 4),
    },
    {
        "id": "Q5", "topic": "逻辑-说谎者",
        "prompt": ("A 说：B 在说谎。B 说：C 在说谎。C 说：A 和 B 都在说谎。"
                   "问谁说真话？一步步推理，最后一行写：答案：说真话的是（字母）"),
        "check": lambda ans: ("B" in ans.upper().replace("答案", "").replace("：", "").replace(":", "")
                              and not re.search(r"答案[:：]\s*A\s*$", ans.strip())
                              and "ABC" not in ans.upper().split("答案")[-1]
                              ) or re.search(r"答案[:：]\s*B\b", ans.strip()) is not None,
    },
    {
        "id": "Q6", "topic": "逻辑-抽屉原理",
        "prompt": ("一副扑克牌去掉大小王共 52 张（4 种花色各 13 张）。至少抽多少张，"
                   "才能保证其中有 4 张花色相同？一步步推理，最后一行写：答案：X 张"),
        "check": lambda ans: find_num(ans, 13),
    },
    {
        "id": "Q7", "topic": "陷阱-数字比较",
        "prompt": ("9.11 和 9.9 哪个数更大？只推理数字大小。最后一行写：答案：X 更大（X 填其中一个数）"),
        "check": lambda ans: ("9.9" in ans and "9.11" not in re.sub(r"9\.9", "", ans.split("答案")[-1])),
    },
    {
        "id": "Q8", "topic": "陷阱-字母计数",
        "prompt": ("英文单词 strawberry 里一共有几个字母 r？请逐个字母数一遍。最后一行写：答案：X 个"),
        "check": lambda ans: find_num(ans, 3),
    },
    {
        "id": "Q9", "topic": "数学-数列找规律",
        "prompt": ("数列 2, 6, 12, 20, 30, ... 的下一项是多少？写出通项规律。最后一行写：答案：X"),
        "check": lambda ans: find_num(ans, 42),
    },
    {
        "id": "Q10", "topic": "技术常识",
        "prompt": ("HTTP 状态码 404 表示什么含义？最后一行写：答案：X（用中文短语概括）"),
        "check": lambda ans: ("未找到" in ans or "不存在" in ans or "not found" in ans.lower()),
    },
]

TOOLS_DEF = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "查询指定城市的实时天气，返回温度（摄氏度）与天气状况",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string", "description": "城市名，如 上海"}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "send_email",
        "description": "发送电子邮件",
        "parameters": {"type": "object",
                       "properties": {"to": {"type": "string"}, "subject": {"type": "string"},
                                      "body": {"type": "string"}},
                       "required": ["to", "subject", "body"]}}},
]

WEATHER_TOOL_RESULT = {"city": "上海", "temp_c": 35, "condition": "晴", "humidity": "60%"}

# ============================================================ 评测执行

def bench_speed(args):
    print("== A. 速度基准 ==", flush=True)
    out = []
    # A1 短prompt长输出: 流式生成速度
    r, err = safe_call(api_chat, args.base_url, args.model,
        [{"role": "user", "content": "请从 1 数到 80，用顿号分隔，不要任何解释。"}],
        max_tokens=300, temperature=0, stream=True, api_key=args.api_key)
    item = {"id": "S1", "name": "短prompt生成(数数1-80)", "ok": False}
    if err:
        item["error"] = err
    else:
        gen = r.get("gen_tokens") or 0
        gen_t = (r.get("elapsed") or 0) - (r.get("ttft") or 0)
        item.update({"ok": True, "ttft_s": round(r.get("ttft") or 0, 3),
                     "ttft_content_s": round(r.get("ttft_content") or 0, 3)
                     if r.get("ttft_content") else None,
                     "gen_tokens": gen, "gen_time_s": round(gen_t, 2),
                     "tok_s": round(gen / gen_t, 1) if gen_t > 0.2 else None,
                     "usage": r.get("usage")})
    out.append(item); print(json.dumps(item, ensure_ascii=False), flush=True)

    # S2 长上下文预填充
    filler = ("数据段落：" + "、".join("第%d条记录值%d" % (i, i * 7 % 97) for i in range(1, 60)) + "。") * 3
    prompt = ("请仔细阅读以下数据（无需复述）：" + filler +
              "\n\n以上数据共约多少条记录？只回答数字，最后一行写：答案：X 条")
    r, err = safe_call(api_chat, args.base_url, args.model,
                       [{"role": "user", "content": prompt}], max_tokens=64, temperature=0, stream=True, api_key=args.api_key)
    item = {"id": "S2", "name": "长上下文预填充(约3000字)", "ok": False}
    if err:
        item["error"] = err
    else:
        ptok = (r.get("usage") or {}).get("prompt_tokens")
        item.update({"ok": True, "ttft_s": round(r.get("ttft") or 0, 3),
                     "prompt_tokens": ptok,
                     "prefill_tok_s": round(ptok / (r.get("ttft") or 1), 1)
                     if ptok and (r.get("ttft") or 0) > 0.05 else None,
                     "text_tail": r.get("text", "")[-80:]})
    out.append(item); print(json.dumps(item, ensure_ascii=False), flush=True)

    # S3 短任务端到端延迟（推理模型思维链会吃 token，预算给足）
    r, err = safe_call(api_chat, args.base_url, args.model,
        [{"role": "user", "content": "回复两个字：收到"}], max_tokens=64, temperature=0, stream=True, api_key=args.api_key)
    item = {"id": "S3", "name": "短回复端到端延迟", "ok": False}
    if err:
        item["error"] = err
    else:
        item.update({"ok": True, "ttft_s": round(r.get("ttft") or 0, 3),
                     "total_s": round(r.get("elapsed") or 0, 3),
                     "text": r.get("text", "")[:40]})
    out.append(item); print(json.dumps(item, ensure_ascii=False), flush=True)
    return out


def bench_code(args):
    print("== B. Code 基准 ==", flush=True)
    out = []
    for task in CODE_TASKS:
        r, err = safe_call(api_chat, args.base_url, args.model,
                           [{"role": "user", "content": task["prompt"]}],
                           max_tokens=args.max_tokens_override or task.get("max_tokens", 1024),
                           temperature=0, timeout=args.timeout, api_key=args.api_key)
        item = {"id": task["id"], "name": task["name"], "ok": False}
        if err:
            item["error"] = err
            out.append(item); print(json.dumps(item, ensure_ascii=False), flush=True)
            continue
        raw = answer_text(r)
        item["raw_len"] = len(raw)
        item["content_len"] = len(r.get("text") or "")
        item["reasoning_len"] = len(r.get("reasoning") or "")
        if task.get("exact") is not None:
            got = re.sub(r"\s+", " ", raw)
            want = re.sub(r"\s+", " ", task["exact"])
            item["ok"] = want in got
            item["got"] = got[:120]
        else:
            # 逐块验证取最优（兼容思维链模型：草稿块也算，取通过数最高的）
            blocks = re.findall(r"```[^\n]*\n(.*?)```", raw, re.S)
            if not blocks:
                blocks = [raw.strip()]
            best = None
            for b in blocks:
                code_b = b.strip()
                if not code_b:
                    continue
                passed, total, fails = run_tests(code_b, task["cases"])
                if best is None or passed > best[0]:
                    best = (passed, total, fails)
                if best[0] == best[1]:
                    break
            if best is None:
                best = (0, len(task["cases"]), ["未找到可验证的代码块"])
            passed, total, fails = best
            item.update({"passed": passed, "total": total,
                         "ok": passed == total, "fails": fails,
                         "n_code_blocks": len(blocks)})
            if task.get("bug_keywords"):
                kws_hit = [kw for kw in task["bug_keywords"]
                           if any(k in raw for k in kw)]
                item["bug_points_hit"] = "%d/2" % len(kws_hit)
        out.append(item); print(json.dumps(item, ensure_ascii=False), flush=True)
    return out


def bench_agent(args):
    print("== C. Agent 基准 ==", flush=True)
    out = []
    # A1 两轮工具调用
    item = {"id": "A1", "name": AGENT_TASKS[0]["name"]}
    msgs = [{"role": "user", "content": "帮我查一下上海现在的天气。你必须使用工具查询，不要自己编造天气数据。"}]
    r, err = safe_call(api_chat, args.base_url, args.model, msgs,
                       max_tokens=args.max_tokens_override or 800,
                       temperature=0, tools=TOOLS_DEF, timeout=args.timeout, api_key=args.api_key)
    if err:
        item["error"] = err
    else:
        tc = r.get("tool_calls") or []
        item["round1_tool_calls"] = [(t.get("function", {}).get("name"),
                                      t.get("function", {}).get("arguments")) for t in tc]
        called_ok = any(t.get("function", {}).get("name") == "get_weather" for t in tc)
        item["round1_called_get_weather"] = called_ok
        if called_ok and tc:
            args_ok = False
            try:
                a = json.loads(tc[0]["function"]["arguments"])
                args_ok = "上海" in str(a.get("city", ""))
            except Exception:
                args_ok = "上海" in str(tc[0]["function"].get("arguments", ""))
            item["round1_args_ok"] = args_ok
            # 第二轮: 回填工具结果
            msgs.append({"role": "assistant",
                         "content": None,
                         "tool_calls": tc})
            msgs.append({"role": "tool",
                         "tool_call_id": tc[0].get("id", "call_1"),
                         "content": json.dumps(WEATHER_TOOL_RESULT, ensure_ascii=False)})
            msgs.append({"role": "user",
                         "content": "根据刚才的查询结果回答：天气热不热？（超过30度答'热'，否则答'凉'，只答一个字）"})
            r2, err2 = safe_call(api_chat, args.base_url, args.model, msgs,
                                 max_tokens=args.max_tokens_override or 400,
                                 temperature=0, timeout=args.timeout, api_key=args.api_key)
            if err2:
                item["round2_error"] = err2
            else:
                ans = answer_text(r2).strip()
                item["round2_answer"] = ans[:40]
                item["round2_ok"] = ans.startswith("热") or "热" in ans[:6]
                item["round2_no_hallucinate"] = not (r2.get("text") or "").strip().startswith(("35", "晴"))
        item["ok"] = bool(called_ok and item.get("round1_args_ok") and item.get("round2_ok"))
    out.append(item); print(json.dumps(item, ensure_ascii=False), flush=True)

    # A2 json_schema
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "person_info", "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "hobbies": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "age", "hobbies"],
                "additionalProperties": False,
            },
        },
    }
    r, err = safe_call(api_chat, args.base_url, args.model,
        [{"role": "user", "content":
          "从这段话中抽取人物信息并按 schema 输出 JSON：王芳今年 34 岁，在杭州做产品经理，业余喜欢爬山、摄影和烘焙。"}],
        max_tokens=args.max_tokens_override or 600, temperature=0,
        response_format=schema, timeout=args.timeout, api_key=args.api_key)
    item = {"id": "A2", "name": AGENT_TASKS[1]["name"]}
    if err:
        item["error"] = err
    else:
        txt = answer_text(r).strip()
        try:
            obj = json.loads(re.sub(r"^```(?:json)?|```$", "", txt, flags=re.M).strip())
            item["parsed"] = obj
            item["ok"] = (obj.get("name") == "王芳" and obj.get("age") == 34
                          and isinstance(obj.get("hobbies"), list)
                          and all(h in obj["hobbies"] for h in ["爬山", "摄影", "烘焙"]))
        except Exception as e:
            item["ok"] = False
            item["parse_error"] = str(e)
            item["raw"] = txt[:150]
    out.append(item); print(json.dumps(item, ensure_ascii=False), flush=True)

    # A3 指令遵循
    r, err = safe_call(api_chat, args.base_url, args.model,
        [{"role": "user", "content":
          "用不超过 5 句话解释什么是 KV cache（面向程序员），最后一行必须单独输出：DONE"}],
        max_tokens=args.max_tokens_override or 800, temperature=0, timeout=args.timeout, api_key=args.api_key)
    item = {"id": "A3", "name": AGENT_TASKS[2]["name"]}
    if err:
        item["error"] = err
    else:
        txt = answer_text(r).strip()
        lines = [l for l in txt.splitlines() if l.strip()]
        last = lines[-1].strip() if lines else ""
        body = lines[:-1] if last.upper().startswith("DONE") else lines
        n_sent = len([s for s in re.split(r"[。！？!?]", " ".join(body)) if s.strip()])
        item.update({"lines": len(lines), "sentences": n_sent,
                     "last_line": last[:30],
                     "ok": last.upper().startswith("DONE") and 1 <= n_sent <= 5})
    out.append(item); print(json.dumps(item, ensure_ascii=False), flush=True)
    return out


def bench_qa(args):
    print("== D. 准确率基准 ==", flush=True)
    out = []
    for task in QA_TASKS:
        r, err = safe_call(api_chat, args.base_url, args.model,
                           [{"role": "user", "content": task["prompt"]}],
                           max_tokens=args.max_tokens_override or 1600,
                           temperature=0, timeout=args.timeout, api_key=args.api_key)
        item = {"id": task["id"], "topic": task["topic"]}
        if err:
            item["error"] = err
        else:
            ans = extract_answer(answer_text(r))
            item["answer_line"] = ans[:60]
            try:
                item["ok"] = bool(task["check"](ans))
            except Exception as e:
                item["ok"] = False
                item["check_error"] = str(e)
        out.append(item); print(json.dumps(item, ensure_ascii=False), flush=True)
    return out


# ============================================================ 报告

def summarize(result):
    speed = result.get("speed", [])
    code = result.get("code", [])
    agent = result.get("agent", [])
    qa = result.get("qa", [])
    if not any((speed, code, agent, qa)):
        return {}

    def ok(l):
        return sum(1 for x in l if x.get("ok"))

    s = {}
    s1 = next((x for x in speed if x["id"] == "S1"), {})
    s2 = next((x for x in speed if x["id"] == "S2"), {})
    s3 = next((x for x in speed if x["id"] == "S3"), {})
    s["tok_s"] = s1.get("tok_s")
    s["ttft_s"] = s1.get("ttft_s")
    s["prefill_tok_s"] = s2.get("prefill_tok_s")
    s["short_total_s"] = s3.get("total_s")
    code_passed = sum(x.get("passed", 1 if x.get("ok") else 0) for x in code)
    code_total = sum(x.get("total", 1) for x in code)
    if code_total:
        s["code"] = "%d/%d" % (code_passed, code_total)
    if agent:
        s["agent"] = "%d/%d" % (ok(agent), len(agent))
    if qa:
        s["qa"] = "%d/%d" % (ok(qa), len(qa))
    denom = (code_total if code_total else 0) + (len(qa) if qa else 0)
    if denom:
        s["overall_qa_code"] = "%.0f%%" % (
            100.0 * ((code_passed if code_total else 0) + (ok(qa) if qa else 0)) / denom)
    return s


def gen_report_md(result, meta, summary):
    L = []
    speed = result.get("speed", [])
    code = result.get("code", [])
    agent = result.get("agent", [])
    qa = result.get("qa", [])
    L.append("# LLM 模型评测报告")
    L.append("")
    L.append("- **被测模型**：%s" % meta["model"])
    L.append("- **端点**：%s" % meta["base_url"])
    L.append("- **量化/上下文**：%s / %s" % (meta.get("quant"), meta.get("ctx")))
    L.append("- **机器**：%s" % meta.get("machine"))
    L.append("- **测试时间**：%s" % meta.get("time"))
    L.append("- **采样**：全部 temperature=0（贪心解码，结果可复现）")
    L.append("")
    L.append("## 一、总分一览")
    L.append("")
    L.append("| 指标 | 结果 |")
    L.append("|---|---|")
    L.append("| 流式生成速度 | %s tok/s |" % summary.get("tok_s"))
    L.append("| 首 token 延迟 TTFT（短输入） | %s s |" % summary.get("ttft_s"))
    L.append("| 长上下文预填充 | %s tok/s |" % summary.get("prefill_tok_s"))
    L.append("| 短回复端到端延迟 | %s s |" % summary.get("short_total_s"))
    L.append("| Code（单测通过） | %s |" % summary.get("code"))
    L.append("| Agent（3 项） | %s |" % summary.get("agent"))
    L.append("| 准确率（10 题） | %s |" % summary.get("qa"))
    L.append("| 综合（Code 50%% + 准确率 50%%） | %s |" % summary.get("overall_qa_code"))
    L.append("")

    L.append("## 二、速度详情")
    L.append("")
    if speed:
        L.append("| 场景 | TTFT | 关键指标 |")
        L.append("|---|---|---|")
        for x in speed:
            if x["id"] == "S1":
                L.append("| 短prompt生成 | %s s | %s tok/s（%s tokens，其中思维链 %s） |" %
                         (x.get("ttft_s"), x.get("tok_s"), x.get("gen_tokens"),
                          (x.get("usage") or {}).get("completion_tokens_details", {}).get("reasoning_tokens", "?")))
            elif x["id"] == "S2":
                L.append("| 长上下文预填充 | %s s | prompt %s tokens，prefill %s tok/s |" %
                         (x.get("ttft_s"), x.get("prompt_tokens"), x.get("prefill_tok_s")))
            elif x["id"] == "S3":
                L.append("| 短回复延迟 | %s s | 端到端 %s s |" % (x.get("ttft_s"), x.get("total_s")))
    else:
        L.append("（本轮未测）")
    L.append("")

    L.append("## 三、Code 详情")
    L.append("")
    if code:
        L.append("| 题目 | 结果 | 说明 |")
        L.append("|---|---|---|")
        for x in code:
            if x.get("error"):
                L.append("| %s | ❌ | %s |" % (x["name"], x["error"]))
            elif x.get("exact") is not None:
                L.append("| %s | %s | 实际输出: `%s` |" %
                         (x["name"], "✅" if x.get("ok") else "❌", x.get("got", "")[:60]))
            else:
                note = x.get("fails") or ""
                if x.get("bug_points_hit"):
                    note = ("bug 识别 %s；" % x["bug_points_hit"]) + note
                L.append("| %s | %s | 单测 %s/%s %s |" %
                         (x["name"], "✅" if x.get("ok") else "❌",
                          x.get("passed"), x.get("total"), note if note else ""))
    else:
        L.append("（本轮未测）")
    L.append("")

    L.append("## 四、Agent 详情")
    L.append("")
    if agent:
        L.append("| 项 | 结果 | 说明 |")
        L.append("|---|---|---|")
        for x in agent:
            note = ""
            if x["id"] == "A1":
                note = "第一轮调用: %s；参数正确: %s；第二轮答: %s" % (
                    x.get("round1_tool_calls"), x.get("round1_args_ok"), x.get("round2_answer"))
            elif x["id"] == "A2":
                note = "解析结果: %s" % (x.get("parsed") or x.get("raw") or x.get("parse_error"))
            elif x["id"] == "A3":
                note = "句子数: %s，末行: %s" % (x.get("sentences"), x.get("last_line"))
            L.append("| %s | %s | %s |" % (x["name"], "✅" if x.get("ok") else "❌", note))
    else:
        L.append("（本轮未测）")
    L.append("")

    L.append("## 五、准确率详情")
    L.append("")
    if qa:
        L.append("| 题号 | 类型 | 结果 | 模型答案 |")
        L.append("|---|---|---|---|")
        for x in qa:
            L.append("| %s | %s | %s | %s |" %
                     (x["id"], x["topic"], "✅" if x.get("ok") else "❌",
                      (x.get("answer_line") or x.get("error") or "")[:50]))
    else:
        L.append("（本轮未测）")
    L.append("")
    return "\n".join(L)


# ============================================================ main

# ============================================================ hard 套件执行

def bench_hard(args):
    """hard 套件: B+ 9 代码题 + H 3 题静态 + H3 + Agent 4 题 + D 3 题 = 20 项。"""
    import hard_tasks as ht
    out = []

    # --- B+ 多语言代码题 ---
    print("== Hard B+: 多语言代码 ==", flush=True)
    for task in ht.HARD_CODE_TASKS:
        r, err = safe_call(api_chat, args.base_url, args.model,
                           [{"role": "user", "content": task["prompt"]}],
                           max_tokens=args.max_tokens_override or 4096,
                           temperature=0, timeout=args.timeout, api_key=args.api_key)
        item = {"id": task["id"], "name": task["name"], "ok": False}
        if err:
            item["error"] = err
            out.append(item); print(json.dumps(item, ensure_ascii=False), flush=True)
            continue
        raw = answer_text(r)
        item["content_len"] = len(r.get("text") or "")
        item["reasoning_len"] = len(r.get("reasoning") or "")
        blocks = re.findall(r"```[^\n]*\n(.*?)```", raw, re.S)
        best = None
        for b in blocks:
            passed, total, fails = run_lang_tests(task["lang"], b.strip(), task["cases"])
            if best is None or passed > best[0]:
                best = (passed, total, fails)
            if best[0] == best[1]:
                break
        if best is None:
            best = (0, len(task["cases"]), ["未找到代码块"])
        item.update({"passed": best[0], "total": best[1], "ok": best[0] == best[1],
                     "fails": best[2], "n_code_blocks": len(blocks), "raw": raw[:4000]})
        out.append(item); print(json.dumps(item, ensure_ascii=False), flush=True)

    # --- H 爬虫/HTML 静态题 ---
    print("== Hard H: 爬虫/HTML ==", flush=True)
    for task in ht.HARD_SCRAPE_TASKS:
        r, err = safe_call(api_chat, args.base_url, args.model,
                           [{"role": "user", "content": task["prompt"]}],
                           max_tokens=args.max_tokens_override or 4096,
                           temperature=0, timeout=args.timeout, api_key=args.api_key)
        item = {"id": task["id"], "name": task["name"], "ok": False}
        if err:
            item["error"] = err
            out.append(item); print(json.dumps(item, ensure_ascii=False), flush=True)
            continue
        raw = answer_text(r)
        item["reasoning_len"] = len(r.get("reasoning") or "")
        blocks = re.findall(r"```[^\n]*\n(.*?)```", raw, re.S)
        best = None
        for b in blocks:
            passed, total, fails = run_lang_tests("python", b.strip(), task["cases"])
            if best is None or passed > best[0]:
                best = (passed, total, fails)
            if best[0] == best[1]:
                break
        if best is None:
            best = (0, len(task["cases"]), ["未找到代码块"])
        item.update({"passed": best[0], "total": best[1], "ok": best[0] == best[1],
                     "fails": best[2], "n_code_blocks": len(blocks), "raw": raw[:4000]})
        out.append(item); print(json.dumps(item, ensure_ascii=False), flush=True)

    # --- H3 分页爬取（mock server 真实 HTTP） ---
    item = ht.run_h3(api_chat, args, ht.H3_PROMPT)
    item["reasoning_len"] = None
    out.append(item); print(json.dumps(item, ensure_ascii=False), flush=True)

    # --- Agent 4 题 ---
    print("== Hard C+: 长 Agent ==", flush=True)
    for agent_fn in (ht.agent_a4_order_chain, ht.agent_a5_error_degrade,
                     ht.agent_a6_instruction_persist, ht.agent_a7_crawl):
        item = agent_fn(api_chat, args)
        out.append(item); print(json.dumps(item, ensure_ascii=False), flush=True)

    # --- D 难题 ---
    print("== Hard D+: 难题 ==", flush=True)
    for task in ht.HARD_QA_TASKS:
        r, err = safe_call(api_chat, args.base_url, args.model,
                           [{"role": "user", "content": task["prompt"]}],
                           max_tokens=args.max_tokens_override or 4096,
                           temperature=0, timeout=args.timeout, api_key=args.api_key)
        item = {"id": task["id"], "topic": task["topic"]}
        if err:
            item["error"] = err
        elif task.get("cases"):
            # 代码形态的 D 题(如正则实现题): 提取代码块跑真实单测
            raw = answer_text(r)
            blocks = re.findall(r"```[^\n]*\n(.*?)```", raw, re.S)
            best = None
            for b in blocks:
                passed, total, fails = run_lang_tests("python", b.strip(), task["cases"])
                if best is None or passed > best[0]:
                    best = (passed, total, fails)
                if best[0] == best[1]:
                    break
            if best is None:
                best = (0, len(task["cases"]), ["未找到代码块"])
            item.update({"passed": best[0], "total": best[1],
                         "ok": best[0] == best[1], "fails": best[2]})
        else:
            ans = extract_answer(answer_text(r))
            item["answer_line"] = ans[:60]
            item["ok"] = bool(task["check"](ans))
        item["raw"] = answer_text(r)[:4000]
        out.append(item); print(json.dumps(item, ensure_ascii=False), flush=True)
    # 难度档标注(仅展示) + infra 错误标记
    for x in out:
        x.setdefault("tier", ht.TIER_OF.get(x["id"], "-"))
        if str(x.get("error") or "").startswith("INFRA:"):
            x["infra"] = True
    return out


GROUP_LABELS = (("hard_code", "代码 B+ (9 题)"), ("hard_scrape", "爬虫 H (4 题)"),
                 ("hard_agent", "Agent C+ (4 题)"), ("hard_puzzle", "难题 D+ (3 题)"))
TIER_LABELS = {"easy": "基础", "mid": "中等", "hard": "难", "-": "-"}


def _group_of(iid):
    if iid[0] in "JPN":
        return "code"
    return {"H": "scrape", "A": "agent"}.get(iid[0], "puzzle")


def summarize_hard(hard):
    """20 题等权: 一道题全部用例通过才算 1 道, 总分直接是 X/20(不做加权/百分制)。"""
    if not hard:
        return {}
    n = len(hard)
    passed = sum(1 for x in hard if x.get("ok"))
    s = {"hard": "%d/%d" % (passed, n),
         "hard_pct": "%.0f%%" % (100.0 * passed / n),
         "hard_cases": "%d/%d" % (sum(x.get("passed", 1 if x.get("ok") else 0) for x in hard),
                                  sum(x.get("total", 1) for x in hard)),
         "hard_errors": sum(1 for x in hard if x.get("infra"))}
    groups = {}
    for x in hard:
        g = _group_of(x["id"])
        groups.setdefault(g, [0, 0])
        groups[g][0] += 1 if x.get("ok") else 0
        groups[g][1] += 1
    for g, (p, t) in groups.items():
        s["hard_" + g] = "%d/%d" % (p, t)
    return s


def gen_hard_report_md(hard, meta, summary):
    """报告口径: X/20 等权数题, 不换算百分制、不加权。"""
    L = []
    L.append("# LLM 模型评测报告（hard 套件 20 题）")
    L.append("")
    L.append("- **被测模型**：%s" % meta["model"])
    L.append("- **端点**：%s" % meta["base_url"])
    L.append("- **机器**：%s" % meta.get("machine"))
    L.append("- **测试时间**：%s" % meta.get("time"))
    L.append("- **采样**：temperature=0（贪心解码，结果可复现）")
    L.append("- **判分口径**：一道题的全部用例通过才算 1 道；总分 = 通过题数 / 总题数，不加权")
    if meta.get("task_hash"):
        L.append("- **题库版本**：`%s`（只可与同版本结果对比）" % meta["task_hash"])
    L.append("")
    L.append("## 总分")
    L.append("")
    L.append("### %s" % summary.get("hard", "-"))
    L.append("")
    L.append("| 维度 | 通过 |")
    L.append("|---|---|")
    for k, label in GROUP_LABELS:
        if k in summary:
            L.append("| %s | %s |" % (label, summary[k]))
    if summary.get("hard_cases"):
        L.append("")
        L.append("> 用例级参考：%s（仅诊断用，不计入总分）" % summary["hard_cases"])
    by_tier = {}
    for x in hard:
        t = x.get("tier", "-")
        by_tier.setdefault(t, [0, 0])
        by_tier[t][0] += 1 if x.get("ok") else 0
        by_tier[t][1] += 1
    if by_tier:
        L.append("")
        L.append("| 难度档 | 通过 |")
        L.append("|---|---|")
        for t in ("easy", "mid", "hard", "-"):
            if t in by_tier:
                L.append("| %s | %d/%d |" % (TIER_LABELS.get(t, t), by_tier[t][0], by_tier[t][1]))
    errs = [x["id"] for x in hard if x.get("infra")]
    if errs:
        L.append("")
        L.append("> ⚠️ %d 道为 ERROR（超时/5xx，已重试仍失败），按未通过计：%s。建议重跑确认。"
                 % (len(errs), "、".join(errs)))
    L.append("")
    L.append("## 逐项明细")
    L.append("")
    L.append("| 题号 | 难度 | 题目 | 结果 | 说明 |")
    L.append("|---|---|---|---|---|")
    for x in hard:
        t = x.get("tier", "-")
        if x.get("infra"):
            res = "⚠️ ERROR"
        elif x.get("ok"):
            res = "✅"
        else:
            p, tt = x.get("passed"), x.get("total")
            res = "❌ %d/%d 用例" % (p, tt) if (p is not None and tt) else "❌"
        if x.get("error"):
            note = x["error"][:110]
        elif x.get("fails"):
            note = "; ".join(x["fails"])[:140]
        elif x.get("answer_line"):
            note = x["answer_line"][:60]
        elif x.get("detail"):
            note = json.dumps(x["detail"], ensure_ascii=False)[:140]
        else:
            note = "✓"
        L.append("| %s | %s | %s | %s | %s |"
                 % (x["id"], TIER_LABELS.get(t, t), x.get("name") or x.get("topic", ""), res, note))
    L.append("")
    return "\n".join(L)


SCHEMA = "hard-item-v2"   # 计分口径版本: v2 = 20 题等权数题(X/20)


def task_hash():
    """题库指纹: 结果只与同指纹的题库结果可比。"""
    import hashlib
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hard_tasks.py")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:1234")
    ap.add_argument("--model", default="", help="模型 ID（--selftest 时可省略）")
    ap.add_argument("--out", default=".temp/bench")
    ap.add_argument("--quant", default="")
    ap.add_argument("--ctx", default="")
    ap.add_argument("--machine", default="本机")
    ap.add_argument("--skip", default="", help="逗号分隔: speed,code,agent,qa")
    ap.add_argument("--timeout", type=int, default=420, help="单请求超时秒数")
    ap.add_argument("--max-tokens-override", type=int, default=0,
                    help="覆盖所有能力题的 max_tokens（0=用题库默认；推理模型建议 4096）")
    ap.add_argument("--api-key", default=os.environ.get("BENCH_API_KEY", ""),
                    help="API key（建议经 BENCH_API_KEY 环境变量传入）")
    ap.add_argument("--suite", default="core", choices=["core", "hard", "all"],
                    help="core=原 21 项；hard=20 题等权套件；all=两者都跑")
    ap.add_argument("--selftest", action="store_true",
                    help="只跑判分器自测: 用参考答案验证 20 道题的判分器是否 100%% 通过")
    args = ap.parse_args()

    if args.selftest:
        import selftest
        sys.exit(0 if selftest.run() else 1)
    if not args.model:
        ap.error("--model 必填（除非使用 --selftest）")

    skip = set(s.strip() for s in args.skip.split(",") if s.strip())
    result = {}
    if args.suite in ("core", "all"):
        if "speed" not in skip:
            result["speed"] = bench_speed(args)
        if "code" not in skip:
            result["code"] = bench_code(args)
        if "agent" not in skip:
            result["agent"] = bench_agent(args)
        if "qa" not in skip:
            result["qa"] = bench_qa(args)

    hard = []
    if args.suite in ("hard", "all"):
        hard = bench_hard(args)
        result["hard"] = hard

    summary = summarize(result)
    if hard:
        summary.update(summarize_hard(hard))
    print("\n== 总结 ==", flush=True)
    print(json.dumps(summary, ensure_ascii=False), flush=True)

    os.makedirs(args.out, exist_ok=True)
    th = task_hash()
    with open(os.path.join(args.out, "result.json"), "w", encoding="utf-8") as f:
        json.dump({"schema": SCHEMA, "task_hash": th, "model": args.model,
                   "suite": args.suite, "summary": summary, "detail": result},
                  f, ensure_ascii=False, indent=2)

    meta = {"model": args.model, "base_url": args.base_url,
            "quant": args.quant or "Q5_K_M(推测,可改)", "ctx": args.ctx or "-",
            "machine": args.machine or "本机", "task_hash": th,
            "time": time.strftime("%Y-%m-%d %H:%M")}
    if hard:
        md = gen_hard_report_md(hard, meta, summary)
    else:
        md = gen_report_md(result, meta, summary)
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("\n报告: %s/report.md  原始数据: %s/result.json" % (args.out, args.out), flush=True)


if __name__ == "__main__":
    main()
