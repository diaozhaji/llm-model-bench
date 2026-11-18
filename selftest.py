#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""判分器自测（`python3 lm_model_bench.py --selftest`）。

20 道题各配一份参考答案，全部通过才算判分器可信——判分器本身错了的话，
所有跨模型分数都不可用（本项目真实踩过：参考答案笔误 + 题面契约不一致
导致模型被集体误判）。

参考答案一律按**题面契约**写（例如 H3 的 crawl_all(base) 内部自己拼
/api/items?page=1），不允许为了迁就判分器而改写——否则自测只会掩盖 bug。
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hard_tasks as ht
from lm_model_bench import run_lang_tests, extract_answer, _pair_check

PASS, FAIL = 0, 0
RESULTS = []


def check(tag, got, expect_ok=True):
    """got = (passed, total, fails)。"""
    global PASS, FAIL
    p, t, fails = got
    ok = (p == t and t > 0) if expect_ok else (p == 0)
    status = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    RESULTS.append((tag, status, "%d/%d" % (p, t), (fails or [])[:1]))
    print("[%s] %-4s %d/%d %s" % (status, tag, p, t, (fails or [])[:1]), flush=True)


def check_bool(tag, ok, note=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print("[PASS] %-4s %s" % (tag, note), flush=True)
    else:
        FAIL += 1
        print("[FAIL] %-4s %s" % (tag, note), flush=True)
    RESULTS.append((tag, "PASS" if ok else "FAIL", "-", note))


# ============================================================ 参考答案: Java
STD_J1 = '''
class IntBox {
    static String intBox(int a, int b) {
        Integer x = a, y = b;
        return (x == y) ? "EQ" : "NE";
    }
}'''

STD_J2 = '''
class Mutator {
    static String mutate(java.util.List<String> list) {
        try {
            for (String s : list) {
                if ("del".equals(s)) { list.remove(s); }
            }
            return "KEPT";
        } catch (java.util.ConcurrentModificationException e) {
            return "CME";
        }
    }
}'''

STD_J3 = '''
class LRUCache {
    private final int capacity;
    private final java.util.LinkedHashMap<Integer,Integer> map;
    LRUCache(int capacity) {
        this.capacity = capacity;
        this.map = new java.util.LinkedHashMap<Integer,Integer>(16, 0.75f, true) {
            protected boolean removeEldestEntry(java.util.Map.Entry<Integer,Integer> e) {
                return size() > LRUCache.this.capacity;
            }
        };
    }
    int get(int key) { return map.getOrDefault(key, -1); }
    void put(int key, int value) { map.put(key, value); }
}'''

STD_J4 = '''
class WC {
    static int wordCount(String text) {
        java.util.Map<String,Integer> freq = new java.util.HashMap<>();
        for (String w : text.split(" ")) {
            freq.merge(w, 1, Integer::sum);
        }
        int total = 0;
        for (int v : freq.values()) total += v;
        return total;
    }
}'''

# ============================================================ 参考答案: Python
STD_P1 = '''
def make_adders():
    adders = []
    for i in range(3):
        def adder(i=i):
            return i + 10
        adders.append(adder)
    return adders
'''

STD_P2 = '''
import asyncio
async def gather_order():
    async def work(i):
        await asyncio.sleep(i * 0.1)
        return str(i)
    out = []
    for fut in asyncio.as_completed([work(1), work(2), work(3)]):
        out.append(await fut)
    return out
'''

STD_P3 = '''
def probe():
    class Base:
        tag = 'B'
    class Sub(Base):
        def __init__(self):
            self.tag = 'S'
    s = Sub()
    a = s.tag
    Base.tag = 'X'
    b = s.tag
    return a + b
'''

# ============================================================ 参考答案: Node
STD_N1 = '''
function order() {
  return new Promise((resolve) => {
    const seq = [];
    seq.push('s');
    process.nextTick(() => seq.push('n'));
    Promise.resolve().then(() => seq.push('p'));
    setTimeout(() => { seq.push('t'); resolve(seq.join('')); }, 0);
  });
}'''

STD_N2 = '''
function varSum() {
  var fns = [];
  for (var i = 0; i < 3; i++) {
    fns.push(() => i);
  }
  var sum = 0;
  for (var f of fns) sum += f();
  return sum;
}'''

# ============================================================ 参考答案: 爬虫/HTML
STD_H1 = '''
import re, html as H

def parse_products(html):
    out = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
        tds = re.findall(r"<td([^>]*)>(.*?)</td>", row, re.S)
        if len(tds) < 3:
            continue
        name = H.unescape(tds[0][1]).replace("\\xa0", " ").strip()
        pm = re.search(r'data-price="(\\d+)"', tds[1][0])
        price = int(pm.group(1)) if pm else None
        status = H.unescape(re.sub(r"<[^>]+>", "", tds[2][1])).replace("\\xa0", " ").strip()
        out.append({"name": name, "price": price, "in_stock": status == "in stock"})
    return out
'''

STD_H2 = '''
import re
from urllib.parse import urlparse, unquote

def extract_requests(log_text):
    out = []
    for line in log_text.splitlines():
        m = re.search(r'"GET\\s+(\\S+)\\s+HTTP/[^"]+"\\s+(\\d{3})', line)
        if not m:
            continue
        url, status = m.group(1), int(m.group(2))
        p = urlparse(url)
        out.append({"path": p.path, "query": unquote(p.query) if p.query else None,
                    "status": status})
    return out
'''

STD_H4 = '''
import re
from urllib.parse import urljoin, urlparse

def collect_links(html, base):
    base_host = urlparse(base).netloc
    out, seen = [], set()
    for m in re.finditer(r'href="([^"]+)"', html):
        href = m.group(1)
        if href.startswith("mailto:") or href.startswith("#"):
            continue
        u = urljoin(base, href).split("#")[0]
        if not u.startswith("http"):
            continue
        if urlparse(u).netloc != base_host:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out
'''

# H3: 严格按题面 —— 收到 base, 自己拼 /api/items?page=1
STD_H3 = '''
import json, urllib.request
from urllib.parse import urljoin

def crawl_all(base_url):
    items, seen = [], set()
    url = base_url.rstrip("/") + "/api/items?page=1"
    while url:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        for it in data.get("items", []):
            if not it.get("name") or it["id"] in seen:
                continue
            seen.add(it["id"])
            items.append(it)
        url = urljoin(base_url, data["next"]) if data.get("next") else None
    return sorted(items, key=lambda x: x["id"])
'''

STD_D1 = '''
import re

def extract_phones(text):
    out = []
    for m in re.finditer(r"1[3-9](?:-?\\d){9}", text):
        s = m.group(0).replace("-", "")
        if len(s) == 11 and s.isdigit():
            out.append(s)
    return out
'''


# ============================================================ 假 api_chat（给 Agent 题用）
class FakeArgs(object):
    base_url = "http://fake.local/v1"
    model = "selftest"
    timeout = 10
    api_key = None
    max_tokens_override = 0


def _tool_call(idx, name, args):
    return {"id": "call_%d" % idx, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


def fake_api_chat_factory(plan, final_text, json_every_turn=False, trace_id="TR-8848"):
    """plan: [(tool_name, args_dict), ...]; 按轮次依次返回工具调用, 用完后返回 final_text。
    json_every_turn=True 时, 每轮正文都带 trace_id(用于 A6 指令保持题)。"""
    def fake(base_url, model, messages, max_tokens=512, temperature=0.0,
             tools=None, response_format=None, timeout=180, stream=False, api_key=None):
        n_assistant = sum(1 for m in messages if m.get("role") == "assistant")
        text = final_text
        if json_every_turn:
            text = json.dumps({"trace_id": trace_id, "result": final_text}, ensure_ascii=False)
        if n_assistant < len(plan):
            name, args = plan[n_assistant]
            return {"text": "", "reasoning": "", "tool_calls": [_tool_call(n_assistant, name, args)],
                    "usage": {}, "elapsed": 0.0}
        return {"text": text, "reasoning": "", "tool_calls": None, "usage": {}, "elapsed": 0.0}
    return fake


def fake_code_chat(code):
    """返回一个只输出代码块的假 API（给 H3 用）。"""
    def fake(base_url, model, messages, max_tokens=512, temperature=0.0,
             tools=None, response_format=None, timeout=180, stream=False, api_key=None):
        return {"text": "```python\n" + code + "\n```", "reasoning": "", "tool_calls": None,
                "usage": {}, "elapsed": 0.0}
    return fake


def run():
    print("== 判分器自测（命中参考答案, 20 题）==", flush=True)

    # ---- 代码题: 真实执行
    for tid, std, lang in [("J1", STD_J1, "java"), ("J2", STD_J2, "java"),
                           ("J3", STD_J3, "java"), ("J4", STD_J4, "java"),
                           ("P1", STD_P1, "python"), ("P2", STD_P2, "python"),
                           ("P3", STD_P3, "python"), ("N1", STD_N1, "node"),
                           ("N2", STD_N2, "node")]:
        task = next(x for x in ht.HARD_CODE_TASKS if x["id"] == tid)
        check(tid, run_lang_tests(lang, std, task["cases"]))

    for tid, std in [("H1", STD_H1), ("H2", STD_H2), ("H4", STD_H4)]:
        task = next(x for x in ht.HARD_SCRAPE_TASKS if x["id"] == tid)
        check(tid, run_lang_tests("python", std, task["cases"]))

    # ---- H3: 走真实 mock server + 题面契约
    item = ht.run_h3(fake_code_chat(STD_H3), FakeArgs(), ht.H3_PROMPT)
    check_bool("H3", bool(item.get("ok")),
               "契约: crawl_all(base)" + ("" if item.get("ok") else " -> %s" % item.get("fails")))

    # ---- D1: 正则实现(真实执行)
    d1 = next(t for t in ht.HARD_QA_TASKS if t["id"] == "D1")
    check("D1", run_lang_tests("python", STD_D1, d1["cases"]))

    # ---- D2/D3: 答案题(直接喂正确答案 + 反向验证)
    d2 = next(t for t in ht.HARD_QA_TASKS if t["id"] == "D2")
    good2 = extract_answer("答案：追上用时 1 小时，距出发点 9 公里")
    bad2 = extract_answer("答案：追上用时 1.5 小时，距出发点 6 公里")
    check_bool("D2", bool(d2["check"](good2)) and not bool(d2["check"](bad2)),
               "正解通过/错解拒绝: %r" % good2)

    d3 = next(t for t in ht.HARD_QA_TASKS if t["id"] == "D3")
    good3 = extract_answer("答案：S1=2,S2=1,S3=1,S4=1,不同=S1")
    bad3 = extract_answer("答案：S1=1,S2=1,S3=1,S4=1,不同=S2")
    check_bool("D3", bool(d3["check"](good3)) and not bool(d3["check"](bad3)),
               "正解通过/错解拒绝: %r" % good3)

    # ---- Agent 题: 用脚本化假后端跑完整多轮流程
    a4_plan = [("get_customer_level", {"customer_id": "C-001"}),
               ("query_discount", {"level": "VIP1"}),
               ("query_inventory", {"sku": "SKU-A"}),
               ("query_price", {"sku": "SKU-A"}),
               ("query_shipping", {"weight_kg": 2})]
    a4_text = "总价 105 元（100×0.9+15）；库存 12 件，够买 2 件。"
    a4 = ht.agent_a4_order_chain(fake_api_chat_factory(a4_plan, a4_text), FakeArgs())
    check_bool("A4", bool(a4.get("ok")), json.dumps(a4.get("detail", {}), ensure_ascii=False))

    a5_plan = [("query_inventory", {"sku": "SKU-B"})]
    a5_text = "SKU-B 库存为 0，缺货，无法下单。建议改买 SKU-C（库存 5 件，单价 80 元）。"
    a5 = ht.agent_a5_error_degrade(fake_api_chat_factory(a5_plan, a5_text), FakeArgs())
    check_bool("A5", bool(a5.get("ok")), json.dumps(a5.get("detail", {}), ensure_ascii=False))

    # A6: 每轮都遵守 JSON + trace_id
    a6 = ht.agent_a6_instruction_persist(
        fake_api_chat_factory([], "已查询", json_every_turn=True), FakeArgs())
    check_bool("A6", bool(a6.get("ok")),
               "保持 %d/%d 轮" % (a6.get("persist_hits", 0), a6.get("n_checks", 0)))

    # A6 反向: 后半程不再输出 JSON -> 必须判不通过
    def a6_drift(base_url, model, messages, **kw):
        n_assistant = sum(1 for m in messages if m.get("role") == "assistant")
        if n_assistant >= 4:
            return {"text": "好的，已经查完了。", "reasoning": "", "tool_calls": None,
                    "usage": {}, "elapsed": 0.0}
        return {"text": json.dumps({"trace_id": "TR-8848", "r": "ok"}), "reasoning": "",
                "tool_calls": None, "usage": {}, "elapsed": 0.0}
    a6b = ht.agent_a6_instruction_persist(a6_drift, FakeArgs())
    check_bool("A6-n", not a6b.get("ok"), "后半程漂移应判不通过(实际 %s)" % a6b.get("ok"))

    root = ht.CRAWL_PAGES["http://mock.local/"]
    a_page = ht.CRAWL_PAGES["http://mock.local/a"]
    a7_plan = [("fetch_page", {"url": "http://mock.local/"}),
               ("parse_links", {"html": root["body"], "base": "http://mock.local/"}),
               ("fetch_page", {"url": "http://mock.local/a"}),
               ("parse_links", {"html": a_page["body"], "base": "http://mock.local/"}),
               ("fetch_page", {"url": "http://mock.local/b"}),
               ("fetch_page", {"url": "http://mock.local/d"}),
               ("fetch_page", {"url": "http://mock.local/c"})]
    a7_text = ("成功抓取(200) 4 个页面：http://mock.local/ 、/a 、/b 、/d；"
               "404 页面 1 个：http://mock.local/c。已跳过外部域链接。")
    a7 = ht.agent_a7_crawl(fake_api_chat_factory(a7_plan, a7_text), FakeArgs())
    check_bool("A7", bool(a7.get("ok")), json.dumps(a7.get("detail", {}), ensure_ascii=False))

    # A7 反向: 抓了外部域(违反题面) 必须判不通过
    a7_bad_plan = list(a7_plan) + [("fetch_page", {"url": "http://ext.local/e"})]
    a7b = ht.agent_a7_crawl(fake_api_chat_factory(a7_bad_plan, a7_text), FakeArgs())
    check_bool("A7-n", not a7b.get("ok"), "抓外域应判不通过(实际 %s)" % a7b.get("ok"))

    # ---- 判分器反向用例: 这些必须被判错, 否则就是假阳性
    q1 = next(t for t in ht.HARD_QA_TASKS if t["id"] == "D2")
    check_bool("pair", _pair_check("苹果 6 斤，香蕉 14 斤", "苹果", 6, "香蕉", 14)
               and not _pair_check("苹果 14 斤，香蕉 6 斤", "苹果", 6, "香蕉", 14),
               "核心套件 Q1 名值配对")

    print("\n== 自测结果: %d PASS / %d FAIL ==" % (PASS, FAIL), flush=True)
    if FAIL:
        print("判分器不可信, 不要用这套题的结果比较模型。", flush=True)
    return FAIL == 0


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
