#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hard 区分度套件 —— 20 题（题库与判分用例）
组 B+: 多语言工程代码 9 题 (Java 4 / Python 3 / Node 2)
组 H : 爬虫/HTML 解析/网页查找 4 题 (纯标准库, 真实执行判分)
组 C+: 长 Agent 4 题 (多轮工具链/报错降级/指令保持/爬虫编排)
组 D+: 准确率难题 3 题

代码题: 每题给模型看的数据内嵌在 prompt 里; 判分用例自带同一份数据,
       模型代码在隔离子进程中执行, 不依赖评测器内部状态。
Agent 题: agent_fn(api_chat, args) -> item dict, 由 bench_hard 驱动。
H3 分页爬取: bench_hard 内置 mock HTTP server。
"""

import json
import re
import threading
import http.server
import socketserver
import urllib.request




def _set_final(item, resp, cap=8000):
    """final_answer 存全文(判分用), *_short 存短版(报告展示用)。
    之前只存 [:200] 导致长答复里的关键数字被截掉, 判成失败。
    resp 可以是字符串, 也可以是 api 返回 dict: 正文为空且思维链很长时标记 starved。"""
    if isinstance(resp, dict):
        ans = (resp.get("text") or "").strip()
        if not ans and len(resp.get("reasoning") or "") >= 3000:
            item["starved"] = True
    else:
        ans = (resp or "").strip()
    item["final_answer"] = ans[:cap]
    item["final_answer_short"] = ans[:200]
    return ans


def _d3_check(ans):
    """D3 判分: 四段输出值全对 + 指出不同段。支持全角/空格/中文冒号。"""
    norm = (ans or "").replace("＝", "=").replace("：", ":").replace("，", ",")
    norm = re.sub(r"[ \t\u3000]", "", norm)
    vals = {int(a): int(b) for a, b in re.findall(r"S([1-4])=(-?\d+)", norm)}
    if vals != {1: 2, 2: 1, 3: 1, 4: 1}:
        return False
    m = re.search(r"不同=S([1-4])", norm)
    return bool(m and m.group(1) == "1")


def find_num(text, n):
    import re as _re
    nums = _re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    for x in nums:
        try:
            if abs(float(x) - n) < 1e-9:
                return True
        except ValueError:
            pass
    return False


# ============================================================ 公共 fixture（题面 + 判分共用同一份数据）

HTML_PRODUCTS = '''<div class="products">
  <table id="list">
    <tr><th>name</th><th>price</th><th>status</th></tr>
    <tr><td>  机械键盘 K8 </td><td data-price="399">¥399.00</td><td><span class="ok">in stock</span></td></tr>
    <tr><td>显示器 27寸</td><td data-price="1299">¥1,299.00</td><td><span class="bad">sold out</span></td></tr>
    <tr><td>鼠标&nbsp;M3</td><td data-price="99">¥99.00</td><td><span class="ok">in stock</span></td></tr>
    <tr class="promo"><td>耳机 E5</td><td data-price="599">¥599.00</td><td><span class="ok">in&nbsp;stock</span></td></tr>
    <tr><td>线材&amp;配件</td><td data-price="45">¥45.00</td><td><span class="ok">in stock</span></td></tr>
    <tr><td>残行缺状态</td><td data-price="7">¥7.00</td></tr>
  </table>
</div>'''

H1_EXPECTED = [
    {"name": "机械键盘 K8", "price": 399, "in_stock": True},
    {"name": "显示器 27寸", "price": 1299, "in_stock": False},
    {"name": "鼠标 M3", "price": 99, "in_stock": True},
    {"name": "耳机 E5", "price": 599, "in_stock": True},
    {"name": "线材&配件", "price": 45, "in_stock": True},
]

NGINX_LOG = '''83.4.11.2 - - [17/Nov/2026:10:23:45 +0800] "GET /search?q=%E6%89%8B%E6%9C%BA HTTP/1.1" 200 5321 "-" "curl/8.4"
10.0.0.5 - - [17/Nov/2026:10:24:01 +0800] "GET /item/1024?from=list&page=2 HTTP/1.1" 200 4096 "-" "Mozilla/5.0"
203.0.113.9 - - [17/Nov/2026:10:24:30 +0800] "GET /search?q=%E9%94%AE%E7%9B%98 HTTP/1.1" 200 8192 "-" "curl/8.4"
83.4.11.2 - - [17/Nov/2026:10:25:12 +0800] "POST /api/cart HTTP/1.1" 405 512 "-" "python-requests/2.31"
10.0.0.5 - - [17/Nov/2026:10:25:44 +0800] "GET /item/2048?q=%E5%86%B7%E5%86%BB HTTP/1.1" 200 2048 "-" "Mozilla/5.0"
2001:db8::1 - - [17/Nov/2026:10:26:10 +0800] "GET /admin/x?a=1 HTTP/1.1" 500 128 "-" "curl/8.4"
this line is broken and has no quoted request'''

H2_EXPECTED = [
    {"path": "/search", "query": "q=手机", "status": 200},
    {"path": "/item/1024", "query": "from=list&page=2", "status": 200},
    {"path": "/search", "query": "q=键盘", "status": 200},
    {"path": "/item/2048", "query": "q=冷冻", "status": 200},
    {"path": "/admin/x", "query": "a=1", "status": 500},
]

LINKS_HTML = '''<html><body>
<a href="/about">About</a>
<a href="https://example.com/a?x=1">A</a>
<a href="https://example.com/b#frag">B</a>
<a href="/about">About duplicate</a>
<a href="https://other-site.org/c">C</a>
<a href="docs/guide.html">Guide</a>
<a href="#top">Top</a>
<a href="mailto:hi@example.com">Mail</a>
</body></html>'''

H4_EXPECTED = [
    "https://example.com/about",
    "https://example.com/a?x=1",
    "https://example.com/b",
    "https://example.com/docs/guide.html",
]


# ============================================================ 组 B+: 多语言代码

HARD_CODE_TASKS = [
    {
        "id": "J1", "lang": "java", "name": "Java: Integer 缓存 == 陷阱",
        "prompt": (
            "请用 Java 实现 `static String intBox(int a, int b)`：装箱两个 int 为 Integer 后用 == 比较，"
            "相等返回 \"EQ\"，否则返回 \"NE\"。只输出一个 ```java 代码块（内含一个类 IntBox，"
            "不要 package 声明，不要任何解释）。"
        ),
        "cases": [
            "String r1=IntBox.intBox(100,100); System.out.println(\"CASE \"+(\"EQ\".equals(r1)?\"PASS\":\"FAIL\")+\" int100\");",
            "String r2=IntBox.intBox(200,200); System.out.println(\"CASE \"+(\"NE\".equals(r2)?\"PASS\":\"FAIL\")+\" int200\");",
        ],
    },
    {
        "id": "J2", "lang": "java", "name": "Java: 集合遍历修改(CME)语义",
        "prompt": (
            "请用 Java 实现 `static String mutate(java.util.List<String> list)`：用增强 for 循环遍历 list，"
            "遇到 \"del\" 元素时通过 list.remove 将其删除，然后继续循环迭代（不得提前 break 或 return）；"
            "若整个遍历正常结束没有 \"del\" 则返回 \"KEPT\"。"
            "删除后继续用增强 for 迭代会触发 ArrayList 的 fail-fast "
            "ConcurrentModificationException，请捕获它并返回 \"CME\"。"
            "只输出一个 ```java 代码块（内含一个类 Mutator，不要 package，不要解释）。"
        ),
        "cases": [
            "String r1=Mutator.mutate(new java.util.ArrayList<>(java.util.Arrays.asList(\"a\",\"del\",\"b\",\"c\"))); System.out.println(\"CASE \"+(\"CME\".equals(r1)?\"PASS\":\"FAIL\")+\" cme\");",
            "String r2=Mutator.mutate(new java.util.ArrayList<>(java.util.Arrays.asList(\"a\",\"b\"))); System.out.println(\"CASE \"+(\"KEPT\".equals(r2)?\"PASS\":\"FAIL\")+\" kept\");",
        ],
    },
    {
        "id": "J3", "lang": "java", "name": "Java: LRU 缓存实现",
        "prompt": (
            "请用 Java 实现 LRU 缓存类 LRUCache（构造参数为容量 capacity）："
            "get(key) 不存在返回 -1，命中时刷新为最近使用；"
            "put(key,value) 已存在的 key 更新值并刷新为最近使用（不触发淘汰）；"
            "插入新 key 超过容量时淘汰最久未使用的条目。"
            "只输出一个 ```java 代码块（内含 LRUCache 类，不要 package，不要解释）。"
        ),
        "cases": [
            (
                "LRUCache c=new LRUCache(2); c.put(1,10); c.put(2,20); c.get(1);"
                "c.put(3,30); int v2=c.get(2);"
                "System.out.println(\"CASE \"+(v2==-1?\"PASS\":\"FAIL\")+\" evict-eldest\");"
            ),
            (
                "LRUCache c2=new LRUCache(2); c2.put(1,10); c2.put(2,20); c2.put(1,11);"
                "int v1=c2.get(1), v2=c2.get(2);"
                "System.out.println(\"CASE \"+(v1==11&&v2==20?\"PASS\":\"FAIL\")+\" put-existing-no-evict\");"
            ),
            (
                "LRUCache c3=new LRUCache(1); c3.put(1,10); c3.get(1); c3.put(2,20);"
                "int v=c3.get(1), v2=c3.get(2);"
                "System.out.println(\"CASE \"+(v==-1&&v2==20?\"PASS\":\"FAIL\")+\" cap1\");"
            ),
            (
                "LRUCache c4=new LRUCache(3); c4.put(1,1); c4.put(2,2); c4.put(3,3);"
                "c4.get(1); c4.get(3); c4.put(4,4); int v2=c4.get(2);"
                "System.out.println(\"CASE \"+(v2==-1?\"PASS\":\"FAIL\")+\" get-refresh\");"
            ),
        ],
    },
    {
        "id": "J4", "lang": "java", "name": "Java: Map.merge/computeIfAbsent 语义",
        "prompt": (
            "请用 Java 实现 `static int wordCount(String text)`：统计文本总单词数（按空格分隔），"
            "内部必须用 HashMap 的 merge 或 computeIfAbsent 完成词频计数（不得用 if-containsKey 手写判空）。"
            "只输出一个 ```java 代码块（内含一个类 WC，不要 package，不要解释）。"
        ),
        "cases": [
            "int n=WC.wordCount(\"a b a c b a\"); System.out.println(\"CASE \"+(n==6?\"PASS\":\"FAIL\")+\" count6\");",
            "int n2=WC.wordCount(\"x\"); System.out.println(\"CASE \"+(n2==1?\"PASS\":\"FAIL\")+\" count1\");",
        ],
    },
    {
        "id": "P1", "lang": "python", "name": "Python: 闭包延迟绑定",
        "prompt": (
            "请用 Python 实现函数 make_adders()：返回含 3 个函数的列表，第 i 个函数调用时返回 i+10"
            "（adders[0]()==10, adders[1]()==11, adders[2]()==12）。必须用循环生成，"
            "且每个函数必须正确捕获自己的 i（不得手写三个字面函数）。"
            "只输出一个 ```python 代码块，不要解释。"
        ),
        "cases": [
            ("make_adders()[0]()", "10"),
            ("make_adders()[1]()", "11"),
            ("make_adders()[2]()", "12"),
        ],
    },
    {
        "id": "P2", "lang": "python", "name": "Python: asyncio 并发时序",
        "prompt": (
            "请用 Python 实现协程 async def gather_order()：并发运行 work(1)/work(2)/work(3)"
            "（work(i) 定义为 await asyncio.sleep(i*0.1) 后返回 str(i)），按完成时间顺序收集返回值，"
            "返回该列表（即 ['1','2','3']）。只输出一个 ```python 代码块，不要解释。"
        ),
        "cases": [
            ("__import__('asyncio').run(gather_order())", "['1', '2', '3']"),
        ],
    },
    {
        "id": "P3", "lang": "python", "name": "Python: 类属性与实例属性继承链",
        "prompt": (
            "请用 Python 实现函数 probe()：定义类 Base（类属性 tag='B'）与 Sub(Base)（__init__ 中"
            "设置实例属性 tag='S'）；创建 s=Sub() 并读 s.tag 存入变量 a；然后执行 Base.tag='X'；"
            "再读 s.tag 存入变量 b；返回字符串 a+b。只输出一个 ```python 代码块，不要解释。"
        ),
        "cases": [
            ("probe()", "'SS'"),
        ],
    },
    {
        "id": "N1", "lang": "node", "name": "Node: 事件循环微任务/宏任务时序",
        "prompt": (
            "请用 Node.js 实现函数 order()：返回 Promise<string>。执行体依次安排四种任务，"
            "每种任务把自己的标记追加到数组：同步代码立即追加 's'；process.nextTick 追加 'n'；"
            "Promise.resolve().then 微任务追加 'p'；setTimeout(cb,0) 宏任务追加 't'。"
            "在宏任务完成后 resolve 数组 join 的字符串（应为 \"snpt\"）。"
            "只输出一个 ```javascript 代码块，不要解释。"
        ),
        "cases": [
            ("(async()=>await order())()", '"snpt"'),
        ],
    },
    {
        "id": "N2", "lang": "node", "name": "Node: 闭包与 var 循环捕获",
        "prompt": (
            "请用 Node.js 实现函数 varSum()：内部用 for 循环（循环变量必须用 var 声明）创建 3 个函数"
            "存入数组，第 i 个函数返回 i；var 的函数级作用域语义下这三个函数共享同一个 i。"
            "循环结束后再给共享 i 的最终值做判断：返回这 3 个函数调用返回值之和。"
            "只输出一个 ```javascript 代码块，不要解释。"
        ),
        "cases": [
            ("varSum()", "9"),
        ],
    },
]


# ============================================================ 组 H: 爬虫/HTML/网页（Python 标准库, 数据内嵌）

HARD_SCRAPE_TASKS = [
    {
        "id": "H1", "lang": "python", "name": "爬虫: HTML 表格结构化解析",
        "prompt": (
            "请用 Python 标准库（html.parser 或 re，禁用第三方库如 bs4/lxml）实现函数 "
            "parse_products(html: str) -> list：解析商品表格 HTML，返回字典列表，"
            "每项 {'name': str, 'price': int, 'in_stock': bool}。要求："
            "name 取每行第一个 <td> 文本，先做 HTML 实体解码（如 &amp; 解成 &）再首尾去空白、"
            "&nbsp; 归一为普通空格后再去首尾空白；不足 3 个 <td> 的行直接跳过；"
            "price 取 data-price 属性的整数；in_stock 为该行第三个 <td> 文本（先归一 &nbsp;）"
            "是否为 'in stock'。\n待解析 HTML：\n```html\n%s```\n"
            "只输出一个 ```python 代码块，不要解释。" % HTML_PRODUCTS
        ),
        "cases": [
            ("parse_products(%r)" % HTML_PRODUCTS, repr(H1_EXPECTED)),
        ],
    },
    {
        "id": "H2", "lang": "python", "name": "爬虫: 日志行 URL 提取与解码",
        "prompt": (
            "请用 Python 标准库（re + urllib.parse，禁用第三方库）实现函数 "
            "extract_requests(log_text: str) -> list：逐行解析类 nginx 访问日志，"
            "只取 GET 请求行，返回字典列表，每项 {'path': str, 'query': str 或 None, 'status': int}："
            "path 为去 query 的 URL 路径；query 为 URL-decode 后的完整查询串原样保留参数名与等号"
            "（如 %%E6%%89%%8B%%E6%%9C%%BA 解码为 手机；from=list&page=2 保持原样；无 query 为 None）；"
            "status 取请求行之后第一个三位数字（不看 POST 行、不看续行）；"
            "无法匹配的行走直接跳过。\n日志：\n```\n%s```\n"
            "只输出一个 ```python 代码块，不要解释。" % NGINX_LOG
        ),
        "cases": [
            ("extract_requests(%r)" % NGINX_LOG, repr(H2_EXPECTED)),
        ],
    },
    {
        "id": "H4", "lang": "python", "name": "爬虫: 链接发现与 URL 规范化",
        "prompt": (
            "请用 Python 标准库（html.parser 或 re + urllib.parse，禁用第三方库）实现函数 "
            "collect_links(html: str, base: str) -> list：提取页面全部 <a href> 并规范化："
            "(1) 相对路径用 base urljoin 成绝对 URL；(2) 去掉 fragment(#...)；"
            "(3) 丢弃非 http(s) 链接（如 mailto:）与纯 fragment 链接（#top）；"
            "(4) 去重（保持首次出现顺序）；(5) 只保留与 base 同一 host 的链接。"
            "返回 URL 字符串列表。\n页面 HTML：\n```html\n%s```\n"
            "base 为 https://example.com/\n"
            "只输出一个 ```python 代码块，不要解释。" % LINKS_HTML
        ),
        "cases": [
            ("collect_links(%r, 'https://example.com/')" % LINKS_HTML,
             repr(H4_EXPECTED)),
        ],
    },
]


# ============================================================ 组 C+: 长 Agent (由 bench_hard 驱动)

INVENTORY_TOOLS = [
    {"type": "function", "function": {
        "name": "query_inventory",
        "description": "查询商品库存，返回 {'stock': int, 'sku': str}",
        "parameters": {"type": "object",
                       "properties": {"sku": {"type": "string", "description": "商品编码"}},
                       "required": ["sku"]}}},
    {"type": "function", "function": {
        "name": "query_price",
        "description": "查询商品单价，返回 {'price': float}",
        "parameters": {"type": "object",
                       "properties": {"sku": {"type": "string"}},
                       "required": ["sku"]}}},
    {"type": "function", "function": {
        "name": "query_discount",
        "description": "查询会员折扣，返回 {'discount': float}（如 0.8 表示 8 折）",
        "parameters": {"type": "object",
                       "properties": {"level": {"type": "string", "description": "会员等级如 VIP1/VIP2"}},
                       "required": ["level"]}}},
    {"type": "function", "function": {
        "name": "query_shipping",
        "description": "查询运费，返回 {'fee': float}",
        "parameters": {"type": "object",
                       "properties": {"weight_kg": {"type": "number"}},
                       "required": ["weight_kg"]}}},
    {"type": "function", "function": {
        "name": "get_customer_level",
        "description": "查询客户会员等级，返回 {'level': str}",
        "parameters": {"type": "object",
                       "properties": {"customer_id": {"type": "string"}},
                       "required": ["customer_id"]}}},
]

FAKE_DB = {
    "inventory": {"SKU-A": {"stock": 12, "sku": "SKU-A"},
                  "SKU-B": {"stock": 0, "sku": "SKU-B"},
                  "SKU-C": {"stock": 5, "sku": "SKU-C"}},
    "price": {"SKU-A": 100.0, "SKU-B": 50.0, "SKU-C": 80.0},
    "discount": {"VIP1": {"discount": 0.9}, "VIP2": {"discount": 0.8}},
    "level": {"C-001": "VIP1", "C-002": "VIP2"},
    "shipping": {1.0: 12.0, 2.0: 15.0},
}


def fake_tool_call(name, args):
    """Agent 题的确定性假后端。返回 (ok, result)。"""
    try:
        if name == "query_inventory":
            r = FAKE_DB["inventory"].get(args["sku"])
            return (r is not None), (r or {"error": "sku not found"})
        if name == "query_price":
            r = FAKE_DB["price"].get(args["sku"])
            return (r is not None), (r or {"error": "sku not found"})
        if name == "query_discount":
            r = FAKE_DB["discount"].get(args["level"])
            return (r is not None), (r or {"error": "unknown level"})
        if name == "query_shipping":
            w = float(args["weight_kg"])
            r = {"fee": FAKE_DB["shipping"].get(round(w, 1), 20.0)}
            return True, r
        if name == "get_customer_level":
            r = FAKE_DB["level"].get(args["customer_id"])
            return (r is not None), ({"level": r} if r else {"error": "customer not found"})
    except Exception as e:
        return False, {"error": "bad args: %s" % e}
    return False, {"error": "unknown tool %s" % name}


def agent_a4_order_chain(api_chat, args):
    """A4 五轮依赖工具链: 会员等级→折扣→库存→价格→运费→总价。
    正确流程: get_customer_level(C-001)=VIP1 → query_discount(VIP1)=0.9
    → query_inventory(SKU-A)=12 → query_price(SKU-A)=100 → query_shipping(2kg)=15
    → 总价 100*0.9+15 = 105"""
    item = {"id": "A4", "name": "Agent: 五轮依赖工具链下单", "ok": False}
    msgs = [{"role": "user", "content":
             "帮我下单：客户 C-001 购买 SKU-A 两件（每件 1kg，总重 2kg）。"
             "请依次查询：客户会员等级、该等级折扣、SKU-A 库存、SKU-A 单价、2kg 运费，"
             "然后计算应付总价（口径：按单件计价演示，总价 = 单价×折扣 + 运费，不乘件数）并报告库存是否足够。"}
            ]
    transcript = []
    for rnd in range(8):
        r, err = api_chat_silent(api_chat, args, msgs, tools=INVENTORY_TOOLS, max_tokens=2000)
        if err:
            item["error"] = err
            break
        tc = (r.get("tool_calls") or [])
        if not tc:
            # 无工具调用 → 应为最终答复
            ans = _set_final(item, r)
            transcript.append({"round": rnd, "final": ans[:200]})
            item["rounds_used"] = rnd
            break
        calls = []
        for c in tc:
            fn = c.get("function", {})
            try:
                a = json.loads(fn.get("arguments") or "{}")
            except Exception:
                a = {}
            ok, result = fake_tool_call(fn.get("name"), a)
            calls.append({"tool": fn.get("name"), "args": a, "result": result})
            msgs.append({"role": "assistant", "content": None, "tool_calls": [c]})
            msgs.append({"role": "tool", "tool_call_id": c.get("id", "c%d" % rnd),
                         "content": json.dumps(result, ensure_ascii=False)})
        transcript.append({"round": rnd, "calls": calls})
    item["transcript"] = transcript
    # 判分: 工具调用序列 + 最终数值
    seq = [c["tool"] for t in transcript for c in t.get("calls", [])]
    item["tool_seq"] = seq
    need_seq = ["get_customer_level", "query_discount", "query_inventory", "query_price", "query_shipping"]
    seq_ok = all(n in seq for n in need_seq)
    # 库存 12 够买 2 件; 总价 = 100*0.9 + 15 = 105
    ans = item.get("final_answer") or ""
    import re as _re
    nums = _re.findall(r"\d+(?:\.\d+)?", ans.replace(",", ""))
    val_ok = any(abs(float(x) - 105.0) < 0.01 for x in nums)
    stock_ok = bool(("12" in ans or "十二" in ans)
                    and any(k in ans for k in ("够", "充足", "可以"))
                    and not any(k in ans for k in ("不够", "不足")))
    item["detail"] = {"seq_ok": seq_ok, "total_105": val_ok, "stock_ok": stock_ok}
    item["ok"] = bool(seq_ok and val_ok and stock_ok)
    return item


def agent_a5_error_degrade(api_chat, args):
    """A5 工具报错降级: SKU-B 库存为 0, 错误工具调用。
    考察: 遇 error/0 库存时不编造、如实说明、给出建议(如换货/缺货)。"""
    item = {"id": "A5", "name": "Agent: 工具报错与缺货降级", "ok": False}
    msgs = [{"role": "user", "content":
             "客户 C-002 想买 SKU-B 三件。请查询库存后告知能否下单；"
             "若不能，如实说明原因并给出替代建议（可查 SKU-C 的库存与价格作为参考）。"}
            ]
    transcript = []
    for rnd in range(8):
        r, err = api_chat_silent(api_chat, args, msgs, tools=INVENTORY_TOOLS, max_tokens=2000)
        if err:
            item["error"] = err
            break
        tc = (r.get("tool_calls") or [])
        if not tc:
            ans = _set_final(item, r)
            item["rounds_used"] = rnd
            break
        calls = []
        for c in tc:
            fn = c.get("function", {})
            try:
                a = json.loads(fn.get("arguments") or "{}")
            except Exception:
                a = {}
            ok, result = fake_tool_call(fn.get("name"), a)
            calls.append({"tool": fn.get("name"), "args": a, "result": result})
            msgs.append({"role": "assistant", "content": None, "tool_calls": [c]})
            msgs.append({"role": "tool", "tool_call_id": c.get("id", "c%d" % rnd),
                         "content": json.dumps(result, ensure_ascii=False)})
        transcript.append({"round": rnd, "calls": calls})
    item["transcript"] = transcript
    ans = item.get("final_answer") or ""
    # 判分: (1) 查过 SKU-B 库存 (2) 未编造可下单 (3) 提及缺货/库存0 (4) 给出建议
    tools_called = [c["tool"] for t in transcript for c in t.get("calls", [])]
    checked_b = any(c.get("args", {}).get("sku") == "SKU-B" and c["tool"] == "query_inventory"
                    for t in transcript for c in t.get("calls", []))
    # 编造判定: 声称能下单/已下单/总价正常算出
    fabricate = bool(re.search(r"已(为你)?下单|下单成功|可以下单|总价\s*[:：]?\s*\d", ans)) and "缺" not in ans and "无法" not in ans
    honest = any(k in ans for k in ("缺货", "无货", "库存为 0", "库存为0", "库存不足",
                                    "库存不够", "售罄", "卖完", "补货", "无法下单"))
    advice = ("SKU-C" in ans.upper()) or any(k in ans for k in
                                             ("替代", "换购", "换成", "推荐", "改买"))
    item["detail"] = {"checked_B": checked_b, "honest": honest, "advice": advice,
                     "fabricated": fabricate}
    item["ok"] = bool(checked_b and honest and advice and not fabricate)
    return item


def agent_a6_instruction_persist(api_chat, args):
    """A6 十轮上下文指令保持: 第 1 轮约束(输出必须 JSON+trace_id), 中间 8 轮工具调用, 第 10 轮检查。"""
    item = {"id": "A6", "name": "Agent: 十轮指令保持", "ok": False}
    trace_id = "TR-8848"
    msgs = [
        {"role": "user", "content":
         "系统约定（必须全程遵守）：你每次给我最终答复时，都必须输出 JSON 对象，"
         "且必须包含字段 \"trace_id\": \"%s\"。现在开始。查一下上海天气。" % trace_id}
    ]
    tools = [{"type": "function", "function": {
        "name": "get_weather",
        "description": "查询天气，返回 {'temp_c': int}",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}]
    r, err = api_chat_silent(api_chat, args, msgs, tools=tools, max_tokens=1500)
    if err:
        item["error"] = err
        return item
    json_hits = []
    def check_json(ans):
        try:
            m = re.search(r"\{.*\}", ans, re.S)
            if not m:
                return False, "no json"
            obj = json.loads(m.group(0))
            return obj.get("trace_id") == trace_id, obj
        except Exception as e:
            return False, "parse: %s" % e
    # round 1 处理
    tc = r.get("tool_calls") or []
    if tc:
        c = tc[0]; fn = c.get("function", {})
        try:
            a = json.loads(fn.get("arguments") or "{}")
        except Exception:
            a = {}
        msgs.append({"role": "assistant", "content": None, "tool_calls": [c]})
        msgs.append({"role": "tool", "tool_call_id": c.get("id", "c0"),
                     "content": json.dumps({"temp_c": 22, "city": a.get("city", "上海")}, ensure_ascii=False)})
        r2, err2 = api_chat_silent(api_chat, args, msgs, tools=tools, max_tokens=1500)
        ans1 = (r2.get("text") or "").strip() if not err2 else ""
    else:
        ans1 = (r.get("text") or "").strip()
    # 关键: 把模型本轮答复写回对话, 否则后面几轮它看不到自己曾说过的约束
    msgs.append({"role": "assistant", "content": ans1})
    ok1, why1 = check_json(ans1)
    json_hits.append(ok1)
    # 中间 8 轮: 闲聊+小工具调用穿插
    smalltalk = [
        "好的，谢谢。顺便问下：1+1 等于几？直接回答。",
        "明白。把这个词翻译成英文：苹果。",
        "继续。查一下北京天气。",
        "好的。今天星期几这种常识不用查工具，直接说一个字：'好'。",
        "接着查深圳天气。",
        "好的。中文数字'五'的英文是？",
        "再查广州天气。",
        "好。数一下 strawberry 里有几个 r？",
    ]
    for i, say in enumerate(smalltalk):
        msgs.append({"role": "user", "content": say})
        r, err = api_chat_silent(api_chat, args, msgs, tools=tools, max_tokens=1500)
        if err:
            break
        tc = r.get("tool_calls") or []
        if tc:
            c = tc[0]; fn = c.get("function", {})
            try:
                a = json.loads(fn.get("arguments") or "{}")
            except Exception:
                a = {}
            msgs.append({"role": "assistant", "content": None, "tool_calls": [c]})
            msgs.append({"role": "tool", "tool_call_id": c.get("id", "c%d" % i),
                         "content": json.dumps({"temp_c": 20 + i, "city": a.get("city", "?")}, ensure_ascii=False)})
            r2, err2 = api_chat_silent(api_chat, args, msgs, tools=tools, max_tokens=1500)
            if err2:
                break
            ans = (r2.get("text") or "").strip()
        else:
            ans = (r.get("text") or "").strip()
        msgs.append({"role": "assistant", "content": ans})
        ok, _ = check_json(ans)
        json_hits.append(ok)
    # 第 10 轮: 最后一次约束检查
    msgs.append({"role": "user", "content": "最后：把至今所有查询过的城市列出来，按我最初的系统约定输出。"})
    r, err = api_chat_silent(api_chat, args, msgs, tools=tools, max_tokens=1500)
    if not err:
        ans10 = (r.get("text") or "").strip()
        ok10, _ = check_json(ans10)
        json_hits.append(ok10)
        _set_final(item, ans10)
    item["json_hits"] = json_hits
    item["n_checks"] = len(json_hits)
    good = sum(1 for x in json_hits if x)
    item["persist_hits"] = good
    item["persist_pct"] = round(100.0 * good / len(json_hits)) if json_hits else 0
    # ok 标准: 首轮 + 末轮都遵守, 且全程保持率 >= 80% (漂移即算没保持住)
    item["ok"] = bool(len(json_hits) >= 2 and json_hits[0] and json_hits[-1]
                      and good >= 0.8 * len(json_hits))
    return item


CRAWLER_TOOLS = [
    {"type": "function", "function": {
        "name": "fetch_page",
        "description": "抓取网页，返回 {'url': str, 'status': int, 'body': str(HTML)}，"
                       "404 时 status=404 且 body 为错误页",
        "parameters": {"type": "object",
                       "properties": {"url": {"type": "string"}},
                       "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "parse_links",
        "description": "从 HTML 提取全部 http(s) 链接（已 urljoin/去 fragment/去重），返回 {'links': []}",
        "parameters": {"type": "object",
                       "properties": {"html": {"type": "string"}, "base": {"type": "string"}},
                       "required": ["html", "base"]}}},
]

CRAWL_PAGES = {
    "http://mock.local/": {"status": 200, "body":
        '<a href="/a">A</a> <a href="/b">B</a> <a href="http://mock.local/d">D</a> <a href="mailto:x@y.z">M</a>'},
    "http://mock.local/a": {"status": 200, "body":
        '<a href="/b">B again</a> <a href="/c">C</a> <a href="http://ext.local/e">E</a>'},
    "http://mock.local/b": {"status": 200, "body": "leaf page, no links"},
    "http://mock.local/c": {"status": 404, "body": "404 not found"},
    "http://mock.local/d": {"status": 200, "body": "leaf d"},
}


def crawl_tool_call(name, args):
    if name == "fetch_page":
        p = CRAWL_PAGES.get(args.get("url"))
        if p is None:
            return {"url": args.get("url"), "status": 404, "body": "404 not found"}
        return {"url": args.get("url"), "status": p["status"], "body": p["body"]}
    if name == "parse_links":
        from urllib.parse import urljoin, urlparse
        links, seen = [], set()
        for m in re.finditer(r'href="([^"]+)"', args.get("html", "")):
            href = m.group(1)
            if href.startswith("mailto:") or href.startswith("#"):
                continue
            u = urljoin(args.get("base", ""), href).split("#")[0]
            if u.startswith("http") and u not in seen:
                seen.add(u)
                links.append(u)
        return {"links": links}
    return {"error": "unknown tool"}


def agent_a7_crawl(api_chat, args):
    """A7 爬虫编排: 从入口页爬出同域全部页面并汇总 (含 404 页处理)。"""
    item = {"id": "A7", "name": "Agent: 爬虫编排多轮抓取", "ok": False}
    msgs = [{"role": "user", "content":
             "从 http://mock.local/ 开始爬取：用 fetch_page 抓入口页，用 parse_links 提取链接，"
             "再逐个抓取同域(http://mock.local)页面（跳过外部域），直到没有新页面。"
             "最后汇总：成功抓取(200)的页面数、404 的页面数，以及全部页面 URL 列表。"}
            ]
    transcript = []
    for rnd in range(14):
        r, err = api_chat_silent(api_chat, args, msgs, tools=CRAWLER_TOOLS, max_tokens=2000)
        if err:
            item["error"] = err
            break
        tc = (r.get("tool_calls") or [])
        if not tc:
            ans = _set_final(item, r)
            item["rounds_used"] = rnd
            break
        calls = []
        for c in tc:
            fn = c.get("function", {})
            try:
                a = json.loads(fn.get("arguments") or "{}")
            except Exception:
                a = {}
            if fn.get("name") == "parse_links":
                result = crawl_tool_call("parse_links", a)
            elif fn.get("name") == "fetch_page":
                result = crawl_tool_call("fetch_page", a)
            else:
                result = {"error": "unknown tool"}
            calls.append({"tool": fn.get("name"), "args": {k: (v[:50] if isinstance(v, str) else v)
                                                            for k, v in a.items()},
                          "result": {k: (v[:40] if isinstance(v, str) else v) for k, v in result.items()}})
            msgs.append({"role": "assistant", "content": None, "tool_calls": [c]})
            msgs.append({"role": "tool", "tool_call_id": c.get("id", "c%d" % rnd),
                         "content": json.dumps(result, ensure_ascii=False)})
        transcript.append({"round": rnd, "calls": calls})
    item["transcript"] = transcript
    # 判分: 抓过入口+4 个链接页(a,b,c,d 中 c 为 404), 最终汇总 4 个 200 页 + 1 个 404
    fetched = set()
    for t in transcript:
        for c in t.get("calls", []):
            if c["tool"] == "fetch_page" and c["args"].get("url"):
                fetched.add(c["args"]["url"].rstrip("/"))
    want = {"http://mock.local", "http://mock.local/a", "http://mock.local/b",
            "http://mock.local/c", "http://mock.local/d"}
    got = {u for u in fetched}
    cover = want & got
    ans = item.get("final_answer") or ""
    # 汇总判定: 5 个页面(含 404 那页)必须全部抓过, 少抓一个就不算完成
    cover_ok = (cover == want)
    listed_ok = all(x in ans for x in ("/a", "/b", "/d", "/c")) or cover_ok
    fetched_c = "http://mock.local/c" in got
    has_404 = ("404" in ans and ("/c" in ans or fetched_c))
    fetched_external = sorted(u for u in got if "ext.local" in u)
    item["detail"] = {"covered": sorted(cover), "n_fetched": len(got),
                      "cover_ok": cover_ok, "summary_lists_pages": listed_ok,
                      "mentions_404": has_404, "fetched_external": fetched_external}
    # 题面要求跳过外部域, 真抓了外域算违反指令
    item["ok"] = bool(cover_ok and listed_ok and has_404 and not fetched_external)
    return item


def api_chat_silent(api_chat, args, msgs, tools=None, max_tokens=2000):
    """安全包装: 调 api_chat 并捕获错误。"""
    try:
        r = api_chat(args.base_url, args.model, msgs, max_tokens=max_tokens,
                     temperature=0, tools=tools, timeout=args.timeout,
                     api_key=args.api_key)
        return r, None
    except Exception as e:
        return None, "%s: %s" % (type(e).__name__, e)


# ============================================================ 组 D+: 准确率难题

HARD_QA_TASKS = [
    {
        "id": "D1", "topic": "正则-实体提取",
        "prompt": (
            "请用 Python 标准库 re 实现函数 extract_phones(text: str) -> list："
            "从文本中提取全部中国大陆手机号，返回按出现顺序排列的字符串列表。"
            "规则：以 1 开头共 11 位、第二位为 3-9；号码可写成分段形式、段与段之间各用一个 '-' 连接"
            "（如 139-0013-9000），需先归一为连续 11 位再提取；"
            "注意不要把订单号、房间号、短客服号误判为手机号。"
            "待处理文本：\n"
            "今天气温 12 度，最高气温 15 度。本月流量已用 8GB。客服电话 95105891。"
            "紧急联系电话 13800138000，备用 139-0013-9000。房间号 1203，订单号 20241117001。"
            "只输出一个 ```python 代码块，不要解释。"
        ),
        "cases": [
            ("extract_phones('今天气温 12 度。客服 95105891。紧急电话 13800138000，备用 139-0013-9000。房间 1203，订单 20241117001。')",
             "['13800138000', '13900139000']"),
            ("extract_phones('联系人 15912345678 或者 18600000000')",
             "['15912345678', '18600000000']"),
            ("extract_phones('无手机号文本 12345678901 是 11 位但 12 开头')",
             "[]"),
        ],
    },
    {
        "id": "D2", "topic": "数学-行程追及",
        "prompt": (
            "甲乙两人从同一地点沿直线同向出发，甲先出发，速度 6 km/h。乙晚 30 分钟出发，"
            "速度 9 km/h。乙出发后多久追上甲？追上时距出发点多远？"
            "一步步推理，最后一行写：答案：追上用时 X 小时，距出发点 Y 公里"
        ),
        "check": lambda ans: find_num(ans, 1) and find_num(ans, 9),
    },
    {
        "id": "D3", "topic": "代码-四段语义辨析(输出预测)", "tier": "hard",
        "prompt": (
            "以下四段 Python 代码各自 print 出的数字是多少？其中恰有一段与其它三段的语义不同。\n"
            "S1: x=[1]; y=x; y+=[2]; print(len(x))\n"
            "S2: x=[1]; y=x; y=y+[2]; print(len(x))\n"
            "S3: x=[1]; y=list(x); y+=[2]; print(len(x))\n"
            "S4: x=[1]; y=x[:]; y+=[2]; print(len(x))\n"
            "最后一行严格按此格式写（顺序固定、四项缺一不可）："
            "答案：S1=?,S2=?,S3=?,S4=?,不同=S?"
        ),
        "check": _d3_check,
    },
]


# ============================================================ H3: 分页爬取 (mock server + 专用驱动)

PAGED_DATA = [
    {"page": 1, "items": [{"id": 101, "name": "a", "val": 1.0}, {"id": 102, "name": "b", "val": 2.5}],
     "next": "/api/items?page=2"},
    {"page": 2, "items": [{"id": 103, "name": "c", "val": None}, {"id": 101, "name": "dup", "val": 1.0}],
     "next": "/api/items?page=3"},
    {"page": 3, "items": [{"id": 104, "name": "d", "val": 4.0}],
     "next": None},
]

H3_PROMPT = (
    "请用 Python 标准库（urllib.request + json，禁用第三方库）实现函数 crawl_all(base_url: str) -> list：\n"
    "从 {base}/api/items?page=1 开始分页爬取 JSON 数据（每页形如 "
    "{{'page': int, 'items': [{{'id': int, 'name': str, 'val': float 或 null}}], 'next': '下一页路径' 或 null}}），\n"
    "跟随 next 直到为 null，聚合全部 items 为一个列表并做清洗：去掉 name 为空字符串的项，"
    "去掉 id 重复的项（保留首次出现的）。返回清洗后的按 id 升序的字典列表。\n"
    "只输出一个 ```python 代码块，不要解释。"
)

H3_EXPECTED = [
    {"id": 101, "name": "a", "val": 1.0},
    {"id": 102, "name": "b", "val": 2.5},
    {"id": 103, "name": "c", "val": None},
    {"id": 104, "name": "d", "val": 4.0},
]


# 难度档标注(仅用于报告分档展示, 不参与计分)
TIERS = {
    "easy": ["J1", "P1", "N2", "J4", "D2"],
    "mid": ["J2", "P2", "N1", "H1", "H2", "H3", "A4", "A5", "P3"],
    "hard": ["J3", "H4", "A6", "A7", "D1", "D3"],
}
TIER_OF = {i: t for t, ids in TIERS.items() for i in ids}


def run_h3(api_chat, args, prompt):
    """H3 驱动: 起 mock server → 生成题目 → 模型出码 → 执行判分 → 关 server。"""
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass
        def do_GET(self):
            if not self.path.startswith("/api/items"):
                self.send_response(404); self.end_headers(); return
            try:
                page = int(self.path.split("page=")[1])
            except Exception:
                self.send_response(400); self.end_headers(); return
            data = PAGED_DATA[page - 1]
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        base = "http://127.0.0.1:%d" % port
        prompt_full = prompt.format(base=base)
        r, err = api_chat_silent(api_chat, args, [{"role": "user", "content": prompt_full}],
                                  max_tokens=4000)
        item = {"id": "H3", "name": "爬虫: 分页爬取编排(真实HTTP)", "ok": False}
        if err:
            item["error"] = err
            return item
        raw = (r.get("text") or "").strip()
        if not raw:
            raw = (r.get("reasoning") or "").strip()
        item["raw"] = raw[:4000]
        if not (r.get("text") or "").strip() and len(r.get("reasoning") or "") >= 3000:
            item["starved"] = True
        blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", raw, re.S)
        best = None
        for b in blocks:
            code = b.strip()
            try:
                prog = ("import json, urllib.request\n" + code + "\n"
                        "import json as _j\n"
                        "out = crawl_all(%r)\n"
                        "print(_j.dumps({'ok': out == %r, 'actual': out}, ensure_ascii=False)[:4000])\n"
                        % (base, H3_EXPECTED))
                import subprocess, tempfile, os, sys as _sys
                with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                                 encoding="utf-8") as f:
                    f.write(prog)
                    path = f.name
                rp = subprocess.run([_sys.executable, path], capture_output=True,
                                    text=True, timeout=30)
                os.unlink(path)
                out = rp.stdout.strip()
                if rp.returncode == 0 and out:
                    obj = json.loads(out.splitlines()[-1])
                    if obj.get("ok"):
                        return dict(item, ok=True, passed=1, total=1, n_code_blocks=len(blocks))
                    last = dict(item, ok=False, passed=0, total=1, fails=["实际=%s" % obj.get("actual", "")[:200]])
                else:
                    err_lines = (rp.stderr or rp.stdout).strip().splitlines()
                    last = dict(item, ok=False, passed=0, total=1,
                                fails=[err_lines[-1][:180] if err_lines else "?"])
                if best is None:
                    best = last
            except Exception as e:
                last = dict(item, ok=False, passed=0, total=1, fails=["评测器异常: %s" % e])
        return best or dict(item, ok=False, passed=0, total=1, fails=["未找到代码块"])
    finally:
        srv.shutdown()
        srv.server_close()
