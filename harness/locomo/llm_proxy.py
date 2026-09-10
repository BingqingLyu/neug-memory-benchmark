"""OpenAI 兼容 LLM 缓存代理（纯 stdlib）。

用途：三个对比系统（Graphiti/Mem0/Cognee）的 LLM 配置指向本代理端点后，
ingestion 抽取调用自动落盘缓存——同一系统换后端时抽取 prompt 相同，
直接命中缓存，实现 benchmark-plan.md §3.1 协议 6 的"抽取缓存跨后端复用"。
对框架代码零侵入（三家都支持 OpenAI 兼容 base_url）。

缓存键 = sha256(path + model + 规范化请求体)。stream=true 不缓存、直接透传。
API key 从环境变量 DASHSCOPE_API_KEY 读取，不落盘、不硬编码。

用法：
  python -m harness.locomo.llm_proxy --port 8787
  各系统 llm base_url 指向 http://127.0.0.1:8787/v1
"""
import argparse
import hashlib
import json
import os
import re
import sqlite3
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("DASHSCOPE_BASE_URL",
                          "https://dashscope.aliyuncs.com/compatible-mode/v1")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..",
                       "results", "cache", "llm_proxy", "cache.db")

_lock = threading.Lock()
_conn = None
_stats = {"hit": 0, "miss": 0, "passthrough": 0,
          "evicted_bad_json": 0, "uncached_bad_json": 0}
# 诊断用：设置后把每个 cacheable POST 的请求体逐行落盘，便于跨进程 diff 出
# 非确定字段（如 litellm 注入的 litellm_call_id/metadata），定位缓存不命中的根因。
_DUMP_BODIES = os.environ.get("BENCH_PROXY_DUMP_BODIES", "")


def _dump_body(path: str, body: dict):
    if not _DUMP_BODIES:
        return
    try:
        import time as _t
        with open(_DUMP_BODIES, "a") as f:
            f.write(json.dumps({"ts": _t.time(), "path": path, "body": body},
                               ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:  # noqa: BLE001 - 诊断落盘绝不能影响请求
        pass


def _db():
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.execute("CREATE TABLE IF NOT EXISTS cache "
                      "(key TEXT PRIMARY KEY, response TEXT)")
    return _conn


def cache_key(path: str, body: dict) -> str:
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(f"{path}|{canonical}".encode()).hexdigest()


def _wants_json(body: dict) -> bool:
    """请求是否要求 JSON 输出（json_object / json_schema 两种 response_format）。

    只有这类响应才需要校验。harness 同时缓存纯文本补全（QA 答案、judge
    判词），它们本来就不该解析成 JSON——不按请求形态收窄会把好数据当坏
    数据丢掉（首次全库扫描就误报了 54205 条）。
    """
    rf = body.get("response_format")
    return isinstance(rf, dict) and rf.get("type") in ("json_object", "json_schema")


def _json_content_ok(data: bytes) -> bool:
    """JSON 模式下的响应 content 能否真的解析。

    模型偶尔吐畸形 JSON（实测一例：`"name": "Samantha'}]}]`，双引号开、
    单引号收）。客户端 openai_generic_client.py:169 直接 json.loads，抛
    JSONDecodeError 后 tenacity 重试——但只要这条坏响应进了缓存，每次重试
    都重放同一份字节，确定性失败到底。这里把"永久毒化"降级成"重掷一次"。

    剥围栏与客户端 _strip_code_fences 逐字一致。结构不认识或判定出错一律
    当有效：校验逻辑自己绝不能成为丢缓存的原因。
    """
    try:
        content = json.loads(data.decode())["choices"][0]["message"]["content"]
    except Exception:  # noqa: BLE001 - 非 chat 响应或结构不同，不归本校验管
        return True
    if not isinstance(content, str) or not content.strip():
        return True
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*[ \t]*\r?\n?", "", stripped)
        stripped = re.sub(r"\r?\n?```[ \t]*$", "", stripped)
    try:
        json.loads(stripped.strip())
    except Exception:  # noqa: BLE001
        return False
    return True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # 静默默认日志
        pass

    def _forward(self, path: str, payload: bytes):
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            self.send_error(500, "DASHSCOPE_API_KEY not set")
            return None
        up_path = path[len("/v1"):] if path.startswith("/v1") else path
        req = urllib.request.Request(
            UPSTREAM + up_path, data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(e.read())
            return None
        except Exception as e:  # URLError / timeout 等
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
            return None

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = self.rfile.read(length)
        cacheable = self.path.rstrip("/") in ("/v1/chat/completions", "/v1/embeddings")
        try:
            body = json.loads(payload)
        except json.JSONDecodeError:
            body = {}
        if body.get("stream"):
            cacheable = False

        if cacheable and self.path.rstrip("/") == "/v1/chat/completions":
            _dump_body(self.path, body)
        key = cache_key(self.path, body) if cacheable else None
        json_mode = _wants_json(body)
        if key:
            with _lock:
                row = _db().execute(
                    "SELECT response FROM cache WHERE key=?", (key,)).fetchone()
            if row and json_mode and not _json_content_ok(row[0].encode()):
                # 历史毒数据自愈：本校验加上之前落盘的坏响应，命中即驱逐，
                # 当 miss 走上游重掷，不需要人工去 DELETE。
                with _lock:
                    _db().execute("DELETE FROM cache WHERE key=?", (key,))
                    _db().commit()
                _stats["evicted_bad_json"] += 1
                print(f"[proxy] evicted malformed cached response "
                      f"{key[:16]}... (re-rolling upstream)", flush=True)
                row = None
            if row:
                _stats["hit"] += 1
                data = row[0].encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("X-Cache", "HIT")
                self.end_headers()
                self.wfile.write(data)
                return
            _stats["miss"] += 1
        else:
            _stats["passthrough"] += 1

        data = self._forward(self.path, payload)
        if data is None:
            return
        if key:
            if json_mode and not _json_content_ok(data):
                # 不落盘：坏响应照样回给客户端（该炸还是炸），但下一次重试
                # 会走上游重掷而不是重放同一份字节。
                _stats["uncached_bad_json"] += 1
                print(f"[proxy] upstream returned malformed JSON for "
                      f"{key[:16]}... -- NOT cached", flush=True)
            else:
                with _lock:
                    _db().execute(
                        "INSERT OR REPLACE INTO cache (key, response) VALUES (?,?)",
                        (key, data.decode()))
                    _db().commit()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Cache", "MISS")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/stats":
            with _lock:
                n = _db().execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            out = {**_stats, "cached_entries": n}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(out).encode())
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise SystemExit("DASHSCOPE_API_KEY not set")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"LLM cache proxy on http://{args.host}:{args.port}/v1 -> {UPSTREAM}")
    print(f"cache db: {os.path.abspath(DB_PATH)}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
