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
_stats = {"hit": 0, "miss": 0, "passthrough": 0}


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

        key = cache_key(self.path, body) if cacheable else None
        if key:
            with _lock:
                row = _db().execute(
                    "SELECT response FROM cache WHERE key=?", (key,)).fetchone()
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
