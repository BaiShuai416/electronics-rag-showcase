"""Serve both RAG demos through a single loopback HTTP server."""

import argparse
import importlib.util
import json
import logging
import mimetypes
import os
import re
import sys
from functools import lru_cache
from getpass import getpass
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from time import monotonic, perf_counter
from urllib.parse import parse_qs, quote, unquote, urlsplit

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError

from accounts import AccountStore, SESSION_SECONDS


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LEARNING = ROOT / "learning"
COOK = ROOT / "data" / "C8" / "cook" / "dishes"
sys.path.insert(0, str(LEARNING))
for config in (ROOT / "code" / "C8" / ".env", ROOT / ".env", LEARNING / ".env"):
    load_dotenv(config, override=False)

LOG = logging.getLogger("rag-showcase")
WORKSPACE = Lock()
AUTH_LOCK = Lock()
AUTH_ATTEMPTS = {}
AUTH_WINDOW = 60
AUTH_LIMIT = 15
LOCAL_HOSTS = {"localhost", "127.0.0.1"}
COOKIE_NAME = "rag_session"
MAX_GENERATIONS = 20
generation_attempts = 0
RECIPES = (
    ("shrimp", "黄油煎虾", "水产", "aquatic/黄油煎虾/黄油煎虾.md"),
    ("eggs", "西葫芦炒鸡蛋", "素菜", "vegetable_dish/西葫芦炒鸡蛋/西葫芦炒鸡蛋.md"),
    ("custard", "鸡蛋羹", "素菜", "vegetable_dish/鸡蛋羹/鸡蛋羹.md"),
    ("soup", "西红柿豆腐汤羹", "素菜", "vegetable_dish/西红柿豆腐汤羹/西红柿豆腐汤羹.md"),
    ("cauliflower", "干锅花菜", "素菜", "vegetable_dish/干锅花菜/干锅花菜.md"),
    ("peppers", "虎皮青椒", "素菜", "vegetable_dish/虎皮青椒/虎皮青椒.md"),
)
MANUALS = {
    model: LEARNING / "迁移资料" / f"{model}.pdf"
    for model in ("DS18B20", "DS18S20")
}
PHOTOS = {
    key: HERE / "dist" / "photos" / (f"{key}.webp" if key == "cauliflower" else f"{key}.jpg")
    for key, _, _, _ in RECIPES
}
STATIC = {
    "/": HERE / "dist" / "recipes.html",
    "/recipes": HERE / "dist" / "recipes.html",
    "/devices": HERE / "dist" / "devices.html",
    "/register": HERE / "dist" / "auth.html",
    "/login": HERE / "dist" / "auth.html",
    "/admin": HERE / "dist" / "admin.html",
    "/assets/styles.css": HERE / "dist" / "styles.css",
    "/assets/app.js": HERE / "dist" / "app.js",
    "/assets/auth.js": HERE / "dist" / "auth.js",
    "/assets/admin.js": HERE / "dist" / "admin.js",
    **{f"/photos/{file.name}": file for file in PHOTOS.values()},
    **{f"/recipes/source/{key}.md": COOK / document for key, _, _, document in RECIPES},
    **{f"/manuals/{model}.pdf": path for model, path in MANUALS.items()},
}
PUBLIC_PAGES = {"/register", "/login", "/assets/styles.css", "/assets/app.js", "/assets/auth.js"}


def has_api_key() -> bool:
    return bool(os.getenv("DEEPSEEK_API_KEY", "").strip())


def import_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def recipe_system():
    module = import_script("showcase_recipe", ROOT / "code" / "C8" / "main.py")
    system = module.RecipeRAGSystem(require_api_key=False)
    system.initialize_system(with_generation=False)
    system.build_knowledge_base()
    return system


@lru_cache(maxsize=1)
def device_module():
    return import_script("showcase_electronics", LEARNING / "11_electronics_vector_search.py")


@lru_cache(maxsize=1)
def device_store():
    return device_module().create_prose_store()


@lru_cache(maxsize=1)
def temperature_evidence():
    return device_module().load_temperature_evidence()


def recipe_sources(documents):
    known_sources = {name: f"/recipes/source/{key}.md" for key, name, _, _ in RECIPES}
    return [
        {
            "title": doc.metadata.get("dish_name", "菜谱原文"),
            "location": Path(doc.metadata.get("source", "菜谱资料")).name,
            "href": known_sources.get(doc.metadata.get("dish_name")),
            "text": doc.page_content,
        }
        for doc in documents
    ]


def manual_source(doc):
    metadata = doc.metadata
    name = metadata["source_name"]
    page = metadata["pdf_page"]
    return {
        "title": " / ".join(metadata["product_models"]),
        "location": f"{name} 第 {page} 页",
        "href": f"/manuals/{quote(name)}#page={page}",
        "text": doc.page_content,
    }


def answer_recipe(question: str, generate: bool) -> dict:
    system = recipe_system()
    documents = []
    if generate:
        if system.generation_module is None:
            system.initialize_generation()
        answer = system.ask_question(question, stream=False, on_evidence=documents.extend)
    else:
        chunks = system.retrieve_chunks(question, question)
        documents = system.data_module.get_parent_documents(chunks)
        answer = (
            "已找到相关菜谱。展开下方原文可查看食材、用量和制作步骤。"
            if documents else "没有找到符合当前条件的菜谱。试试输入一个完整菜名。"
        )
    return {"answer": answer, "route": "混合检索", "mode": "answer" if generate else "preview", "sources": recipe_sources(documents)}


def answer_device(question: str, generate: bool) -> dict:
    module = device_module()
    model, temperature = module.parse_asked_question(question)
    if temperature is not None:
        count, row = module.select_temperature_error(temperature_evidence(), model, temperature)
        return {
            "answer": row.page_content if row else "已核对的温度误差记录中，没有覆盖该条件的条目。",
            "route": "温度误差查表",
            "mode": "table",
            "model": model,
            "matched": count,
            "sources": [manual_source(row)] if row else [],
        }
    store, parents = device_store()
    _, parent_hits = module.retrieve_prose(store, model, question)
    documents = [parents[parent_id] for parent_id in parent_hits]
    if generate and documents:
        messages, references = module.build_answer_messages(question, documents)
        answer = module.generate_answer(messages, references)
    elif documents:
        answer = "已回取相关手册章节。展开证据查看英文原文，或切换到生成回答获取中文说明。"
    else:
        answer = "检索结果中没有可供回答的手册章节。"
    return {
        "answer": answer, "route": "正文检索", "mode": "answer" if generate else "preview",
        "model": model, "sources": [manual_source(doc) for doc in documents],
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "RAGShowcase"
    sys_version = ""

    def send_bytes(self, status: int, data: bytes, content_type: str, **headers):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        for name, value in headers.items():
            self.send_header(name.replace("_", "-"), value)
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, status: int, value: dict, **headers):
        self.send_bytes(
            status, json.dumps(value, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8", **headers,
        )

    def hostname(self):
        try:
            return urlsplit("//" + self.headers.get("Host", "")).hostname or ""
        except ValueError:
            return ""

    def valid_host(self):
        host = self.hostname()
        return host in LOCAL_HOSTS or host.endswith(".trycloudflare.com")

    def local_admin_access(self):
        forwarded = ("CF-Connecting-IP", "CF-Ray", "X-Forwarded-For", "X-Forwarded-Proto")
        return self.hostname() in LOCAL_HOSTS and not any(name in self.headers for name in forwarded)

    def session_token(self):
        try:
            cookies = SimpleCookie(self.headers.get("Cookie", ""))
        except CookieError:
            return None
        cookie = cookies.get(COOKIE_NAME)
        if cookie and re.fullmatch(r"[A-Za-z0-9_-]{43}", cookie.value):
            return cookie.value
        return None

    def session_cookie(self, token=""):
        max_age = SESSION_SECONDS if token else 0
        cookie = f"{COOKIE_NAME}={token}; Path=/; HttpOnly; SameSite=Lax; Max-Age={max_age}"
        if self.hostname().endswith(".trycloudflare.com"):
            cookie += "; Secure"
        return cookie

    def redirect(self, destination):
        self.send_bytes(303, b"", "text/plain", Location=destination)

    def auth_throttled(self):
        address = self.headers.get("CF-Connecting-IP", self.client_address[0])
        now = monotonic()
        with AUTH_LOCK:
            expired = [ip for ip, (end, _) in AUTH_ATTEMPTS.items() if end <= now]
            for ip in expired:
                del AUTH_ATTEMPTS[ip]
            if address not in AUTH_ATTEMPTS and len(AUTH_ATTEMPTS) >= 2048:
                return True
            end, count = AUTH_ATTEMPTS.get(address, (now + AUTH_WINDOW, 0))
            if count >= AUTH_LIMIT:
                return True
            AUTH_ATTEMPTS[address] = (end, count + 1)
            return False

    def authenticate_request(self, path, payload):
        accounts = self.server.accounts
        token = self.session_token()
        if path == "/api/auth/logout":
            accounts.revoke_session(token)
            return self.send_json(200, {"ok": True}, Set_Cookie=self.session_cookie())
        if self.auth_throttled():
            return self.send_json(429, {"error": "登录或注册尝试过多，请稍后再试。"}, Retry_After="60")
        username, password = payload.get("username"), payload.get("password")
        is_admin = False
        if path == "/api/auth/register":
            try:
                user_id = accounts.create_user(username, payload.get("display_name"), password)
            except ValueError as error:
                return self.send_json(400, {"error": str(error)})
        else:
            user = accounts.authenticate(username, password)
            if not user:
                return self.send_json(401, {"error": "账号或密码有误。"})
            is_admin = bool(user["is_admin"])
            if is_admin and not self.local_admin_access():
                return self.send_json(403, {"error": "请在本机管理后台登录管理员账号。"})
            user_id = user["id"]
        new_token = accounts.create_session(user_id)
        accounts.revoke_session(token)
        return self.send_json(200, {"is_admin": is_admin}, Set_Cookie=self.session_cookie(new_token))

    def do_GET(self):
        if not self.valid_host():
            return self.send_json(403, {"error": "访问地址不受支持。"})
        request = urlsplit(self.path)
        path = unquote(request.path).rstrip("/") or "/"
        if path == "/healthz":
            return self.send_json(200, {"ok": True})
        admin_route = path in {"/admin", "/api/admin/overview", "/assets/admin.js"}
        if path not in STATIC and path not in {"/api/catalog", "/api/admin/overview"}:
            return self.send_json(404, {"error": "没有找到该页面或资料。"})
        viewer = None
        if path not in PUBLIC_PAGES:
            if admin_route and not self.local_admin_access():
                return self.send_json(403, {"error": "管理后台仅支持本机访问。"})
            viewer = self.server.accounts.viewer(self.session_token())
            if not viewer:
                if path in {"/", "/recipes", "/devices"}:
                    target = "/devices" if path == "/devices" else "/recipes"
                    return self.redirect(f"/register?next={quote(target, safe='')}")
                if path == "/admin":
                    return self.redirect("/login?next=%2Fadmin")
                return self.send_json(401, {"error": "请先登录。"})
            if admin_route and not viewer["is_admin"]:
                if path == "/admin":
                    return self.redirect("/login?next=%2Fadmin")
                return self.send_json(403, {"error": "请使用管理员账号登录。"})
        if path == "/api/admin/overview":
            try:
                params = parse_qs(request.query, max_num_fields=4)
                user_page = int(params.get("users", ["1"])[0])
                question_page = int(params.get("questions", ["1"])[0])
                if not (1 <= user_page <= 1_000_000 and 1 <= question_page <= 1_000_000):
                    raise ValueError()
            except ValueError:
                return self.send_json(400, {"error": "页码有误。"})
            return self.send_json(200, self.server.accounts.overview(user_page, question_page))
        if path == "/api/catalog":
            return self.send_json(200, {
                "recipes": [{"key": key, "name": name, "category": category, "image": f"/photos/{PHOTOS[key].name}"} for key, name, category, _ in RECIPES],
                "generation": has_api_key(),
                "remaining": max(0, MAX_GENERATIONS - generation_attempts),
                "viewer": {"display_name": viewer["display_name"]},
            })
        file = STATIC.get(path)
        if not file or not file.is_file():
            return self.send_json(404, {"error": "没有找到该页面或资料。"})
        content_type = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
        if file.suffix == ".md":
            content_type = "text/plain; charset=utf-8"
        elif file.suffix == ".js":
            content_type = "text/javascript; charset=utf-8"
        elif content_type.startswith("text/"):
            content_type += "; charset=utf-8"
        self.send_bytes(200, file.read_bytes(), content_type)

    def do_POST(self):
        global generation_attempts
        origin = self.headers.get("Origin")
        host = self.headers.get("Host")
        if not self.valid_host() or (origin and origin not in {f"http://{host}", f"https://{host}"}):
            return self.send_json(403, {"error": "请从当前演示页面提交。"})
        path = urlsplit(self.path).path
        actions = {
            "/api/recipes/ask": ("recipes", answer_recipe),
            "/api/devices/ask": ("devices", answer_device),
        }
        auth_request = path in {"/api/auth/register", "/api/auth/login", "/api/auth/logout"}
        if not auth_request and path not in actions:
            return self.send_json(404, {"error": "接口不存在。"})
        if not auth_request:
            viewer = self.server.accounts.viewer(self.session_token())
            if not viewer:
                return self.send_json(401, {"error": "请先登录。"})
        if self.headers.get_content_type() != "application/json":
            return self.send_json(415, {"error": "请求格式有误。"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 8192:
                raise ValueError()
            self.connection.settimeout(20)
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError()
            # Reject malformed Unicode before storing or echoing strings.
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
        except (ValueError, TimeoutError, RecursionError):
            return self.send_json(400, {"error": "请求内容有误。"})
        if auth_request:
            return self.authenticate_request(path, payload)
        try:
            question = payload["question"].strip()
            mode = payload.get("mode", "preview")
            if not question or len(question) > 300 or mode not in ("preview", "answer"):
                raise ValueError()
        except (ValueError, TypeError, KeyError, AttributeError, TimeoutError):
            return self.send_json(400, {"error": "请输入 1 到 300 字的问题。"})

        project, action = actions[path]
        accounts = self.server.accounts
        question_id = accounts.log_question(viewer["id"], project, question)
        if not WORKSPACE.acquire(blocking=False):
            accounts.finish_question(question_id, "busy")
            return self.send_json(429, {"error": "演示正在处理另一条问题，稍后重试。"})
        completed = False
        try:
            # 参数查表在本机完成，即便选了生成回答也不扣额度。
            generate = mode == "answer"
            if action is answer_device:
                try:
                    _, temperature = device_module().parse_asked_question(question)
                except ValueError as error:
                    return self.send_json(400, {"error": str(error)})
                generate = generate and temperature is None
            if generate:
                if not has_api_key():
                    return self.send_json(503, {"error": "当前可以使用证据预览。生成回答需要在电脑上配置 DeepSeek。"})
                if generation_attempts >= MAX_GENERATIONS:
                    return self.send_json(429, {"error": "本次展示的生成额度已用完，可以继续使用证据预览和参数查表。"})
                generation_attempts += 1
            started = perf_counter()
            result = action(question, generate)
            result["elapsed_ms"] = round((perf_counter() - started) * 1000)
            result["remaining"] = MAX_GENERATIONS - generation_attempts
            accounts.finish_question(question_id, "answered", result["route"])
            completed = True
        except APIStatusError as error:
            return self.send_json(502, {"error": f"生成服务返回 HTTP {error.status_code}，稍后可重试或使用证据预览。"})
        except APIConnectionError:
            return self.send_json(502, {"error": "生成服务连接失败，稍后可重试或使用证据预览。"})
        except SystemExit as error:
            return self.send_json(502, {"error": str(error)})
        except Exception:
            LOG.exception("处理演示请求失败")
            return self.send_json(500, {"error": "本地问答服务处理失败，请检查运行窗口。"})
        finally:
            WORKSPACE.release()
            if not completed:
                accounts.finish_question(question_id, "failed")
        self.send_json(200, result)


def main():
    parser = argparse.ArgumentParser(description="菜谱与器件手册浏览器演示")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--create-admin", action="store_true", help="在本机创建一个后台管理员")
    args = parser.parse_args()
    accounts = AccountStore(HERE / ".data" / "showcase.sqlite3")
    if args.create_admin:
        try:
            username = input("管理员账号：")
            name = input("管理员名称：")
            password = getpass("管理员密码：")
            if password != getpass("再输入一次密码："):
                raise ValueError("两次输入的密码不同。")
            accounts.create_user(username, name, password, admin=True)
        except ValueError as error:
            raise SystemExit(str(error)) from None
        print("管理员已创建。")
        return
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.accounts = accounts
        print(f"菜谱 http://127.0.0.1:{args.port}/recipes", flush=True)
        print(f"器件 http://127.0.0.1:{args.port}/devices", flush=True)
        print(f"本机后台 http://127.0.0.1:{args.port}/admin", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
