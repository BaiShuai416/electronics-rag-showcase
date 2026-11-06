"""Keep the loopback demo and its temporary public tunnel in one terminal."""

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


HERE = Path(__file__).resolve().parent


def stop(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main():
    parser = argparse.ArgumentParser(description="启动两套问答页面的临时公网演示")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    bundled = HERE / ".runtime" / "cloudflared.exe"
    cloudflared = str(bundled) if bundled.is_file() else shutil.which("cloudflared")
    if not cloudflared:
        raise SystemExit("请先准备 cloudflared，下载位置和启动步骤见 showcase/README.md。")

    local_url = f"http://127.0.0.1:{args.port}"
    server = subprocess.Popen(
        [sys.executable, "-X", "utf8", str(HERE / "server.py"), "--port", str(args.port)],
        cwd=HERE,
    )
    tunnel = None
    try:
        for _ in range(40):
            if server.poll() is not None:
                raise SystemExit("本地问答服务未能启动，请检查端口和 Python 环境。")
            try:
                with urlopen(local_url + "/healthz", timeout=1) as response:
                    if response.status == 200:
                        break
            except (URLError, TimeoutError):
                time.sleep(0.25)
        else:
            raise SystemExit("等待本地问答服务超时。")

        tunnel = subprocess.Popen(
            [cloudflared, "tunnel", "--no-autoupdate", "--protocol", "http2", "--url", local_url],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace",
        )
        public_url = None
        for line in tunnel.stdout:
            print(line.rstrip(), flush=True)
            match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com\b", line)
            if match and public_url is None:
                public_url = match.group()
                print(f"\n菜谱演示 {public_url}/recipes\n器件演示 {public_url}/devices", flush=True)
                print("保持此窗口运行。按 Ctrl+C 停止演示与隧道。\n", flush=True)
        code = tunnel.wait()
        if public_url is None or code:
            raise SystemExit("临时隧道已退出，请查看上方的连接日志。")
    except KeyboardInterrupt:
        print("\n正在结束临时演示。")
    finally:
        stop(tunnel)
        stop(server)


if __name__ == "__main__":
    main()
