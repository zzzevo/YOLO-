# -*- coding: utf-8 -*-
"""进程内验证：Gradio 6 + 自签证书的 HTTPS 能不能正常服务（排除'进程被会话杀死'的干扰）。"""
import ssl
import time
import urllib.request

import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("# SSL 测试")
    box = gr.Textbox(value="hello")


def try_fetch(url, ctx=None):
    try:
        r = urllib.request.urlopen(url, timeout=15, context=ctx)
        return f"OK {r.status} 长度 {len(r.read())}"
    except Exception as e:
        return f"失败 {type(e).__name__}: {str(e)[:120]}"


for tag, kw in (("HTTP", {}),
                ("HTTPS", {"ssl_keyfile": "certs/key.pem", "ssl_certfile": "certs/cert.pem",
                           "ssl_verify": False})):
    demo.launch(server_name="127.0.0.1", server_port=7881 if tag == "HTTP" else 7882,
                prevent_thread_lock=True, quiet=True, show_error=True, **kw)
    time.sleep(3)
    url = f"http://127.0.0.1:7881/" if tag == "HTTP" else "https://127.0.0.1:7882/"
    ctx = None
    if tag == "HTTPS":
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    print(f"{tag}: {try_fetch(url, ctx)}")
    try:
        demo.close()
    except Exception as e:
        print(f"  关闭时: {type(e).__name__}")
    time.sleep(1)
