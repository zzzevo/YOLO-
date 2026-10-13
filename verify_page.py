# -*- coding: utf-8 -*-
"""验证手机页内容（避免 here-string 引号问题）。"""
import ssl
import urllib.request

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

for host in ("127.0.0.1", "192.168.43.202"):
    try:
        r = urllib.request.urlopen(f"https://{host}:7868/", timeout=25, context=ctx)
        b = r.read().decode("utf-8", "replace")
        checks = {
            "版本号": "buildstamp" in b,
            "二维码": "phone_qr" in b or "data:image/png;base64" in b,
            "地址文本": "https://192.168.43.202:7868" in b,
            "通道自检": "通道自检" in b,
            "视频检测": "上传视频检测" in b,
            "抓一帧": "抓一帧" in b,
            "镜像CSS": "scaleX(-1)" in b,
        }
        print(f"{host}: OK {r.status} | " + " ".join(f"{k}={'Y' if v else 'N'}" for k, v in checks.items()))
    except Exception as e:
        print(f"{host}: FAIL {type(e).__name__} {str(e)[:70]}")
