import re
import urllib.request

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

URLS = [
    "https://www.bing.com/images/search?q=pink+basketball&count=35",
    "https://www.bing.com/images/search?q=pink+basketball&count=35&first=1",
    "https://cn.bing.com/images/search?q=pink+basketball&count=35",
]

for url in URLS:
    print("---", url[:80])
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        html = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "replace")
        print(f"    长度 {len(html)} | count('murl')={html.count('murl')} | "
              f"count('turl')={html.count('turl')} | count('.jpg')={html.count('.jpg')} | "
              f"count('mediaurl')={html.count('mediaurl')}")
        for pat in [r"https?://[^\"\\ ]{15,90}\.jpg", r"murl&quot;:&quot;([^&]{10,80})",
                    r'"murl":"([^"]{10,80})', r"turl&quot;:&quot;([^&]{10,80})"]:
            m = re.findall(pat, html)[:3]
            if m:
                print("    样例:", [x[:70] for x in m])
        if len(html) > 1000:
            open("_bing_dump.html", "w", encoding="utf-8").write(html[:300000])
            print("    已保存 _bing_dump.html")
    except Exception as e:
        print("    失败:", type(e).__name__, str(e)[:80])
