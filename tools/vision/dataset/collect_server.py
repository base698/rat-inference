#!/usr/bin/env python3
"""Red Bull dataset collection page.

Serves a live view of the rig camera (proxying rt_200 /raw-frame, which has
no overlays) with single-shot and burst capture into datasets/redbull/raw/.

Run on the Jetson host:  python3 tools/vision/dataset/collect_server.py
Then open:               http://jetson:8010
"""
import io
import time
import threading
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RT200 = "http://127.0.0.1:8000"
OUT = Path(__file__).resolve().parents[3] / "datasets" / "redbull" / "raw"
OUT.mkdir(parents=True, exist_ok=True)

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>redbull collect</title>
<style>
 body{background:#111;color:#eee;font-family:monospace;text-align:center;margin:0;padding:16px}
 img{max-width:96vw;max-height:70vh;border:2px solid #444;border-radius:8px}
 button{font-size:22px;padding:14px 28px;margin:12px 8px;border-radius:8px;border:0;cursor:pointer}
 #cap{background:#F6C042}
 #burst{background:#46D39A}
 #msg{font-size:18px;color:#F6C042;min-height:24px}
</style></head><body>
<h2>Red Bull collector &rarr; datasets/redbull/raw/ <span id=count></span></h2>
<img id=view src=/frame>
<div>
 <button id=cap onclick=cap()>Capture (space)</button>
 <button id=burst onclick=burst()>Burst 10 (1/s)</button>
</div>
<div id=msg></div>
<script>
 const v=document.getElementById("view"),m=document.getElementById("msg"),c=document.getElementById("count");
 setInterval(()=>{v.src="/frame?t="+Date.now()},350);
 async function refreshCount(){c.textContent="("+await (await fetch("/count")).text()+" imgs)";}
 refreshCount();
 async function cap(){const r=await fetch("/capture",{method:"POST"});m.textContent=await r.text();refreshCount();}
 async function burst(){for(let i=0;i<10;i++){await cap();m.textContent+=" ["+(i+1)+"/10]";await new Promise(s=>setTimeout(s,1000));}}
 document.addEventListener("keydown",e=>{if(e.code==="Space"){e.preventDefault();cap();}});
</script></body></html>"""

def latest_frame():
    with urllib.request.urlopen(RT200 + "/raw-frame", timeout=5) as r:
        return r.read()

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="text/plain"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else body.encode())

    def do_GET(self):
        if self.path.startswith("/frame"):
            try:
                self._send(200, latest_frame(), "image/jpeg")
            except Exception as e:
                self._send(503, f"no frame: {e}")
        elif self.path == "/count":
            self._send(200, str(len(list(OUT.glob("*.jpg")))))
        else:
            self._send(200, PAGE, "text/html")

    def do_POST(self):
        if self.path == "/capture":
            try:
                data = latest_frame()
                if len(data) < 1000:
                    self._send(503, "empty frame — is ratbot running?")
                    return
                name = "capture_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] + ".jpg"
                (OUT / name).write_bytes(data)
                self._send(200, "saved " + name)
            except Exception as e:
                self._send(503, f"capture failed: {e}")
        else:
            self._send(404, "nope")

if __name__ == "__main__":
    print(f"collect server on :8010 -> {OUT}")
    ThreadingHTTPServer(("0.0.0.0", 8010), H).serve_forever()
