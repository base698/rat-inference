#!/usr/bin/env python3
"""Yes/no review UI for the Red Bull dataset.

Shows each image from the input dir with its proposed box (from
.proposals/<stem>.txt, produced by autolabel_redbull.py). Keyboard:
  y / enter  accept  -> image + label move into images/{train,val} + labels/{train,val}
                        (every 5th accept goes to val)
  n / x      reject  -> image moves to datasets/redbull/rejected/
  s          skip    -> leave for later

Usage:
  python3 tools/vision/dataset/review_server.py                    # reviews raw/
  python3 tools/vision/dataset/review_server.py --input datasets/redbull/synthetic
Then open http://localhost:8020
"""
import argparse
import json
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DS = ROOT / "datasets/redbull"

ap = argparse.ArgumentParser()
ap.add_argument("--input", default=str(DS / "raw"))
ap.add_argument("--port", type=int, default=8020)
args = ap.parse_args()

IN = Path(args.input)
PROP = IN / ".proposals"
REJ = DS / "rejected"
for d in (REJ, DS / "images/train", DS / "images/val", DS / "labels/train", DS / "labels/val"):
    d.mkdir(parents=True, exist_ok=True)

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def pending():
    return sorted(p for p in IN.iterdir() if p.suffix.lower() in EXTS)


def accepted_count():
    return len(list((DS / "images/train").glob("*"))) + len(list((DS / "images/val").glob("*")))


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>redbull review</title>
<style>
 body{background:#111;color:#eee;font-family:monospace;text-align:center;margin:0;padding:12px}
 #wrap{position:relative;display:inline-block}
 img{max-width:94vw;max-height:74vh;display:block}
 #box{position:absolute;border:3px solid #22FF88;box-shadow:0 0 0 1px #000;pointer-events:none}
 button{font-size:20px;padding:12px 26px;margin:10px 6px;border-radius:8px;border:0;cursor:pointer}
 #yes{background:#46D39A}#no{background:#C03A2B;color:#fff}#skip{background:#555;color:#fff}
 #stat{color:#F6C042;min-height:22px}
</style></head><body>
<h3 id=head></h3>
<div id=wrap><img id=im><div id=box hidden></div></div>
<div>
 <button id=yes onclick=vote("accept")>Yes (y)</button>
 <button id=no onclick=vote("reject")>No (n)</button>
 <button id=skip onclick=next(1)>Skip (s)</button>
</div>
<div id=stat></div>
<script>
let cur=null,idx=0;
async function load(){
  const r=await fetch("/next?skip="+idx); const d=await r.json();
  const head=document.getElementById("head");
  if(!d.name){head.textContent="done — nothing pending";document.getElementById("im").src="";document.getElementById("box").hidden=true;return;}
  cur=d; head.textContent=d.name+"  ·  "+d.remaining+" pending  ·  "+d.accepted+" accepted";
  const im=document.getElementById("im");
  im.onload=()=>{
    const b=document.getElementById("box");
    if(d.box){
      const [x,y,w,h]=d.box, W=im.clientWidth, H=im.clientHeight;
      b.style.left=(x-w/2)*W+"px"; b.style.top=(y-h/2)*H+"px";
      b.style.width=w*W+"px"; b.style.height=h*H+"px"; b.hidden=false;
    } else { b.hidden=true; head.textContent+="  ·  NO BOX (probably reject)"; }
  };
  im.src="/img/"+encodeURIComponent(d.name)+"?t="+Date.now();
}
function next(skip){ if(skip)idx++; load(); }
async function vote(v){
  if(!cur||!cur.name)return;
  const r=await fetch("/vote",{method:"POST",body:JSON.stringify({name:cur.name,vote:v})});
  document.getElementById("stat").textContent=await r.text(); idx=0; load();
}
document.addEventListener("keydown",e=>{
  if(e.key==="y"||e.key==="Enter")vote("accept");
  else if(e.key==="n"||e.key==="x")vote("reject");
  else if(e.key==="s")next(1);
});
load();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="text/plain"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else body.encode())

    def do_GET(self):
        if self.path.startswith("/next"):
            skip = 0
            if "skip=" in self.path:
                try:
                    skip = int(self.path.split("skip=")[1])
                except ValueError:
                    pass
            items = pending()
            if not items:
                self._send(200, json.dumps({"name": None}), "application/json")
                return
            p = items[skip % len(items)]
            box = None
            prop = PROP / (p.stem + ".txt")
            if prop.exists():
                txt = prop.read_text().strip()
                if txt:
                    parts = txt.splitlines()[0].split()
                    box = [float(v) for v in parts[1:5]]
            self._send(200, json.dumps({
                "name": p.name, "box": box,
                "remaining": len(items), "accepted": accepted_count(),
            }), "application/json")
        elif self.path.startswith("/img/"):
            name = self.path[5:].split("?")[0]
            from urllib.parse import unquote
            f = IN / unquote(name)
            if f.exists():
                ext = f.suffix.lower().lstrip(".")
                self._send(200, f.read_bytes(), f"image/{'jpeg' if ext=='jpg' else ext}")
            else:
                self._send(404, "gone")
        else:
            self._send(200, PAGE, "text/html")

    def do_POST(self):
        if self.path != "/vote":
            self._send(404, "nope")
            return
        length = int(self.headers.get("Content-Length", 0))
        d = json.loads(self.rfile.read(length))
        src = IN / d["name"]
        prop = PROP / (src.stem + ".txt")
        if not src.exists():
            self._send(410, "already handled")
            return
        if d["vote"] == "accept":
            if not (prop.exists() and prop.read_text().strip()):
                self._send(400, "no box — reject instead, or label by hand with labeler.py")
                return
            split = "val" if accepted_count() % 5 == 4 else "train"
            shutil.move(str(src), DS / "images" / split / src.name)
            shutil.move(str(prop), DS / "labels" / split / (src.stem + ".txt"))
            self._send(200, f"accepted -> {split}")
        else:
            shutil.move(str(src), REJ / src.name)
            if prop.exists():
                prop.unlink()
            self._send(200, "rejected")


if __name__ == "__main__":
    print(f"review UI on http://localhost:{args.port}  input={IN}")
    ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()
