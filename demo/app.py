#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""The demo: a held-out duel replays while the net paints what it sees.

Everything the viewer sees runs through the EXPORTED JSON via the pure-numpy
reference forward - the demo exercises the same artifact a hardware target
would consume, not the PyTorch model.

    python demo/app.py                      # -> http://127.0.0.1:5051

Modes (button cycles): float dense masks (default), INT8 dense masks (the
A/B beat - you shouldn't be able to tell), coarse M1 heatmap, off.
A telemetry strip below the canvas draws itself as the net watches: per-class
pixel counts (ships, lasers) over time, from the float dense prediction.
The dense models come from the frozen deploy/ bundle - the demo eats the
same artifact a hardware port would.

Frames are PNGs composited server-side and cached; the browser is a player.
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                                    # repo-local run without install
sys.path.insert(0, str(ROOT / "deploy"))
from spotter import CLASSES, PALETTE                             # noqa: E402
from spotter.reference import forward_heatmap                    # noqa: E402
from spotter.reference import load_doc as load_doc_m1            # noqa: E402
from spotter.render import load_replay_scenes, render            # noqa: E402
from spotter_forward import (forward_dense, forward_dense_int8,  # noqa: E402
                             load_doc)

from PIL import Image                                            # noqa: E402

app = FastAPI()
STATE = {}
_CACHE = {}

PAGE = """<!doctype html><meta charset="utf-8"><title>spotter - the net that sees the game</title>
<style>
 body{background:#0d0d14;color:#ebebe4;font:14px system-ui;margin:24px;text-align:center}
 canvas{image-rendering:pixelated;border:1px solid #2e2e3a}
 #c{width:640px;height:384px} #t{width:640px;height:120px;margin-top:6px}
 button{background:#1c1c28;color:#ebebe4;border:1px solid #2e2e3a;padding:6px 14px;margin:8px 4px;cursor:pointer}
 .legend span{display:inline-block;margin:0 10px 0 4px}
 .sw{display:inline-block;width:10px;height:10px;margin-right:4px}
</style>
<h2>spotter - the net that sees the game</h2>
<div><canvas id=c width=320 height=192></canvas></div>
<div><canvas id=t width=640 height=120></canvas></div>
<div>
 <button id=play>pause</button>
 <button id=mode>mode: dense</button>
 <span id=fno></span>
</div>
<div class=legend id=legend></div>
<p>held-out episode - the net never trained on any frame of this duel.<br>
every pixel of overlay and telemetry comes from the exported JSON via the numpy reference, not PyTorch.</p>
<script>
const c=document.getElementById('c').getContext('2d');
const t=document.getElementById('t').getContext('2d');
const MODES=['dense','int8','heatmap','off'];
let n=0,i=0,run=true,mi=0,pal=null;
const stats={};
const img=new Image();
img.onload=()=>c.drawImage(img,0,0);
fetch('/api/meta').then(r=>r.json()).then(m=>{
  n=m.frames;pal=m.palette;
  document.getElementById('legend').innerHTML=m.classes.slice(1).map((k,j)=>
    `<span><span class=sw style="background:rgb(${pal[j+1]})"></span>${k}</span>`).join('');
  tick();
});
function drawTelemetry(){
  t.fillStyle='#12121a';t.fillRect(0,0,640,120);
  const keys=Object.keys(stats).map(Number).sort((a,b)=>a-b);
  if(!keys.length)return;
  const sx=640/n;
  [[1,120],[2,120],[3,30]].forEach(([cls,scale])=>{   // ships /120px, lasers /30px
    t.strokeStyle=`rgb(${pal[cls]})`;t.beginPath();
    keys.forEach((k,j)=>{
      const y=118-Math.min(116,stats[k][cls]/scale*116);
      j?t.lineTo(k*sx,y):t.moveTo(k*sx,y);
    });
    t.stroke();
  });
  t.fillStyle='#96958c';t.font='11px system-ui';
  t.fillText('pixels seen per class - the curve draws itself as the net watches',8,14);
}
function tick(){
  if(run){
    document.getElementById('fno').textContent=`frame ${i}/${n}`;
    const m=MODES[mi];
    img.src=(m==='dense'?'/dense/':m==='int8'?'/int8/':m==='heatmap'?'/overlay/':'/frame/')+i+'.png';
    if(!(i in stats))fetch('/api/stats/'+i).then(r=>r.json()).then(s=>{stats[i]=s.counts;});
    drawTelemetry();
    i=(i+1)%n;
  }
  setTimeout(tick,60);
}
document.getElementById('play').onclick=e=>{run=!run;e.target.textContent=run?'pause':'play'};
document.getElementById('mode').onclick=e=>{mi=(mi+1)%3;e.target.textContent='mode: '+MODES[mi];};
</script>"""


def _frame_im(i: int) -> Image.Image:
    im, _ = render(STATE["scenes"][i])
    return im


def _x(im: Image.Image) -> np.ndarray:
    return (np.asarray(im.convert("RGB")).astype(np.float32) / 255.0
            ).transpose(2, 0, 1)


def _tint(im: Image.Image, pred: np.ndarray) -> Image.Image:
    tint = np.zeros((im.height, im.width, 4), dtype=np.uint8)
    for cid in range(1, len(CLASSES)):
        tint[pred == cid] = (*PALETTE[cid], 130)
    im.alpha_composite(Image.fromarray(tint))
    return im.convert("RGB")


def _dense(i: int):
    """-> (composited PIL image, per-class pixel counts) from the dense net."""
    im = _frame_im(i).convert("RGBA")
    pred = forward_dense(_x(im), STATE["dense"]).argmax(0)
    counts = np.bincount(pred.ravel(), minlength=len(CLASSES)).tolist()
    return _tint(im, pred), counts


def _int8(i: int) -> Image.Image:
    im = _frame_im(i).convert("RGBA")
    _, pred = forward_dense_int8(_x(im), STATE["int8"])
    return _tint(im, pred)


def _heatmap(i: int) -> Image.Image:
    im = _frame_im(i).convert("RGBA")
    if STATE["patch"] is None:
        return im
    hm = forward_heatmap(_x(im), STATE["patch"])
    e = np.exp(hm - hm.max(axis=0, keepdims=True))
    conf = e / e.sum(axis=0, keepdims=True)
    cls = hm.argmax(axis=0)
    tint = np.zeros((im.height, im.width, 4), dtype=np.uint8)
    for (r, col), c in np.ndenumerate(cls):
        if c == 0:
            continue
        a = int(180 * conf[c, r, col])
        y0, x0 = 8 * r + 12, 8 * col + 12  # the cell's own 8x8 tile
        tint[y0:y0 + 8, x0:x0 + 8] = (*PALETTE[c], a)
    im.alpha_composite(Image.fromarray(tint))
    return im.convert("RGB")


def _png(im: Image.Image) -> Response:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png")


def _dense_cached(i: int):
    i %= len(STATE["scenes"])
    key = ("d", i)
    if key not in _CACHE:
        im, counts = _dense(i)
        _CACHE[key] = (_png(im), counts)
    return _CACHE[key]


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.get("/api/meta")
def meta():
    return {"frames": len(STATE["scenes"]), "classes": CLASSES,
            "palette": {k: v for k, v in PALETTE.items() if k != 255}}


@app.get("/api/stats/{i}")
def stats(i: int):
    return {"counts": _dense_cached(i)[1]}


@app.get("/frame/{i}.png")
def frame(i: int):
    key = ("f", i % len(STATE["scenes"]))
    if key not in _CACHE:
        _CACHE[key] = _png(_frame_im(key[1]))
    return _CACHE[key]


@app.get("/dense/{i}.png")
def dense(i: int):
    return _dense_cached(i)[0]


@app.get("/int8/{i}.png")
def int8(i: int):
    key = ("q", i % len(STATE["scenes"]))
    if key not in _CACHE:
        _CACHE[key] = _png(_int8(key[1]))
    return _CACHE[key]


@app.get("/overlay/{i}.png")
def overlay(i: int):
    key = ("o", i % len(STATE["scenes"]))
    if key not in _CACHE:
        _CACHE[key] = _png(_heatmap(key[1]))
    return _CACHE[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense",
                    default=str(ROOT / "deploy/models/spotter_dense.json"))
    ap.add_argument("--int8",
                    default=str(ROOT / "deploy/models/spotter_dense_int8.json"))
    ap.add_argument("--patch", default=str(ROOT / "runs/m1/spotter_patch.json"))
    ap.add_argument("--replay",
                    default=str(ROOT / "assets/replays/seed_005.json"))
    ap.add_argument("--port", type=int, default=5051)
    args = ap.parse_args()
    STATE["dense"] = load_doc(args.dense)
    STATE["int8"] = load_doc(args.int8)
    try:
        STATE["patch"] = load_doc_m1(args.patch)
    except FileNotFoundError:
        STATE["patch"] = None  # heatmap view needs a local M1 run
    STATE["scenes"] = load_replay_scenes(args.replay)
    print(f"{len(STATE['scenes'])} frames; http://127.0.0.1:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
