#!/usr/bin/env python3
"""
验证码框选标注服务（纯标准库，无第三方依赖）

为什么不用 FastAPI
------------------
目标服务器（Debian 12）没有 pip / ensurepip，装不了第三方包。
用 http.server + 自写路由，零依赖，直接 python3 跑。

安全设计
--------
1. **访问令牌**：必须带 ?k=<token>，否则 403
2. **不暴露文件路径**：所有响应只返回图片内容或 JSON，不回显服务器路径
3. **禁止目录穿越**：文件名白名单校验
4. **不能查看已标注的图**：/api/next 只发未标注的图，/api/image 对已标注图返回 403
5. **监听 127.0.0.1**：由 nginx/openresty 反代，不直接暴露

用法
----
    python3 label_server.py --port 8899 --token YOUR_SECRET
"""

from __future__ import annotations

import argparse
import hmac
import json
import mimetypes
import os
import re
import secrets
import socketserver
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
DATA_DIR.mkdir(exist_ok=True)

DATASETS = {
    "live4": {"dir": "_live4", "glob": "*.jpg",
              "out": "data/boxes_live4.jsonl",
              "label": "真实 4 字（新字形，优先核对）"},
    "review": {"dir": "_live4_review", "glob": "*.jpg",
               "out": "data/boxes_review.jsonl",
               "label": "4 字复核（语序判错的）"},
    "phrase": {"dir": "captcha_dataset/phrase", "glob": "*.jpg",
               "out": "data/glyph_boxes.jsonl", "label": "phrase 词组点选（旧 3 字）"},
    "word": {"dir": "captcha_dataset/word", "glob": "*_bg.jpg",
             "out": "data/boxes_word.jsonl", "label": "word 选字"},
    "icon": {"dir": "captcha_dataset/icon", "glob": "*_bg.jpg",
             "out": "data/boxes_icon.jsonl", "label": "icon 选图标"},
}

STATE = {"key": "live4", "token": "", "require_login": True}

# 会话表：sid -> {"user": str, "expire": float}
SESSIONS: dict[str, dict] = {}
SESSION_TTL = 86400          # 24 小时

# 账号密码（部署时用 --users 覆盖，或改这里）
# 未传 --users 时用下面这组默认账号，正式部署请改掉
DEFAULT_USER = "LeapYa"
DEFAULT_PASS = "Lbl#2026kQ7vx"
USERS: dict[str, str] = {DEFAULT_USER: DEFAULT_PASS}


def _check_login(user: str, pwd: str) -> bool:
    if not USERS:
        return False
    expect = USERS.get(user)
    if expect is None:
        return False
    return hmac.compare_digest(expect, pwd)


def _new_session(user: str) -> str:
    sid = secrets.token_urlsafe(24)
    SESSIONS[sid] = {"user": user, "expire": time.time() + SESSION_TTL}
    # 顺手清过期会话
    now = time.time()
    for k in [k for k, v in SESSIONS.items() if v["expire"] <= now]:
        SESSIONS.pop(k, None)
    return sid

SAFE_NAME = re.compile(r"^[A-Za-z0-9_\-]+\.jpg$")
SAFE_NAME_Q = re.compile(r"^[A-Za-z0-9_\-]+\.png$")


# ---------------------------------------------------------------- 工具
def ds_cfg(key: str | None = None) -> dict:
    return DATASETS[key or STATE["key"]]


def list_images(key: str | None = None) -> list[str]:
    cfg = ds_cfg(key)
    d = HERE / cfg["dir"]
    if not d.exists():
        return []
    return sorted(p.name for p in d.glob(cfg["glob"]))


def out_path(key: str | None = None) -> Path:
    return HERE / ds_cfg(key)["out"]


def load_confirmed(key: str | None = None) -> dict[str, dict]:
    """只统计人工核对过的（排除 live4 的 OCR 预填）。

    live4 数据集里，boxes_live4.jsonl 同时存了 OCR 预填和人工核对的结果：
      - 预填记录：无 reviewed/ts
      - 人工保存过的记录：reviewed=True（旧记录以 ts 存在为标志）
    用记录级标志判断，boxes[].src 只表示「这个框的字是否被人工改过」。
    """
    done = load_done(key)
    k = key or STATE["key"]
    if k not in ("live4", "review"):
        return done
    out = {}
    for img, r in done.items():
        # 跳过：算作已处理（避免反复弹回同一张），但不作为训练数据
        if r.get("skip"):
            out[img] = r
            continue
        # 记录级标志是唯一判据：人工保存过的记录带 reviewed=True（旧记录用 ts）。
        # ⚠️ 不要再用「全部框 src==ocr」推断未核对 —— 预填全对、一个字没改就
        #    保存的记录框仍是 src=ocr，会被漏算；反过来「部分框改过」的
        #    未浏览记录又会被多算。src 只用来区分训练时哪些框是人改的。
        if r.get("reviewed") or r.get("ts"):
            out[img] = r
    return out


def load_done(key: str | None = None) -> dict[str, dict]:
    p = out_path(key)
    done: dict[str, dict] = {}
    if not p.exists():
        return done
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                img = r.get("img")
                if img:
                    done[img] = r
            except Exception:
                continue
    except Exception:
        pass
    return done


def save_all(done: dict[str, dict], key: str | None = None) -> None:
    p = out_path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for k in sorted(done):
            fh.write(json.dumps(done[k], ensure_ascii=False) + "\n")
    tmp.replace(p)


# 繁简映射
T2S = {
    "決": "决", "說": "说", "國": "国", "學": "学", "會": "会", "體": "体",
    "實": "实", "對": "对", "開": "开", "關": "关", "發": "发", "現": "现",
    "電": "电", "車": "车", "門": "门", "東": "东", "馬": "马", "鳥": "鸟",
    "魚": "鱼", "龍": "龙", "雲": "云", "風": "风", "飛": "飞", "氣": "气",
    "長": "长", "萬": "万", "與": "与", "書": "书", "畫": "画", "筆": "笔",
    "號": "号", "線": "线", "點": "点", "題": "题", "設": "设", "計": "计",
    "試": "试", "認": "认", "識": "识", "語": "语", "讀": "读", "寫": "写",
    "頭": "头", "臉": "脸", "腳": "脚", "貝": "贝", "頁": "页", "見": "见",
    "愛": "爱", "親": "亲", "師": "师", "醫": "医", "藥": "药", "廠": "厂",
    "廣": "广", "場": "场", "農": "农", "產": "产", "業": "业", "團": "团",
    "島": "岛", "灣": "湾", "縣": "县", "鄉": "乡", "鐵": "铁", "銀": "银",
    "錢": "钱", "價": "价", "買": "买", "賣": "卖", "費": "费", "貨": "货",
}


def to_simplified(ch: str) -> str:
    return T2S.get(ch, ch)


# ---------------------------------------------------------------- HTML
HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>验证码框选标注</title>
<style>
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
     background:#0f1115;color:#e6e6e6}
.bar{display:flex;align-items:center;gap:12px;padding:10px 16px;background:#171a21;
     border-bottom:1px solid #262b36;position:sticky;top:0;z-index:10;flex-wrap:wrap}
.bar h1{font-size:15px;margin:0;font-weight:600}
.prog{font-size:13px;color:#8b95a7}
.prog b{color:#4ade80}
.spacer{flex:1}
button{background:#2a3040;color:#dfe4ee;border:1px solid #3a4256;padding:7px 14px;
       border-radius:7px;cursor:pointer;font-size:13px}
button:hover{background:#333b4d}
button.primary{background:#2563eb;border-color:#2563eb;color:#fff}
button.primary:hover{background:#1d4ed8}
select{background:#2a3040;color:#dfe4ee;border:1px solid #3a4256;padding:7px 10px;
       border-radius:7px;font-size:13px}
.wrap{display:flex;gap:18px;padding:18px;align-items:flex-start;flex-wrap:wrap}
.stage{position:relative;border:1px solid #2b3240;border-radius:10px;overflow:hidden;
       background:#000;user-select:none}
canvas{display:block;cursor:crosshair;max-width:100%}
.side{width:330px;flex:none;max-width:100%}
.card{background:#171a21;border:1px solid #262b36;border-radius:10px;padding:12px;
      margin-bottom:12px}
.card h2{font-size:13px;margin:0 0 10px;color:#9aa4b8;font-weight:600}
.boxrow{display:flex;align-items:center;gap:8px;padding:6px 0;
        border-bottom:1px solid #21262f;font-size:13px}
.boxrow .grip{width:16px;flex:0 0 auto;color:#64748b;cursor:grab;font-size:13px;
  line-height:1;user-select:none;text-align:center;letter-spacing:-2px}
.boxrow .grip:active{cursor:grabbing}
.boxrow.drop-line{outline:2px dashed #38bdf8;outline-offset:-2px}
.boxrow[draggable="true"]{transition:opacity .12s}
.boxrow:last-child{border-bottom:none}
.idx{width:22px;height:22px;border-radius:50%;display:flex;align-items:center;
     justify-content:center;font-size:11px;background:#2563eb;color:#fff;flex:none}
.boxrow input{width:56px;background:#0f1115;border:1px solid #333b4d;color:#fff;
              border-radius:5px;padding:4px 6px;text-align:center;font-size:15px}
.boxrow .coord{color:#6b7689;font-size:11px;flex:1}
.boxrow .del{color:#f87171;cursor:pointer;font-size:16px}
.hint{font-size:12px;color:#6b7689;line-height:1.7}
.hint kbd{background:#262b36;padding:1px 5px;border-radius:4px;font-family:monospace;
          font-size:11px}
.empty{color:#4b5563;font-size:13px;padding:14px 0;text-align:center}
.phrase-card{border-color:#2563eb55;background:#141a26}
.ques-card{border-color:#f59e0b55;background:#1c1810}
.ques-row{display:flex;gap:10px;justify-content:center;flex-wrap:wrap}
.ques-item{text-align:center}
.ques-item img{width:72px;height:72px;image-rendering:pixelated;
               border:1px solid #3a4256;border-radius:6px;
               background:#fff;padding:2px}
.ques-item .qlabel{font-size:11px;color:#fbbf24;margin-top:3px}
.ques-hint{font-size:12px;color:#a16207;line-height:1.6;margin-top:8px}
.ph-hint{font-size:11px;color:#60a5fa;font-weight:400}
.phrase-input{width:100%;background:#0f1115;border:2px solid #3b82f6;color:#fff;
              border-radius:8px;padding:10px 12px;font-size:22px;letter-spacing:3px;
              text-align:center;outline:none}

/* ---- 排序按钮（手机拖拽不可用，用按钮代替）---- */
.boxrow .mv{display:flex;flex-direction:column;gap:1px;flex:none}
.boxrow .mv button{padding:0;width:20px;height:15px;font-size:10px;line-height:13px;
  border-radius:3px;background:#232a38;border:1px solid #333b4d;color:#9aa4b8}
.boxrow .mv button:hover{background:#2f394b;color:#e6e6e6}
.boxrow .mv button:disabled{opacity:.25;cursor:default}

/* ---- 窄屏（手机）---- */
@media (max-width:760px){
  .bar{padding:8px 10px;gap:8px}
  .bar h1{font-size:14px}
  .prog{font-size:12px}
  .bar button{padding:8px 10px;font-size:12px}
  .bar select{padding:7px 8px;font-size:12px;max-width:46vw}

  .wrap{flex-direction:column;padding:10px;gap:12px}
  .stage{width:100%}
  canvas{width:100%!important;height:auto!important;max-width:100%}
  .side{width:100%;flex:1 1 auto}

  .card{padding:10px}
  .card h2{font-size:12px;margin-bottom:8px}

  .boxrow{gap:6px;padding:8px 0}
  .boxrow input{width:52px;font-size:18px;padding:8px 4px}
  .boxrow .coord{font-size:10px}
  .boxrow .grip{display:none}          /* 手机不显示拖拽把手 */
  .boxrow .del{font-size:20px;padding:0 6px}

  /* 手机上操作按钮全宽，两列排 */
  .ops{display:grid;grid-template-columns:1fr 1fr;gap:8px}
  .ops button{width:100%;padding:12px 8px;font-size:14px}

  .phrase-input{font-size:20px;padding:12px}

  /* 提示在小屏收起，省空间 */
  .hint{font-size:11px}
}

/* ---- 超窄屏 ---- */
@media (max-width:400px){
  .boxrow .coord{display:none}
  .ops{grid-template-columns:1fr}
}

.phrase-input:focus{border-color:#60a5fa;box-shadow:0 0 0 3px #3b82f633}
.match-ok{color:#4ade80}
.match-part{color:#fbbf24}
.match-none{color:#f87171}
#toast{position:fixed;bottom:24px;left:50%;transform:translateX(-50%);background:#1f2937;
       border:1px solid #374151;color:#e5e7eb;padding:10px 18px;border-radius:8px;
       font-size:13px;opacity:0;transition:.2s;pointer-events:none;z-index:99}
#toast.show{opacity:1}
</style>
</head>
<body>
<div class="bar">
  <h1>验证码框选标注</h1>
  <span class="prog" id="prog">加载中…</span>
  <div class="spacer"></div>
  <select id="dsSelect" title="切换数据集"></select>
  <button id="btnPrev">← 上一张</button>
  <button id="btnUndo">撤销</button>
  <button id="btnLogout" title="退出登录">退出</button>
  <button id="btnUncertain" title="字形被遮挡/看不清">存疑</button>
  <button id="btnSkip">跳过</button>
  <button class="primary" id="btnDone">完成 →</button>
</div>
<div class="wrap">
  <div class="stage"><canvas id="cv"></canvas></div>
  <div class="side">
    <div class="card" id="quesCard" style="display:none">
      <h2>要找的目标 <span id="quesCount" class="ph-hint"></span></h2>
      <div class="ques-row" id="quesRow"></div>
      <div class="ques-hint" id="quesHint">
        在左边底图里找到这几个字，按上方顺序依次框选
      </div>
    </div>
    <div class="card phrase-card" id="phraseCard">
      <h2>词组快填 <span id="phHint" class="ph-hint"></span></h2>
      <input id="phrase" class="phrase-input" placeholder="打词组，如：决定性"
             autocomplete="off" spellcheck="false">
      <div class="hint" style="margin-top:8px">
        按<b>框的顺序</b>分配 → <b>Enter</b> 保存并下一张
      </div>
    </div>
    <div class="card">
      <h2>框选列表 <span id="cnt" style="color:#4ade80"></span></h2>
      <div id="list"><div class="empty">还没有框选</div></div>
    </div>
    <div class="card">
      <h2>操作</h2>
      <div class="hint">
        1. 图上<b>拖拽</b>框字（按词语顺序：第一个字先框）<br>
        2. 顶部打<b>词组</b>，自动分配<br>
        3. <b>完成 →</b> 保存并切下一张<br><br>
        <kbd>Enter</kbd> 保存并下一张 &nbsp; <kbd>Ctrl</kbd>+<kbd>Z</kbd> 撤销<br>
        <kbd>Esc</kbd> 取消拖拽<br><br>
        框松一点没关系，别把两个字框进一个框。
      </div>
    </div>
  </div>
</div>
<div id="toast"></div>
<script>
const TOKEN = new URLSearchParams(location.search).get('k') || '';
function api(p, opt) {
  const sep = p.includes('?') ? '&' : '?';
  const url = p + (TOKEN ? sep + 'k=' + encodeURIComponent(TOKEN) : '');
  return fetch(url, opt).then(r => r.json());
}
function imgUrl(n) {
  const sep = '?';
  return '/api/image/' + encodeURIComponent(n) + sep + 't=' + Date.now() +
         (TOKEN ? '&k=' + encodeURIComponent(TOKEN) : '');
}
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
let currentImage = null;
let state = {img:null,boxes:[],dragging:null,scale:2,total:0,done:0,index:0,
             uncertain:false,note:'',dsKey:'',sel:-1,resizing:null,dragStart:null};

function toast(m,ms=1800){const t=document.getElementById('toast');
  t.textContent=m;t.classList.add('show');clearTimeout(t._t);
  t._t=setTimeout(()=>t.classList.remove('show'),ms);}

function setProg(){
  document.getElementById('prog').innerHTML =
    '第 <b>'+(state.index+1)+'</b> / '+state.total+' 张 &nbsp;·&nbsp; 已存 <b>'+state.done+'</b>';
}
function updateUncertainBtn(){
  const b=document.getElementById('btnUncertain');
  if(state.uncertain){b.style.background='#b45309';b.style.borderColor='#d97706';
    b.textContent='存疑 ✓';}
  else{b.style.background='';b.style.borderColor='';b.textContent='存疑';}
}

async function loadDatasets(){
  const d = await api('/api/datasets');
  const sel=document.getElementById('dsSelect');
  sel.innerHTML=(d.datasets||[]).map(x=>
    '<option value="'+x.key+'"'+(x.current?' selected':'')+'>'+
    x.label+' ('+x.done+'/'+x.total+')</option>').join('');
  state.dsKey=d.current;
  isIcon = (d.current === 'icon');
  // phrase 型用词组快填；word/icon 型用 ques 面板
  const pc = document.getElementById('phraseCard');
  if (pc) pc.style.display = (d.current === 'phrase') ? '' : 'none';
  sel.onchange=async()=>{
    const r=await api('/api/switch',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({key:sel.value})});
    if(r.ok){toast('已切到 '+r.label);state.boxes=[];currentImage=null;
      isIcon = (sel.value === 'icon');
      if (pc) pc.style.display = (sel.value === 'phrase') ? '' : 'none';
      await loadNext('');await loadDatasets();}
    else toast('切换失败');
  };
}

let isIcon = false;

async function loadQues(name){
  const card=document.getElementById('quesCard');
  const row=document.getElementById('quesRow');
  const cnt=document.getElementById('quesCount');
  const hint=document.getElementById('quesHint');
  if(!name||!name.endsWith('_bg.jpg')){ card.style.display='none'; return; }
  try{
    const d=await api('/api/ques/'+encodeURIComponent(name));
    if(!d.ques||!d.ques.length){ card.style.display='none'; return; }
    card.style.display='';
    cnt.textContent='('+d.ques.length+' 个)';
    row.innerHTML=d.ques.map((q,i)=>
      '<div class="ques-item">'+
      '<img src="/api/quesimg/'+encodeURIComponent(q)+'?t='+Date.now()+
      (TOKEN?'&k='+encodeURIComponent(TOKEN):'')+'" alt="'+q+'">'+
      '<div class="qlabel">第 '+(i+1)+' 个</div></div>').join('');
    hint.textContent = isIcon
      ? '在底图里找到这几个图标，按上方顺序依次框选'
      : '在底图里找到这几个字，按上方顺序依次框选（框里填对应的字）';
  }catch(e){ card.style.display='none'; }
}

async function loadImage(name,index){
  state.img=name;
  if(typeof index==='number')state.index=index;
  await loadQues(name);
  const ex=await api('/api/existing/'+encodeURIComponent(name));
  state.boxes=(ex&&ex.boxes)?ex.boxes.map(b=>({...b})):[];
  state.uncertain=!!(ex&&ex.uncertain);
  state.note=(ex&&ex.note)||'';
  updateUncertainBtn();
  const ph=document.getElementById('phrase');
  ph.value=state.boxes.map(b=>b.char||'').join('');
  syncPhraseHint();renderList();setProg();
  const img=new Image();
  img.onload=()=>{
    currentImage=img;state.scale=2;
    cv.width=img.width*state.scale;cv.height=img.height*state.scale;
    cv.style.width=cv.width+'px';cv.style.height=cv.height+'px';
    ctx.imageSmoothingEnabled=false;redraw();
  };
  img.src=imgUrl(name);
}

async function loadNext(after){
  const d=await api('/api/next?after='+encodeURIComponent(after||''));
  if(!d.img){
    state.img=null;
    document.getElementById('prog').textContent='全部完成！共 '+d.total+' 张';
    ctx.clearRect(0,0,cv.width,cv.height);currentImage=null;
    toast('全部标注完成 🎉');return;
  }
  state.total=d.total;state.done=d.done;
  await loadImage(d.img,d.index);
}

function redraw(){
  if(!currentImage)return;
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.drawImage(currentImage,0,0,cv.width,cv.height);
  state.boxes.forEach((b,i)=>{
    const sel = (i===state.sel);
    ctx.lineWidth=sel?3:2;
    ctx.strokeStyle=sel?'#fbbf24':'#22d3ee';
    const X=b.x*state.scale,Y=b.y*state.scale,W=b.w*state.scale,H=b.h*state.scale;
    ctx.strokeRect(X,Y,W,H);
    ctx.fillStyle=sel?'#f59e0b':'#2563eb';ctx.beginPath();
    ctx.arc(X+9,Y+9,9,0,7);ctx.fill();
    ctx.fillStyle='#fff';ctx.font='bold 11px sans-serif';
    ctx.textAlign='center';ctx.textBaseline='middle';
    ctx.fillText(String(i+1),X+9,Y+9);
    // 选中时画 8 个缩放手柄
    if(sel){
      ctx.fillStyle='#fbbf24';
      const hs=5;
      [[X,Y],[X+W/2,Y],[X+W,Y],[X+W,Y+H/2],[X+W,Y+H],[X+W/2,Y+H],[X,Y+H],[X,Y+H/2]]
        .forEach(([hx,hy])=>{ctx.fillRect(hx-hs,hy-hs,hs*2,hs*2);});
    }
  });
  if(state.dragging){
    const d=state.dragging;
    const x=Math.min(d.x0,d.x1)*state.scale,y=Math.min(d.y0,d.y1)*state.scale;
    const w=Math.abs(d.x1-d.x0)*state.scale,h=Math.abs(d.y1-d.y0)*state.scale;
    ctx.setLineDash([5,4]);ctx.strokeStyle='#fbbf24';ctx.lineWidth=2;
    ctx.strokeRect(x,y,w,h);ctx.setLineDash([]);
  }
}

function pos(e){
  const r=cv.getBoundingClientRect();
  const sx=cv.width/r.width, sy=cv.height/r.height;
  return {x:(e.clientX-r.left)*sx/state.scale,
          y:(e.clientY-r.top)*sy/state.scale};
}
// 手柄命中半径（屏幕像素，换算到图像坐标）
const HANDLE_PX = 9;

function hitHandle(b,p){
  // 返回命中的手柄名，或 null。判断顺序：四角 → 四边 → 框内
  const tol = HANDLE_PX/state.scale;
  const x0=b.x, y0=b.y, x1=b.x+b.w, y1=b.y+b.h;
  const nearL=Math.abs(p.x-x0)<=tol, nearR=Math.abs(p.x-x1)<=tol;
  const nearT=Math.abs(p.y-y0)<=tol, nearB=Math.abs(p.y-y1)<=tol;
  const inX=p.x>=x0-tol && p.x<=x1+tol;
  const inY=p.y>=y0-tol && p.y<=y1+tol;
  if(!(inX&&inY)) return null;
  if(nearL&&nearT) return 'nw';
  if(nearR&&nearT) return 'ne';
  if(nearL&&nearB) return 'sw';
  if(nearR&&nearB) return 'se';
  if(nearT) return 'n';
  if(nearB) return 's';
  if(nearL) return 'w';
  if(nearR) return 'e';
  // 框内
  if(p.x>x0&&p.x<x1&&p.y>y0&&p.y<y1) return 'move';
  return null;
}

function boxAt(p){
  // 从后往前找（后画的在上层）
  for(let i=state.boxes.length-1;i>=0;i--){
    if(hitHandle(state.boxes[i],p)) return i;
  }
  return -1;
}

function applyResize(b,mode,p,orig){
  // orig: 拖动开始时的框快照
  let {x,y,w,h}=orig;
  if(mode==='move'){
    x = orig.x + (p.x-state.dragStart.x);
    y = orig.y + (p.y-state.dragStart.y);
  }else{
    if(mode.includes('w')){ const nx=Math.min(p.x, orig.x+orig.w-8); w=orig.x+orig.w-nx; x=nx; }
    if(mode.includes('e')){ w=Math.max(8, p.x-orig.x); }
    if(mode.includes('n')){ const ny=Math.min(p.y, orig.y+orig.h-8); h=orig.y+orig.h-ny; y=ny; }
    if(mode.includes('s')){ h=Math.max(8, p.y-orig.y); }
  }
  // 限制在图内
  const iw=cv.width/state.scale, ih=cv.height/state.scale;
  w=Math.min(w, iw-x); h=Math.min(h, ih-y);
  if(x<0){ w+=x; x=0; }
  if(y<0){ h+=y; y=0; }
  b.x=Math.round(x); b.y=Math.round(y);
  b.w=Math.max(8,Math.round(w)); b.h=Math.max(8,Math.round(h));
}

cv.addEventListener('mousedown',e=>{
  if(!state.img||e.button!==0)return;e.preventDefault();
  const p=pos(e);
  const i=boxAt(p);
  if(i>=0){
    // 选中并开始拖动/缩放
    const mode=hitHandle(state.boxes[i],p);
    state.sel=i;
    state.dragStart={x:p.x,y:p.y};
    state.resizing={i,mode,orig:{...state.boxes[i]}};
    renderList();redraw();
    return;
  }
  // 空白处：取消选中，开始画新框
  state.sel=-1;
  state.dragging={x0:p.x,y0:p.y,x1:p.x,y1:p.y};
  redraw();
});
cv.addEventListener('mousemove',e=>{
  // 拖动/缩放已有框
  if(state.resizing){
    e.preventDefault();
    const p=pos(e);
    applyResize(state.boxes[state.resizing.i], state.resizing.mode, p, state.resizing.orig);
    updateCoords();
    redraw();
    return;
  }
  if(!state.dragging)return;e.preventDefault();
  const p=pos(e);
  state.dragging.x1=Math.max(0,Math.min(p.x,cv.width/state.scale));
  state.dragging.y1=Math.max(0,Math.min(p.y,cv.height/state.scale));
  redraw();
});
function finishDrag(){
  // 缩放/拖动已有框结束
  if(state.resizing){
    state.resizing=null;
    updateCoords(); redraw();
    return;
  }
  if(!state.dragging) return;
  const d=state.dragging; state.dragging=null;
  const x=Math.round(Math.max(0,Math.min(d.x0,d.x1)));
  const y=Math.round(Math.max(0,Math.min(d.y0,d.y1)));
  const w=Math.round(Math.abs(d.x1-d.x0)), h=Math.round(Math.abs(d.y1-d.y0));
  if(w<6||h<6){ redraw(); return; }
  state.boxes.push({x,y,w,h,char:''});
  state.sel=state.boxes.length-1;
  renderList(); redraw(); syncPhraseHint();
  document.getElementById('phrase').focus();
}
// 拖拽结束统一走 finishDrag，canvas 和 document 都监听：
// 拖拽中鼠标移出画布时由 document 兜底，避免丢框（实测出现过只保存 1 个框）
cv.addEventListener('mouseup', finishDrag);
document.addEventListener('mouseup', finishDrag);

// 完整重建列表（数量/顺序变化时才调用）
function renderList(){
  const list=document.getElementById('list');
  const cnt=document.getElementById('cnt');
  cnt.textContent=state.boxes.length?'('+state.boxes.length+')':'';
  if(!state.boxes.length){
    list.innerHTML='<div class="empty">还没有框选</div>';
    return;
  }
  list.innerHTML=state.boxes.map((b,i)=>
    '<div class="boxrow" data-sel="'+i+'" data-row="'+i+'" draggable="true" '+
    'style="'+(i===state.sel?'background:#f59e0b22':'')+'">'+
    '<div class="grip" title="拖动调整顺序">⋮⋮</div>'+
    '<div class="idx">'+(i+1)+'</div>'+
    '<input data-i="'+i+'" value="'+(b.char||'')+'" maxlength="2" placeholder="字" draggable="false">'+
    '<div class="mv">'+
      '<button data-mv="up" data-i="'+i+'"'+(i===0?' disabled':'')+'>▲</button>'+
      '<button data-mv="down" data-i="'+i+'"'+(i===state.boxes.length-1?' disabled':'')+'>▼</button>'+
    '</div>'+
    '<div class="coord" data-coord="'+i+'">'+b.x+','+b.y+' '+b.w+'×'+b.h+'</div>'+
    '<div class="del" data-del="'+i+'">×</div></div>').join('');

  list.querySelectorAll('input').forEach(inp=>{
    // 只改数据，不重建 DOM —— 否则输入框被销毁，光标丢失
    inp.addEventListener('input',e=>{
      const i=+e.target.dataset.i;
      if(state.boxes[i]) state.boxes[i].char=e.target.value.trim();
      syncPhraseHint();
    });
    inp.addEventListener('keydown',e=>{
      if(e.key==='Enter'){e.preventDefault();saveAndNext(false);}
    });
    inp.addEventListener('focus',e=>{
      const i=+e.target.dataset.i;
      if(state.sel!==i){ state.sel=i; updateSelHighlight(); redraw(); }
    });
  });
  list.querySelectorAll('[data-sel]').forEach(el=>{
    el.addEventListener('click',e=>{
      if(e.target.dataset.del!==undefined) return;
      if(e.target.tagName==='INPUT') return;
      state.sel=+el.dataset.sel;
      updateSelHighlight(); redraw();
    });
  });
  list.querySelectorAll('[data-del]').forEach(el=>{
    el.addEventListener('click',()=>{
      state.boxes.splice(+el.dataset.del,1);
      if(state.sel>=state.boxes.length) state.sel=state.boxes.length-1;
      renderList(); redraw(); syncPhraseHint();
    });
  });
  // 上移 / 下移（手机友好，不依赖 HTML5 拖拽）
  list.querySelectorAll('[data-mv]').forEach(btn=>{
    btn.addEventListener('click',e=>{
      e.stopPropagation();
      const i=+btn.dataset.i;
      const dir=btn.dataset.mv==='up'?-1:1;
      const j=i+dir;
      if(j<0||j>=state.boxes.length) return;
      const tmp=state.boxes[i];
      state.boxes[i]=state.boxes[j];
      state.boxes[j]=tmp;
      state.sel=j;
      renderList(); redraw(); syncPhraseHint();
    });
  });
  bindDragSort(list);
}

// ---- 框选列表拖拽排序
let _dragFrom=-1, _dragOver=-1;
function bindDragSort(list){
  const rows=[...list.querySelectorAll('.boxrow')];
  rows.forEach(row=>{
    row.addEventListener('dragstart',e=>{
      _dragFrom=+row.dataset.row;
      e.dataTransfer.effectAllowed='move';
      try{e.dataTransfer.setData('text/plain',String(_dragFrom));}catch(_){}
      row.style.opacity='0.35';
    });
    row.addEventListener('dragend',()=>{
      row.style.opacity='';
      list.querySelectorAll('.boxrow').forEach(r=>r.classList.remove('drop-line'));
      _dragFrom=-1; _dragOver=-1;
    });
    row.addEventListener('dragover',e=>{
      e.preventDefault();
      e.dataTransfer.dropEffect='move';
      const to=+row.dataset.row;
      if(_dragFrom<0||to===_dragFrom) return;
      list.querySelectorAll('.boxrow').forEach(r=>r.classList.remove('drop-line'));
      // 拖到目标行的上半 → 插到前面；下半 → 插到后面
      const rect=row.getBoundingClientRect();
      const after=(e.clientY-rect.top)>rect.height/2;
      row.classList.add('drop-line');
      _dragOver=after?to+1:to;
    });
    row.addEventListener('dragleave',e=>{
      if(!row.contains(e.relatedTarget)) row.classList.remove('drop-line');
    });
    row.addEventListener('drop',e=>{
      e.preventDefault();
      if(_dragFrom<0) return;
      let to=_dragOver;
      if(to<0) to=+row.dataset.row;
      // 取出被拖的元素
      const moved=state.boxes.splice(_dragFrom,1)[0];
      // 修正插入位置（移除后索引会变）
      let ins=to>_dragFrom?to-1:to;
      ins=Math.max(0,Math.min(state.boxes.length,ins));
      state.boxes.splice(ins,0,moved);
      state.sel=ins;
      renderList(); redraw(); syncPhraseHint();
      _dragFrom=-1; _dragOver=-1;
    });
  });
}

// 只更新选中高亮，不动输入框
function updateSelHighlight(){
  document.querySelectorAll('#list .boxrow').forEach(el=>{
    const i=+el.dataset.sel;
    el.style.background = (i===state.sel) ? '#f59e0b22' : '';
  });
}

// 只更新坐标文字（缩放/拖动时调用），避免重建输入框
function updateCoords(){
  state.boxes.forEach((b,i)=>{
    const el=document.querySelector('#list [data-coord="'+i+'"]');
    if(el) el.textContent=b.x+','+b.y+' '+b.w+'×'+b.h;
  });
}

function applyPhrase(){
  const inp=document.getElementById('phrase');
  const chars=Array.from((inp.value||'').replace(/\s+/g,''));
  state.boxes.forEach((b,i)=>{b.char=chars[i]||'';});
  // 只同步输入框的值，不重建 DOM（否则焦点会丢）
  document.querySelectorAll('#list input[data-i]').forEach(el=>{
    const i=+el.dataset.i;
    const v=state.boxes[i]?state.boxes[i].char:'';
    if(el.value!==v) el.value=v;
  });
  syncPhraseHint();
}
function syncPhraseHint(){
  const inp=document.getElementById('phrase');
  const chars=Array.from((inp.value||'').replace(/\s+/g,''));
  const n=state.boxes.length,el=document.getElementById('phHint');
  if(!chars.length){el.textContent=n?'('+n+' 个框)':'';el.className='ph-hint';return;}
  if(chars.length===n){el.textContent='✓ 正好 '+n+' 个字';el.className='ph-hint match-ok';}
  else if(chars.length<n){el.textContent=chars.length+'/'+n+'，末尾留空';
    el.className='ph-hint match-part';}
  else{el.textContent=chars.length+' > '+n+'，多余的丢弃';el.className='ph-hint match-none';}
}
document.getElementById('phrase').addEventListener('input',applyPhrase);
document.getElementById('phrase').addEventListener('keydown',e=>{
  if(e.key==='Enter'){e.preventDefault();saveAndNext(false);}
});

async function saveAndNext(skip){
  if(!state.img)return;
  const wasUncertain=state.uncertain;
  const body={img:state.img,boxes:skip?[]:state.boxes,
              uncertain:wasUncertain,note:state.note};
  const d=await api(skip?'/api/skip':'/api/save',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  state.done=d.saved;
  state.boxes=[];state.uncertain=false;state.note='';
  updateUncertainBtn();
  const ph=document.getElementById('phrase');ph.value='';syncPhraseHint();
  toast(skip?'已跳过':('已保存 '+body.boxes.length+' 个框'+
        (wasUncertain?'（存疑）':'')));
  await loadNext(state.img);
  ph.focus();
}

document.getElementById('btnPrev').onclick=async()=>{
  if(!state.img)return;
  // 先保存当前这张，再切上一张（否则改动会丢）
  const cur=state.img, curBoxes=state.boxes.slice();
  await api('/api/prev?cur='+encodeURIComponent(cur)).then(async d=>{
    if(!d.img||d.at_start){toast('已经是第一张');return;}
    // 保存当前（静默，不弹提示）
    if(curBoxes.length){
      await api('/api/save',{method:'POST',body:JSON.stringify({
        img:cur, boxes:curBoxes, uncertain:state.uncertain, note:state.note})});
    }
    await loadImage(d.img,d.index);
    toast('已保存并返回上一张');
  });
};
document.getElementById('btnLogout').onclick=async()=>{
  location.href='/api/logout';
};
document.getElementById('btnDone').onclick=()=>saveAndNext(false);
document.getElementById('btnSkip').onclick=()=>saveAndNext(true);
document.getElementById('btnUndo').onclick=()=>{
  if(state.boxes.length){state.boxes.pop();renderList();redraw();syncPhraseHint();}
};
document.getElementById('btnUncertain').onclick=()=>{
  state.uncertain=!state.uncertain;updateUncertainBtn();
  toast(state.uncertain?'已标记存疑':'已取消存疑');
};
document.addEventListener('keydown',e=>{
  const inPh=e.target&&e.target.id==='phrase';
  if(e.key==='Enter'&&!inPh&&!e.target.matches('input')){
    e.preventDefault();saveAndNext(false);}
  if(e.key==='Escape'&&state.dragging){state.dragging=null;redraw();}
  if((e.ctrlKey||e.metaKey)&&e.key==='z'){e.preventDefault();
    if(state.boxes.length){state.boxes.pop();renderList();redraw();syncPhraseHint();}}
});
(async()=>{
  const s=await api('/api/state');
  if(s.next)await loadNext('');
  else document.getElementById('prog').textContent='全部完成！共 '+s.total+' 张';
  await loadDatasets();
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------- 路由
LOGIN_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>登录 · 验证码框选标注</title>
<style>
*{box-sizing:border-box}
body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
     font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
     background:#0f1115;color:#e6e6e6}
.box{width:340px;background:#171a21;border:1px solid #262b36;border-radius:12px;
     padding:28px 24px}
h1{font-size:17px;margin:0 0 20px;font-weight:600;text-align:center}
label{display:block;font-size:12px;color:#8b95a7;margin:14px 0 6px}
input{width:100%;background:#0f1115;border:1px solid #333b4d;color:#fff;
      border-radius:8px;padding:11px 12px;font-size:14px;outline:none}
input:focus{border-color:#3b82f6;box-shadow:0 0 0 3px #3b82f633}
button{width:100%;margin-top:22px;background:#2563eb;border:none;color:#fff;
       border-radius:8px;padding:12px;font-size:14px;cursor:pointer;font-weight:600}
button:hover{background:#1d4ed8}
button:disabled{opacity:.6;cursor:default}
#err{margin-top:14px;font-size:13px;color:#f87171;text-align:center;min-height:18px}
</style>
</head>
<body>
<form class="box" id="f">
  <h1>验证码框选标注</h1>
  <label>账号</label>
  <input id="u" autocomplete="username" required>
  <label>密码</label>
  <input id="p" type="password" autocomplete="current-password" required>
  <button id="btn" type="submit">登录</button>
  <div id="err"></div>
</form>
<script>
const f=document.getElementById('f');
f.addEventListener('submit',async e=>{
  e.preventDefault();
  const btn=document.getElementById('btn'),err=document.getElementById('err');
  btn.disabled=true;err.textContent='';
  try{
    const r=await fetch('/api/login',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({user:document.getElementById('u').value,
                           pwd:document.getElementById('p').value})});
    const d=await r.json();
    if(d.ok){location.href='/';}
    else{err.textContent=d.error||'登录失败';btn.disabled=false;}
  }catch(ex){err.textContent='网络错误';btn.disabled=false;}
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "LabelTool/1.0"

    def log_message(self, fmt, *args):
        # 不记录 query（含 token）
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # ---- 鉴权
    def _session_id(self) -> str | None:
        """从 Cookie 里取会话 ID。"""
        raw = self.headers.get("Cookie") or ""
        for part in raw.split(";"):
            part = part.strip()
            if part.startswith("sid="):
                sid = part[4:]
                if sid in SESSIONS and SESSIONS[sid]["expire"] > time.time():
                    return sid
        return None

    def _authed(self) -> bool:
        """登录态校验。

        优先用账号密码登录后的会话 Cookie；
        也兼容 URL token（给脚本/自动化用）。
        """
        if self._session_id():
            return True
        tok = STATE.get("token")
        if tok:
            q = parse_qs(urlparse(self.path).query)
            if q.get("k", [""])[0] == tok:
                return True
            return False
        # 既没配 token 也没登录要求时，才放行
        return not STATE.get("require_login", True)

    def _set_cookie(self, sid: str) -> dict:
        return {"Set-Cookie":
                f"sid={sid}; Path=/; HttpOnly; SameSite=Strict; Max-Age=86400"}

    def _send(self, code: int, body: bytes, ctype: str,
              extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if extra:
            for k, v in extra.items():
                self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _err(self, code: int, msg: str):
        self._json({"error": msg}, code)

    # ---- GET
    def do_GET(self):
        u = urlparse(self.path)
        path = u.path

        if path in ("/", "/index.html"):
            # 未登录先给登录页
            if STATE.get("require_login", True):
                if self._session_id():
                    self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
                else:
                    self._send(200, LOGIN_HTML.encode("utf-8"),
                               "text/html; charset=utf-8")
            else:
                self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return

        if path == "/api/login":
            # 登录用 POST，GET 只做跳转
            self._send(302, b"", "text/plain", {"Location": "/"})
            return

        if path == "/api/logout":
            sid = self._session_id()
            if sid:
                SESSIONS.pop(sid, None)
            self._send(302, b"", "text/plain",
                       {"Location": "/",
                        "Set-Cookie": "sid=; Path=/; Max-Age=0"})
            return

        if not self._authed():
            self._err(403, "forbidden")
            return

        if path == "/api/state":
            imgs = list_images()
            done = load_confirmed()
            remain = [i for i in imgs if i not in done]
            self._json({"total": len(imgs), "done": len(done),
                        "remaining": len(remain),
                        "next": remain[0] if remain else None})
            return

        if path == "/api/datasets":
            out = []
            for k, cfg in DATASETS.items():
                d = HERE / cfg["dir"]
                if not d.exists():
                    continue
                imgs = sorted(p.name for p in d.glob(cfg["glob"]))
                done = load_confirmed(k)
                out.append({"key": k, "label": cfg["label"],
                            "total": len(imgs), "done": len(done),
                            "current": k == STATE["key"]})
            self._json({"datasets": out, "current": STATE["key"]})
            return

        if path == "/api/prev":
            # 上一张（按文件名顺序回退一格）。已登录用户可用。
            q = parse_qs(u.query)
            cur = q.get("cur", [""])[0]
            imgs = list_images()
            if cur in imgs:
                i = imgs.index(cur)
                if i == 0:
                    self._json({"img": imgs[0], "index": 0,
                                "total": len(imgs), "at_start": True})
                    return
                prev = imgs[i - 1]
                self._json({"img": prev, "index": i - 1, "total": len(imgs)})
                return
            self._json({"img": None, "error": "current not found"})
            return

        if path == "/api/next":
            q = parse_qs(u.query)
            after = q.get("after", [""])[0]
            imgs = list_images()
            done = load_confirmed()
            remain = [n for n in imgs if n not in done]
            if not remain:
                self._json({"img": None, "total": len(imgs),
                            "done": len(done), "all_done": True})
                return
            seq = imgs
            if after in imgs and after not in done:
                i = imgs.index(after) + 1
                seq = imgs[i:] + imgs[:i]
            pick = next((n for n in seq if n not in done), remain[0])
            self._json({"img": pick, "index": imgs.index(pick),
                        "total": len(imgs), "done": len(done)})
            return

        if path.startswith("/api/ques/"):
            # 取某张底图对应的 ques 小图列表（word / icon 型）
            name = path[len("/api/ques/"):]
            if not SAFE_NAME.match(name):
                self._err(400, "bad name")
                return
            cfg = ds_cfg()
            d = HERE / cfg["dir"]
            stem = name.replace("_bg.jpg", "")
            qs = sorted(p.name for p in d.glob(f"{stem}_ques_*.png"))
            self._json({"bg": name, "ques": qs, "count": len(qs)})
            return

        if path.startswith("/api/quesimg/"):
            # ques 小图本体
            name = path[len("/api/quesimg/"):]
            if not SAFE_NAME_Q.match(name):
                self._err(400, "bad name")
                return
            cfg = ds_cfg()
            fp = HERE / cfg["dir"] / name
            if not fp.exists() or not fp.is_file():
                self._err(404, "not found")
                return
            try:
                body = fp.read_bytes()
            except Exception:
                self._err(500, "read error")
                return
            self._send(200, body, "image/png", {"Cache-Control": "no-store"})
            return

        if path.startswith("/api/existing/"):
            name = path[len("/api/existing/"):]
            if not SAFE_NAME.match(name):
                self._err(400, "bad name")
                return
            # 登录后可读取已有标注（支持「上一张」回看并修改）
            done = load_done()
            self._json(done.get(name))
            return

        if path.startswith("/api/image/"):
            name = path[len("/api/image/"):]
            if not SAFE_NAME.match(name):
                self._err(400, "bad name")
                return
            # 登录后可查看任意图（含已标注，用于「上一张」回看）
            cfg = ds_cfg()
            fp = HERE / cfg["dir"] / name
            if not fp.exists() or not fp.is_file():
                self._err(404, "not found")
                return
            try:
                body = fp.read_bytes()
            except Exception:
                self._err(500, "read error")
                return
            self._send(200, body, "image/jpeg",
                       {"Cache-Control": "no-store"})
            return

        self._err(404, "not found")

    # ---- POST
    def do_POST(self):
        u = urlparse(self.path)

        # 登录接口不需要已有会话
        if u.path == "/api/login":
            n = int(self.headers.get("Content-Length") or 0)
            if n > 64 * 1024:
                self._err(413, "too large")
                return
            raw = self.rfile.read(n) if n else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
            except Exception:
                self._err(400, "bad json")
                return
            user = str(body.get("user") or "").strip()
            pwd = str(body.get("pwd") or "")
            if not _check_login(user, pwd):
                # 简单延迟，降低暴力破解速度
                time.sleep(0.6)
                self._json({"ok": False, "error": "账号或密码错误"}, 401)
                return
            sid = _new_session(user)
            self._send(200,
                       json.dumps({"ok": True, "user": user}).encode("utf-8"),
                       "application/json; charset=utf-8",
                       self._set_cookie(sid))
            return

        if not self._authed():
            self._err(403, "forbidden")
            return
        n = int(self.headers.get("Content-Length") or 0)
        if n > 2 * 1024 * 1024:
            self._err(413, "too large")
            return
        raw = self.rfile.read(n) if n else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            self._err(400, "bad json")
            return

        if u.path == "/api/switch":
            key = body.get("key", "")
            if key not in DATASETS:
                self._err(400, "unknown dataset")
                return
            if not (HERE / DATASETS[key]["dir"]).exists():
                self._err(404, "dir missing")
                return
            STATE["key"] = key
            self._json({"ok": True, "key": key,
                        "label": DATASETS[key]["label"]})
            return

        if u.path == "/api/save":
            img = body.get("img")
            boxes = body.get("boxes") or []
            if not img or not SAFE_NAME.match(img):
                self._err(400, "bad img")
                return
            clean = []
            for b in boxes:
                try:
                    x, y, w, h = int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
                except Exception:
                    continue
                if w < 3 or h < 3:
                    continue
                raw_ch = (b.get("char") or "").strip()
                # 保留 src（ocr=预填 / manual=已核对），否则无法统计真实进度
                src = b.get("src") or "manual"
                if src not in ("ocr", "manual", "corrected"):
                    src = "manual"
                clean.append({"x": x, "y": y, "w": w, "h": h,
                              "char": to_simplified(raw_ch) if raw_ch else "",
                              "char_raw": raw_ch,
                              "src": src})
            rec = {"img": img, "boxes": clean,
                   "order": body.get("order") or list(range(len(clean))),
                   "uncertain": bool(body.get("uncertain")),
                   "note": (body.get("note") or "").strip()[:200],
                   "reviewed": True,
                   "ts": int(time.time())}
            done = load_done()
            done[img] = rec
            save_all(done)
            self._json({"ok": True, "saved": len(load_confirmed()),
                        "boxes": len(clean)})
            return

        if u.path == "/api/skip":
            img = body.get("img")
            if not img or not SAFE_NAME.match(img):
                self._err(400, "bad img")
                return
            done = load_done()
            done[img] = {"img": img, "boxes": [], "skip": True,
                         "ts": int(time.time())}
            save_all(done)
            self._json({"ok": True, "saved": len(done)})
            return

        self._err(404, "not found")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--token", default="",
                    help="URL 令牌（给脚本用，可留空）")
    ap.add_argument("--dataset", default="live4")
    ap.add_argument("--users", default="",
                    help="账号密码，格式 user:pass,user2:pass2")
    ap.add_argument("--no-login", action="store_true",
                    help="关闭登录校验（仅本地调试）")
    args = ap.parse_args()

    if args.dataset in DATASETS:
        STATE["key"] = args.dataset
    STATE["token"] = args.token
    STATE["require_login"] = not args.no_login

    if args.users:
        USERS.clear()
        for pair in args.users.split(","):
            if ":" in pair:
                u, p = pair.split(":", 1)
                USERS[u.strip()] = p.strip()
    if USERS == {DEFAULT_USER: DEFAULT_PASS} and not args.no_login:
        print(f"⚠ 正在使用默认账号 {DEFAULT_USER}，正式部署请用 --users 改成自己的")

    print(f"数据集: {STATE['key']}  图片: {len(list_images())} 张")
    print(f"已标: {len(load_done())} 张")
    print(f"监听: http://{args.host}:{args.port}/")
    print(f"登录: {'开启' if STATE['require_login'] else '关闭'}"
          + (f"，账号 {list(USERS)}" if USERS else ""))
    if args.token:
        print("URL 令牌已启用（脚本可用 ?k=xxx）")

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())