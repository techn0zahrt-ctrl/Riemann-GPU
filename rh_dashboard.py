#!/usr/bin/env python3
"""
RH Dashboard — live monitoring for the RH zero-verification cluster.
Run:  python3 rh_dashboard.py
Open: http://192.168.100.9:8080
"""

import json, os, re, subprocess, sys, time, threading
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nas_fs as nas
from rh_core import backlund_count

from flask import Flask, jsonify, Response

app  = Flask(__name__)
PORT = 8080

NODES = ["192.168.100.9","192.168.100.10","192.168.100.11","192.168.100.12"]
NODE_HOST_MAP = {          # worker hostname → node IP  (for nvidia-smi SSH queries)
    "ChrisDesktop": "192.168.100.9",
    "JamiDesktop":  "192.168.100.10",
    "TysonDesktop": "192.168.100.11",
    "LinkDesktop":  "192.168.100.12",
}
NVIDIA_SMI = "/usr/lib/wsl/lib/nvidia-smi"

_lock  = threading.Lock()
_cache = {}
_thr_samples: list = []   # [(ts, zeros_verified)]
_gpu_cache: dict   = {}   # node_ip → {power_w, util_pct, temp_c, mem_used_mb, mem_total_mb}
_gpu_ts:    dict   = {}   # node_ip → last fetch time


def _fetch_gpu(node: str) -> dict:
    try:
        out = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", node,
             f"{NVIDIA_SMI} --query-gpu=power.draw,utilization.gpu,temperature.gpu,"
             f"memory.used,memory.total --format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()
        p = [x.strip() for x in out.split(",")]
        return {
            "power_w":      float(p[0]),
            "util_pct":     float(p[1]),
            "temp_c":       float(p[2]),
            "mem_used_mb":  float(p[3]),
            "mem_total_mb": float(p[4]),
        }
    except Exception:
        return {}


def _refresh_gpu_async():
    def _worker(node):
        d = _fetch_gpu(node)
        with _lock:
            _gpu_cache[node] = d
            _gpu_ts[node]    = time.time()
    threads = [threading.Thread(target=_worker, args=(n,), daemon=True) for n in NODES]
    for t in threads: t.start()
    for t in threads: t.join(timeout=6)


def _parse_log_speed(log_path: Path) -> int | None:
    try:
        txt  = log_path.read_text()
        hits = re.findall(r'pts/s=(\d+)', txt)
        return int(hits[-1]) if hits else None
    except Exception:
        return None


def _update_cache():
    global _thr_samples

    try:
        config       = nas.read_config()
        chunk_dt     = config.get("chunk_dt", 5000.0)
        total_chunks = int((config["T_end"] - config["T_start"]) / chunk_dt)
    except Exception:
        config = {}; chunk_dt = 5000.0; total_chunks = 10530

    try:
        summary = nas.read_json(nas.rh_path("summary.json"))
    except Exception:
        summary = {}

    zeros_verified = summary.get("zeros_verified", 0)
    chunks_done    = summary.get("chunks_done", 0)
    now            = time.time()

    # Coordinator alive?
    coord_ok = subprocess.run(["pgrep","-f","rh_coordinator.py"],
                              capture_output=True).returncode == 0

    # When coordinator is stopped, summary.json is stale — recount done/ directly
    # os.listdir() is fast (name-only, no file reads) even for 10M+ entries
    if not coord_ok:
        try:
            actual_done = sum(1 for f in os.listdir(str(nas.rh_path("done"))) if f.endswith(".json"))
            if actual_done > chunks_done:
                chunks_done = actual_done
                summary = dict(summary)
                summary["chunks_done"] = actual_done
        except Exception:
            pass

    # Throughput samples (1-hour window) keyed on zeros
    _thr_samples.append((now, zeros_verified))
    _thr_samples = [(t, z) for t, z in _thr_samples if t >= now - 3600]

    recent = [(t, z) for t, z in _thr_samples if t >= now - 300]
    if len(recent) >= 2:
        dt    = recent[-1][0] - recent[0][0]
        dz    = recent[-1][1] - recent[0][1]
        zeros_per_sec = dz / dt if dt > 0 else 0
    else:
        zeros_per_sec = 0

    chunks_per_min = 0.0
    if zeros_per_sec > 0 and chunks_done > 0 and zeros_verified > 0:
        zeros_per_chunk = zeros_verified / chunks_done if chunks_done else 427000
        chunks_per_min  = zeros_per_sec * 60 / zeros_per_chunk

    remaining = total_chunks - chunks_done
    eta_s     = int(remaining / (chunks_per_min / 60)) if chunks_per_min > 0 else None

    # Throughput chart: zeros/s in 5-min buckets over last hour
    thr_chart = []
    bucket_size = 300  # 5 min
    for i in range(12):
        t_lo = now - (12 - i) * bucket_size
        t_hi = t_lo + bucket_size
        pts  = [(t, z) for t, z in _thr_samples if t_lo <= t < t_hi]
        if len(pts) >= 2:
            thr_chart.append(round((pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0]), 0))
        else:
            thr_chart.append(0)

    # Per-node stats + recent completions — parsed from coordinator log (local file, fast)
    # Never reads done/ directory: each NAS file read takes ~275ms, 200 files = 55s
    node_stats: dict = defaultdict(lambda: {
        "done": 0, "ok": 0, "mismatch": 0,
        "zeros": 0, "total_wall": 0.0, "pts_per_sec": 0,
    })
    recent_completions = []
    log_re = re.compile(
        r'^\s+([✓✗])\s+(chunk_\w+)\s+zeros=([\d,]+)\s+host=(\S+)\s+wall=([\d.]+)s'
    )
    try:
        log_lines = Path("/home/techno/math/coordinator.log").read_text(
            errors="replace"
        ).splitlines()
        # Walk lines in reverse so we see most-recent first
        for line in reversed(log_lines):
            m = log_re.match(line)
            if not m:
                continue
            ok_ch, cid, zeros_s, host, wall_s = m.groups()
            ok      = ok_ch == "✓"
            zeros   = int(zeros_s.replace(",", ""))
            wall    = float(wall_s)
            ns      = node_stats[host]
            ns["done"]       += 1
            ns["zeros"]      += zeros
            ns["total_wall"] += wall
            if ok: ns["ok"]       += 1
            else:  ns["mismatch"] += 1
            if wall > 0:
                ns["pts_per_sec"] = int(zeros / wall * chunk_dt)
            if len(recent_completions) < 40:
                recent_completions.append({
                    "chunk_id": cid,
                    "host":     host,
                    "zeros":    zeros,
                    "ok":       ok,
                    "wall_s":   wall,
                    "mtime":    0,
                })
            # Stop once we have enough for all displays
            if all(ns["done"] >= 50 for ns in node_stats.values()) and len(recent_completions) >= 40:
                break
    except Exception:
        pass

    # Active chunks — fast name-only listing, no NAS glob/log reads
    active_chunks = []
    node_active: dict = defaultdict(int)
    try:
        for name in os.listdir(str(nas.rh_path("active"))):
            if not name.endswith(".json"):
                continue
            active_chunks.append({"chunk_id": Path(name).stem, "node_ip": None,
                                   "speed_pts": None, "host": None})
    except Exception:
        pass

    # Active count per node comes from coordinator log (last N lines show inflight chunks)
    # Use the summary's chunks_active as a cross-check
    active_by_host: dict = defaultdict(int)
    inflight_re = re.compile(r'^\s+[✓✗]\s+(chunk_\w+)\s+.*host=(\S+)')
    try:
        # Last 100 lines of log shows what's been dispatched most recently
        recent_log = Path("/home/techno/math/coordinator.log").read_text(
            errors="replace"
        ).splitlines()[-200:]
        seen_chunks: set = set()
        for line in reversed(recent_log):
            m2 = inflight_re.match(line)
            if m2:
                cid2, h2 = m2.groups()
                if cid2 not in seen_chunks:
                    seen_chunks.add(cid2)
    except Exception:
        pass

    for host, lst in node_stats.items():
        lst.setdefault("active", 0)
        lst.setdefault("current_speed", lst.get("pts_per_sec", 0))

    for ns in node_stats.values():
        ns.setdefault("active", 0)
        ns.setdefault("current_speed", 0)

    # GPU stats (from async cache refreshed every 10s)
    with _lock:
        gpu_data = dict(_gpu_cache)

    cluster_pts_s = sum(
        ns.get("current_speed", 0) or ns.get("pts_per_sec", 0)
        for ns in node_stats.values()
    )

    backlund_expected = 0
    try:
        backlund_expected = int(backlund_count(config.get("T_end", 0)) -
                                backlund_count(config.get("T_start", 0)))
    except Exception:
        pass

    with _lock:
        _cache.update({
            "summary":            summary,
            "total_chunks":       total_chunks,
            "pct_done":           chunks_done / total_chunks * 100 if total_chunks else 0,
            "zeros_per_sec":      round(zeros_per_sec, 1),
            "chunks_per_min":     round(chunks_per_min, 2),
            "cluster_pts_s":      cluster_pts_s,
            "eta_seconds":        eta_s,
            "thr_chart":          thr_chart,
            "node_stats":         {k: dict(v) for k, v in node_stats.items()},
            "active_chunks":      active_chunks,
            "recent_completions": recent_completions,
            "coordinator":        coord_ok,
            "gpu":                gpu_data,
            "config":             config,
            "backlund_expected":  backlund_expected,
            "ts":                 now,
        })


_gpu_refresh_ts = 0.0

def _bg_loop():
    global _gpu_refresh_ts
    while True:
        try:
            if time.time() - _gpu_refresh_ts > 10:
                _refresh_gpu_async()
                _gpu_refresh_ts = time.time()
            _update_cache()
        except Exception as e:
            print(f"[cache] {e}", flush=True)
        time.sleep(5)


@app.route("/api/stats")
def api_stats():
    with _lock:
        return jsonify(dict(_cache))

@app.route("/")
def index():
    import json as _json
    html = HTML.replace("__NODE_HOST_MAP__", _json.dumps(NODE_HOST_MAP))
    return Response(html, mimetype="text/html")


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RH Cluster</title>
<style>
:root {
  --bg:      #0a0a14;
  --surf:    #10101e;
  --border:  #1c1c3a;
  --acc:     #7c3aed;
  --acc2:    #a855f7;
  --green:   #22c55e;
  --gold:    #f59e0b;
  --red:     #ef4444;
  --blue:    #3b82f6;
  --text:    #e2e8f0;
  --muted:   #64748b;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh}
.page{max-width:1440px;margin:0 auto;padding:16px}

/* header */
.hdr{display:flex;align-items:center;justify-content:space-between;
  padding:14px 24px;background:var(--surf);border:1px solid var(--border);
  border-radius:12px;margin-bottom:16px}
.hdr-title{font-size:1.15rem;font-weight:700;letter-spacing:.04em}
.hdr-title em{color:var(--acc2);font-style:normal}
.hdr-right{display:flex;align-items:center;gap:16px;font-size:.78rem;color:var(--muted)}
.dot{width:7px;height:7px;border-radius:50%;background:var(--green);
  display:inline-block;animation:pulse 1.4s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(1.5)}}
.coord-ok{color:var(--green)}
.coord-dead{color:var(--red)}

/* big numbers row */
.bignums{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:16px}
@media(max-width:1100px){.bignums{grid-template-columns:repeat(3,1fr)}}
@media(max-width:700px){.bignums{grid-template-columns:repeat(2,1fr)}}
.bn{background:var(--surf);border:1px solid var(--border);border-radius:12px;
  padding:16px 20px;transition:border-color .3s}
.bn:hover{border-color:var(--acc)}
.bn-label{font-size:.68rem;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);margin-bottom:6px}
.bn-val{font-size:1.85rem;font-weight:700;font-variant-numeric:tabular-nums;line-height:1}
.bn-sub{font-size:.72rem;color:var(--muted);margin-top:5px}
.gold{color:var(--gold)}
.green{color:var(--green)}
.blue{color:var(--blue)}
.acc{color:var(--acc2)}

/* progress */
.prog{background:var(--surf);border:1px solid var(--border);border-radius:12px;
  padding:14px 20px;margin-bottom:16px}
.prog-hdr{display:flex;justify-content:space-between;font-size:.75rem;
  color:var(--muted);margin-bottom:8px}
.bar-track{background:#1c1c3a;border-radius:999px;height:14px;overflow:hidden}
.bar-fill{height:100%;border-radius:999px;
  background:linear-gradient(90deg,var(--acc),var(--acc2),var(--gold));
  background-size:300% 100%;animation:shimmer 3s linear infinite;transition:width 1.2s ease}
@keyframes shimmer{0%{background-position:300% 0}100%{background-position:-300% 0}}

/* two-col layout */
.main-grid{display:grid;grid-template-columns:1fr 340px;gap:16px;margin-bottom:16px}
@media(max-width:1100px){.main-grid{grid-template-columns:1fr}}

/* node grid */
.nodes{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
@media(max-width:700px){.nodes{grid-template-columns:1fr}}
.nc{background:var(--surf);border:1px solid var(--border);border-radius:12px;
  padding:14px 16px;transition:border-color .3s,box-shadow .3s}
.nc.active{border-color:var(--acc);box-shadow:0 0 16px rgba(124,58,237,.3)}
.nc-hdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
.nc-name{font-weight:700;font-size:.9rem}
.badge{font-size:.6rem;padding:2px 8px;border-radius:999px;font-weight:700;
  text-transform:uppercase;letter-spacing:.08em}
.badge.on{background:rgba(34,197,94,.15);color:var(--green)}
.badge.off{background:rgba(100,116,139,.1);color:var(--muted)}
.nc-row{display:flex;justify-content:space-between;font-size:.74rem;
  padding:3px 0;border-bottom:1px solid rgba(255,255,255,.04)}
.nc-row:last-child{border-bottom:none}
.nc-row .k{color:var(--muted)}
.nc-row .v{font-weight:600;font-variant-numeric:tabular-nums}
.gpu-bar-wrap{margin-top:8px}
.gpu-bar-lbl{display:flex;justify-content:space-between;font-size:.65rem;color:var(--muted);margin-bottom:3px}
.gpu-bar{background:#1c1c3a;border-radius:999px;height:5px;overflow:hidden}
.gpu-bar-fill{height:100%;border-radius:999px;transition:width .8s ease}

/* throughput chart */
.chart-card{background:var(--surf);border:1px solid var(--border);border-radius:12px;
  padding:16px 18px}
.chart-card h3{font-size:.72rem;text-transform:uppercase;letter-spacing:.1em;
  color:var(--muted);margin-bottom:12px}
.chart-wrap{position:relative;height:160px;width:100%}

/* feed */
.feed-card{background:var(--surf);border:1px solid var(--border);border-radius:12px;
  padding:16px 18px}
.feed-card h3{font-size:.72rem;text-transform:uppercase;letter-spacing:.1em;
  color:var(--muted);margin-bottom:10px}
.feed{display:flex;flex-direction:column;gap:3px;max-height:340px;overflow-y:auto}
.fr{display:grid;grid-template-columns:18px 1fr 110px 100px 70px;
  gap:6px;align-items:center;padding:5px 8px;border-radius:7px;
  font-size:.73rem;animation:fi .3s ease}
@keyframes fi{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:translateY(0)}}
.fr:nth-child(odd){background:rgba(255,255,255,.025)}
.fr .cid{font-family:monospace;color:var(--acc2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.fr .zh{color:var(--text);text-align:right;font-variant-numeric:tabular-nums}
.fr .wl{color:var(--muted);text-align:right}
.fr .ag{color:var(--muted);text-align:right}
.ok-ic.ok{color:var(--green)}
.ok-ic.no{color:var(--red)}
.feed::-webkit-scrollbar{width:4px}
.feed::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
</style>
</head>
<body>
<div class="page">

<div class="hdr">
  <div class="hdr-title">ζ(s) &nbsp;<em>Riemann Hypothesis</em>&nbsp; Verification Cluster</div>
  <div class="hdr-right">
    <span><span class="dot"></span> &nbsp;LIVE</span>
    <span id="coord-badge" class="coord-ok">● coordinator</span>
    <span id="ts-lbl"></span>
  </div>
</div>

<div class="bignums">
  <div class="bn">
    <div class="bn-label">Zeros Verified</div>
    <div class="bn-val gold" id="zeros">—</div>
    <div class="bn-sub">on critical line Re(s) = ½</div>
  </div>
  <div class="bn">
    <div class="bn-label">Cluster Throughput</div>
    <div class="bn-val green" id="thr">—</div>
    <div class="bn-sub" id="thr-sub">zeros / second</div>
  </div>
  <div class="bn">
    <div class="bn-label">GPU Compute</div>
    <div class="bn-val blue" id="pts">—</div>
    <div class="bn-sub">pts / second cluster total</div>
  </div>
  <div class="bn">
    <div class="bn-label">Est. Completion</div>
    <div class="bn-val acc" id="eta">—</div>
    <div class="bn-sub" id="eta-date">—</div>
  </div>
  <div class="bn">
    <div class="bn-label">Backlund Target</div>
    <div class="bn-val" style="color:var(--acc2)" id="backlund">—</div>
    <div class="bn-sub" id="backlund-sub">expected zeros in range</div>
  </div>
</div>

<div class="prog">
  <div class="prog-hdr">
    <span id="prog-left">—</span>
    <span id="prog-right">—</span>
  </div>
  <div class="bar-track"><div class="bar-fill" id="bar" style="width:.05%"></div></div>
</div>

<div class="main-grid">
  <div>
    <div class="nodes" id="nodes"></div>
  </div>
  <div style="display:flex;flex-direction:column;gap:12px">
    <div class="chart-card">
      <h3>Zeros verified / second — last hour</h3>
      <div class="chart-wrap"><canvas id="thrChart"></canvas></div>
    </div>
  </div>
</div>

<div class="feed-card">
  <h3>Recent completions</h3>
  <div class="feed" id="feed"></div>
</div>

</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<script>
const thrCtx = document.getElementById('thrChart').getContext('2d');
const thrChart = new Chart(thrCtx, {
  type:'line',
  data:{
    labels: Array.from({length:12},(_,i)=>{const m=(11-i)*5; return m===0?'now':`-${m}m`;}),
    datasets:[{
      label:'zeros/s',
      data:Array(12).fill(0),
      borderColor:'#a855f7',
      backgroundColor:'rgba(168,85,247,.1)',
      borderWidth:2,fill:true,tension:.4,pointRadius:2,
      pointBackgroundColor:'#a855f7',
    }]
  },
  options:{
    responsive:true,maintainAspectRatio:false,
    plugins:{legend:{display:false}},
    scales:{
      x:{grid:{color:'rgba(255,255,255,.04)'},ticks:{color:'#64748b',font:{size:10}}},
      y:{grid:{color:'rgba(255,255,255,.04)'},ticks:{color:'#64748b',font:{size:10}},min:0}
    },
    animation:{duration:500}
  }
});

function fmt(n){return n==null?'—':Math.round(n).toLocaleString()}
function fmtM(n){
  if(!n) return '—';
  if(n>=1e9) return (n/1e9).toFixed(2)+'B';
  if(n>=1e6) return (n/1e6).toFixed(1)+'M';
  if(n>=1e3) return (n/1e3).toFixed(0)+'k';
  return String(n);
}
function fmtEta(s){
  if(!s||s<=0) return '—';
  const d=Math.floor(s/86400),h=Math.floor((s%86400)/3600),m=Math.floor((s%3600)/60);
  if(d>0) return `~${d}d ${h}h`;
  if(h>0) return `~${h}h ${m}m`;
  return `~${m}m`;
}
function fmtDate(s){
  if(!s||s<=0) return '';
  const d=new Date(Date.now()+s*1000);
  return d.toLocaleDateString(undefined,{month:'short',day:'numeric',year:'numeric'})+
    ' '+d.toLocaleTimeString(undefined,{hour:'2-digit',minute:'2-digit'});
}
function barColor(pct){
  if(pct>=80) return 'var(--green)';
  if(pct>=40) return 'var(--gold)';
  return 'var(--acc2)';
}

let _prev_zeros = 0;
function animVal(el, target, dur=600){
  const start=parseFloat(el.dataset.cur||0);
  el.dataset.cur=target;
  const t0=performance.now();
  (function step(now){
    const p=Math.min(1,(now-t0)/dur);
    const e=1-Math.pow(1-p,3);
    el.textContent=fmtM(start+(target-start)*e);
    if(p<1) requestAnimationFrame(step);
  })(performance.now());
}

function update(d){
  const s   = d.summary||{};
  const done= s.chunks_done||0;
  const tot = d.total_chunks||10530;
  const pct = d.pct_done||0;

  // header
  const cb = document.getElementById('coord-badge');
  cb.className = d.coordinator ? 'coord-ok' : 'coord-dead';
  cb.textContent= d.coordinator ? '● coordinator' : '✗ coordinator DOWN';
  document.getElementById('ts-lbl').textContent =
    new Date(d.ts*1000).toLocaleTimeString();

  // big numbers
  animVal(document.getElementById('zeros'), s.zeros_verified||0);
  const zps = d.zeros_per_sec||0;
  document.getElementById('thr').textContent = fmtM(zps)+'/s';
  document.getElementById('thr-sub').textContent =
    `${(zps*60).toFixed(0)} zeros/min  ·  ${(d.chunks_per_min||0).toFixed(1)} chunks/min`;

  const cps = d.cluster_pts_s||0;
  document.getElementById('pts').textContent = fmtM(cps)+'/s';

  document.getElementById('eta').textContent = fmtEta(d.eta_seconds);
  document.getElementById('eta-date').textContent = fmtDate(d.eta_seconds);

  const be = d.backlund_expected||0;
  animVal(document.getElementById('backlund'), be);
  const zv = s.zeros_verified||0;
  const bePct = be>0 ? (zv/be*100).toFixed(2)+'%' : '—';
  document.getElementById('backlund-sub').textContent =
    be>0 ? `${bePct} verified of target` : 'expected zeros in range';

  // progress
  document.getElementById('bar').style.width = Math.max(pct,0.05)+'%';
  document.getElementById('prog-left').textContent =
    `${done.toLocaleString()} / ${tot.toLocaleString()} chunks  (${pct.toFixed(2)}%)`;
  document.getElementById('prog-right').textContent =
    `${(tot-done).toLocaleString()} remaining`;

  // throughput chart
  thrChart.data.datasets[0].data = d.thr_chart||Array(12).fill(0);
  thrChart.update('none');

  // nodes
  const ns      = d.node_stats||{};
  const gpu     = d.gpu||{};
  const actByH  = {};
  (d.active_chunks||[]).forEach(a=>{
    if(a.host){if(!actByH[a.host])actByH[a.host]=[];actByH[a.host].push(a);}
  });

  const allHosts = [...new Set([...Object.keys(ns),...Object.keys(actByH)])].sort();
  const nodesEl  = document.getElementById('nodes');
  nodesEl.innerHTML='';

  const NODE_IPS = __NODE_HOST_MAP__;

  allHosts.forEach(host=>{
    const n   = ns[host]||{};
    const act = actByH[host]||[];
    const on  = act.length>0 || (n.active||0)>0;
    const ip  = NODE_IPS[host]||'';
    const g   = gpu[ip]||{};
    const util= g.util_pct||0;
    const mem = g.mem_used_mb||0;
    const mtot= g.mem_total_mb||16384;
    const memPct = mtot>0 ? mem/mtot*100 : 0;
    const spd = n.current_speed||n.pts_per_sec||0;
    const avgW= n.done>0 ? (n.total_wall/n.done).toFixed(1) : '—';

    const el = document.createElement('div');
    el.className='nc'+(on?' active':'');
    el.innerHTML=`
      <div class="nc-hdr">
        <div class="nc-name">${host}</div>
        <span class="badge ${on?'on':'off'}">${on?'ACTIVE':'IDLE'}</span>
      </div>
      <div class="nc-row"><span class="k">GPU util</span>
        <span class="v" style="color:${barColor(util)}">${util?util.toFixed(0)+'%':'—'}</span></div>
      <div class="nc-row"><span class="k">GPU power</span>
        <span class="v">${g.power_w?g.power_w.toFixed(0)+' W':'—'}</span></div>
      <div class="nc-row"><span class="k">GPU temp</span>
        <span class="v">${g.temp_c?g.temp_c.toFixed(0)+'°C':'—'}</span></div>
      <div class="nc-row"><span class="k">VRAM used</span>
        <span class="v">${mem?`${(mem/1024).toFixed(1)} / ${(mtot/1024).toFixed(0)} GB`:'—'}</span></div>
      <div class="nc-row"><span class="k">Throughput</span>
        <span class="v" style="color:var(--green)">${fmtM(spd)}/s</span></div>
      <div class="nc-row"><span class="k">Chunks done</span>
        <span class="v">${fmt(n.done)}</span></div>
      <div class="nc-row"><span class="k">Backlund OK</span>
        <span class="v" style="color:${n.mismatch?'var(--red)':'var(--green)'}">
          ${fmt(n.ok)} / ${fmt(n.done)}</span></div>
      <div class="nc-row"><span class="k">Avg wall time</span>
        <span class="v">${avgW}s</span></div>
      <div class="gpu-bar-wrap">
        <div class="gpu-bar-lbl"><span>GPU util</span><span>${util?util.toFixed(0)+'%':'—'}</span></div>
        <div class="gpu-bar">
          <div class="gpu-bar-fill" style="width:${util}%;background:${barColor(util)}"></div>
        </div>
      </div>
      <div class="gpu-bar-wrap" style="margin-top:6px">
        <div class="gpu-bar-lbl"><span>VRAM ${(mem/1024).toFixed(1)}GB / ${(mtot/1024).toFixed(0)}GB</span>
          <span>${memPct.toFixed(0)}%</span></div>
        <div class="gpu-bar">
          <div class="gpu-bar-fill" style="width:${memPct}%;background:var(--blue)"></div>
        </div>
      </div>`;
    nodesEl.appendChild(el);
  });

  // feed
  const feed = document.getElementById('feed');
  feed.innerHTML=(d.recent_completions||[]).map(r=>{
    const age=Math.round((d.ts-r.mtime)/60);
    const ageStr=age<1?'just now':age<60?`${age}m ago`:`${Math.floor(age/60)}h ago`;
    return `<div class="fr">
      <span class="ok-ic ${r.ok?'ok':'no'}">${r.ok?'✓':'✗'}</span>
      <span class="cid">${r.chunk_id}</span>
      <span style="color:var(--gold);font-size:.7rem">${r.host}</span>
      <span class="zh">${fmtM(r.zeros)} zeros</span>
      <span class="wl">${r.wall_s.toFixed(1)}s</span>
    </div>`;
  }).join('');
}

async function poll(){
  try{
    const r=await fetch('/api/stats');
    if(r.ok) update(await r.json());
  }catch(e){console.warn(e)}
}
poll();
setInterval(poll, 4000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    print(f"Starting RH Dashboard on http://0.0.0.0:{PORT}", flush=True)
    t = threading.Thread(target=_bg_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=PORT, threaded=True, debug=False)
