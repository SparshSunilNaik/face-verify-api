# app1.py — FastAPI Face Verify (PyTorch + facenet-pytorch)
# Backend API with API-key protection, verbose CLI metrics, and UI pages (/ui, /mobile)

import io, os, base64, threading, logging, time
from typing import Optional, Tuple, Dict
import numpy as np
import cv2
from PIL import Image, ImageOps
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# ================== Config ==================
DEVICE = "cpu"
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.30"))  # cosine distance
MIN_FACE_PROB = float(os.getenv("MIN_FACE_PROB", "0.95"))
REJECT_MULTI_FACE = os.getenv("REJECT_MULTI_FACE", "1") == "1"
REF_IMAGE_PATH = os.getenv("REF_IMAGE_PATH", "me.jpg")
MAX_SIDE = int(os.getenv("MAX_SIDE", "1600"))  # downsize large uploads for speed/stability
VERBOSE = os.getenv("VERBOSE", "1") == "1"

# API key protection
API_KEY = os.getenv("API_KEY", "")                        # set this in your shell
API_KEY_HEADER = os.getenv("API_KEY_HEADER", "x-api-key") # header name to check
OPEN_PATHS = {"/health", "/ui", "/mobile"}                # these do NOT require key

# Logging
logging.basicConfig(
    level=logging.INFO if VERBOSE else logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("face-api")

# Globals
DETECTOR: MTCNN = None
EMB_MODEL: InceptionResnetV1 = None
REF_EMB: Optional[np.ndarray] = None
REF_LOCK = threading.Lock()

app = FastAPI(title="Face Verify API (Backend)", version="3.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# -------------- API-key middleware --------------
@app.middleware("http")
async def api_key_guard(request: Request, call_next):
    # Allow CORS preflight and public pages
    if request.method == "OPTIONS" or request.url.path in OPEN_PATHS:
        return await call_next(request)
    key = request.headers.get(API_KEY_HEADER)
    if not API_KEY or key != API_KEY:
        log.warning(f"[auth] unauthorized ip={request.client.host if request.client else 'unknown'} path={request.url.path}")
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    return await call_next(request)

# ------------------- IO Helpers -------------------
def _pil_from_bytes_exif(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    try:
        im = ImageOps.exif_transpose(im)  # respect EXIF rotation
    except Exception:
        pass
    im = im.convert("RGB")
    if max(im.size) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return im

def _read_bgr_from_bytes_with_meta(data: bytes) -> Tuple[np.ndarray, Tuple[int,int], float]:
    t0 = time.time()
    im = _pil_from_bytes_exif(data)
    w, h = im.size
    bgr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    decode_ms = (time.time() - t0) * 1000.0
    return bgr, (w, h), decode_ms

def _read_bgr_from_path(path: str) -> Tuple[np.ndarray, Tuple[int,int], float]:
    with open(path, "rb") as f:
        data = f.read()
    return _read_bgr_from_bytes_with_meta(data)

# ------------------- Core Logic -------------------
def _detect_best_face_with_rotations(pil_img: Image.Image) -> Tuple[torch.Tensor, float, int, int]:
    """
    Try 0/90/270/180 degrees; pick orientation with highest prob.
    Returns (aligned_tensor[3,160,160], best_prob, strong_count, best_deg).
    """
    rotations = [0, 90, 270, 180]
    best = None
    for deg in rotations:
        img = pil_img if deg == 0 else pil_img.rotate(deg, expand=True)
        _, probs = DETECTOR.detect(img)
        if probs is None or len(probs) == 0:
            continue
        probs = np.asarray(probs, dtype="float32")
        prob = float(np.max(probs))
        strong = int(np.sum(probs >= MIN_FACE_PROB))
        if (best is None) or (prob > best["prob"]):
            best = {"img": img, "prob": prob, "strong": strong, "deg": deg}
    if best is None:
        raise ValueError("No face detected")
    if not np.isfinite(best["prob"]) or best["prob"] < MIN_FACE_PROB:
        raise ValueError(f"Low face confidence: {best['prob']:.3f} (< {MIN_FACE_PROB})")
    if REJECT_MULTI_FACE and best["strong"] > 1:
        raise ValueError(f"Multiple faces detected ({best['strong']}); submit a single-person image.")
    aligned = DETECTOR(best["img"])
    if aligned is None:
        raise ValueError("Face alignment failed")
    return aligned, best["prob"], best["strong"], best["deg"]

def _to_embedding(bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    t_det0 = time.time()
    pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    aligned, best_prob, strong_count, best_deg = _detect_best_face_with_rotations(pil_img)
    detect_ms = (time.time() - t_det0) * 1000.0

    t_emb0 = time.time()
    aligned = aligned.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = EMB_MODEL(aligned).cpu().numpy()[0].astype("float32")
    embed_ms = (time.time() - t_emb0) * 1000.0

    n = float(np.linalg.norm(emb))
    if not np.isfinite(n) or n <= 0:
        raise ValueError("Invalid embedding norm")
    emb = emb / n

    return emb, {
        "best_prob": best_prob,
        "strong_faces": strong_count,
        "best_rotation_deg": best_deg,
        "detect_ms": detect_ms,
        "embed_ms": embed_ms
    }

def _cosine_distance(e1: np.ndarray, e2: np.ndarray) -> float:
    return float(1.0 - float(np.dot(e1, e2)))

def _verify_emb_vs_ref(live_emb: np.ndarray, threshold: Optional[float] = None) -> Dict[str, float]:
    if threshold is None:
        threshold = DEFAULT_THRESHOLD
        thr_src = "default"
    else:
        threshold = float(threshold)
        thr_src = "custom"
    with REF_LOCK:
        if REF_EMB is None:
            raise RuntimeError("Reference embedding not loaded. Upload or point to me.jpg.")
        ref_emb = REF_EMB
    dist = _cosine_distance(ref_emb, live_emb)
    sim = 1.0 - dist
    return {
        "verified": bool(dist <= threshold),
        "distance": float(dist),
        "similarity": float(sim),
        "threshold": float(threshold),
        "threshold_source": thr_src,
        "metric": "cosine(1 - dot)",
        "backbone": "InceptionResnetV1(vggface2)",
        "detector": "MTCNN(160x160)",
        "min_face_prob": MIN_FACE_PROB,
        "ref_path": REF_IMAGE_PATH
    }

def _load_reference_from_path(path: str):
    global REF_EMB, REF_IMAGE_PATH
    bgr, (w, h), dec_ms = _read_bgr_from_path(path)
    emb, info = _to_embedding(bgr)
    with REF_LOCK:
        REF_EMB = emb
        REF_IMAGE_PATH = path
    log.info(
        f"[REF] Loaded: {path} | img={w}x{h} decode_ms={dec_ms:.1f} "
        f"detect_ms={info['detect_ms']:.1f} embed_ms={info['embed_ms']:.1f} "
        f"best_prob={info['best_prob']:.3f} strong_faces={info['strong_faces']} rot={info['best_rotation_deg']}°"
    )

def _load_reference_from_bytes(data: bytes, tag: str = "uploaded"):
    bgr, (w, h), dec_ms = _read_bgr_from_bytes_with_meta(data)
    emb, info = _to_embedding(bgr)
    with REF_LOCK:
        global REF_EMB, REF_IMAGE_PATH
        REF_EMB = emb
        REF_IMAGE_PATH = tag
    log.info(
        f"[REF] Loaded upload: {tag} | img={w}x{h} decode_ms={dec_ms:.1f} "
        f"detect_ms={info['detect_ms']:.1f} embed_ms={info['embed_ms']:.1f} "
        f"best_prob={info['best_prob']:.3f} strong_faces={info['strong_faces']} rot={info['best_rotation_deg']}°"
    )

def _log_verdict(request: Request, stage: str, result: Dict[str, float], extra: Dict[str, float],
                 img_wh: Tuple[int,int], decode_ms: float, total_ms: float):
    client = request.client.host if request.client else "unknown"
    log.info(
        f"[{stage}] ip={client} img={img_wh[0]}x{img_wh[1]} "
        f"decode_ms={decode_ms:.1f} detect_ms={extra.get('detect_ms',0):.1f} embed_ms={extra.get('embed_ms',0):.1f} total_ms={total_ms:.1f} "
        f"best_prob={extra.get('best_prob',0):.3f} strong_faces={int(extra.get('strong_faces',0))} rot={int(extra.get('best_rotation_deg',0))}° "
        f"distance={result.get('distance'):.4f} similarity={result.get('similarity'):.4f} "
        f"thr={result.get('threshold'):.3f}({result.get('threshold_source')}) verdict={'MATCH' if result.get('verified') else 'NO_MATCH'} "
        f"ref='{result.get('ref_path')}'"
    )

# ------------------- Startup -------------------
@app.on_event("startup")
def _startup():
    global DETECTOR, EMB_MODEL
    DETECTOR = MTCNN(
        image_size=160, margin=20, post_process=True, device=DEVICE,
        keep_all=False, select_largest=True
    )
    EMB_MODEL = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
    try:
        if os.path.isfile(REF_IMAGE_PATH):
            _load_reference_from_path(REF_IMAGE_PATH)
            log.info(f"[Startup] Reference ready: {REF_IMAGE_PATH}")
        else:
            log.warning(f"[Startup] REF_IMAGE_PATH '{REF_IMAGE_PATH}' not found. Load later via /reload-ref-*")
    except Exception as e:
        log.error(f"[Startup] Failed to load reference: {e}")

# ------------------- Schemas -------------------
class VerifyB64Body(BaseModel):
    live_image_b64: str
    custom_threshold: Optional[float] = None

# ------------------- Endpoints -------------------
@app.get("/health", response_class=PlainTextResponse)
def health():
    with REF_LOCK:
        ref_ok = REF_EMB is not None
    return f"ok (ref_loaded={ref_ok}, thr={DEFAULT_THRESHOLD}, min_prob={MIN_FACE_PROB}, max_side={MAX_SIDE})"

# 1) Plain text verdict: "Match" / "No Match"
@app.post("/verify-live-text", response_class=PlainTextResponse)
async def verify_live_text(
    request: Request,
    live_file: UploadFile = File(...),
    custom_threshold: Optional[float] = Form(default=None),
):
    t0 = time.time()
    try:
        data = await live_file.read()
        bgr, (w, h), decode_ms = _read_bgr_from_bytes_with_meta(data)
        live_emb, info = _to_embedding(bgr)
        out = _verify_emb_vs_ref(live_emb, custom_threshold)
        total_ms = (time.time() - t0) * 1000.0
        _log_verdict(request, "verify-live-text", out, info, (w, h), decode_ms, total_ms)
        return "Match" if out["verified"] else "No Match"
    except Exception as e:
        log.warning(f"[verify-live-text] error: {e}")
        return PlainTextResponse(str(e), status_code=400)

# 2) JSON verdict (distance, threshold, etc.)
@app.post("/verify-live-file")
async def verify_live_file(
    request: Request,
    live_file: UploadFile = File(...),
    custom_threshold: Optional[float] = Form(default=None),
):
    t0 = time.time()
    try:
        data = await live_file.read()
        bgr, (w, h), decode_ms = _read_bgr_from_bytes_with_meta(data)
        live_emb, info = _to_embedding(bgr)
        out = _verify_emb_vs_ref(live_emb, custom_threshold)
        total_ms = (time.time() - t0) * 1000.0
        _log_verdict(request, "verify-live-file", out, info, (w, h), decode_ms, total_ms)
        return JSONResponse(out)
    except Exception as e:
        log.warning(f"[verify-live-file] error: {e}")
        return JSONResponse({"error": str(e)}, status_code=400)

# 3) Base64 JSON body
@app.post("/verify-live-b64")
async def verify_live_b64(
    request: Request,
    body: VerifyB64Body
):
    t0 = time.time()
    try:
        b64 = body.live_image_b64
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        data = base64.b64decode(b64)
        bgr, (w, h), decode_ms = _read_bgr_from_bytes_with_meta(data)
        live_emb, info = _to_embedding(bgr)
        out = _verify_emb_vs_ref(live_emb, body.custom_threshold)
        total_ms = (time.time() - t0) * 1000.0
        _log_verdict(request, "verify-live-b64", out, info, (w, h), decode_ms, total_ms)
        return JSONResponse(out)
    except Exception as e:
        log.warning(f"[verify-live-b64] error: {e}")
        return JSONResponse({"error": str(e)}, status_code=400)

# Reference management (protected)
@app.post("/reload-ref-file")
async def reload_ref_file(ref_file: UploadFile = File(...)):
    try:
        data = await ref_file.read()
        _load_reference_from_bytes(data, tag=f"uploaded:{ref_file.filename}")
        return {"ok": True, "ref_path": REF_IMAGE_PATH}
    except Exception as e:
        log.warning(f"[reload-ref-file] error: {e}")
        return JSONResponse({"error": str(e)}, status_code=400)

@app.post("/reload-ref-path")
async def reload_ref_path(ref_image_path: str = Form(...)):
    try:
        _load_reference_from_path(ref_image_path)
        return {"ok": True, "ref_path": REF_IMAGE_PATH}
    except Exception as e:
        log.warning(f"[reload-ref-path] error: {e}")
        return JSONResponse({"error": str(e)}, status_code=400)

# ========== Desktop webcam test UI (public GET) ==========
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!doctype html>
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Face Verify - Desktop (API-key aware)</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 24px; }
  video, canvas { max-width: 420px; width: 100%; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,.15); }
  button { padding: 10px 16px; border-radius: 10px; border: 0; margin-right: 8px; cursor: pointer; }
  #status { margin-top: 12px; font-weight: 600; }
  input[type=text]{ padding:8px 10px; border-radius: 8px; border:1px solid #ccc; width: 260px; }
</style>
<h2>Desktop Webcam Test</h2>
<p>Snaps one frame and calls <code>/verify-live-text</code> (needs API key).</p>

<label>API key (or add ?k=... in URL):</label><br/>
<input id="k" type="text" placeholder="x-api-key value" />
<button id="savek">Save Key</button>
<br/><br/>

<video id="cam" autoplay playsinline muted></video><br/>
<button id="open">Open Camera</button>
<button id="snap">Snap & Verify</button>
<div id="status"></div>

<script>
let stream=null;
const video=document.getElementById('cam');
const statusEl=document.getElementById('status');
const kInput=document.getElementById('k');
const qs = new URLSearchParams(location.search);
const preK = qs.get('k') || localStorage.getItem('face_api_key') || "";
kInput.value = preK;

document.getElementById('savek').onclick=()=>{
  localStorage.setItem('face_api_key', kInput.value.trim());
  statusEl.textContent = "Saved API key";
};

document.getElementById('open').onclick=async()=>{
  try{
    stream = await navigator.mediaDevices.getUserMedia({video:true});
    video.srcObject = stream; await video.play();
    statusEl.textContent = "Camera ready";
  }catch(e){ statusEl.textContent = "Camera error: "+e.message; }
};

document.getElementById('snap').onclick=async()=>{
  const key = (kInput.value || "").trim();
  if(!key){ statusEl.textContent="Set API key first"; return; }
  if(!video.videoWidth){ statusEl.textContent="Open camera first"; return; }
  const c=document.createElement('canvas'); c.width=video.videoWidth; c.height=video.videoHeight;
  c.getContext('2d').drawImage(video,0,0);
  const blob = await new Promise(r=>c.toBlob(r,'image/jpeg',0.95));
  const file = new File([blob],'live.jpg',{type:'image/jpeg'});
  const form=new FormData(); form.append('live_file', file);
  statusEl.textContent="Verifying…";
  try{
    const res = await fetch('/verify-live-text',{
      method:'POST',
      headers:{ 'x-api-key': key },
      body: form
    });
    const text = await res.text();
    statusEl.textContent = "Result: "+text;
  }catch(e){ statusEl.textContent="Error: "+e.message; }
};
</script>
"""

# ========== Phone camera test UI (public GET) ==========
@app.get("/mobile", response_class=HTMLResponse)
def mobile():
    return """
<!doctype html>
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Face Verify - Mobile (API-key aware)</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 18px; }
  input, button { font-size: 16px; }
  img { max-width: 100%; border-radius: 12px; margin-top: 10px; }
  button { padding: 10px 16px; border-radius: 10px; border: 0; margin-top: 12px; }
  #status { margin-top: 12px; font-weight: 600; }
  input[type=text]{ padding:8px 10px; border-radius: 8px; border:1px solid #ccc; width: 260px; }
</style>
<h2>Mobile Test</h2>
<p>Pick/Take a photo and call <code>/verify-live-text</code> (needs API key).</p>

<label>API key (or add ?k=...):</label><br/>
<input id="k" type="text" placeholder="x-api-key value" />
<button id="savek">Save Key</button>

<p></p>
<input id="pick" type="file" accept="image/*" capture="user">
<button id="send">Send & Verify</button>
<div id="status"></div>
<img id="preview" alt="" />

<script>
const pick=document.getElementById('pick'), statusEl=document.getElementById('status'), prev=document.getElementById('preview');
const kInput=document.getElementById('k');
const qs = new URLSearchParams(location.search);
const preK = qs.get('k') || localStorage.getItem('face_api_key') || "";
kInput.value = preK;

document.getElementById('savek').onclick=()=>{
  localStorage.setItem('face_api_key', kInput.value.trim());
  statusEl.textContent = "Saved API key";
};

pick.onchange=()=>{
  const f=pick.files[0];
  if(!f){ prev.src=''; return; }
  const r=new FileReader();
  r.onload=()=>prev.src=r.result; r.readAsDataURL(f);
};

document.getElementById('send').onclick=async()=>{
  const key = (kInput.value || "").trim();
  if(!key){ statusEl.textContent="Set API key first"; return; }
  const f=pick.files[0];
  if(!f){ statusEl.textContent="Choose/take a photo first"; return; }
  statusEl.textContent="Uploading…";
  const form=new FormData(); form.append('live_file', f);
  try{
    const res = await fetch('/verify-live-text', { method:'POST', headers:{'x-api-key': key}, body: form });
    const text = await res.text();
    statusEl.textContent = "Result: "+text;
  }catch(e){ statusEl.textContent="Error: "+e.message; }
};
</script>
"""

# Run with: uvicorn app1:app --host 0.0.0.0 --port 8001
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app1:app", host="0.0.0.0", port=8001, reload=False)
