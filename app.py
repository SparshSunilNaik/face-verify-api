# app.py — FastAPI Face Verify (PyTorch + facenet-pytorch)
# EXIF fix + rotation fallback + desktop/mobile UIs + mobile autoclose / lock / RETRY
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
DEVICE = "cpu"                      # CPU-only for portability
DEFAULT_THRESHOLD = 0.30            # cosine distance; lower => stricter
MIN_FACE_PROB = 0.95                # reject weak detections
REJECT_MULTI_FACE = True            # avoid wrong face in crowded images
REF_IMAGE_PATH = os.getenv("REF_IMAGE_PATH", "me.jpg")  # keep me.jpg next to app.py or set env
MAX_SIDE = int(os.getenv("MAX_SIDE", "1600"))  # downsize very large uploads for stability/speed
VERBOSE = os.getenv("VERBOSE", "1") == "1"

API_KEY = os.getenv("API_KEY", "sparsh_naik")  # your key (used elsewhere if you add more protected routes)
API_KEY_HEADER = "x-api-key"
# Allow these without API key (so the phone page works out-of-the-box)
OPEN_PATHS = {"/health", "/ui", "/mobile", "/verify-live-text"}

# Logging setup
logging.basicConfig(
    level=logging.INFO if VERBOSE else logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("face-api")

# Globals (initialized at startup)
DETECTOR: MTCNN = None
EMB_MODEL: InceptionResnetV1 = None
REF_EMB: Optional[np.ndarray] = None
REF_LOCK = threading.Lock()         # guard ref embedding swaps
# ============================================

app = FastAPI(title="Face Verify API (FaceNet/PyTorch)", version="2.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---- API key middleware ----
@app.middleware("http")
async def api_key_guard(request: Request, call_next):
    # Allow CORS preflight and public pages
    if request.method == "OPTIONS" or request.url.path in OPEN_PATHS:
        return await call_next(request)

    got = request.headers.get(API_KEY_HEADER)
    if got != API_KEY:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)

    return await call_next(request)

# ------------------- IO Helpers -------------------
def _pil_from_bytes_exif(data: bytes) -> Image.Image:
    """Load PIL, apply EXIF orientation, convert to RGB, and downsize if huge."""
    im = Image.open(io.BytesIO(data))
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    im = im.convert("RGB")
    if max(im.size) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return im

def _pil_from_path_exif(path: str) -> Image.Image:
    with open(path, "rb") as f:
        data = f.read()
    return _pil_from_bytes_exif(data)

def _read_bgr_from_bytes(data: bytes) -> np.ndarray:
    im = _pil_from_bytes_exif(data)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

def _read_bgr_from_b64(b64str: str) -> np.ndarray:
    if "," in b64str:
        b64str = b64str.split(",", 1)[1]
    data = base64.b64decode(b64str)
    return _read_bgr_from_bytes(data)

def _read_bgr_from_path(path: str) -> np.ndarray:
    im = _pil_from_path_exif(path)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

# ------------------- Core Logic -------------------
def _detect_best_face_with_rotations(pil_img: Image.Image) -> Tuple[torch.Tensor, float, int]:
    """
    Try 0/90/270/180 degrees; pick the orientation with highest prob.
    Returns (aligned_tensor[3,160,160], best_prob, strong_count).
    """
    rotations = [0, 90, 270, 180]
    best = None

    for deg in rotations:
        img = pil_img if deg == 0 else pil_img.rotate(deg, expand=True)
        boxes, probs = DETECTOR.detect(img)
        if probs is None or len(probs) == 0:
            continue

        probs = np.asarray(probs, dtype="float32")
        prob = float(np.max(probs))
        strong = int(np.sum(probs >= MIN_FACE_PROB))

        if (best is None) or (prob > best["prob"]):
            best = {"img": img, "prob": prob, "strong": strong}

    if best is None:
        raise ValueError("No face detected")

    if not np.isfinite(best["prob"]) or best["prob"] < MIN_FACE_PROB:
        raise ValueError(f"Low face confidence: {best['prob']:.3f} (< {MIN_FACE_PROB})")

    if REJECT_MULTI_FACE and best["strong"] > 1:
        raise ValueError(f"Multiple faces detected ({best['strong']}); please submit a single-person image.")

    aligned = DETECTOR(best["img"])  # torch.Tensor [3,160,160] or None
    if aligned is None:
        raise ValueError("Face alignment failed")

    return aligned, best["prob"], best["strong"]

def _to_embedding(bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """Detect+align with rotation fallback, embed with FaceNet. Returns (embedding, info)."""
    pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    aligned, best_prob, strong_count = _detect_best_face_with_rotations(pil_img)
    aligned = aligned.unsqueeze(0).to(DEVICE)  # [1,3,160,160]

    with torch.no_grad():
        emb = EMB_MODEL(aligned).cpu().numpy()[0].astype("float32")

    n = float(np.linalg.norm(emb))
    if not np.isfinite(n) or n <= 0:
        raise ValueError("Invalid embedding norm")
    emb = emb / n

    return emb, {"best_prob": best_prob, "strong_faces": strong_count}

def _cosine_distance(e1: np.ndarray, e2: np.ndarray) -> float:
    # embeddings are L2-normalized → cosine distance = 1 - dot
    return float(1.0 - float(np.dot(e1, e2)))

def _verify_emb_vs_ref(live_emb: np.ndarray, threshold: Optional[float] = None) -> Dict[str, float]:
    if threshold is None:
        threshold = DEFAULT_THRESHOLD
    with REF_LOCK:
        if REF_EMB is None:
            raise RuntimeError("Reference embedding not loaded. Upload me.jpg or call /reload-ref-*.")  # noqa: E501
        ref_emb = REF_EMB
    dist = _cosine_distance(ref_emb, live_emb)
    return {
        "verified": bool(dist <= threshold),
        "distance": float(dist),
        "threshold": float(threshold),
        "metric": "cosine(1 - dot)",
        "backbone": "InceptionResnetV1(vggface2)",
        "detector": "MTCNN(160x160)",
        "min_face_prob": MIN_FACE_PROB,
        "ref_path": REF_IMAGE_PATH
    }

def _load_reference_from_path(path: str):
    global REF_EMB, REF_IMAGE_PATH
    bgr = _read_bgr_from_path(path)
    emb, info = _to_embedding(bgr)
    with REF_LOCK:
        REF_EMB = emb
        REF_IMAGE_PATH = path
    log.info(f"[REF] Loaded from path: {path} | prob={info['best_prob']:.3f} strong_faces={info['strong_faces']}")

def _load_reference_from_bytes(data: bytes, path_label: str = "uploaded"):
    global REF_EMB, REF_IMAGE_PATH
    bgr = _read_bgr_from_bytes(data)
    emb, info = _to_embedding(bgr)
    with REF_LOCK:
        REF_EMB = emb
        REF_IMAGE_PATH = path_label
    log.info(f"[REF] Loaded from upload: {path_label} | prob={info['best_prob']:.3f} strong_faces={info['strong_faces']}")

def _log_verdict(request: Request, stage: str, result: Dict[str, float], extra: Dict[str, float]):
    client = request.client.host if request.client else "unknown"
    log.info(
        f"[{stage}] ip={client} "
        f"distance={result.get('distance'):.4f} thr={result.get('threshold'):.3f} "
        f"best_prob={extra.get('best_prob'):.3f} strong_faces={int(extra.get('strong_faces',0))} "
        f"verdict={'MATCH' if result.get('verified') else 'NO_MATCH'} "
        f"ref='{result.get('ref_path')}'"
    )

# ------------------- Startup -------------------
@app.on_event("startup")
def _startup():
    global DETECTOR, EMB_MODEL
    DETECTOR = MTCNN(
        image_size=160,
        margin=20,
        post_process=True,
        device=DEVICE,
        keep_all=False,
        select_largest=True
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
    ref_image_b64: str
    live_image_b64: str
    custom_threshold: Optional[float] = None

# ------------------- Endpoints -------------------
@app.get("/health", response_class=PlainTextResponse)
def health():
    with REF_LOCK:
        ref_ok = REF_EMB is not None
    return f"ok (ref_loaded={ref_ok}, thr={DEFAULT_THRESHOLD}, min_prob={MIN_FACE_PROB}, max_side={MAX_SIDE})"

# === Live-only endpoints using server-side me.jpg ===
@app.post("/verify-live-text", response_class=PlainTextResponse)
async def verify_live_text(
    request: Request,
    live_file: UploadFile = File(...),
    custom_threshold: Optional[float] = Form(default=None),
):
    t0 = time.time()
    try:
        live_bgr = _read_bgr_from_bytes(await live_file.read())
        live_emb, info = _to_embedding(live_bgr)
        out = _verify_emb_vs_ref(live_emb, custom_threshold)
        _log_verdict(request, "verify-live-text", out, info)
        return "Match" if out["verified"] else "No Match"
    except Exception as e:
        log.warning(f"[verify-live-text] error: {e}")
        return PlainTextResponse(str(e), status_code=400)
    finally:
        log.debug(f"[verify-live-text] took {(time.time()-t0)*1000:.1f}ms")

@app.post("/verify-live-file")
async def verify_live_file(
    request: Request,
    live_file: UploadFile = File(...),
    custom_threshold: Optional[float] = Form(default=None),
):
    t0 = time.time()
    try:
        live_bgr = _read_bgr_from_bytes(await live_file.read())
        live_emb, info = _to_embedding(live_bgr)
        out = _verify_emb_vs_ref(live_emb, custom_threshold)
        _log_verdict(request, "verify-live-file", out, info)
        return JSONResponse(out)
    except Exception as e:
        log.warning(f"[verify-live-file] error: {e}")
        return JSONResponse({"error": str(e)}, status_code=400)
    finally:
        log.debug(f"[verify-live-file] took {(time.time()-t0)*1000:.1f}ms")

# (Optional) Debug endpoint: see distance to help tune thresholds
@app.post("/verify-live-file-debug")
async def verify_live_file_debug(
    request: Request,
    live_file: UploadFile = File(...),
    custom_threshold: Optional[float] = Form(default=None),
):
    try:
        live_bgr = _read_bgr_from_bytes(await live_file.read())
        live_emb, info = _to_embedding(live_bgr)
        out = _verify_emb_vs_ref(live_emb, custom_threshold)
        _log_verdict(request, "verify-live-file-debug", out, info)
        return JSONResponse({
            "distance": out["distance"],
            "threshold": out["threshold"],
            "verified": out["verified"],
            "best_prob": info["best_prob"],
            "strong_faces": info["strong_faces"]
        })
    except Exception as e:
        log.warning(f"[verify-live-file-debug] error: {e}")
        return JSONResponse({"error": str(e)}, status_code=400)

# === Reference management ===
@app.post("/reload-ref-file")
async def reload_ref_file(ref_file: UploadFile = File(...)):
    try:
        data = await ref_file.read()
        _load_reference_from_bytes(data, path_label=f"uploaded:{ref_file.filename}")
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

# === Pairwise endpoints disabled to prevent bypass ===
@app.post("/verify-text", response_class=PlainTextResponse)
async def verify_text(
    ref_file: UploadFile = File(...),
    live_file: UploadFile = File(...),
    custom_threshold: Optional[float] = Form(default=None),
):
    raise HTTPException(status_code=403, detail="Pairwise verification disabled. Use /verify-live-text.")

@app.post("/verify-file")
async def verify_file(
    ref_file: UploadFile = File(...),
    live_file: UploadFile = File(...),
    custom_threshold: Optional[float] = Form(default=None),
):
    raise HTTPException(status_code=403, detail="Pairwise verification disabled. Use /verify-live-file.")

# ========== Desktop webcam test UI ==========
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!doctype html>
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Face Verify - Desktop</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 24px; }
  video, canvas { max-width: 420px; width: 100%; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,.15); }
  button { padding: 10px 16px; border-radius: 10px; border: 0; margin-right: 8px; cursor: pointer; }
  #status { margin-top: 12px; font-weight: 600; }
</style>
<h2>Desktop Webcam Test</h2>
<p>This snaps one frame from your webcam and calls <code>/verify-live-text</code>.</p>
<video id="cam" autoplay playsinline muted></video><br/>
<button id="open">Open Camera</button>
<button id="snap">Snap & Verify</button>
<div id="status"></div>
<script>
let stream=null;
const video=document.getElementById('cam'), statusEl=document.getElementById('status');

document.getElementById('open').onclick=async()=>{
  try{
    stream = await navigator.mediaDevices.getUserMedia({video:true});
    video.srcObject = stream; await video.play();
    statusEl.textContent = "Camera ready";
  }catch(e){ statusEl.textContent = "Camera error: "+e.message; }
};

document.getElementById('snap').onclick=async()=>{
  if(!video.videoWidth){ statusEl.textContent="Open camera first"; return; }
  const c=document.createElement('canvas'); c.width=video.videoWidth; c.height=video.videoHeight;
  c.getContext('2d').drawImage(video,0,0);
  const blob = await new Promise(r=>c.toBlob(r,'image/jpeg',0.95));
  const file = new File([blob],'live.jpg',{type:'image/jpeg'});
  const form=new FormData(); form.append('live_file', file);
  statusEl.textContent="Verifying…";
  try{
    const res = await fetch('/verify-live-text',{method:'POST', body:form});
    const text = await res.text();
    statusEl.textContent = "Result: "+text;
  }catch(e){ statusEl.textContent="Error: "+e.message; }
};
</script>
"""

# ========== Phone camera test UI with RETRY ==========
@app.get("/mobile", response_class=HTMLResponse)
def mobile():
    return """
<!doctype html>
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Face Verify - Mobile</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 18px; line-height: 1.35; }
  .card { padding: 16px; border-radius: 14px; box-shadow: 0 6px 20px rgba(0,0,0,.12); }
  input, button { font-size: 16px; }
  img { max-width: 100%; border-radius: 12px; margin-top: 10px; }
  button { padding: 12px 16px; border-radius: 10px; border: 0; margin-top: 12px; font-weight: 600; }
  .primary { background: #2563eb; color: white; }
  .danger { background: #ef4444; color: white; }
  .muted { opacity: .7; }
  .hidden { display: none; }
  #status { margin-top: 12px; font-weight: 700; }
  #count { font-variant: tabular-nums; }
</style>
<div class="card">
  <h2>Mobile Verification</h2>
  <p class="muted">Take a photo and send for verification.</p>

  <input id="pick" type="file" accept="image/*" capture="user">
  <div style="display:flex; gap:8px; flex-wrap:wrap;">
    <button id="send" class="primary">Send & Verify</button>
    <button id="retry" class="hidden">Retry</button>
  </div>
  <div id="status"></div>
  <img id="preview" alt="" />
</div>

<script>
(function(){
  const $ = s => document.querySelector(s);
  const pick = $("#pick");
  const sendBtn = $("#send");
  const retryBtn = $("#retry");
  const statusEl = $("#status");
  const preview = $("#preview");

  // Config via query params
  const url = new URL(location.href);
  const AUTO_CLOSE_SEC = Math.max(0, parseInt(url.searchParams.get("autoclose") || "10", 10));
  const CLOSE_URL = url.searchParams.get("close_url"); // optional fallback redirect

  let popGuardActive = false;

  // Preview
  pick.addEventListener("change", ()=>{
    const f = pick.files && pick.files[0];
    if (!f) { preview.src = ""; return; }
    const r = new FileReader();
    r.onload = ()=> preview.src = r.result;
    r.readAsDataURL(f);
  });

  function enablePopGuard(){
    if (popGuardActive) return;
    window.onbeforeunload = (e)=>{
      e.preventDefault();
      e.returnValue = "";
      return "";
    };
    history.pushState(null, "", location.href);
    window.addEventListener("popstate", backPush);
    popGuardActive = true;
  }
  function disablePopGuard(){
    window.onbeforeunload = null;
    window.removeEventListener("popstate", backPush);
    popGuardActive = false;
  }
  function backPush(){ history.pushState(null, "", location.href); }

  // Try to close the tab politely, with fallbacks
  async function tryCloseOrRedirect(){
    window.close();
    window.open('','_self'); window.close();
    setTimeout(()=>{
      if (CLOSE_URL) {
        location.href = CLOSE_URL;
      } else {
        location.href = "about:blank";
      }
    }, 250);
  }

  function startCountdownAndClose(seconds){
    let left = seconds;
    const base = "Match ✓ — closing in ";
    statusEl.innerHTML = base + "<span id='count'>" + left + "</span>s";
    const timer = setInterval(()=>{
      left -= 1;
      const span = document.getElementById("count");
      if (span) span.textContent = left;
      if (left <= 0) {
        clearInterval(timer);
        statusEl.textContent = "Closing…";
        tryCloseOrRedirect();
      }
    }, 1000);
  }

  function lockOnNoMatch(){
    pick.disabled = true;
    sendBtn.disabled = true;
    sendBtn.classList.add("danger");
    sendBtn.textContent = "Verification Failed";
    statusEl.innerHTML = "No Match ✗ — please try again.";

    // Show Retry
    retryBtn.classList.remove("hidden");
    retryBtn.textContent = "Retry";

    // Guard navigation
    enablePopGuard();
  }

  function resetForRetry(){
    // Re-enable input & button
    pick.disabled = false;
    sendBtn.disabled = false;
    sendBtn.classList.remove("danger");
    sendBtn.textContent = "Send & Verify";

    // Clear status and preview
    statusEl.textContent = "";
    preview.src = "";

    // Hide Retry, remove guards
    retryBtn.classList.add("hidden");
    disablePopGuard();

    // Clear file input (so user can re-pick)
    try { pick.value = ""; } catch(e){}
  }

  retryBtn.addEventListener("click", ()=>{
    resetForRetry();
  });

  // Main submit
  sendBtn.addEventListener("click", async ()=>{
    const f = pick.files && pick.files[0];
    if (!f) { statusEl.textContent = "Choose/take a photo first"; return; }

    statusEl.textContent = "Uploading…";
    sendBtn.disabled = true;
    retryBtn.classList.add("hidden"); // hide retry while sending

    const form = new FormData();
    form.append("live_file", f);

    try{
      const res = await fetch("/verify-live-text", { method: "POST", body: form });
      const text = (await res.text()).trim();
      if (!res.ok){
        statusEl.textContent = "Error: " + text;
        // enable retry after error
        sendBtn.disabled = false;
        retryBtn.classList.remove("hidden");
        retryBtn.textContent = "Retry";
        return;
      }

      if (text === "Match"){
        if (AUTO_CLOSE_SEC > 0){
          startCountdownAndClose(AUTO_CLOSE_SEC);
        } else {
          statusEl.textContent = "Match ✓";
        }
      } else {
        lockOnNoMatch();
      }
    }catch(e){
      statusEl.textContent = "Error: " + (e?.message || e);
      // allow retry on network errors
      sendBtn.disabled = false;
      retryBtn.classList.remove("hidden");
      retryBtn.textContent = "Retry";
    }
  });
})();
</script>
"""
# Optional: run with python app.py (instead of uvicorn CLI)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
