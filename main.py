import io, base64
from typing import Optional
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---- InsightFace (ONNX) ----
from insightface.app import FaceAnalysis

# -------------- Config --------------
MODEL_PACK = "buffalo_l"    # ArcFace + SCRFD pack
DET_SIZE = (640, 640)       # detector input
COSINE_THRESHOLD = 0.35     # default; lower => stricter
# ------------------------------------

app = FastAPI(title="Face Verify API (InsightFace/ONNX)", version="1.0.0")

# CORS open for hackathon; lock later if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Global FaceAnalysis (loads on startup once)
face_app: FaceAnalysis = None

@app.on_event("startup")
def _warm_models():
    global face_app
    face_app = FaceAnalysis(name=MODEL_PACK, providers=['CPUExecutionProvider'])
    # ctx_id = -1 -> CPU; det_size controls detector input resolution
    face_app.prepare(ctx_id=-1, det_size=DET_SIZE)

# ---------- Helpers ----------
def _read_bgr_from_path(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img

def _read_bgr_from_bytes(data: bytes) -> np.ndarray:
    im = Image.open(io.BytesIO(data)).convert("RGB")
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

def _read_bgr_from_b64(b64str: str) -> np.ndarray:
    if "," in b64str:  # handle data URL
        b64str = b64str.split(",", 1)[1]
    data = base64.b64decode(b64str)
    return _read_bgr_from_bytes(data)

def _largest_face(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

def _embedding_from_bgr(bgr: np.ndarray) -> np.ndarray:
    """
    Detects faces, selects the largest, and returns a L2-normalized embedding.
    """
    # InsightFace expects BGR; .get handles det+align+embedding internally
    faces = face_app.get(bgr)
    f = _largest_face(faces)
    if f is None or f.embedding is None:
        raise ValueError("No face (with embedding) detected.")
    emb = f.embedding.astype("float32")
    # Ensure L2 normalization (usually already normalized, but be safe)
    n = np.linalg.norm(emb)
    if n > 0:
        emb = emb / n
    return emb

def _cosine_distance(e1: np.ndarray, e2: np.ndarray) -> float:
    # embeddings are L2-normalized -> cosine distance = 1 - dot
    return float(1.0 - float(np.dot(e1, e2)))

def _verify_embeddings(e1: np.ndarray, e2: np.ndarray, threshold: Optional[float] = None):
    if threshold is None:
        threshold = COSINE_THRESHOLD
    dist = _cosine_distance(e1, e2)
    return {
        "verified": bool(dist <= threshold),
        "distance": float(dist),
        "threshold": float(threshold),
        "metric": "cosine(1 - dot)",
        "model_pack": MODEL_PACK
    }

def _verify_bgr_pair(ref_bgr: np.ndarray, live_bgr: np.ndarray, custom_threshold: Optional[float] = None):
    e1 = _embedding_from_bgr(ref_bgr)
    e2 = _embedding_from_bgr(live_bgr)
    return _verify_embeddings(e1, e2, custom_threshold)

# ---------- Schemas ----------
class VerifyB64Body(BaseModel):
    ref_image_b64: str
    live_image_b64: str
    custom_threshold: Optional[float] = None

class VerifyPathBody(BaseModel):
    ref_image_path: str
    live_image_path: str
    custom_threshold: Optional[float] = None

# ---------- Endpoints ----------
@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"

# A) Multipart files -> JSON
@app.post("/verify-file")
async def verify_file(
    ref_file: UploadFile = File(...),
    live_file: UploadFile = File(...),
    custom_threshold: Optional[float] = Form(default=None),
):
    try:
        ref_bgr = _read_bgr_from_bytes(await ref_file.read())
        live_bgr = _read_bgr_from_bytes(await live_file.read())
        out = _verify_bgr_pair(ref_bgr, live_bgr, custom_threshold)
        return JSONResponse(out)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

# B) Base64 JSON -> JSON
@app.post("/verify-b64")
def verify_b64(body: VerifyB64Body):
    try:
        ref_bgr = _read_bgr_from_b64(body.ref_image_b64)
        live_bgr = _read_bgr_from_b64(body.live_image_b64)
        out = _verify_bgr_pair(ref_bgr, live_bgr, body.custom_threshold)
        return JSONResponse(out)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

# C) Local paths -> JSON
@app.post("/verify-path")
def verify_path(body: VerifyPathBody):
    try:
        ref_bgr = _read_bgr_from_path(body.ref_image_path)
        live_bgr = _read_bgr_from_path(body.live_image_path)
        out = _verify_bgr_pair(ref_bgr, live_bgr, body.custom_threshold)
        return JSONResponse(out)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

# D) Plain text only -> "Match" / "No Match"
@app.post("/verify-text", response_class=PlainTextResponse)
async def verify_text(
    ref_file: UploadFile = File(...),
    live_file: UploadFile = File(...),
    custom_threshold: Optional[float] = Form(default=None),
):
    try:
        ref_bgr = _read_bgr_from_bytes(await ref_file.read())
        live_bgr = _read_bgr_from_bytes(await live_file.read())
        out = _verify_bgr_pair(ref_bgr, live_bgr, custom_threshold)
        return "Match" if out["verified"] else "No Match"
    except Exception as e:
        return PlainTextResponse(str(e), status_code=400)
