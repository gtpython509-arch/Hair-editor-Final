"""FastAPI application for HairCLIP v3 production deployment & testing.

Run locally or on Hostinger VPS:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Literal, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

from cloth_service import (
    ClothModelNotReadyError,
    ClothTryOnPipeline,
)
from hairclip_service import (
    FaceNotFoundError,
    HairCLIPPipeline,
    InvalidStyleError,
    ModelNotReadyError,
    SessionNotFoundError,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="HairCLIP v3 Hair Change API",
    version="1.0.0",
    description="Production FastAPI wrapper for FFHQ alignment, e4e inversion, and HairCLIP hairstyle transfer.",
)

_CORS_RAW = os.environ.get("HAIRCLIP_CORS_ORIGINS", "*")
_CORS_ORIGINS = [o.strip() for o in _CORS_RAW.split(",") if o.strip()] or ["*"]
_API_KEY = os.environ.get("HAIRCLIP_API_KEY", "").strip()
_MAX_UPLOAD_BYTES = int(os.environ.get("HAIRCLIP_MAX_UPLOAD_MB", "15")) * 1024 * 1024

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=_CORS_ORIGINS != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = HairCLIPPipeline()
cloth_pipeline = ClothTryOnPipeline()


def _require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    if _API_KEY and x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


async def _read_upload(upload: UploadFile) -> bytes:
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail=f"Empty file: {upload.filename}")
    if len(data) > _MAX_UPLOAD_BYTES:
        mb = _MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"File exceeds {mb} MB limit")
    return data


@app.on_event("startup")
def _on_startup() -> None:
    removed = pipeline.cleanup_old_sessions()
    if removed:
        logging.getLogger(__name__).info("Cleaned up %d stale sessions", removed)


def _handle_service_error(exc: Exception) -> HTTPException:
    if isinstance(exc, (FaceNotFoundError, InvalidStyleError, ValueError)):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, SessionNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (ModelNotReadyError, ClothModelNotReadyError)):
        return HTTPException(status_code=503, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


_HOME_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Hair &amp; Cloth Editor</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 620px; margin: 48px auto; padding: 0 20px; background: #ffffff; color: #111827; }
    h1 { font-size: 28px; font-weight: 700; margin: 0 0 8px; letter-spacing: -0.02em; }
    .sub { color: #4b5563; font-size: 14px; margin: 0 0 22px; }
    .dropzone { display: block; border: 1.5px dashed #9ca3af; border-radius: 8px; padding: 26px 16px; text-align: center; background: #f9fafb; cursor: pointer; transition: border-color .15s, background .15s; margin-bottom: 12px; }
    .dropzone.dragover { border-color: #2563eb; background: #eff6ff; }
    .dropzone input { display: none; }
    .dropzone .link { text-decoration: underline; font-weight: 600; color: #111827; }
    .dropzone .hint { font-size: 13px; color: #4b5563; }
    .dropzone .fname { margin-top: 8px; font-size: 12px; color: #2563eb; font-weight: 500; }
    .opt-row { display: flex; align-items: center; gap: 8px; margin: 10px 0; font-size: 14px; font-weight: 500; cursor: pointer; }
    .opt-row input[type="checkbox"] { width: 16px; height: 16px; accent-color: #2563eb; cursor: pointer; }
    .opt-row span.note { color: #6b7280; font-weight: 400; font-size: 13px; }
    .panel { display: none; margin: 6px 0 12px 24px; }
    .panel.open { display: block; }
    select { width: 100%; padding: 8px 10px; border: 1px solid #d1d5db; border-radius: 6px; font-size: 14px; background: #fff; color: #111827; margin-bottom: 8px; }
    button.primary { width: 100%; padding: 12px 16px; border: 1px solid #9ca3af; border-radius: 6px; background: #e5e7eb; color: #111827; font-size: 14px; font-weight: 600; cursor: pointer; margin-top: 6px; }
    button.primary:hover:not(:disabled) { background: #d1d5db; }
    button.primary:disabled { opacity: 0.6; cursor: wait; }
    #msg { font-size: 13px; color: #4b5563; margin: 12px 0 0; min-height: 18px; }
    #outWrap { display: none; margin-top: 18px; }
    #out { width: 100%; border-radius: 8px; border: 1px solid #e5e7eb; }
    .dl { display: inline-block; margin-top: 8px; font-size: 13px; color: #2563eb; text-decoration: none; font-weight: 500; }
  </style>
</head>
<body>
  <h1>Hair &amp; Cloth Editor</h1>
  <p class="sub">Upload your photo to change hairstyle, outfit, or both in one place.</p>

  <label class="dropzone" id="userZone">
    <input id="userImg" type="file" accept=".jpg,.jpeg,.png,.webp,image/*" />
    <span class="link">Choose a photo</span> <span class="hint">or drag &amp; drop here (.jpg, .png, .webp)</span>
    <div class="fname" id="userName"></div>
  </label>

  <label class="opt-row">
    <input type="checkbox" id="chkHair" checked />
    <span>Change Hair Style <span class="note">(Upload a hairstyle reference photo)</span></span>
  </label>
  <div class="panel open" id="hairPanel">
    <label class="dropzone" id="styleZone">
      <input id="styleImg" type="file" accept=".jpg,.jpeg,.png,.webp,image/*" />
      <span class="link">Choose a hair style photo</span> <span class="hint">or drag &amp; drop here</span>
      <div class="fname" id="styleName"></div>
    </label>
  </div>

  <label class="opt-row">
    <input type="checkbox" id="chkCloth" />
    <span>Change Cloth / Outfit <span class="note">(Upload a garment photo)</span></span>
  </label>
  <div class="panel" id="clothPanel">
    <label class="dropzone" id="clothZone">
      <input id="clothImg" type="file" accept=".jpg,.jpeg,.png,.webp,image/*" />
      <span class="link">Choose a cloth photo</span> <span class="hint">or drag &amp; drop here</span>
      <div class="fname" id="clothName"></div>
    </label>
    <select id="clothCat">
      <option value="tops">Category: Tops</option>
      <option value="bottoms">Category: Bottoms</option>
      <option value="one-pieces">Category: One-Pieces</option>
    </select>
  </div>

  <button class="primary" id="go">Apply Changes</button>
  <p id="msg"></p>
  <div id="outWrap">
    <img id="out" alt="Result" />
    <a id="dlLink" class="dl" download="result.png">Download result.png</a>
  </div>

  <script>
    function bindDropzone(zoneId, inputId, nameId) {
      const zone = document.getElementById(zoneId);
      const inp = document.getElementById(inputId);
      const lbl = document.getElementById(nameId);
      const update = () => { lbl.textContent = inp.files[0] ? inp.files[0].name : ''; };
      inp.addEventListener('change', update);
      zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
      zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
      zone.addEventListener('drop', e => {
        e.preventDefault();
        zone.classList.remove('dragover');
        if (e.dataTransfer.files.length) { inp.files = e.dataTransfer.files; update(); }
      });
    }
    bindDropzone('userZone', 'userImg', 'userName');
    bindDropzone('styleZone', 'styleImg', 'styleName');
    bindDropzone('clothZone', 'clothImg', 'clothName');

    const chkHair = document.getElementById('chkHair');
    const chkCloth = document.getElementById('chkCloth');
    chkHair.onchange = () => document.getElementById('hairPanel').classList.toggle('open', chkHair.checked);
    chkCloth.onchange = () => document.getElementById('clothPanel').classList.toggle('open', chkCloth.checked);

    document.getElementById('go').onclick = async () => {
      const btn = document.getElementById('go');
      const msg = document.getElementById('msg');
      const u = document.getElementById('userImg').files[0];
      const s = chkHair.checked ? document.getElementById('styleImg').files[0] : null;
      const c = chkCloth.checked ? document.getElementById('clothImg').files[0] : null;
      if (!u) { msg.textContent = 'Please choose your photo first.'; return; }
      if (!s && !c) { msg.textContent = 'Select at least one option (Hair Style or Cloth) and upload its photo.'; return; }

      const fd = new FormData();
      fd.append('user_image', u);
      if (s) fd.append('style_image', s);
      if (c) fd.append('cloth_image', c);
      fd.append('category', document.getElementById('clothCat').value);

      btn.disabled = true;
      msg.textContent = 'Processing... (CPU may take 1-3 minutes)';
      try {
        const r = await fetch('/api/v1/try-on?response_format=png', { method: 'POST', body: fd });
        if (!r.ok) {
          const e = await r.json().catch(() => ({}));
          msg.textContent = 'Error: ' + (e.detail || r.status);
          return;
        }
        const url = URL.createObjectURL(await r.blob());
        document.getElementById('out').src = url;
        document.getElementById('dlLink').href = url;
        document.getElementById('outWrap').style.display = 'block';
        msg.textContent = 'Done.';
      } finally {
        btn.disabled = false;
      }
    };
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    """Landing page with links to /docs, /health, and a 2-input test form."""
    return _HOME_HTML


@app.get("/health")
def health() -> dict:
    """Check readiness for both HairCLIP (hair) and FASHN VTON v1.5 (cloth)."""
    hair = pipeline.readiness()
    cloth = cloth_pipeline.readiness()
    ready = hair["ready"] and cloth["ready"]
    return {
        "status": "ok" if ready else "partial_or_missing",
        "ready": ready,
        "hair": hair,
        "cloth": cloth,
    }


@app.post("/api/v1/style/check", dependencies=[Depends(_require_api_key)])
async def check_style(
    style_photo: UploadFile = File(..., description="Portrait photo whose hairstyle you want to transfer"),
    response_format: Literal["json", "png"] = Query("json"),
):
    """Validate a style photo (face detection + hair mask fraction) before running full transfer."""
    try:
        res = pipeline.check_style(await _read_upload(style_photo))
    except Exception as exc:
        raise _handle_service_error(exc) from exc

    if response_format == "png":
        return Response(content=res["report_png"], media_type="image/png")
    return JSONResponse(
        {
            "verdict": res["verdict"],
            "message": res["message"],
            "hair_fraction": res["hair_fraction"],
            "hair_percent": res["hair_percent"],
            "note": res["note"],
            "report_png_b64": base64.b64encode(res["report_png"]).decode("ascii"),
        }
    )


@app.post("/api/v1/invert", dependencies=[Depends(_require_api_key)])
async def invert_person(
    person_photo: UploadFile = File(..., description="User portrait photo to invert once"),
):
    """Align and invert a user photo into a reusable `session_id`."""
    try:
        res = pipeline.invert_person(await _read_upload(person_photo))
    except Exception as exc:
        raise _handle_service_error(exc) from exc

    return {
        "session_id": res["session_id"],
        "latent_shape": res["latent_shape"],
        "width": res["width"],
        "height": res["height"],
        "note": res["note"],
        "aligned_png_b64": base64.b64encode(res["aligned_png"]).decode("ascii"),
    }


@app.post("/api/v1/edit", dependencies=[Depends(_require_api_key)])
async def edit_hair(
    session_id: str = Form(..., description="Session ID returned by /api/v1/invert"),
    style_photo: Optional[UploadFile] = File(None, description="Optional portrait style photo"),
    hair_text: str = Form("", description="Optional hairstyle text, e.g. 'curly bob'"),
    color_text: str = Form("", description="Optional hair color text, e.g. 'ash blonde'"),
    strength: float = Form(0.10, ge=0.01, le=0.50),
    mode: Literal["hair", "head", "crop"] = Form("hair"),
    dilate: int = Form(25, ge=0, le=80),
    feather: int = Form(41, ge=5, le=121),
    color_match: bool = Form(True),
    response_format: Literal["png", "json"] = Query("png"),
):
    """Apply a style photo and/or text prompt onto an existing inverted session."""
    try:
        style_bytes = await _read_upload(style_photo) if style_photo is not None else None
        res = pipeline.edit_hair(
            session_id,
            style_bytes=style_bytes,
            hair_text=hair_text,
            color_text=color_text,
            strength=strength,
            mode=mode,
            dilate=dilate,
            feather=feather,
            color_match=color_match,
        )
    except Exception as exc:
        raise _handle_service_error(exc) from exc

    if response_format == "png":
        return Response(content=res["result_png"], media_type="image/png")
    return JSONResponse(
        {
            "session_id": res["session_id"],
            "mode": res["mode"],
            "strength": res["strength"],
            "hair_fraction": res["hair_fraction"],
            "style_note": res["style_note"],
            "result_png_b64": base64.b64encode(res["result_png"]).decode("ascii"),
            "comparison_png_b64": base64.b64encode(res["comparison_png"]).decode("ascii"),
        }
    )


@app.post("/api/v1/hair-change", dependencies=[Depends(_require_api_key)])
async def hair_change(
    user_image: UploadFile = File(..., description="User portrait image"),
    style_image: UploadFile = File(..., description="Style portrait image"),
    response_format: Literal["png", "json"] = Query("png"),
):
    """Transfer the hairstyle from `style_image` onto `user_image`."""
    try:
        cloth_pipeline.unload()
        user_bytes = await _read_upload(user_image)
        style_bytes = await _read_upload(style_image)
        res = pipeline.hair_change(
            user_bytes,
            style_bytes=style_bytes,
        )
    except Exception as exc:
        raise _handle_service_error(exc) from exc

    if response_format == "png":
        return Response(content=res["result_png"], media_type="image/png")
    return JSONResponse(
        {
            "session_id": res["session_id"],
            "person_note": res["person_note"],
            "style_note": res["style_note"],
            "hair_fraction": res["hair_fraction"],
            "result_png_b64": base64.b64encode(res["result_png"]).decode("ascii"),
            "comparison_png_b64": base64.b64encode(res["comparison_png"]).decode("ascii"),
        }
    )


@app.post("/api/v1/cloth-change", dependencies=[Depends(_require_api_key)])
async def cloth_change(
    user_image: UploadFile = File(..., description="Person portrait/full-body image"),
    cloth_image: UploadFile = File(..., description="Garment photo (model or flat-lay)"),
    category: Literal["tops", "bottoms", "one-pieces"] = Form("tops"),
    garment_photo_type: Literal["model", "flat-lay"] = Form("model"),
    num_timesteps: int = Form(20, ge=10, le=50),
    guidance_scale: float = Form(1.5, ge=1.0, le=3.0),
    seed: int = Form(42),
    segmentation_free: bool = Form(True),
    response_format: Literal["png", "json"] = Query("png"),
):
    """Virtual try-on: render `user_image` wearing `cloth_image` using FASHN VTON v1.5."""
    try:
        pipeline.unload()
        user_bytes = await _read_upload(user_image)
        cloth_bytes = await _read_upload(cloth_image)
        res = cloth_pipeline.try_on(
            user_bytes,
            cloth_bytes,
            category=category,
            garment_photo_type=garment_photo_type,
            num_timesteps=num_timesteps,
            guidance_scale=guidance_scale,
            seed=seed,
            segmentation_free=segmentation_free,
        )
    except Exception as exc:
        raise _handle_service_error(exc) from exc

    if response_format == "png":
        return Response(content=res["result_png"], media_type="image/png")
    return JSONResponse(
        {
            "width": res["width"],
            "height": res["height"],
            "category": res["category"],
            "garment_photo_type": res["garment_photo_type"],
            "result_png_b64": base64.b64encode(res["result_png"]).decode("ascii"),
        }
    )
