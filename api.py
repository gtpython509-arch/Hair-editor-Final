"""FastAPI application for HairCLIP v3 production deployment & testing.

Run locally or on Hostinger VPS:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import base64
import logging
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = HairCLIPPipeline()


def _handle_service_error(exc: Exception) -> HTTPException:
    if isinstance(exc, (FaceNotFoundError, InvalidStyleError, ValueError)):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, SessionNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ModelNotReadyError):
        return HTTPException(status_code=503, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


_HOME_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>HairCLIP v3 API</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 820px; margin: 32px auto; padding: 0 16px; background: #0f172a; color: #f8fafc; }
    .card { background: #1e293b; border-radius: 10px; padding: 18px; margin-bottom: 16px; }
    a { color: #60a5fa; }
    .row { display: flex; flex-wrap: wrap; gap: 14px; align-items: flex-end; }
    label { display: flex; flex-direction: column; gap: 6px; font-size: 13px; color: #cbd5e1; }
    input, button { padding: 8px 10px; border-radius: 6px; border: 1px solid #334155; background: #0f172a; color: #f8fafc; }
    button { background: #2563eb; border-color: #2563eb; cursor: pointer; font-weight: 600; padding: 10px 18px; }
    img { max-width: 100%; border-radius: 8px; margin-top: 12px; }
  </style>
</head>
<body>
  <h2>HairCLIP v3 Hair Change API</h2>
  <div class="card">
    <p>Swagger Docs: <a href="/docs">/docs</a> &nbsp;|&nbsp; Health: <a href="/health">/health</a></p>
    <p id="healthStatus" style="font-size:13px;color:#94a3b8">Checking model status...</p>
  </div>
  <div class="card">
    <h3>Hair Change (<code>POST /api/v1/hair-change</code>)</h3>
    <div class="row">
      <label>1. User Image <input id="userImg" type="file" accept="image/*" /></label>
      <label>2. Style Image <input id="styleImg" type="file" accept="image/*" /></label>
      <button id="go">Run Hair Change</button>
    </div>
    <p id="msg" style="font-size:13px;color:#94a3b8;margin-top:10px"></p>
    <img id="out" />
  </div>
  <script>
    fetch('/health').then(r => r.json()).then(d => {
      document.getElementById('healthStatus').textContent =
        d.ready ? 'Models ready.' : 'Models not downloaded yet (run python setup_models.py).';
    });
    document.getElementById('go').onclick = async () => {
      const u = document.getElementById('userImg').files[0];
      const s = document.getElementById('styleImg').files[0];
      const msg = document.getElementById('msg');
      if (!u || !s) { msg.textContent = 'Select both User Image and Style Image.'; return; }
      const fd = new FormData();
      fd.append('user_image', u);
      fd.append('style_image', s);
      msg.textContent = 'Running...';
      const r = await fetch('/api/v1/hair-change?response_format=png', { method: 'POST', body: fd });
      if (!r.ok) { const e = await r.json().catch(() => ({})); msg.textContent = 'Error: ' + (e.detail || r.status); return; }
      document.getElementById('out').src = URL.createObjectURL(await r.blob());
      msg.textContent = 'Done.';
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
    """Check service readiness, downloaded weights, and loaded models."""
    status = pipeline.readiness()
    return {"status": "ok" if status["ready"] else "models_missing", **status}


@app.post("/api/v1/style/check")
async def check_style(
    style_photo: UploadFile = File(..., description="Portrait photo whose hairstyle you want to transfer"),
    response_format: Literal["json", "png"] = Query("json"),
):
    """Validate a style photo (face detection + hair mask fraction) before running full transfer."""
    try:
        res = pipeline.check_style(await style_photo.read())
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


@app.post("/api/v1/invert")
async def invert_person(
    person_photo: UploadFile = File(..., description="User portrait photo to invert once"),
):
    """Align and invert a user photo into a reusable `session_id`."""
    try:
        res = pipeline.invert_person(await person_photo.read())
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


@app.post("/api/v1/edit")
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
        style_bytes = await style_photo.read() if style_photo is not None else None
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


@app.post("/api/v1/hair-change")
async def hair_change(
    user_image: UploadFile = File(..., description="User portrait image"),
    style_image: UploadFile = File(..., description="Style portrait image"),
    response_format: Literal["png", "json"] = Query("png"),
):
    """Transfer the hairstyle from `style_image` onto `user_image`."""
    try:
        user_bytes = await user_image.read()
        style_bytes = await style_image.read()
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
