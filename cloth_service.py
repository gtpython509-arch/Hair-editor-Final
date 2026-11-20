"""Production-ready FASHN VTON v1.5 virtual try-on (cloth change) service.

Designed for CPU-only (Hostinger KVM 4) or GPU deployment with lazy loading,
thread-safe inference, and automatic image normalization.
"""

from __future__ import annotations

import gc
import io
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


class ClothTryOnError(Exception):
    """Base error for FASHN VTON cloth-change service."""


class ClothModelNotReadyError(ClothTryOnError):
    """Raised when `fashn_vton` or its downloaded weights are missing."""


@dataclass(frozen=True)
class ClothTryOnConfig:
    base_dir: Path
    weights_dir: Path
    repo_dir: Path
    max_side: int = 1536

    @classmethod
    def from_env(cls) -> "ClothTryOnConfig":
        base = Path(os.environ.get("HAIRCLIP_BASE_DIR", Path(__file__).resolve().parent))
        weights = Path(os.environ.get("FASHN_WEIGHTS_DIR", base / "weights" / "fashn_vton"))
        repo = Path(os.environ.get("FASHN_REPO_DIR", base / "repos" / "fashn-vton-1.5"))
        return cls(base_dir=base, weights_dir=weights, repo_dir=repo)


def _load_pil_rgb(data: bytes, max_side: int = 1536) -> Image.Image:
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")
    if max_side and max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    return im


def _encode_png(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


class ClothTryOnPipeline:
    """Thread-safe, lazy-loading wrapper around FASHN VTON v1.5 `TryOnPipeline`."""

    def __init__(self, config: Optional[ClothTryOnConfig] = None) -> None:
        self.cfg = config or ClothTryOnConfig.from_env()
        self._lock = threading.RLock()
        self._pipeline: Any = None

    def unload(self) -> None:
        """Release loaded FASHN VTON pipeline from memory to free RAM."""
        with self._lock:
            self._pipeline = None
            gc.collect()

    def readiness(self) -> Dict[str, Any]:
        missing = []
        try:
            import fashn_vton  # noqa: F401
        except ImportError:
            missing.append("package:fashn_vton")
        if not (self.cfg.weights_dir / "model.safetensors").exists():
            missing.append(str(self.cfg.weights_dir / "model.safetensors"))
        return {
            "ready": len(missing) == 0,
            "missing": missing,
            "loaded": self._pipeline is not None,
        }

    def _ensure_pipeline(self) -> Any:
        with self._lock:
            if self._pipeline is None:
                status = self.readiness()
                if not status["ready"]:
                    raise ClothModelNotReadyError(
                        "FASHN VTON v1.5 is not ready. Run `python setup_models.py` first. "
                        f"Missing: {', '.join(status['missing'])}"
                    )
                from fashn_vton import TryOnPipeline

                logger.info("Loading FASHN VTON v1.5 from %s", self.cfg.weights_dir)
                self._pipeline = TryOnPipeline(weights_dir=str(self.cfg.weights_dir))
            return self._pipeline

    def try_on(
        self,
        user_bytes: bytes,
        cloth_bytes: bytes,
        *,
        category: str = "tops",
        garment_photo_type: str = "model",
        num_timesteps: int = 20,
        guidance_scale: float = 1.5,
        seed: int = 42,
        segmentation_free: bool = True,
    ) -> Dict[str, Any]:
        """Render `user_bytes` wearing `cloth_bytes`."""
        if category not in ("tops", "bottoms", "one-pieces"):
            raise ValueError("category must be one of: 'tops', 'bottoms', 'one-pieces'")
        if garment_photo_type not in ("model", "flat-lay"):
            raise ValueError("garment_photo_type must be 'model' or 'flat-lay'")

        pipe = self._ensure_pipeline()
        person = _load_pil_rgb(user_bytes, max_side=self.cfg.max_side)
        garment = _load_pil_rgb(cloth_bytes, max_side=self.cfg.max_side)

        with self._lock:
            result = pipe(
                person_image=person,
                garment_image=garment,
                category=category,
                garment_photo_type=garment_photo_type,
                num_samples=1,
                num_timesteps=int(num_timesteps),
                guidance_scale=float(guidance_scale),
                seed=int(seed),
                segmentation_free=bool(segmentation_free),
            )

        out_img = result.images[0]
        return {
            "width": out_img.width,
            "height": out_img.height,
            "category": category,
            "garment_photo_type": garment_photo_type,
            "result_png": _encode_png(out_img),
        }
