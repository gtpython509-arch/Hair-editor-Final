"""Production-ready HairCLIP v3 inference service.

Encapsulates FFHQ face alignment, e4e W+ inversion, style-photo validation,
and HairCLIPMapper hairstyle/color transfer without notebook or Colab globals.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import sys
import threading
import uuid
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- Errors


class HairCLIPError(Exception):
    """Base error for HairCLIP service."""


class FaceNotFoundError(HairCLIPError):
    """Raised when no face is detected in an uploaded portrait or style photo."""


class InvalidStyleError(HairCLIPError):
    """Raised when the style photo has insufficient visible hair or invalid inputs."""


class SessionNotFoundError(HairCLIPError):
    """Raised when an inversion session_id does not exist."""


class ModelNotReadyError(HairCLIPError):
    """Raised when required repos or model weights are missing."""


# --------------------------------------------------------------------------- Config


@dataclass(frozen=True)
class HairCLIPConfig:
    base_dir: Path
    repos_dir: Path
    weights_dir: Path
    work_dir: Path
    device: str
    crop_size: int = 1024

    @classmethod
    def from_env(cls) -> "HairCLIPConfig":
        base = Path(os.environ.get("HAIRCLIP_BASE_DIR", Path(__file__).resolve().parent))
        repos = Path(os.environ.get("HAIRCLIP_REPOS_DIR", base / "repos"))
        weights = Path(os.environ.get("HAIRCLIP_WEIGHTS_DIR", base / "weights"))
        work = Path(os.environ.get("HAIRCLIP_WORK_DIR", base / "work"))
        device = os.environ.get("HAIRCLIP_DEVICE", "auto")
        return cls(
            base_dir=base,
            repos_dir=repos,
            weights_dir=weights,
            work_dir=work,
            device=device,
        )

    @property
    def hairclip_repo(self) -> Path:
        return self.repos_dir / "HairCLIP"

    @property
    def e4e_repo(self) -> Path:
        return self.repos_dir / "encoder4editing"

    @property
    def hairclip_ckpt(self) -> Path:
        return self.weights_dir / "hairclip.pt"

    @property
    def e4e_ckpt(self) -> Path:
        return self.weights_dir / "e4e_ffhq_encode.pt"

    @property
    def landmarks_dat(self) -> Path:
        return self.weights_dir / "shape_predictor_68_face_landmarks.dat"

    @property
    def parsenet_ckpt(self) -> Path:
        return self.hairclip_repo / "pretrained_models" / "parsenet.pth"

    @property
    def hairstyle_list(self) -> Path:
        return self.hairclip_repo / "mapper" / "hairstyle_list.txt"


# --------------------------------------------------------------------------- Image & Alignment


def load_rgb_from_bytes(data: bytes, bg: int = 128) -> Tuple[np.ndarray, bool]:
    """Load image bytes through PIL, flattening any alpha channel onto neutral gray."""
    im = Image.open(io.BytesIO(data))
    had_alpha = im.mode in ("RGBA", "LA", "P")
    if had_alpha:
        im = im.convert("RGBA")
        canvas = Image.new("RGB", im.size, (bg, bg, bg))
        canvas.paste(im, mask=im.split()[-1])
        im = canvas
    else:
        im = im.convert("RGB")
    return np.array(im), had_alpha


def encode_png(rgb_or_bgr: np.ndarray, *, is_rgb: bool = True) -> bytes:
    bgr = cv2.cvtColor(rgb_or_bgr, cv2.COLOR_RGB2BGR) if is_rgb else rgb_or_bgr
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Failed to encode PNG image.")
    return buf.tobytes()


def find_face(rgb: np.ndarray, detector: Any, predictor: Any) -> Tuple[Optional[np.ndarray], int]:
    dets = detector(rgb, 1)
    if len(dets) == 0:
        return None, 0
    largest = max(dets, key=lambda r: r.width() * r.height())
    shape = predictor(rgb, largest)
    lm = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)
    return lm, len(dets)


def ffhq_quad(lm: np.ndarray) -> np.ndarray:
    eye_l, eye_r = lm[36:42].mean(0), lm[42:48].mean(0)
    eye_avg = (eye_l + eye_r) * 0.5
    eye_to_eye = eye_r - eye_l
    mouth_avg = (lm[48] + lm[54]) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    return np.stack([c - x - y, c - x + y, c + x + y, c + x - y]).astype(np.float32)


def warp_to_crop(rgb: np.ndarray, quad: np.ndarray, size: int = 1024) -> Tuple[np.ndarray, int]:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    dst = np.array([[0, 0], [0, size], [size, size], [size, 0]], dtype=np.float32)
    qw = float(np.hypot(*(quad[3] - quad[0])))
    shrink = max(1, int(np.floor(qw / size / 2)))
    if shrink > 1:
        h, w = bgr.shape[:2]
        small = cv2.resize(bgr, (max(1, w // shrink), max(1, h // shrink)), interpolation=cv2.INTER_AREA)
    else:
        small = bgr
    mat = cv2.getPerspectiveTransform(quad / shrink, dst)
    crop = cv2.warpPerspective(small, mat, (size, size), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    return crop, shrink


def align_bytes(
    image_bytes: bytes,
    detector: Any,
    predictor: Any,
    *,
    size: int = 1024,
    label: str = "image",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    rgb, had_alpha = load_rgb_from_bytes(image_bytes)
    notes = []
    if had_alpha:
        notes.append("transparency flattened")
    h, w = rgb.shape[:2]
    if min(h, w) < 256:
        notes.append(f"small input ({w}x{h})")

    lm, n_faces = find_face(rgb, detector, predictor)
    if lm is None:
        raise FaceNotFoundError(
            f"No face found in {label}. Upload a front-facing portrait with eyes, nose, and mouth visible."
        )
    if n_faces > 1:
        notes.append(f"{n_faces} faces detected, using largest")

    quad = ffhq_quad(lm)
    crop_bgr, shrink = warp_to_crop(rgb, quad, size=size)
    if shrink > 1:
        notes.append(f"shrink={shrink}")
    return crop_bgr, quad, rgb, ", ".join(notes)


# --------------------------------------------------------------------------- Compositing Helpers


def _build_blend_mask(
    mask_new: Any,
    mask_old: Any,
    *,
    size: int,
    mode: str,
    dilate: int,
    feather: int,
) -> np.ndarray:
    import torch

    if mode == "head":
        m = np.ones((size, size), dtype=np.float32)
        b = max(feather, 8)
        m[:b, :] = 0
        m[-b:, :] = 0
        m[:, :b] = 0
        m[:, -b:] = 0
    else:
        m = torch.clamp(mask_new + mask_old, 0, 1)[0, 0].cpu().numpy()
        m = (m > 0.5).astype(np.uint8)
        if dilate > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
            m = cv2.dilate(m, k)
        m = m.astype(np.float32)
    return cv2.GaussianBlur(m, (feather | 1, feather | 1), 0)


def _match_color(src: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
    sel = mask < 0.1
    if sel.sum() < 1000:
        return src
    out = src.copy()
    for c in range(3):
        s_vals, r_vals = src[..., c][sel], ref[..., c][sel]
        sd = s_vals.std() if s_vals.std() > 1e-5 else 1.0
        out[..., c] = (src[..., c] - s_vals.mean()) * (r_vals.std() / sd) + r_vals.mean()
    return np.clip(out, 0, 255)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "edit"


# --------------------------------------------------------------------------- Pipeline


class HairCLIPPipeline:
    """Thread-safe, lazy-loading wrapper around e4e inversion + HairCLIP editing."""

    def __init__(self, config: Optional[HairCLIPConfig] = None) -> None:
        self.cfg = config or HairCLIPConfig.from_env()
        self.cfg.work_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._device: Optional[str] = None
        self._detector: Any = None
        self._predictor: Any = None
        self._avg_lab: Any = None
        self._e4e_net: Any = None
        self._mapper_net: Any = None
        self._clip_mod: Any = None

    @staticmethod
    def _patch_compat() -> None:
        import torch

        if not hasattr(Image, "ANTIALIAS"):
            Image.ANTIALIAS = Image.LANCZOS  # type: ignore[attr-defined]
        orig_load = torch.load
        if not getattr(orig_load, "_hairclip_patched", False):
            def _patched_load(*args: Any, **kwargs: Any) -> Any:
                kwargs.setdefault("weights_only", False)
                if not torch.cuda.is_available():
                    kwargs.setdefault("map_location", "cpu")
                return orig_load(*args, **kwargs)

            _patched_load._hairclip_patched = True  # type: ignore[attr-defined]
            torch.load = _patched_load

        import clip

        orig_clip_load = clip.load
        if not getattr(orig_clip_load, "_hairclip_patched", False):
            def _patched_clip_load(name: Any, device: Any = "cpu", *args: Any, **kwargs: Any) -> Any:
                if not torch.cuda.is_available() and str(device).startswith("cuda"):
                    device = "cpu"
                return orig_clip_load(name, device=device, *args, **kwargs)

            _patched_clip_load._hairclip_patched = True  # type: ignore[attr-defined]
            clip.load = _patched_clip_load

        if not torch.cuda.is_available():
            if not getattr(torch.Tensor.cuda, "_hairclip_patched", False):
                def _cpu_tensor(self: Any, *args: Any, **kwargs: Any) -> Any:
                    return self

                _cpu_tensor._hairclip_patched = True  # type: ignore[attr-defined]
                torch.Tensor.cuda = _cpu_tensor  # type: ignore[method-assign]

            if not getattr(torch.nn.Module.cuda, "_hairclip_patched", False):
                def _cpu_module(self: Any, *args: Any, **kwargs: Any) -> Any:
                    return self

                _cpu_module._hairclip_patched = True  # type: ignore[attr-defined]
                torch.nn.Module.cuda = _cpu_module  # type: ignore[method-assign]

    def _resolve_device(self) -> str:
        if self._device is None:
            import torch

            if self.cfg.device == "auto":
                self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self.cfg.device
        return self._device

    def readiness(self) -> Dict[str, Any]:
        missing = [
            str(p)
            for p in (
                self.cfg.hairclip_repo,
                self.cfg.e4e_repo,
                self.cfg.hairclip_ckpt,
                self.cfg.e4e_ckpt,
                self.cfg.landmarks_dat,
                self.cfg.parsenet_ckpt,
            )
            if not p.exists()
        ]
        return {
            "ready": len(missing) == 0,
            "missing": missing,
            "loaded": {
                "dlib": self._detector is not None,
                "parsenet": self._avg_lab is not None,
                "e4e": self._e4e_net is not None,
                "hairclip": self._mapper_net is not None,
            },
        }

    def _require_paths(self, *paths: Path) -> None:
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise ModelNotReadyError(
                "Missing required model files or repositories. Run `python setup_models.py` first. "
                f"Missing: {', '.join(missing)}"
            )

    def _ensure_dlib(self) -> Tuple[Any, Any]:
        with self._lock:
            if self._detector is None:
                self._require_paths(self.cfg.landmarks_dat)
                import dlib

                self._detector = dlib.get_frontal_face_detector()
                self._predictor = dlib.shape_predictor(str(self.cfg.landmarks_dat))
            return self._detector, self._predictor

    def _ensure_parsenet(self) -> Any:
        with self._lock:
            if self._avg_lab is None:
                self._require_paths(self.cfg.hairclip_repo, self.cfg.parsenet_ckpt)
                self._patch_compat()
                hc_path = str(self.cfg.hairclip_repo)
                if hc_path not in sys.path:
                    sys.path.insert(0, hc_path)
                from criteria.parse_related_loss.average_lab_color_loss import AvgLabLoss

                device = self._resolve_device()
                self._avg_lab = AvgLabLoss(Namespace(parsenet_weights=str(self.cfg.parsenet_ckpt))).to(device).eval()
            return self._avg_lab

    def _ensure_e4e(self) -> Any:
        with self._lock:
            if self._e4e_net is None:
                self._require_paths(self.cfg.e4e_repo, self.cfg.e4e_ckpt)
                self._patch_compat()
                import torch

                e4e_path = str(self.cfg.e4e_repo)
                if e4e_path not in sys.path:
                    sys.path.insert(0, e4e_path)
                from models.psp import pSp

                device = self._resolve_device()
                ckpt = torch.load(str(self.cfg.e4e_ckpt), map_location="cpu")
                opts = dict(ckpt["opts"]) if isinstance(ckpt["opts"], dict) else vars(ckpt["opts"])
                opts["checkpoint_path"] = str(self.cfg.e4e_ckpt)
                opts["device"] = device
                self._e4e_net = pSp(Namespace(**opts)).eval().to(device)
            return self._e4e_net

    def _ensure_mapper(self) -> Any:
        with self._lock:
            if self._mapper_net is None:
                self._require_paths(
                    self.cfg.hairclip_repo,
                    self.cfg.hairclip_ckpt,
                    self.cfg.parsenet_ckpt,
                    self.cfg.hairstyle_list,
                )
                self._patch_compat()
                import torch

                hc_path = str(self.cfg.hairclip_repo)
                if hc_path not in sys.path:
                    sys.path.insert(0, hc_path)
                from mapper.hairclip_mapper import HairCLIPMapper

                device = self._resolve_device()
                ckpt = torch.load(str(self.cfg.hairclip_ckpt), map_location="cpu")
                opts = ckpt["opts"]
                opts = dict(opts) if isinstance(opts, dict) else vars(opts).copy()
                opts.update(
                    {
                        "checkpoint_path": str(self.cfg.hairclip_ckpt),
                        "parsenet_weights": str(self.cfg.parsenet_ckpt),
                        "stylegan_size": self.cfg.crop_size,
                        "device": device,
                        "editing_type": "both",
                        "input_type": "text",
                        "hairstyle_description": str(self.cfg.hairstyle_list),
                        "color_description": "",
                    }
                )
                self._mapper_net = HairCLIPMapper(Namespace(**opts)).eval().to(device)
            return self._mapper_net

    def _ensure_clip(self) -> Any:
        if self._clip_mod is None:
            import clip

            self._clip_mod = clip
        return self._clip_mod

    # ------------------------------------------------------------------ Public API Methods

    def check_style(self, style_bytes: bytes) -> Dict[str, Any]:
        """Validate an uploaded style photo and return hair fraction, verdict, and report PNG."""
        import torch
        from torchvision import transforms

        detector, predictor = self._ensure_dlib()
        avg_lab = self._ensure_parsenet()
        device = self._resolve_device()

        ref_crop_bgr, _, ref_rgb, note = align_bytes(
            style_bytes, detector, predictor, size=self.cfg.crop_size, label="style photo"
        )
        ref_crop_rgb = cv2.cvtColor(ref_crop_bgr, cv2.COLOR_BGR2RGB)
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
        tensor = tf(Image.fromarray(ref_crop_rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            rmask = avg_lab.gen_hair_mask(tensor)
        frac = float(rmask.mean())

        if frac < 0.01:
            verdict = "FAIL"
            message = "Barely any hair found; choose a portrait where hair is clearly visible."
        elif frac < 0.05:
            verdict = "WEAK"
            message = "Little hair visible; expect a faint transfer effect."
        else:
            verdict = "GOOD"
            message = "Style photo is usable."

        masked_vis = ref_crop_rgb.astype(np.float32) * rmask[0, 0].cpu().numpy()[..., None]
        strip_bgr = np.concatenate(
            [
                cv2.resize(cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGR), (512, 512), interpolation=cv2.INTER_AREA),
                cv2.resize(ref_crop_bgr, (512, 512), interpolation=cv2.INTER_AREA),
                cv2.resize(
                    cv2.cvtColor(masked_vis.astype(np.uint8), cv2.COLOR_RGB2BGR),
                    (512, 512),
                    interpolation=cv2.INTER_AREA,
                ),
            ],
            axis=1,
        )
        return {
            "verdict": verdict,
            "message": message,
            "hair_fraction": round(frac, 4),
            "hair_percent": round(frac * 100.0, 2),
            "note": note,
            "report_png": encode_png(strip_bgr, is_rgb=False),
        }

    def invert_person(self, person_bytes: bytes) -> Dict[str, Any]:
        """Align the person photo, invert to W+ latent, and persist a reusable session."""
        import torch
        from torchvision import transforms

        detector, predictor = self._ensure_dlib()
        e4e_net = self._ensure_e4e()
        device = self._resolve_device()

        crop_bgr, quad, orig_rgb, note = align_bytes(
            person_bytes, detector, predictor, size=self.cfg.crop_size, label="your photo"
        )
        session_id = uuid.uuid4().hex[:12]
        session_dir = self.cfg.work_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(session_dir / "aligned_person.png"), crop_bgr)
        cv2.imwrite(str(session_dir / "original.png"), cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR))
        np.save(str(session_dir / "quad.npy"), quad)
        (session_dir / "meta.json").write_text(
            json.dumps(
                {
                    "height": int(orig_rgb.shape[0]),
                    "width": int(orig_rgb.shape[1]),
                    "size": self.cfg.crop_size,
                    "note": note,
                }
            ),
            encoding="utf-8",
        )

        tf = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )
        pil_crop = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            _, w = e4e_net(tf(pil_crop).unsqueeze(0).to(device).float(), randomize_noise=False, return_latents=True)
        torch.save(w.cpu(), str(session_dir / "w.pt"))

        return {
            "session_id": session_id,
            "latent_shape": list(w.shape),
            "width": int(orig_rgb.shape[1]),
            "height": int(orig_rgb.shape[0]),
            "note": note,
            "aligned_png": encode_png(crop_bgr, is_rgb=False),
        }

    def edit_hair(
        self,
        session_id: str,
        *,
        style_bytes: Optional[bytes] = None,
        hair_text: str = "",
        color_text: str = "",
        strength: float = 0.10,
        mode: str = "hair",
        dilate: int = 25,
        feather: int = 41,
        color_match: bool = True,
    ) -> Dict[str, Any]:
        """Apply a style photo and/or text hairstyle/color onto an inverted session."""
        import torch
        from torchvision import transforms

        if not style_bytes and not hair_text.strip() and not color_text.strip():
            raise InvalidStyleError("Provide a style photo, hair_text, or color_text.")
        if mode not in ("hair", "head", "crop"):
            raise ValueError(f"Unsupported mode: {mode!r}")

        session_dir = self.cfg.work_dir / session_id
        if not (session_dir / "w.pt").exists():
            raise SessionNotFoundError(f"Session {session_id!r} not found. Call /api/v1/invert first.")

        avg_lab = self._ensure_parsenet()
        mapper_net = self._ensure_mapper()
        device = self._resolve_device()
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])

        none_t = torch.tensor([0]).unsqueeze(0).to(device)
        hair_tensor = none_t
        style_note = ""
        hair_fraction = None

        if style_bytes:
            detector, predictor = self._ensure_dlib()
            ref_crop_bgr, _, _, style_note = align_bytes(
                style_bytes, detector, predictor, size=self.cfg.crop_size, label="style photo"
            )
            ref_crop_rgb = cv2.cvtColor(ref_crop_bgr, cv2.COLOR_BGR2RGB)
            t = tf(Image.fromarray(ref_crop_rgb)).unsqueeze(0).to(device)
            with torch.no_grad():
                rmask = avg_lab.gen_hair_mask(t)
            hair_fraction = float(rmask.mean())
            if hair_fraction < 0.01:
                raise InvalidStyleError(
                    "Barely any hair found in style photo (<1%). Pick a portrait with clearly visible hair."
                )
            hair_tensor = t * rmask

        hair_tok = none_t
        color_tok = none_t
        if hair_text.strip() or color_text.strip():
            clip_mod = self._ensure_clip()
            if hair_text.strip():
                hair_tok = clip_mod.tokenize([hair_text.strip()]).to(device)
            if color_text.strip():
                color_tok = clip_mod.tokenize([color_text.strip()]).to(device)

        w = torch.load(str(session_dir / "w.pt"), map_location="cpu").to(device).float()
        aligned_pil = Image.open(str(session_dir / "aligned_person.png")).convert("RGB")
        aligned_rgb = np.array(aligned_pil).astype(np.float32)
        with torch.no_grad():
            mask_old = avg_lab.gen_hair_mask(tf(aligned_pil).unsqueeze(0).to(device))
            w_hat = w + float(strength) * mapper_net.mapper(w, hair_tok, color_tok, hair_tensor, none_t)
            x_hat, _ = mapper_net.decoder(
                [w_hat],
                input_is_latent=True,
                return_latents=True,
                randomize_noise=False,
                truncation=1,
            )
            mask_new = avg_lab.gen_hair_mask(x_hat.clamp(-1, 1))

        res_rgb = ((x_hat[0].clamp(-1, 1) + 1.0) * 127.5).permute(1, 2, 0).cpu().numpy()

        if mode == "crop":
            final_rgb = res_rgb.astype(np.uint8)
            comp_rgb = np.concatenate([aligned_rgb.astype(np.uint8), final_rgb], axis=1)
        else:
            meta = json.loads((session_dir / "meta.json").read_text(encoding="utf-8"))
            quad = np.load(str(session_dir / "quad.npy")).astype(np.float32)
            size = int(meta["size"])
            dst = np.array([[0, 0], [0, size], [size, size], [size, 0]], dtype=np.float32)
            m_inv = np.linalg.inv(cv2.getPerspectiveTransform(quad, dst))

            orig_bgr = cv2.imread(str(session_dir / "original.png"), cv2.IMREAD_COLOR)
            orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            oh, ow = orig_rgb.shape[:2]

            blend_m = _build_blend_mask(mask_new, mask_old, size=size, mode=mode, dilate=dilate, feather=feather)
            src = _match_color(res_rgb, aligned_rgb, blend_m) if color_match else res_rgb
            warped = cv2.warpPerspective(
                src.astype(np.uint8), m_inv, (ow, oh), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT
            )
            warped_m = np.clip(
                cv2.warpPerspective(blend_m, m_inv, (ow, oh), flags=cv2.INTER_LINEAR, borderValue=0.0), 0.0, 1.0
            )[..., None]
            final_rgb = (orig_rgb * (1.0 - warped_m) + warped.astype(np.float32) * warped_m).astype(np.uint8)
            comp_rgb = np.concatenate([orig_rgb.astype(np.uint8), final_rgb], axis=1)

        final_png = encode_png(final_rgb, is_rgb=True)
        comp_png = encode_png(comp_rgb, is_rgb=True)
        tag = _slug(hair_text or "style")
        (session_dir / f"final_{tag}.png").write_bytes(final_png)

        return {
            "session_id": session_id,
            "mode": mode,
            "strength": strength,
            "hair_fraction": round(hair_fraction, 4) if hair_fraction is not None else None,
            "style_note": style_note,
            "result_png": final_png,
            "comparison_png": comp_png,
        }

    def hair_change(self, person_bytes: bytes, **edit_kwargs: Any) -> Dict[str, Any]:
        """One-shot invert + edit convenience method."""
        inv = self.invert_person(person_bytes)
        out = self.edit_hair(inv["session_id"], **edit_kwargs)
        out["person_note"] = inv["note"]
        return out
