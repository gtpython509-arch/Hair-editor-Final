"""Idempotent setup script to clone HairCLIP/e4e repos and download model weights."""

from __future__ import annotations

import bz2
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

from hairclip_service import HairCLIPConfig


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def main() -> None:
    cfg = HairCLIPConfig.from_env()
    cfg.repos_dir.mkdir(parents=True, exist_ok=True)
    cfg.weights_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.hairclip_repo.exists():
        _run(["git", "clone", "https://github.com/wtybest/HairCLIP.git", str(cfg.hairclip_repo)])
    if not cfg.e4e_repo.exists():
        _run(["git", "clone", "https://github.com/omertov/encoder4editing.git", str(cfg.e4e_repo)])

    # Pull LFS assets (parsenet.pth) if pointer-sized
    if cfg.parsenet_ckpt.exists() and cfg.parsenet_ckpt.stat().st_size < 5_000_000:
        _run(["git", "lfs", "pull"], cwd=cfg.hairclip_repo)

    # Patch e4e's StyleGAN2 ops with HairCLIP's ops
    src_op = cfg.hairclip_repo / "models" / "stylegan2" / "op"
    dst_op = cfg.e4e_repo / "models" / "stylegan2" / "op"
    dst_op.mkdir(parents=True, exist_ok=True)
    for py_file in src_op.glob("*.py"):
        shutil.copy2(py_file, dst_op / py_file.name)

    if not cfg.hairclip_ckpt.exists():
        hf_hub_download("public-data/HairCLIP", "hairclip.pt", local_dir=str(cfg.weights_dir))
    if not cfg.e4e_ckpt.exists():
        hf_hub_download("public-data/e4e", "e4e_ffhq_encode.pt", local_dir=str(cfg.weights_dir))

    if not cfg.landmarks_dat.exists():
        bz2_path = cfg.weights_dir / "shape_predictor_68_face_landmarks.dat.bz2"
        urllib.request.urlretrieve(
            "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
            str(bz2_path),
        )
        with bz2.open(bz2_path, "rb") as src, open(cfg.landmarks_dat, "wb") as dst:
            shutil.copyfileobj(src, dst)
        bz2_path.unlink(missing_ok=True)

    import clip
    clip.load("ViT-B/32", device="cpu", jit=False)

    print("Setup complete. Weights directory:", cfg.weights_dir)


if __name__ == "__main__":
    main()
