"""Assemble le dossier `space_build/` à pousser sur Hugging Face Spaces.

    python deploy/build_space.py
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "space_build"
shutil.rmtree(OUT, ignore_errors=True)
shutil.copytree(ROOT / "recsys", OUT / "recsys", ignore=shutil.ignore_patterns("__pycache__"))
shutil.copy(ROOT / "app.py", OUT / "app.py")
shutil.copy(ROOT / "deploy" / "requirements.txt", OUT / "requirements.txt")
shutil.copy(ROOT / "deploy" / "SPACE_README.md", OUT / "README.md")
print("Dossier prêt :", OUT)
