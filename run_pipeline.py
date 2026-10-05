"""Point d'entrée : entraîne, évalue et exporte les tables BI.

    python run_pipeline.py                      # MovieLens 100K
    python run_pipeline.py --dataset synthetic  # hors-ligne
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from recsys.bi_export import export_bi, plot_metrics
from recsys.config import Config
from recsys.pipeline import run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ml-100k", choices=["ml-100k", "synthetic"])
    p.add_argument("--index", default="flat", choices=["flat", "hnsw", "ivf"])
    p.add_argument("--n-candidates", type=int, default=100)
    p.add_argument("--retr-epochs", type=int, default=40)
    p.add_argument("--ranker-epochs", type=int, default=30)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-artifacts", action="store_true")
    a = p.parse_args()

    cfg = Config(dataset=a.dataset, index_type=a.index, n_candidates=a.n_candidates,
                 retr_epochs=a.retr_epochs, ranker_epochs=a.ranker_epochs, output_dir=a.output_dir, seed=a.seed)
    res = run(cfg)

    out = Path(cfg.output_dir)
    bi_dir = export_bi(res, cfg.output_dir)
    plot_metrics(res["metrics"], str(out / "metrics.png"))
    if a.save_artifacts:
        res["model"].save(out / "artifacts")

    table = res["metrics"].pivot_table(index=["model"], columns=["metric", "k"], values="value")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print("\n=== Résultats (test) ===")
        print(table)
        print("\n=== Diversité ===")
        print(res["diversity"])
    (out / "metrics.json").write_text(json.dumps(res["metrics"].to_dict("records"), indent=2))
    print(f"\nTables BI : {bi_dir.resolve()}")


if __name__ == "__main__":
    main()
