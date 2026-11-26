"""Experimento D - robustez: blur e oclusão em 4 níveis (0=original, 1=baixa, 2=média, 3=alta).
Threshold = o congelado na validação (results/thresholds.json, gerado por evaluate.py).

  python robustness.py                       # 3 seeds, IITD e CASIA
  python robustness.py --seeds 42 --datasets IITD
"""
import argparse
import json
import os
import pandas as pd
import torch
import yaml
import metrics as M
from extract_embeddings import extract_all_embeddings, files_from_pairs, load_model
from evaluate import DATASETS

MODES = ["blur", "occlusion"]
LEVELS = [0, 1, 2, 3]
LEVEL_NAMES = {0: "original", 1: "baixa", 2: "média", 3: "alta"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    seeds = a.seeds or cfg.get("seeds", [42, 123, 2026])
    res_dir, proto = cfg["paths"]["results_dir"], cfg["paths"]["protocols_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(res_dir, "thresholds.json")) as f:
        thr_all = json.load(f)  # chaves são strings

    rows = []
    for seed in seeds:
        thr = thr_all[str(seed)]["threshold"]
        model = load_model(cfg, os.path.join(res_dir, f"mobilenetv3_seed{seed}.pth"), device)
        for ds in a.datasets:
            pairs = pd.read_csv(os.path.join(proto, DATASETS[ds][1]))
            files = files_from_pairs(pairs)
            for mode in MODES:
                for lvl in LEVELS:
                    emb, st = extract_all_embeddings(model, files, cfg, device, deg_mode=mode, deg_level=lvl)
                    sc, _ = M.compute_pair_scores(pairs, emb)
                    m = M.compute_metrics(sc, thr)
                    rows.append({"seed": seed, "dataset": ds, "modo": mode, "nivel": lvl,
                                 "nivel_nome": LEVEL_NAMES[lvl], "EER": m["EER"], "AUC": m["AUC"],
                                 "FMR": m["FMR"], "FNMR": m["FNMR"], "FTA": st["FTA"]})
                    print(f"seed {seed} {ds} {mode} nível {lvl}: EER={m['EER']:.4f} FNMR={m['FNMR']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(res_dir, "robustness_results.csv"), index=False)
    summ = df.groupby(["dataset", "modo", "nivel", "nivel_nome"])[["EER", "AUC", "FMR", "FNMR", "FTA"]] \
             .agg(["mean", "std"]).round(4)
    summ.to_csv(os.path.join(res_dir, "robustness_summary.csv"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(MODES), figsize=(10, 4), sharey=True)
    for ax, mode in zip(axes, MODES):
        for ds in a.datasets:
            g = df[(df.modo == mode) & (df.dataset == ds)].groupby("nivel")["EER"].agg(["mean", "std"]).fillna(0)
            ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", capsize=3, label=ds)
        ax.set_title(mode); ax.set_xlabel("nível (0=orig, 3=alta)"); ax.set_xticks(LEVELS); ax.grid(alpha=.3)
    axes[0].set_ylabel("EER"); axes[0].legend()
    plt.tight_layout(); plt.savefig(os.path.join(res_dir, "robustness.png"), dpi=300)


if __name__ == "__main__":
    main()