"""Métricas, gráficos e análise de erros (sem dependência de PyTorch)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

MIN_IMPOSTORS_FOR_TAR = 100  # abaixo disso TAR@FAR=1% não é confiável


def compute_pair_scores(pairs_df, emb_dict):
    """Similaridade de cosseno (embeddings já L2-normalizados -> produto interno).
    Pares com alguma amostra sem embedding (falha de aquisição) são descartados."""
    rows, dropped = [], 0
    for r in pairs_df.itertuples(index=False):
        a, b = r.amostra_1, r.amostra_2
        if a not in emb_dict or b not in emb_dict:
            dropped += 1
            continue
        rows.append({"amostra_1": a, "amostra_2": b, "tipo": r.tipo,
                     "score": float(np.dot(emb_dict[a], emb_dict[b]))})
    return pd.DataFrame(rows, columns=["amostra_1", "amostra_2", "tipo", "score"]), dropped


def _labels_scores(scores_df):
    return (scores_df["tipo"] == "genuine").astype(int).values, scores_df["score"].values


def find_eer_threshold(scores_df):
    """EER e o limiar correspondente (usar SOMENTE com scores de validação)."""
    y, s = _labels_scores(scores_df)
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2.0), float(thr[i])


def compute_metrics(scores_df, threshold):
    """EER/AUC/TAR@FAR1% dependem só dos scores; FMR/FNMR usam o limiar CONGELADO."""
    y, s = _labels_scores(scores_df)
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    gen = s[y == 1]
    imp = s[y == 0]
    tar = None
    if len(imp) >= MIN_IMPOSTORS_FOR_TAR:
        ok = np.where(fpr <= 0.01)[0]
        tar = float(tpr[ok[-1]]) if len(ok) else 0.0
    return {
        "EER": float((fpr[i] + fnr[i]) / 2.0),
        "AUC": float(auc(fpr, tpr)),
        "Threshold": float(threshold),
        "FMR": float(np.mean(imp >= threshold)),
        "FNMR": float(np.mean(gen < threshold)),
        "TAR_at_FAR_1pct": tar,
        "N_genuine": int(len(gen)),
        "N_impostor": int(len(imp)),
    }


def aggregate(dicts):
    """Média e desvio-padrão (ddof=1) entre seeds, para chaves numéricas."""
    mean, std = {}, {}
    for k in dicts[0]:
        vals = [d[k] for d in dicts if isinstance(d.get(k), (int, float)) and d[k] is not None]
        if len(vals) == len(dicts) and vals:
            mean[k] = float(np.mean(vals))
            std[k] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return mean, std


def plot_roc(curves, out_path, title="Curva ROC"):
    """curves: lista de (rótulo, scores_df)."""
    plt.figure(figsize=(6, 5))
    for label, df in curves:
        y, s = _labels_scores(df)
        fpr, tpr, _ = roc_curve(y, s)
        plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC={auc(fpr, tpr):.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("FMR (FAR)")
    plt.ylabel("TAR (1 - FNMR)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_distributions(scores_df, threshold, out_path, title="Distribuição de scores"):
    gen = scores_df.loc[scores_df["tipo"] == "genuine", "score"]
    imp = scores_df.loc[scores_df["tipo"] == "impostor", "score"]
    plt.figure(figsize=(7, 5))
    plt.hist(imp, bins=50, alpha=0.6, density=True, label="Impostor", color="crimson")
    plt.hist(gen, bins=50, alpha=0.6, density=True, label="Genuíno", color="royalblue")
    plt.axvline(threshold, color="black", ls="--", lw=2,
                label=f"Threshold (validação) = {threshold:.3f}")
    plt.xlabel("Similaridade de cosseno")
    plt.ylabel("Densidade")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _resolve(path, data_dir):
    if os.path.exists(path) or not data_dir:
        return path
    return os.path.join(data_dir, path)


def save_error_cases(scores_df, threshold, csv_path, png_path, data_dir=None, n=3):
    """Os n falsos matches e n falsos non-matches mais graves (CSV + figura)."""
    fm = scores_df[(scores_df.tipo == "impostor") & (scores_df.score >= threshold)] \
        .sort_values("score", ascending=False).head(n).copy()
    fm["erro_tipo"] = "False Match"
    fnm = scores_df[(scores_df.tipo == "genuine") & (scores_df.score < threshold)] \
        .sort_values("score").head(n).copy()
    fnm["erro_tipo"] = "False Non-Match"
    cases = pd.concat([fm, fnm])
    cases.to_csv(csv_path, index=False)
    if len(cases) == 0:
        return cases
    from PIL import Image
    fig, axes = plt.subplots(len(cases), 2, figsize=(6, 2.3 * len(cases)), squeeze=False)
    for row, c in enumerate(cases.itertuples(index=False)):
        for col, p in enumerate((c.amostra_1, c.amostra_2)):
            ax = axes[row][col]
            ax.axis("off")
            try:
                ax.imshow(Image.open(_resolve(p, data_dir)), cmap="gray")
            except Exception:
                ax.text(0.5, 0.5, "imagem\nnão encontrada", ha="center", va="center")
            if col == 0:
                ax.set_title(f"{c.erro_tipo} | score={c.score:.3f} | thr={threshold:.3f}",
                             fontsize=8, loc="left")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()
    return cases