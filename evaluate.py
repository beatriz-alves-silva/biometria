"""Experimentos A (IITD, base principal) e B/C (CASIA, cross-dataset sem fine-tuning).

Para cada seed: threshold = EER na VALIDAÇÃO do IITD -> congelado -> aplicado ao teste IITD
e ao teste CASIA. Reporta média ± desvio-padrão entre seeds.

  python evaluate.py                      # seeds de config.yaml (padrão 42,123,2026)
  python evaluate.py --seeds 42           # só uma seed
  python evaluate.py --no-rs              # ablação: sem Rubber Sheet (saída em results/ablation_no_rs)
"""
import argparse
import json
import os
import shutil
import numpy as np
import pandas as pd
import torch
import yaml
import metrics as M
from extract_embeddings import extract_all_embeddings, files_from_pairs, get_hardware, load_model

DATASETS = {  # nome -> (pares de validação para definir threshold, pares de teste)
    "IITD": ("val_pairs.csv", "test_pairs.csv"),
    "CASIA": (None, "casia_test_pairs.csv"),
}


def relativize(df, root):
    if not root:
        return df
    df = df.copy()
    for c in ("amostra_1", "amostra_2"):
        df[c] = df[c].map(lambda p: os.path.relpath(p, root) if os.path.isabs(p) else p)
    return df


def ckpt_for(cfg, seed, out_dir, ablation):
    """Na ablação, usa checkpoints retreinados (out_dir) se existirem; senão, ablação só em teste."""
    name = f"mobilenetv3_seed{seed}.pth"
    p = os.path.join(out_dir, name)
    if ablation and os.path.exists(p):
        return p, "retreinado sem Rubber Sheet"
    if ablation:
        print("  [aviso] checkpoint sem RS não encontrado: ablação APENAS em teste "
              "(modelo treinado com RS). Declarar isso no artigo.")
        return os.path.join(cfg["paths"]["results_dir"], name), "somente em teste (modelo treinado com RS)"
    return os.path.join(cfg["paths"]["results_dir"], name), "completo"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--no-rs", action="store_true", help="ablação: desliga o Rubber Sheet")
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    seeds = a.seeds or cfg.get("seeds", [42, 123, 2026])
    main_seed = 42 if 42 in seeds else seeds[0]
    ablation = a.no_rs
    use_rs = False if ablation else cfg["preprocess"]["use_rubber_sheet"]
    res_dir = cfg["paths"]["results_dir"]
    out_dir = os.path.join(res_dir, "ablation_no_rs") if ablation else res_dir
    os.makedirs(out_dir, exist_ok=True)
    proto = cfg["paths"]["protocols_dir"]
    data_root = cfg["paths"].get("data_dir")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hw = get_hardware()

    per_seed = {ds: {} for ds in DATASETS}
    thresholds = {}
    canonical = {}
    info = {}

    for seed in seeds:
        ckpt, ablation_mode = ckpt_for(cfg, seed, out_dir, ablation)
        print(f"\n=== seed {seed} | {os.path.basename(ckpt)} | RS={use_rs} ===")
        model = load_model(cfg, ckpt, device)
        n_params = int(sum(p.numel() for p in model.parameters()))
        size_mb = os.path.getsize(ckpt) / 2**20

        # 1) threshold: EER na validação do IITD, depois CONGELADO
        val_pairs = pd.read_csv(os.path.join(proto, DATASETS["IITD"][0]))
        v_emb, _ = extract_all_embeddings(model, files_from_pairs(val_pairs), cfg, device, use_rs=use_rs)
        v_scores, _ = M.compute_pair_scores(val_pairs, v_emb)
        val_eer, thr = M.find_eer_threshold(v_scores)
        thresholds[seed] = {"threshold": thr, "val_EER": val_eer}
        print(f"  threshold congelado (val IITD) = {thr:.4f} | EER val = {val_eer:.4f}")

        # 2) teste nas duas bases com o MESMO threshold
        for ds, (_, test_csv) in DATASETS.items():
            path = os.path.join(proto, test_csv)
            if not os.path.exists(path):
                print(f"  [aviso] {path} não existe; pulando {ds}")
                continue
            pairs = pd.read_csv(path)
            emb, st = extract_all_embeddings(model, files_from_pairs(pairs), cfg, device, use_rs=use_rs)
            scores, dropped = M.compute_pair_scores(pairs, emb)
            m = M.compute_metrics(scores, thr)
            m.update({"FTA": st["FTA"], "pares_descartados": dropped,
                      "Tempo_inferencia_ms": st["avg_model_ms"],
                      "Tempo_total_com_preproc_ms": st["avg_total_ms"],
                      "Num_parametros": n_params, "Tamanho_modelo_MB": size_mb})
            per_seed[ds][str(seed)] = m
            sc = relativize(scores, data_root)
            sc.to_csv(os.path.join(out_dir, f"scores_{ds.lower()}_seed{seed}.csv"), index=False)
            if seed == main_seed:
                canonical[ds] = (scores, thr)
            print(f"  {ds}: EER={m['EER']:.4f} AUC={m['AUC']:.4f} FMR={m['FMR']:.4f} "
                  f"FNMR={m['FNMR']:.4f} FTA={m['FTA']:.4f} {m['Tempo_inferencia_ms']:.2f} ms")
        info = {"CNN": cfg["model"]["backbone"],
                "Pre_treinamento": "ImageNet" if cfg["model"]["pretrained"] else "Nenhum",
                "Num_parametros": n_params, "Dimensao_embedding": cfg["model"]["embedding_dim"],
                "Matching": "Similaridade de cosseno (embeddings L2-normalizados)",
                "Regra_threshold": "EER na validação do IITD, congelado antes do teste",
                "Rubber_Sheet": use_rs, "Modo_ablacao": ablation_mode,
                "Hardware": hw, "Seeds": seeds, "Seed_dos_graficos": main_seed}

    # 3) arquivos finais (nomes pedidos no PDF) a partir da seed principal
    for ds, (scores, thr) in canonical.items():
        suffix = "" if ds == "IITD" else "_casia"
        scores.pipe(relativize, data_root).to_csv(os.path.join(out_dir, f"scores{suffix}.csv"), index=False)
        M.plot_roc([(ds, scores)], os.path.join(out_dir, f"roc{suffix}.png"), f"Curva ROC - {ds}")
        M.plot_distributions(scores, thr, os.path.join(out_dir, f"distributions{suffix}.png"),
                             f"Distribuição de scores - {ds}")
    if "IITD" in canonical and not ablation:
        M.save_error_cases(canonical["IITD"][0], canonical["IITD"][1],
                           os.path.join(out_dir, "error_cases.csv"),
                           os.path.join(out_dir, "error_cases.png"), data_dir=data_root)
        if "CASIA" in canonical:
            M.save_error_cases(canonical["CASIA"][0], canonical["CASIA"][1],
                               os.path.join(out_dir, "error_cases_casia.csv"),
                               os.path.join(out_dir, "error_cases_casia.png"), data_dir=data_root)
    if len(canonical) == 2:
        M.plot_roc([(k, v[0]) for k, v in canonical.items()],
                   os.path.join(out_dir, "roc_both.png"), "ROC: IITD (principal) vs CASIA (cross-dataset)")

    # 4) metrics.json + tabela da seção 26 (média ± desvio entre seeds)
    out, rows = {"info": info, "thresholds": thresholds}, []
    for ds, runs in per_seed.items():
        if not runs:
            continue
        mean, std = M.aggregate(list(runs.values()))
        out[ds] = {"per_seed": runs, "mean": mean, "std": std}
        row = {"Modalidade": "Íris", "Dataset": ds + (" (principal)" if ds == "IITD" else " (cross-dataset)"),
               **{k: info[k] for k in ("CNN", "Pre_treinamento", "Num_parametros", "Dimensao_embedding", "Matching", "Hardware")}}
        for k in ("Threshold", "EER", "AUC", "FMR", "FNMR", "TAR_at_FAR_1pct", "FTA",
                  "Tempo_inferencia_ms", "Tamanho_modelo_MB"):
            if k in mean:
                row[k] = f"{mean[k]:.4f} ± {std[k]:.4f}"
        rows.append(row)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4, ensure_ascii=False)
    with open(os.path.join(out_dir, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "results_table.csv"), index=False)
    print(f"\nPronto. Resultados em {out_dir}/")


if __name__ == "__main__":
    main()