"""Extração de embeddings (CNN -> vetor L2-normalizado) e medição de tempo.

Uso como script:
  python extract_embeddings.py --seed 42 --pairs protocols/test_pairs.csv --out results/embeddings/iitd_seed42.npz
"""
import argparse
import os
import platform
import time
import numpy as np
import pandas as pd
import torch
import yaml
from model import MobileNetV3Iris
from preprocess import preprocess_image


def get_hardware():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def load_model(cfg, ckpt_path, device):
    model = MobileNetV3Iris(backbone_name=cfg["model"]["backbone"], pretrained=False,
                            embedding_dim=cfg["model"]["embedding_dim"]).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.eval()


def extract_all_embeddings(model, filepaths, cfg, device, deg_mode="none", deg_level=0,
                           use_rs=None, warmup=5):
    """Retorna (dict arquivo->embedding, estatísticas).
    FTA (image-level) = imagens que falharam no pré-processamento / total.
    Tempos: avg_model_ms (só CNN) e avg_total_ms (pré-processamento + CNN)."""
    img_size = tuple(cfg["preprocess"]["img_size"])
    if use_rs is None:
        use_rs = cfg["preprocess"]["use_rubber_sheet"]
    cuda = device.type == "cuda"
    emb, failed, t_model, t_total = {}, [], [], []
    model.eval()

    if warmup and len(filepaths):  # evita contar a inicialização no tempo de inferência
        with torch.no_grad():
            dummy = torch.zeros(1, 3, *img_size, device=device)
            for _ in range(warmup):
                model(dummy)
            if cuda:
                torch.cuda.synchronize()

    with torch.no_grad():
        for fp in filepaths:
            t0 = time.perf_counter()
            tensor = preprocess_image(fp, img_size, use_rs, deg_mode, deg_level)
            if tensor is None:
                failed.append(fp)
                continue
            inp = tensor.unsqueeze(0).to(device)
            if cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            out = model(inp)
            if cuda:
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            emb[fp] = out.squeeze(0).cpu().numpy()
            t_model.append((t2 - t1) * 1e3)
            t_total.append((t2 - t0) * 1e3)

    n = len(filepaths)
    return emb, {
        "FTA": len(failed) / max(1, n),
        "n_images": n,
        "n_failed": len(failed),
        "failed": failed,
        "avg_model_ms": float(np.mean(t_model)) if t_model else 0.0,
        "avg_total_ms": float(np.mean(t_total)) if t_total else 0.0,
    }


def files_from_pairs(pairs_df):
    return sorted(set(pairs_df["amostra_1"]) | set(pairs_df["amostra_2"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pairs", required=True, help="CSV de pares (amostra_1, amostra_2, tipo)")
    ap.add_argument("--out", required=True, help="arquivo .npz de saída")
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = os.path.join(cfg["paths"]["results_dir"], f"mobilenetv3_seed{a.seed}.pth")
    model = load_model(cfg, ckpt, device)
    files = files_from_pairs(pd.read_csv(a.pairs))
    emb, st = extract_all_embeddings(model, files, cfg, device)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, files=np.array(list(emb.keys())), embeddings=np.stack(list(emb.values())))
    print(f"{len(emb)} embeddings salvos em {a.out} | FTA={st['FTA']:.4f} | "
          f"{st['avg_model_ms']:.2f} ms/img (CNN)")


if __name__ == "__main__":
    main()