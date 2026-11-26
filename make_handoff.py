"""Gera handoff_para_dupla5.zip com o que a outra dupla (CASIA como base principal) precisa
para usar os MESMOS pares/pré-processamento e consolidar resultados.
  python make_handoff.py            # sem pesos
  python make_handoff.py --weights  # inclui os .pth
"""
import argparse
import glob
import json
import os
import zipfile
import yaml


def ficha(cfg, res_dir):
    with open(os.path.join(res_dir, "metrics.json"), encoding="utf-8") as f:
        info = json.load(f)["info"]
    return f"""# Ficha da CNN - Dupla 6 (Íris, base principal IIT Delhi)
- Arquitetura: {info['CNN']} (família MobileNet)
- Origem dos pesos / pré-treino: torchvision, {info['Pre_treinamento']}
- Camadas congeladas / treinadas: PREENCHER (ver train.py; freeze_backbone em model.py)
- Função de perda: CosFace (s=30, m=0.30) sobre embeddings L2-normalizados (ver model.py)
- Dimensão do embedding: {info['Dimensao_embedding']}
- Nº de parâmetros: {info['Num_parametros']:,}
- Matching: {info['Matching']}
- Threshold: {info['Regra_threshold']}
- Rubber Sheet: {info['Rubber_Sheet']} | Seeds: {info['Seeds']} | Hardware: {info['Hardware']}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--weights", action="store_true")
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    res, proto = cfg["paths"]["results_dir"], cfg["paths"]["protocols_dir"]
    files = ["preprocess.py", "config.yaml", "model.py"]
    files += glob.glob(os.path.join(proto, "*.csv"))
    files += [p for pat in ("scores*.csv", "metrics.json", "thresholds.json", "results_table.csv")
              for p in glob.glob(os.path.join(res, pat))]
    if a.weights:
        files += glob.glob(os.path.join(res, "*.pth"))
    with zipfile.ZipFile("handoff_para_dupla5.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            if os.path.exists(p):
                z.write(p)
        z.writestr("FICHA_CNN.md", ficha(cfg, res))
    print("Gerado: handoff_para_dupla5.zip")


if __name__ == "__main__":
    main()