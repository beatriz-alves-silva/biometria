import os
import glob
import random
import itertools
import pandas as pd
import yaml

def parse_iitd(root_dir):
    records = []
    for subj_dir in sorted(os.listdir(root_dir)):
        full_subj = os.path.join(root_dir, subj_dir)
        if not os.path.isdir(full_subj):
            continue
        for fname in sorted(os.listdir(full_subj)):
            if not fname.lower().endswith(('.bmp', '.jpg', '.png')):
                continue
            # Identifica olho esquerdo (L) ou direito (R)
            upper = fname.upper()
            if 'L' in upper:
                eye = 'L'
            elif 'R' in upper:
                eye = 'R'
            else:
                # Fallback padrão IITD: imagens 1-5 são L, 6-10 são R
                num = int(''.join(filter(str.isdigit, fname.split('.')[0]))[-2:])
                eye = 'L' if num <= 5 else 'R'
            
            identity = f"IITD_{subj_dir}_{eye}"
            records.append({"filepath": os.path.join(full_subj, fname), "identity": identity})
    return pd.DataFrame(records)

def parse_casia_interval(root_dir):
    records = []
    for subj_dir in sorted(os.listdir(root_dir)):
        full_subj = os.path.join(root_dir, subj_dir)
        if not os.path.isdir(full_subj):
            continue
        for eye in ['L', 'R']:
            eye_dir = os.path.join(full_subj, eye)
            if not os.path.isdir(eye_dir):
                continue
            for fname in sorted(os.listdir(eye_dir)):
                if fname.lower().endswith(('.jpg', '.bmp', '.png')):
                    identity = f"CASIA_{subj_dir}_{eye}"
                    records.append({"filepath": os.path.join(eye_dir, fname), "identity": identity})
    return pd.DataFrame(records)

def generate_pairs(df, max_impostor_per_sample=5, seed=42):
    rng = random.Random(seed)
    genuine_pairs = []
    impostor_pairs = []
    
    grouped = df.groupby("identity")["filepath"].apply(list).to_dict()
    identities = list(grouped.keys())
    
    # Todos os pares genuínos possíveis dentro da mesma identidade (mesmo indivíduo + mesmo olho)
    for ident, files in grouped.items():
        for a, b in itertools.combinations(files, 2):
            genuine_pairs.append({"amostra_1": a, "amostra_2": b, "tipo": "genuine"})
            
    # Amostragem balanceada e determinística de pares impostores
    all_samples = [(ident, f) for ident, files in grouped.items() for f in files]
    for ident_a, file_a in all_samples:
        other_idents = [i for i in identities if i != ident_a]
        sampled_idents = rng.sample(other_idents, min(max_impostor_per_sample, len(other_idents)))
        for ident_b in sampled_idents:
            file_b = rng.choice(grouped[ident_b])
            impostor_pairs.append({"amostra_1": file_a, "amostra_2": file_b, "tipo": "impostor"})
            
    return pd.DataFrame(genuine_pairs + impostor_pairs)

def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    seed = cfg["seed"]
    os.makedirs(cfg["paths"]["protocols_dir"], exist_ok=True)
    
    # 1. Base Principal: IIT Delhi Iris (Split 70% treino / 15% val / 15% teste por IDENTIDADE)
    df_iitd = parse_iitd(cfg["paths"]["iitd_root"])
    identities = sorted(df_iitd["identity"].unique())
    rng = random.Random(seed)
    rng.shuffle(identities)
    
    n = len(identities)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    
    train_ids = set(identities[:n_train])
    val_ids = set(identities[n_train:n_train + n_val])
    test_ids = set(identities[n_train + n_val:])
    
    df_train = df_iitd[df_iitd["identity"].isin(train_ids)].copy()
    df_val = df_iitd[df_iitd["identity"].isin(val_ids)].copy()
    df_test = df_iitd[df_iitd["identity"].isin(test_ids)].copy()
    
    # Mapeia classes numéricas apenas para o treino
    id_to_label = {ident: idx for idx, ident in enumerate(sorted(train_ids))}
    df_train["label"] = df_train["identity"].map(id_to_label)
    
    df_train.to_csv(os.path.join(cfg["paths"]["protocols_dir"], "train.csv"), index=False)
    df_val.to_csv(os.path.join(cfg["paths"]["protocols_dir"], "validation.csv"), index=False)
    df_test.to_csv(os.path.join(cfg["paths"]["protocols_dir"], "test.csv"), index=False)
    
    val_pairs = generate_pairs(df_val, max_impostor_per_sample=10, seed=seed)
    test_pairs = generate_pairs(df_test, max_impostor_per_sample=10, seed=seed)
    
    val_pairs.to_csv(os.path.join(cfg["paths"]["protocols_dir"], "val_pairs.csv"), index=False)
    test_pairs.to_csv(os.path.join(cfg["paths"]["protocols_dir"], "test_pairs.csv"), index=False)
    
    # 2. Base Secundária (Cross-Dataset): CASIA-Iris-V3 Interval (100% avaliação sem fine-tuning)
    if os.path.exists(cfg["paths"]["casia_root"]):
        df_casia = parse_casia_interval(cfg["paths"]["casia_root"])
        casia_pairs = generate_pairs(df_casia, max_impostor_per_sample=5, seed=seed)
        casia_pairs.to_csv(os.path.join(cfg["paths"]["protocols_dir"], "casia_test_pairs.csv"), index=False)
        print(f"CASIA pares gerados: {len(casia_pairs)}")

    print(f"Identidades IITD -> Treino: {len(train_ids)} | Val: {len(val_ids)} | Teste: {len(test_ids)}")
    print(f"Pares Val: {len(val_pairs)} | Pares Teste IITD: {len(test_pairs)}")

if __name__ == "__main__":
    main()