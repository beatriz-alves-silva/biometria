# Verificação biométrica de íris com MobileNetV3 (IITD → CASIA-Iris-V3 Interval)

Projeto Final – Biometria (UTFPR-PG). Dupla 6. **Versão avaliada:** `final-v1.0` / commit `PREENCHER`.

## 1. Modalidade e datasets

Os datasets **não** estão neste repositório. Organização local esperada:
```
data/
├── IITD_Iris/001/01_L.bmp ...
└── CASIA-Iris-Interval/001/L/S1001L01.jpg ...
```
Olho esquerdo e direito são **identidades distintas** (seção 10.1 enunciado).

## 2. Modelo
- CNN: MobileNetV3-Small.
- Camadas congeladas / treinadas: PREENCHER.
- Cabeça de embedding: Linear(→512)-BN-Hardswish-Dropout(0.2)-Linear(→128)-BN, normalização L2. **Embedding = 128**.
- Perda: CosFace (s=30, m=0.30) em `model.py` (`CosFaceHead`); a cabeça só é usada no treino. Em inferência só se usa o embedding.
- Nº de parâmetros / tamanho do modelo / tempo de inferência: ver `results/results_table.csv`.

## 3. Pipeline
Dado bruto → controle de qualidade (FTA) → pré-processamento → segmentação pupila/limbo → Rubber Sheet → normalização
→ CNN → embedding → cosseno → score → threshold → genuíno/impostor. Entrada: PREENCHER (ex.: 64×512, 3 canais replicados).
Detalhes em `preprocess.py` e `config.yaml`.

## 4. Matching e threshold
Similaridade de cosseno entre embeddings L2-normalizados. Threshold = ponto de EER nos scores da **validação do IITD**,
**congelado** e aplicado ao teste IITD e ao CASIA. O teste nunca é usado para escolher threshold/hiperparâmetros.

## 5. Protocolo
Divisão por **identidade** (sem vazamento), `seed = 42`. Arquivos em `protocols/`:
`train.csv`, `validation.csv`, `test.csv`, `val_pairs.csv`, `test_pairs.csv`, `casia_test_pairs.csv`.
Pares com colunas `amostra_1,amostra_2,tipo` (`genuine`/`impostor`). Nº de pares: PREENCHER.
Os pares de cada base são os **mesmos** usados pela outra dupla.

## 6. Ambiente
Python 3.13 `pip install -r requirements.txt` para instalar pacotes.
Seeds de treino: 42, 123, 2026.

## 7. Passo a passo
```bash
pip install -r requirements.txt
python dataset_setup.py                 # (se aplicável) gera/organiza protocolos a partir de data/
python train.py --seed 42               # repetir para 123 e 2026  (gera results/mobilenetv3_seed*.pth)
python evaluate.py                      # Exp. A e C: scores, metrics.json, roc.png, distributions.png, error_cases.*
python robustness.py                    # Exp. D: blur e oclusão, 4 níveis
python evaluate.py --no-rs              # Ablação: sem Rubber Sheet (idealmente após retreinar sem RS)
python make_handoff.py                  # pacote para a outra dupla
```
Saídas principais em `results/`: `scores.csv` (IITD), `scores_casia.csv`, `metrics.json` (média±desvio das 3 seeds),
`results_table.csv` (tabela da seção 26), `roc.png`, `distributions.png`, `robustness_*.{csv,png}`, `error_cases*.{csv,png}`.

## 8. Estrutura
```
README.md  requirements.txt  config.yaml
preprocess.py  model.py  train.py  extract_embeddings.py  evaluate.py  metrics.py  robustness.py  make_handoff.py
protocols/   results/
```