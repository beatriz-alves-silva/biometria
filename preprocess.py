import cv2
import numpy as np
import torch
from torchvision import transforms

def segment_and_rubber_sheet(img_gray, out_h=64, out_w=256):
    """
    Segmentação simplificada pupila/limbo + Normalização Rubber Sheet de Daugman.
    """
    h, w = img_gray.shape
    blur = cv2.GaussianBlur(img_gray, (7, 7), 0)
    
    # Estima pupila (região mais escura próxima ao centro)
    # Threshold adaptativo baseado no percentil baixo de intensidade
    thresh_val = np.percentile(blur, 12)
    _, mask = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY_INV)
    
    # Limpeza morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Fallback: centro da imagem caso falhe a detecção
    cx, cy = w // 2, h // 2
    r_pupil = min(h, w) * 0.15
    
    if contours:
        # Pega o maior contorno próximo ao centro
        best_cnt = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(best_cnt)
        if 10 < radius < min(h, w) * 0.35:
            cx, cy, r_pupil = int(x), int(y), radius

    # Raio estimado do limbo (aprox. 2.6x o raio da pupila, limitado pela borda)
    r_iris = min(r_pupil * 2.6, cx - 2, cy - 2, w - cx - 2, h - cy - 2)
    if r_iris <= r_pupil + 5:
        r_iris = min(h, w) * 0.42
        r_pupil = r_iris * 0.35

    # Mapeamento Polar -> Cartesiano (Rubber Sheet)
    theta = np.linspace(0, 2 * np.pi, out_w, endpoint=False)
    r = np.linspace(0, 1, out_h)
    
    theta_grid, r_grid = np.meshgrid(theta, r)
    radius_grid = (1 - r_grid) * r_pupil + r_grid * r_iris
    
    map_x = (cx + radius_grid * np.cos(theta_grid)).astype(np.float32)
    map_y = (cy + radius_grid * np.sin(theta_grid)).astype(np.float32)
    
    rubber_sheet = cv2.remap(img_gray, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    
    # Equalização de histograma (CLAHE) para realçar textura da íris
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    rubber_sheet = clahe.apply(rubber_sheet)
    return rubber_sheet

def apply_degradation(img_gray, mode="none", level=0):
    """
    Experimento D - Robustez para Íris: blur e/ou oclusão em 4 níveis:
    level 0: original
    level 1: baixa degradação
    level 2: média degradação
    level 3: alta degradação
    """
    if level == 0 or mode == "none":
        return img_gray
    
    out = img_gray.copy()
    h, w = out.shape
    
    if mode == "blur":
        k_sizes = {1: 5, 2: 11, 3: 19}
        k = k_sizes[level]
        out = cv2.GaussianBlur(out, (k, k), 0)
        
    elif mode == "occlusion":
        # Simula oclusão por pálpebra/cílios (faixas escuras no topo e base)
        ratios = {1: 0.15, 2: 0.28, 3: 0.40}
        r = ratios[level]
        top_h = int(h * r)
        bot_h = int(h * (1 - r * 0.7))
        out[:top_h, :] = 20
        out[bot_h:, :] = 20
        
    return out

def preprocess_image(img_path, img_size=(128, 128), use_rubber_sheet=True, deg_mode="none", deg_level=0):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Aplica degradação na imagem bruta antes do pipeline (simula aquisição degradada)
    img = apply_degradation(img, mode=deg_mode, level=deg_level)
    
    if use_rubber_sheet:
        proc = segment_and_rubber_sheet(img, out_h=img_size[0], out_w=img_size[1])
    else:
        # Ablação: sem Rubber Sheet, apenas CLAHE + resize direto
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        proc = clahe.apply(img)
        proc = cv2.resize(proc, (img_size[1], img_size[0]))
    
    # Converte para 3 canais (esperado pela MobileNetV3 pré-treinada no ImageNet)
    proc_rgb = cv2.cvtColor(proc, cv2.COLOR_GRAY2RGB)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(proc_rgb)