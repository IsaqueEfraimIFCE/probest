# -*- coding: utf-8 -*-
"""
04_distribuicoes_features.py
----------------------------
Gera a DISTRIBUICAO (histograma) de cada uma das 8 variaveis usadas no
modelo, considerando todos os pacientes juntos (sem separar por classe e
sem linha de media).

Salva um arquivo PNG por variavel na pasta "imagens".
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1) CARREGAR A BASE
# ---------------------------------------------------------------------------
medidas = [
    "raio", "textura", "perimetro", "area", "suavidade",
    "compacidade", "concavidade", "pontos_concavos", "simetria", "dimensao_fractal",
]
colunas = ["id", "diagnostico"]
for sufixo in ["media", "se", "pior"]:
    for m in medidas:
        colunas.append(f"{m}_{sufixo}")

CAMINHO_DADOS = os.path.join("breast+cancer+wisconsin+diagnostic", "wdbc.data")
df = pd.read_csv(CAMINHO_DADOS, header=None, names=colunas)

# As 8 variaveis escolhidas (mesmas do modelo).
FEATURES = [
    "pontos_concavos_pior",
    "perimetro_pior",
    "pontos_concavos_media",
    "raio_pior",
    "perimetro_media",
    "area_pior",
    "raio_media",
    "area_media",
]

# ---------------------------------------------------------------------------
# 2) UM HISTOGRAMA POR VARIAVEL, SEPARADO POR CLASSE (Benigno x Maligno)
# ---------------------------------------------------------------------------
os.makedirs("imagens", exist_ok=True)

for var in FEATURES:
    print("=" * 60)
    print(f"DISTRIBUICAO DE '{var}' POR DIAGNOSTICO")
    print("=" * 60)
    print(df.groupby("diagnostico")[var].describe().round(4))

    benigno = df.loc[df["diagnostico"] == "B", var]
    maligno = df.loc[df["diagnostico"] == "M", var]

    plt.figure(figsize=(9, 5))
    plt.hist(benigno, bins=30, alpha=0.6, label="Benigno",
             color="#2e86de", edgecolor="white")
    plt.hist(maligno, bins=30, alpha=0.6, label="Maligno",
             color="#e74c3c", edgecolor="white")
    plt.title(f"Distribuicao de '{var}' por diagnostico")
    plt.xlabel(var)
    plt.ylabel("Frequencia (numero de pacientes)")
    plt.legend()
    plt.tight_layout()
    saida = os.path.join("imagens", f"distribuicao_{var}.png")
    plt.savefig(saida, dpi=120)
    plt.close()
    print(f"Grafico salvo em '{saida}'.\n")
