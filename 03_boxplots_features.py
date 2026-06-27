# -*- coding: utf-8 -*-
"""
03_boxplots_features.py
-----------------------
Gera UM unico grafico com os box plots de TODAS as 30 features, separados
por classe (Benigno x Maligno), para ajudar a decidir quais variaveis usar.

Por que padronizar?
  As features tem escalas muito diferentes (ex.: area chega a 4254, enquanto
  suavidade fica em ~0.13). Se colocassemos os valores brutos no mesmo grafico,
  as caixas pequenas sumiriam. Por isso usamos a PADRONIZACAO (z-score):
        z = (valor - media) / desvio_padrao
  Assim todas as features ficam na mesma escala (media 0, desvio 1) e podem
  ser comparadas lado a lado.

Como ler o grafico:
  Para cada feature ha duas caixas (Benigno e Maligno). Se as duas caixas
  estao BEM SEPARADAS, a feature distingue bem as classes -> boa feature.
  Se estao quase sobrepostas, a feature ajuda pouco.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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

# Nome amigavel para a legenda
df["Diagnostico"] = df["diagnostico"].map({"B": "Benigno", "M": "Maligno"})

# ---------------------------------------------------------------------------
# 2) PADRONIZAR AS 30 FEATURES (z-score)
# ---------------------------------------------------------------------------
features = [c for c in colunas if c not in ("id", "diagnostico")]
df_padronizado = df.copy()
df_padronizado[features] = (df[features] - df[features].mean()) / df[features].std()

# ---------------------------------------------------------------------------
# 3) TRANSFORMAR PARA FORMATO "LONGO" (necessario para um unico grafico)
# ---------------------------------------------------------------------------
# Em vez de 30 colunas, criamos 3 colunas: feature, valor e Diagnostico.
df_longo = df_padronizado.melt(
    id_vars="Diagnostico",
    value_vars=features,
    var_name="Feature",
    value_name="Valor (padronizado)",
)

# ---------------------------------------------------------------------------
# 4) DESENHAR UM UNICO GRAFICO COM TODAS AS FEATURES
# ---------------------------------------------------------------------------
os.makedirs("imagens", exist_ok=True)
plt.figure(figsize=(12, 14))
sns.boxplot(
    data=df_longo,
    y="Feature",
    x="Valor (padronizado)",
    hue="Diagnostico",
    palette={"Benigno": "#4CAF50", "Maligno": "#E53935"},
    orient="h",
)
plt.title("Box plot de todas as 30 features (padronizadas), por diagnostico")
plt.xlabel("Valor padronizado (z-score)")
plt.ylabel("")
plt.legend(title="Diagnostico", loc="lower right")
plt.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("imagens/boxplots_todas_features.png", dpi=120)
plt.close()

print("Grafico salvo em 'imagens/boxplots_todas_features.png'.")
print("Dica: as features cujas caixas verde e vermelha estao MAIS SEPARADAS")
print("sao as que melhor distinguem Benigno de Maligno.")
