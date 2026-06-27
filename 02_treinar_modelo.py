# -*- coding: utf-8 -*-
"""
02_treinar_modelo.py
--------------------
Treinamento do modelo Naive Bayes para classificar tumores como
Benigno (B) ou Maligno (M), usando a base Breast Cancer Wisconsin (WDBC).

Passos (conforme pedido no trabalho):
  1. Carregar a base de dados.
  2. Selecionar as variaveis preditoras (features).
  3. Dividir os dados em treino e teste.
  4. Treinar um modelo Naive Bayes.
  5. Avaliar (Acuracia, Precisao, Recall, F1-Score e Matriz de Confusao).
  6. Salvar o modelo treinado com joblib (para usar na aplicacao web).
"""

import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)

# ---------------------------------------------------------------------------
# 1) CARREGAR A BASE DE DADOS
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

# ---------------------------------------------------------------------------
# 2) SELECIONAR AS VARIAVEIS PREDITORAS (FEATURES)
# ---------------------------------------------------------------------------
# A base tem 30 variaveis. Para o modelo ficar simples e o formulario da
# aplicacao web nao ficar gigante, escolhemos 6 variaveis.
#
# JUSTIFICATIVA: na analise exploratoria (arquivo 01), calculamos a correlacao
# de cada variavel com o diagnostico. Quanto mais perto de 1, melhor a variavel
# separa Benigno de Maligno. Escolhemos as 6 variaveis com MAIOR correlacao:
#
#   1. pontos_concavos_pior   -> 0.794
#   2. perimetro_pior         -> 0.783
#   3. pontos_concavos_media  -> 0.777
#   4. raio_pior              -> 0.776
#   5. perimetro_media        -> 0.743
#   6. area_pior              -> 0.734
FEATURES = [
    "pontos_concavos_pior",   # correlacao 0.794
    "perimetro_pior",         # correlacao 0.783
    "pontos_concavos_media",  # correlacao 0.777
    "raio_pior",              # correlacao 0.776
    "perimetro_media",        # correlacao 0.743
    "area_pior",              # correlacao 0.734
]

X = df[FEATURES]                              # variaveis de entrada
y = (df["diagnostico"] == "M").astype(int)   # alvo: 1 = Maligno, 0 = Benigno

print("Features usadas no modelo:")
for f in FEATURES:
    print(f"  - {f}")

# ---------------------------------------------------------------------------
# 3) DIVIDIR EM TREINO E TESTE
# ---------------------------------------------------------------------------
# 80% para treinar, 20% para testar. random_state deixa o resultado reproduzivel.
# stratify=y mantem a mesma proporcao de classes nos dois conjuntos.
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTamanho do treino: {len(X_treino)}  |  Tamanho do teste: {len(X_teste)}")

# ---------------------------------------------------------------------------
# 4) TREINAR O MODELO NAIVE BAYES
# ---------------------------------------------------------------------------
# GaussianNB = Naive Bayes para variaveis numericas continuas (assume que
# cada variavel segue uma distribuicao Normal/Gaussiana dentro de cada classe).
modelo = GaussianNB()
modelo.fit(X_treino, y_treino)

# ---------------------------------------------------------------------------
# 5) AVALIAR O MODELO
# ---------------------------------------------------------------------------
y_previsto = modelo.predict(X_teste)

acuracia = accuracy_score(y_teste, y_previsto)
precisao = precision_score(y_teste, y_previsto)
recall = recall_score(y_teste, y_previsto)
f1 = f1_score(y_teste, y_previsto)

print("\n" + "=" * 60)
print("DESEMPENHO DO MODELO (no conjunto de teste)")
print("=" * 60)
print(f"Acuracia : {acuracia:.4f}  ({acuracia*100:.1f}%)")
print(f"Precisao : {precisao:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")

print("\nRelatorio completo:")
print(classification_report(y_teste, y_previsto,
                            target_names=["Benigno", "Maligno"]))

# Matriz de confusao
matriz = confusion_matrix(y_teste, y_previsto)
print("Matriz de Confusao:")
print("                 Previsto Benigno  Previsto Maligno")
print(f"Real Benigno          {matriz[0,0]:5d}            {matriz[0,1]:5d}")
print(f"Real Maligno          {matriz[1,0]:5d}            {matriz[1,1]:5d}")

# Salvar a matriz de confusao como imagem
os.makedirs("imagens", exist_ok=True)
disp = ConfusionMatrixDisplay(confusion_matrix=matriz,
                              display_labels=["Benigno", "Maligno"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Matriz de Confusao - Naive Bayes")
plt.tight_layout()
plt.savefig("imagens/matriz_confusao.png")
plt.close()
print("\nMatriz de confusao salva em 'imagens/matriz_confusao.png'.")

# ---------------------------------------------------------------------------
# 6) SALVAR O MODELO TREINADO (joblib)
# ---------------------------------------------------------------------------
# Salvamos o modelo E a lista de features, para a aplicacao web saber a
# ordem correta dos campos. Tambem salvamos os valores medios por classe,
# usados como exemplos no formulario.
pacote = {
    "modelo": modelo,
    "features": FEATURES,
}
joblib.dump(pacote, "modelo_naive_bayes.joblib")
print("\nModelo salvo em 'modelo_naive_bayes.joblib'.")

# Para ajudar no formulario web: mostra faixa (min/media/max) de cada feature
print("\nFaixa de valores de cada feature (ajuda a preencher o formulario):")
print(df[FEATURES].describe().loc[["min", "mean", "max"]].round(2).T)
