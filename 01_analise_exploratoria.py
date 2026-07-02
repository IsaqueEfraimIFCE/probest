# -*- coding: utf-8 -*-
"""
01_analise_exploratoria.py
--------------------------
Analise Exploratoria dos Dados (EDA) da base Breast Cancer Wisconsin (WDBC).

O que este arquivo faz (passo a passo):
  1. Carrega a base de dados.
  2. Mostra estatisticas descritivas das variaveis.
  3. Investiga a distribuicao das classes (Benigno x Maligno).
  4. Calcula a correlacao de cada variavel com o diagnostico para
     ajudar a JUSTIFICAR a escolha das features usadas no modelo.
  5. Gera alguns graficos simples salvos na pasta "imagens".

Tudo foi escrito de forma simples para ser facilmente explicado.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # permite salvar graficos sem abrir janela
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1) CARREGAR A BASE DE DADOS
# ---------------------------------------------------------------------------
# O arquivo wdbc.data nao tem cabecalho. Pelas instrucoes da base (wdbc.names):
#   coluna 1  -> ID do paciente
#   coluna 2  -> Diagnostico (M = Maligno, B = Benigno)
#   colunas 3 a 32 -> 30 variaveis numericas (10 medidas x 3 versoes)
#
# As 10 medidas basicas sao calculadas a partir da imagem do nucleo da celula:
medidas = [
    "raio", "textura", "perimetro", "area", "suavidade",
    "compacidade", "concavidade", "pontos_concavos", "simetria", "dimensao_fractal",
]
# Cada medida aparece em 3 versoes: media (mean), erro padrao (se) e pior (worst)
colunas = ["id", "diagnostico"]
for sufixo in ["media", "se", "pior"]:
    for m in medidas:
        colunas.append(f"{m}_{sufixo}")

CAMINHO_DADOS = os.path.join(
    "breast+cancer+wisconsin+diagnostic", "wdbc.data"
)
df = pd.read_csv(CAMINHO_DADOS, header=None, names=colunas)

print("=" * 70)
print("1) VISAO GERAL DA BASE")
print("=" * 70)
print(f"Numero de observacoes (linhas): {df.shape[0]}")
print(f"Numero de colunas: {df.shape[1]} (1 id + 1 alvo + 30 variaveis)")
print(f"Valores faltantes na base: {df.isnull().sum().sum()}")
print("\nPrimeiras 5 linhas:")
print(df.head())

# ---------------------------------------------------------------------------
# 2) ESTATISTICAS DESCRITIVAS
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("2) ESTATISTICAS DESCRITIVAS (algumas variaveis)")
print("=" * 70)
# describe() mostra: contagem, media, desvio padrao, minimo, quartis e maximo
algumas = ["raio_media", "textura_media", "area_media",
           "concavidade_media", "pontos_concavos_media"]
print(df[algumas].describe().round(2))

# ---------------------------------------------------------------------------
# 3) DISTRIBUICAO DAS CLASSES
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("3) DISTRIBUICAO DAS CLASSES")
print("=" * 70)
contagem = df["diagnostico"].value_counts()
print("B = Benigno, M = Maligno")
print(contagem)
total = len(df)
print(f"\nBenigno: {contagem.get('B', 0)} "
      f"({100*contagem.get('B', 0)/total:.1f}%)")
print(f"Maligno: {contagem.get('M', 0)} "
      f"({100*contagem.get('M', 0)/total:.1f}%)")

# ---------------------------------------------------------------------------
# 4) ASSOCIACAO COM O DIAGNOSTICO (R2 / ETA-QUADRADO)
#    -> JUSTIFICATIVA DA ESCOLHA DAS FEATURES
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("4) VARIAVEIS MAIS RELACIONADAS COM O DIAGNOSTICO (R2 / ETA-QUADRADO)")
print("=" * 70)
# O diagnostico e uma variavel QUALITATIVA (B/M). Para medir a associacao
# entre uma variavel quantitativa e uma qualitativa usamos o eta-quadrado (R2),
# que e a medida correta nesse caso (independente do numero de categorias).
#
# Formula (conforme PDF):
#     var_dentro = (n_B * var_B + n_M * var_M) / N
#     R2 = 1 - var_dentro / var_total
# Usamos ddof=0 (variancia populacional) em todos os calculos, de forma
# consistente. Assim o R2 fica no intervalo [0, 1] (nunca negativo).
variaveis = [c for c in colunas if c not in ("id", "diagnostico")]
benignos   = df[df["diagnostico"] == "B"]
malignos   = df[df["diagnostico"] == "M"]
N = len(df)


def eta2(feature):
    var_total  = df[feature].var(ddof=0)
    var_dentro = (len(benignos) * benignos[feature].var(ddof=0) +
                  len(malignos) * malignos[feature].var(ddof=0)) / N
    return 1 - var_dentro / var_total


correlacoes = pd.Series(
    {v: eta2(v) for v in variaveis}
).sort_values(ascending=False)
print("R2 (eta-quadrado) de TODAS as variaveis em relacao ao diagnostico:")
print(correlacoes.round(3).to_string())

# ---------------------------------------------------------------------------
# 5) GRAFICOS SIMPLES
# ---------------------------------------------------------------------------
os.makedirs("imagens", exist_ok=True)

# Grafico 1: distribuicao das classes
plt.figure(figsize=(5, 4))
contagem.rename({"B": "Benigno", "M": "Maligno"}).plot(
    kind="bar", color=["#4CAF50", "#E53935"])
plt.title("Distribuicao das classes")
plt.ylabel("Quantidade de pacientes")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("imagens/distribuicao_classes.png")
plt.close()

# Grafico 2: R2 (eta-quadrado) de todas as variaveis
plt.figure(figsize=(8, 10))
correlacoes.iloc[::-1].plot(kind="barh", color="#1E88E5")
plt.title("Todas as variaveis x R2 (eta-quadrado) com o diagnostico")
plt.xlabel("R2")
plt.tight_layout()
plt.savefig("imagens/correlacoes.png")
plt.close()

print("\nGraficos salvos na pasta 'imagens/'.")
print("Analise exploratoria concluida com sucesso!")
