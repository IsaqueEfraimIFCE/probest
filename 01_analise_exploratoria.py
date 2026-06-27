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
print("A base e razoavelmente equilibrada (nao ha classe muito rara).")

# ---------------------------------------------------------------------------
# 4) CORRELACAO COM O DIAGNOSTICO -> JUSTIFICATIVA DA ESCOLHA DAS FEATURES
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("4) VARIAVEIS MAIS RELACIONADAS COM O DIAGNOSTICO")
print("=" * 70)
# Transformamos o alvo em numero: Maligno = 1, Benigno = 0
df["alvo_num"] = (df["diagnostico"] == "M").astype(int)
# Correlacao (em modulo) de cada variavel com o alvo.
# Quanto mais perto de 1, mais a variavel ajuda a separar as classes.
variaveis = [c for c in colunas if c not in ("id", "diagnostico")]
correlacoes = df[variaveis].corrwith(df["alvo_num"]).abs().sort_values(ascending=False)
print("Top 12 variaveis mais correlacionadas com o diagnostico:")
print(correlacoes.head(12).round(3))

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

# Grafico 2: top variaveis correlacionadas
plt.figure(figsize=(7, 5))
correlacoes.head(10).iloc[::-1].plot(kind="barh", color="#1E88E5")
plt.title("Top 10 variaveis x correlacao com o diagnostico")
plt.xlabel("Correlacao (em modulo)")
plt.tight_layout()
plt.savefig("imagens/correlacoes.png")
plt.close()

print("\nGraficos salvos na pasta 'imagens/'.")
print("Analise exploratoria concluida com sucesso!")
