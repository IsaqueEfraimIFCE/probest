# Trabalho Final - Probabilidade e Estatística
## Diagnóstico de Câncer de Mama com Naive Bayes

Curso de Engenharia da Computação — IFCE
Base de dados: **Breast Cancer Wisconsin Diagnostic (WDBC)**

Este projeto cobre todo o ciclo de um problema de Machine Learning:
analisar os dados → treinar um modelo → salvar o modelo → usar o modelo
em uma aplicação web.

---

## 1. O que cada arquivo faz

| Arquivo | Função |
|---|---|
| `01_analise_exploratoria.py` | Análise exploratória: estatísticas descritivas, distribuição das classes e correlações. Gera gráficos na pasta `imagens/`. |
| `02_treinar_modelo.py` | Treina o modelo **Naive Bayes**, avalia (acurácia, precisão, recall, F1, matriz de confusão) e salva o modelo em `modelo_naive_bayes.joblib`. |
| `app.py` | Aplicação web em **Flask** (back-end): carrega o modelo e faz as previsões. |
| `templates/index.html` | Página web com o formulário (front-end). |
| `static/style.css` | Estilo (cores e layout) da página. |
| `modelo_naive_bayes.joblib` | Modelo treinado salvo (gerado pelo script de treino). |
| `breast+cancer+wisconsin+diagnostic/` | Base de dados original (`wdbc.data` e `wdbc.names`). |

---

## 2. Como executar (passo a passo)

### Passo 0 — Instalar as bibliotecas
```bash
pip install -r requirements.txt
```

### Passo 1 — Análise exploratória (opcional, mas recomendado)
```bash
python 01_analise_exploratoria.py
```
Mostra as estatísticas no terminal e salva gráficos em `imagens/`.

### Passo 2 — Treinar e salvar o modelo
```bash
python 02_treinar_modelo.py
```
Treina o Naive Bayes, mostra o desempenho e cria o arquivo
`modelo_naive_bayes.joblib`.

### Passo 3 — Rodar a aplicação web
```bash
python app.py
```
Depois abra no navegador: **http://127.0.0.1:5050**
Preencha o formulário e clique em **Prever**.

---

## 3. Sobre a base de dados

- **569** pacientes (observações).
- **30** variáveis numéricas, calculadas a partir da imagem do núcleo das
  células de um exame (punção por agulha fina).
- As 30 variáveis vêm de **10 medidas básicas** (raio, textura, perímetro,
  área, suavidade, compacidade, concavidade, pontos côncavos, simetria e
  dimensão fractal), cada uma em **3 versões**: média (*mean*), erro
  padrão (*se*) e pior valor (*worst*).
- **Alvo (diagnóstico):** `B` = Benigno, `M` = Maligno.
- **Distribuição das classes:** 357 Benignos (62,7%) e 212 Malignos (37,3%).
- **Não há valores faltantes.**

---

## 4. Seleção das variáveis (features)

Das 30 variáveis, escolhemos as **8 com maior R² (eta-quadrado) com o
diagnóstico** (calculado em `r2_features.py`). Quanto mais perto de 1,
melhor a variável separa Benigno de Maligno:

| Variável | R² |
|---|---|
| `pontos_concavos_pior` | 0,630 |
| `perimetro_pior` | 0,613 |
| `pontos_concavos_media` | 0,603 |
| `raio_pior` | 0,603 |
| `perimetro_media` | 0,552 |
| `area_pior` | 0,538 |
| `raio_media` | 0,533 |
| `area_media` | 0,503 |

Usar poucas variáveis bem escolhidas mantém o modelo simples, deixa o
formulário da web pequeno e ainda assim alcança ótimo desempenho.

---

## 5. Pré-processamento

A base já é "limpa", então o pré-processamento foi mínimo:
- A coluna **ID** foi descartada (não ajuda no diagnóstico).
- O alvo foi convertido em número: **Maligno = 1**, **Benigno = 0**.
- Os dados foram divididos em **80% treino** e **20% teste**
  (com `stratify` para manter a proporção das classes).
- Naive Bayes Gaussiano **não exige normalização**, então não foi necessário
  padronizar as variáveis.

---

## 6. Modelo e resultados

- **Algoritmo:** Naive Bayes Gaussiano (`GaussianNB` do scikit-learn).
  Ele assume que as variáveis são independentes e seguem uma distribuição
  Normal dentro de cada classe; usa o **Teorema de Bayes** para calcular a
  probabilidade de cada classe.

Resultados no conjunto de teste (114 pacientes):

| Métrica | Valor |
|---|---|
| Acurácia | ~88,6% |
| Precisão | ~0,94 |
| Recall | ~0,74 |
| F1-Score | ~0,83 |

A **matriz de confusão** é salva em `imagens/matriz_confusao.png`.

---

## 7. Fluxo da aplicação

1. O usuário acessa a página.
2. Preenche as 8 medidas no formulário.
3. Clica em **Prever**.
4. O back-end (Flask) recebe os dados e carrega o modelo salvo.
5. O modelo Naive Bayes faz a previsão.
6. A página mostra **Tumor Benigno** ou **Tumor Maligno**, com a
   probabilidade (confiança) associada.

