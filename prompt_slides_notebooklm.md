# Prompt para NotebookLM — Apresentação de 15 slides

> Cole o texto abaixo no NotebookLM (na função de criação de apresentação/visão geral em slides).
> Os arquivos do projeto (PDF do trabalho, README, scripts `.py`, imagens da pasta `imagens/`)
> devem estar adicionados como fontes do notebook para que o conteúdo seja fiel.

---

## INSTRUÇÃO PARA O NOTEBOOKLM

Crie uma apresentação de slides com **exatamente 15 slides**, em **português do Brasil**, com tom
**acadêmico, claro e didático**, adequada para a apresentação oral de um trabalho final de faculdade.
Use as fontes adicionadas (especificação do trabalho, README e código-fonte) como base e **não invente
dados** — use apenas os números e fatos presentes nas fontes.

Para cada slide gere: um **título curto**, de **3 a 5 bullets objetivos** (frases curtas, não parágrafos)
e uma **nota do apresentador** (1 a 2 frases) com o que falar. Quando houver imagem disponível na pasta
`imagens/`, indique qual usar. Mantenha consistência visual: tema limpo, cores que remetam a saúde/dados
(azul e verde), fonte legível.

Tema do trabalho: **Diagnóstico de Câncer de Mama com Naive Bayes** — Probabilidade e Estatística,
Engenharia da Computação, IFCE (Prof. Valberto Feitosa). Base: Breast Cancer Wisconsin Diagnostic (WDBC).

Estruture os 15 slides exatamente assim:

---

**Slide 1 — Capa**
- Título: "Diagnóstico de Câncer de Mama com Naive Bayes"
- Subtítulo: Trabalho Final de Probabilidade e Estatística — IFCE
- Curso de Engenharia da Computação / Prof. Valberto Feitosa
- Nomes da equipe e data
- Nota: apresentar o tema e que o projeto cobre o ciclo completo de Machine Learning.

**Slide 2 — Objetivo do trabalho**
- Desenvolver uma solução completa de Machine Learning, da análise dos dados à aplicação web
- Etapas: analisar dados → selecionar features → treinar Naive Bayes → salvar modelo → app web
- Problema: classificar tumores como Benigno ou Maligno
- Nota: explicar que o foco é o ciclo completo, não apenas o modelo.

**Slide 3 — A base de dados (WDBC)**
- 569 pacientes (observações) e 30 variáveis numéricas + 1 alvo (diagnóstico)
- Variáveis extraídas de imagem do núcleo celular (punção por agulha fina)
- Alvo: B = Benigno, M = Maligno
- Sem valores faltantes
- Nota: citar a fonte oficial (UCI Machine Learning Repository).

**Slide 4 — As 30 variáveis**
- 10 medidas básicas: raio, textura, perímetro, área, suavidade, compacidade, concavidade, pontos côncavos, simetria, dimensão fractal
- Cada medida em 3 versões: média (mean), erro padrão (se) e pior valor (worst)
- 10 × 3 = 30 variáveis preditoras
- Nota: explicar de onde vêm as 30 colunas do dataset.

**Slide 5 — Análise exploratória (EDA)**
- Estatísticas descritivas (média, desvio, mín/máx, quartis) via `describe()`
- Verificação de valores faltantes (nenhum)
- Script: `01_analise_exploratoria.py`
- Nota: mostrar que os dados foram compreendidos antes de modelar.

**Slide 6 — Distribuição das classes**
- 357 Benignos (62,7%) e 212 Malignos (37,3%)
- Base razoavelmente equilibrada (sem classe muito rara)
- Imagem: `imagens/distribuicao_classes.png`
- Nota: justificar por que não houve necessidade de balanceamento.

**Slide 7 — Correlação com o diagnóstico**
- Alvo convertido em número (Maligno = 1, Benigno = 0)
- Correlação de cada variável com o diagnóstico (quanto mais perto de 1, melhor separa as classes)
- Imagem: `imagens/correlacoes.png` (top 10 variáveis)
- Nota: esta análise embasou a escolha das features.

**Slide 8 — Box plots por classe**
- 30 features padronizadas (z-score) comparadas Benigno × Maligno
- Caixas bem separadas = feature que distingue bem as classes
- Imagem: `imagens/boxplots_todas_features.png`
- Nota: reforço visual da capacidade de separação das variáveis.

**Slide 9 — Seleção e justificativa das features**
- Das 30, foram escolhidas as 6 de maior correlação com o diagnóstico:
- pontos_concavos_pior (0,794), perimetro_pior (0,783), pontos_concavos_media (0,777)
- raio_pior (0,776), perimetro_media (0,743), area_pior (0,734)
- Modelo simples + formulário enxuto + ótimo desempenho
- Nota: justificar a escolha com base estatística, não arbitrária.

**Slide 10 — Pré-processamento**
- Coluna ID descartada (não ajuda no diagnóstico)
- Alvo convertido: Maligno = 1, Benigno = 0
- Divisão 80% treino / 20% teste com `stratify` (mantém proporção das classes)
- GaussianNB não exige normalização → sem padronização no modelo
- Nota: explicar por que o pré-processamento foi mínimo.

**Slide 11 — O modelo Naive Bayes**
- Algoritmo: Naive Bayes Gaussiano (`GaussianNB` do scikit-learn)
- Baseado no Teorema de Bayes; assume variáveis independentes e com distribuição Normal por classe
- Treinado em `02_treinar_modelo.py`
- Nota: conectar com o conteúdo de Probabilidade e Estatística (Teorema de Bayes).

**Slide 12 — Avaliação do modelo**
- Resultados no teste (114 pacientes): Acurácia ~93,9%
- Precisão ~0,95 | Recall ~0,88 | F1-Score ~0,91
- Métricas calculadas com scikit-learn
- Nota: explicar cada métrica em uma frase (acurácia, precisão, recall, F1).

**Slide 13 — Matriz de confusão**
- Compara o real (Benigno/Maligno) com o previsto
- Mostra acertos e os tipos de erro (falsos positivos / falsos negativos)
- Imagem: `imagens/matriz_confusao.png`
- Nota: comentar a importância de minimizar falsos negativos em diagnóstico.

**Slide 14 — Aplicação web (Flask)**
- Modelo salvo com joblib (`modelo_naive_bayes.joblib`) e carregado no `app.py`
- Front-end: página HTML com formulário das 6 variáveis e botão "Prever"
- Back-end (Flask): recebe os dados, aplica o modelo e retorna o resultado
- Fluxo: usuário preenche → Prever → "Tumor Benigno" ou "Tumor Maligno" (com probabilidade)
- Nota: demonstrar o app rodando em http://127.0.0.1:5050, se possível.

**Slide 15 — Conclusão**
- Ciclo completo de ML realizado: EDA → seleção de features → treino → modelo salvo → app web
- Naive Bayes alcançou ~94% de acurácia com apenas 6 variáveis
- Projeto educacional — não substitui diagnóstico médico real
- Tecnologias: Python, scikit-learn, Pandas, joblib, Flask, HTML/CSS
- Nota: encerrar reforçando o aprendizado e abrir para perguntas.

---

### Diretrizes finais
- Mantenha cada slide enxuto (texto que caiba na tela, sem parágrafos longos).
- Use os números exatamente como nas fontes (569, 30, 357/212, 93,9%, etc.).
- Prefira verbos no passado para o que foi feito ("selecionamos", "treinamos", "avaliamos").
- Inclua as imagens citadas nos slides 6, 7, 8 e 13.
