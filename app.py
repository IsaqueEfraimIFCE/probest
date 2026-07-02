# -*- coding: utf-8 -*-
"""
app.py
------
Aplicacao web (back-end) usando Flask.

O que faz:
  1. Carrega o modelo Naive Bayes salvo (modelo_naive_bayes.joblib).
  2. Mostra uma pagina HTML com um formulario.
  3. Recebe os dados digitados pelo usuario.
  4. Usa o modelo para prever: Tumor Benigno ou Tumor Maligno.
  5. Mostra o resultado (e a probabilidade) na tela.

Para rodar:  python app.py
Depois abra no navegador: http://127.0.0.1:5050
"""

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Carrega o modelo treinado uma unica vez, quando a aplicacao inicia.
pacote = joblib.load("modelo_naive_bayes.joblib")
modelo = pacote["modelo"]
FEATURES = pacote["features"]

# Rotulo amigavel de cada feature, usado nos titulos dos campos do formulario.
CAMPOS = {
    "pontos_concavos_pior":  "Pontos concavos (pior)",
    "perimetro_pior":        "Perimetro (pior)",
    "pontos_concavos_media": "Pontos concavos (media)",
    "raio_pior":             "Raio (pior)",
    "perimetro_media":       "Perimetro (media)",
    "area_pior":             "Area (pior)",
    "raio_media":            "Raio (media)",
    "area_media":            "Area (media)",
}


@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None      # texto: "Benigno" ou "Maligno"
    erro = None           # mensagem de erro, se houver
    valores = {}          # guarda o que o usuario digitou (para nao apagar)

    if request.method == "POST":
        try:
            # 1) Le os valores do formulario na ORDEM correta das features.
            entrada = []
            for f in FEATURES:
                valor_texto = request.form.get(f, "").replace(",", ".")
                valores[f] = valor_texto
                entrada.append(float(valor_texto))

            # 2) Faz a previsao. Montamos uma tabela (DataFrame) com os nomes
            #    das colunas iguais aos do treino.
            entrada_df = pd.DataFrame([entrada], columns=FEATURES)
            previsao = modelo.predict(entrada_df)[0]          # 0 = Benigno, 1 = Maligno

            # 3) Traduz o resultado para texto.
            resultado = "Maligno" if previsao == 1 else "Benigno"

        except ValueError:
            erro = "Por favor, preencha todos os campos com numeros validos."

    return render_template(
        "index.html",
        campos=CAMPOS,
        resultado=resultado,
        erro=erro,
        valores=valores,
    )


if __name__ == "__main__":
    # debug=True facilita o desenvolvimento (mostra erros e recarrega sozinho).
    # threaded=True permite atender varias conexoes ao mesmo tempo. Sem isso,
    # o navegador (que mantem a conexao aberta) pode travar a pagina no Windows.
    # use_reloader=False evita o "processo filho" do reloader, que no Windows
    # (especialmente rodando pelo Thonny) costuma ficar orfao segurando a porta
    # 5000 e faz a pagina ficar "carregando" para sempre.
    app.run(host="127.0.0.1", port=5050, debug=True, threaded=True, use_reloader=False)
