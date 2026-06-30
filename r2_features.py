import pandas as pd

FEATURE_NAMES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean",
    "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se", "concave_points_se",
    "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst", "concave_points_worst",
    "symmetry_worst", "fractal_dimension_worst",
]

COLUMNS = ["id", "diagnosis"] + FEATURE_NAMES

df = pd.read_csv("wdbc.data", header=None, names=COLUMNS)

benign    = df[df["diagnosis"] == "B"]
malignant = df[df["diagnosis"] == "M"]
N = len(df)


def eta2(feature):
    """
    Eta-quadrado: medida correta para associacao entre variavel quantitativa
    e variavel qualitativa (independente do numero de categorias).

    Formula (conforme PDF):
        var_dentro = (n_B * var_B + n_M * var_M) / N
        eta2 = 1 - var_dentro / var_total

    Usa ddof=1 em todos os calculos (variancia amostral), de forma consistente.
    """
    var_total  = df[feature].var(ddof=1)
    var_dentro = (len(benign)    * benign[feature].var(ddof=1) +
                  len(malignant) * malignant[feature].var(ddof=1)) / N
    return 1 - var_dentro / var_total


resultados = sorted(
    [(f, eta2(f)) for f in FEATURE_NAMES],
    key=lambda x: x[1],
    reverse=True
)

print("=" * 55)
print("Eta-quadrado (R2) — Breast Cancer Wisconsin")
print("Metodo: 1 - var_dentro / var_total  (ddof=1)")
print("=" * 55)
print("{:<4} {:<30} {:>8}".format("Rank", "Feature", "R2"))
print("-" * 55)
for i, (feature, r2) in enumerate(resultados, 1):
    print("{:<4} {:<30} {:>8.6f}".format(i, feature, r2))
print("=" * 55)

results_df = pd.DataFrame(resultados, columns=["feature", "r2"])
results_df.index = range(1, len(results_df) + 1)
results_df.to_csv("r2_features.csv", index_label="rank")
print("\nResultados salvos em 'r2_features.csv'.")
