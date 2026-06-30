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


def r2(feature):
    var_total  = df[feature].var(ddof=0)
    var_dentro = (len(benign)    * benign[feature].var(ddof=0) +
                  len(malignant) * malignant[feature].var(ddof=0)) / N
    return 1 - var_dentro / var_total


resultados = sorted(
    [(f, r2(f)) for f in FEATURE_NAMES],
    key=lambda x: x[1],
    reverse=True
)

print("=" * 55)
print("R2 — Breast Cancer Wisconsin")
print("Metodo: 1 - var_dentro / var_total  (ddof=0)")
print("=" * 55)
print("{:<4} {:<30} {:>8}".format("Rank", "Feature", "R2"))
print("-" * 55)
for i, (feature, val) in enumerate(resultados, 1):
    print("{:<4} {:<30} {:>8.6f}".format(i, feature, val))
print("=" * 55)

results_df = pd.DataFrame(resultados, columns=["feature", "r2"])
results_df.index = range(1, len(results_df) + 1)
results_df.to_csv("r2_features_ddof0.csv", index_label="rank")
print("\nResultados salvos em 'r2_features_ddof0.csv'.")
