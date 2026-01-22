# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, spearmanr
from pathlib import Path

# Caminho do CSV (ajuste se necessário)
caminho_csv = r"Amostra.csv" 

# 1) Leitura e limpeza básica
p = Path(caminho_csv)
if not p.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {caminho_csv}")

dados = pd.read_csv(caminho_csv)
dados = dados.replace([np.inf, -np.inf], np.nan).dropna(subset=["NDVI", "DP"])
dados = dados[(dados["NDVI"] >= -1) & (dados["NDVI"] <= 1)]
dados = dados[dados["DP"] > 0]

# 2) Classificação por faixas de densidade
def classificar_dp(dp):
    if dp < 47:
        return "Baixa (DP < 47)"
    elif dp < 95:
        return "Média (47 ≤ DP < 95)"
    else:
        return "Alta (DP ≥ 95)"

dados["classe_DP"] = dados["DP"].apply(classificar_dp)

# 3) Ordenação das classes
ordem = {"Baixa (DP < 47)": 0, "Média (47 ≤ DP < 95)": 1, "Alta (DP ≥ 95)": 2}
grupos = list(dados.groupby("classe_DP"))
grupos.sort(key=lambda kv: ordem.get(kv[0], 99))

# 4) Plots lado a lado com linha de tendência destacada e r de Pearson
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
resultados = []  # vamos preencher para a tabela resumo

for ax, (classe, grupo) in zip(axes, grupos):
    X = grupo["DP"].values.reshape(-1, 1)
    y = grupo["NDVI"].values

    # Correlações
    r_p, p_p = pearsonr(grupo["DP"], grupo["NDVI"])
    r_s, p_s = spearmanr(grupo["DP"], grupo["NDVI"])

    # Guarda resultados para a tabela
    resultados.append(
        {"Classe": classe, "n": len(grupo),
         "Pearson_r": r_p, "Pearson_p": p_p,
         "Spearman_rho": r_s, "Spearman_p": p_s}
    )

    # Regressão linear para a linha de tendência
    modelo = LinearRegression().fit(X, y)
    x_grid = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    y_grid = modelo.predict(x_grid)

    # Dispersão
    ax.scatter(X, y, alpha=0.35, color="teal", edgecolors="none", s=12, label="Dados")

    # Linha de tendência destacada + R de Pearson no rótulo
    ax.plot(x_grid, y_grid, color="black", linewidth=3,
            label=f"Tendência linear (R = {r_p:.2f})")

    # Layout
    ax.set_title(f"{classe} (N = {len(grupo)})")
    ax.set_xlabel("Densidade populacional (Hab/Ha)")
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

axes[0].set_ylabel("NDVI")
plt.suptitle("NDVI × Densidade populacional por classe", fontsize=14)
plt.tight_layout()
plt.show()

# 5) Tabela resumo
tabela = pd.DataFrame(resultados)[
    ["Classe", "n", "Pearson_r", "Pearson_p", "Spearman_rho", "Spearman_p"]
].sort_values(by="Classe", key=lambda s: s.map(ordem))

print("\nResumo das correlações por classe:")
print(tabela.to_string(index=False, float_format=lambda v: f"{v:.6f}" if isinstance(v, float) else str(v)))

