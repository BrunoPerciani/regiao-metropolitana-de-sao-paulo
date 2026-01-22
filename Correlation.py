# Import libraries for data handling, math operations, plotting, regression, and correlation metrics.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, spearmanr
from pathlib import Path

# Define the CSV file path.
caminho_csv = r"Paste your path here" 

# Create a Path object to validate that the file exists before reading.
p = Path(caminho_csv)
# Stop execution with a clear error if the CSV path is invalid or missing.
if not p.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {caminho_csv}")

# Read the CSV into a DataFrame for analysis.
dados = pd.read_csv(caminho_csv)
# Replace infinite values with NaN and drop rows missing the required columns (NDVI and DP).
dados = dados.replace([np.inf, -np.inf], np.nan).dropna(subset=["NDVI", "DP"])
# Filter NDVI to its valid physical range between -1 and 1.
dados = dados[(dados["NDVI"] >= -1) & (dados["NDVI"] <= 1)]
# Remove non-positive population density values to avoid invalid interpretations.
dados = dados[dados["DP"] > 0]

# Map each DP value into a density class based on predefined thresholds.
def classificar_dp(dp):
    if dp < 47:
        return "Baixa (DP < 47)"
    elif dp < 95:
        return "Média (47 ≤ DP < 95)"
    else:
        return "Alta (DP ≥ 95)"

# Create a new column assigning each record to a DP class label.
dados["classe_DP"] = dados["DP"].apply(classificar_dp)

# Define the desired ordering for the class panels and summary table.
ordem = {"Baixa (DP < 47)": 0, "Média (47 ≤ DP < 95)": 1, "Alta (DP ≥ 95)": 2}
# Group data by the DP class to generate one subplot per class.
grupos = list(dados.groupby("classe_DP"))
# Sort groups according to the predefined class order to keep plots consistent.
grupos.sort(key=lambda kv: ordem.get(kv[0], 99))

# Create three side-by-side subplots sharing the same y-axis for direct NDVI comparison.
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
# Initialize a list to store per-class statistics for the final summary table.
resultados = []  # vamos preencher para a tabela resumo

# Loop through each class group, computing correlations and plotting scatter + trend line.
for ax, (classe, grupo) in zip(axes, grupos):
    # Prepare the independent variable (DP) as a 2D array for scikit-learn.
    X = grupo["DP"].values.reshape(-1, 1)
    # Extract NDVI as the dependent variable.
    y = grupo["NDVI"].values

    # Compute Pearson correlation (linear association) between DP and NDVI.
    r_p, p_p = pearsonr(grupo["DP"], grupo["NDVI"])
    # Compute Spearman correlation (rank-based monotonic association) between DP and NDVI.
    r_s, p_s = spearmanr(grupo["DP"], grupo["NDVI"])

    # Store class-level results to generate a consolidated correlation summary table later.
    resultados.append(
        {"Classe": classe, "n": len(grupo),
         "Pearson_r": r_p, "Pearson_p": p_p,
         "Spearman_rho": r_s, "Spearman_p": p_s}
    )

    # Fit a linear regression model to estimate the trend line within this class.
    modelo = LinearRegression().fit(X, y)
    # Create a smooth DP grid spanning the class range to plot a continuous trend line.
    x_grid = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    # Predict NDVI values along the DP grid using the fitted regression model.
    y_grid = modelo.predict(x_grid)

    # Plot the scatter of NDVI vs DP with transparency to reduce overplotting.
    ax.scatter(X, y, alpha=0.35, color="teal", edgecolors="none", s=12, label="Dados")

    # Plot the fitted trend line and include Pearson R in the legend label.
    ax.plot(x_grid, y_grid, color="black", linewidth=3,
            label=f"Tendência linear (R = {r_p:.2f})")

    # Set subplot title with class name and sample size for quick interpretation.
    ax.set_title(f"{classe} (N = {len(grupo)})")
    # Label the x-axis with population density units.
    ax.set_xlabel("Densidade populacional (Hab/Ha)")
    # Fix y-limits to the valid NDVI range for consistent visual scaling.
    ax.set_ylim(-1, 1)
    # Enable a light grid to help read values without cluttering the plot.
    ax.grid(True, alpha=0.3)
    # Display the legend describing points and the fitted trend line.
    ax.legend()

# Label the shared y-axis only once to keep the layout clean.
axes[0].set_ylabel("NDVI")
# Add an overall title describing the relationship being analyzed across classes.
plt.suptitle("NDVI × Densidade populacional por classe", fontsize=14)
# Adjust spacing to avoid overlap between titles, labels, and panels.
plt.tight_layout()
# Render the figure window.
plt.show()

# Build a summary DataFrame selecting the key metrics for each class.
tabela = pd.DataFrame(resultados)[
    ["Classe", "n", "Pearson_r", "Pearson_p", "Spearman_rho", "Spearman_p"]
].sort_values(by="Classe", key=lambda s: s.map(ordem))

# Print a readable header before showing the correlation summary in the console.
print("\nResumo das correlações por classe:")
# Print the table with consistent float formatting for easier comparison across classes.
print(tabela.to_string(index=False, float_format=lambda v: f"{v:.6f}" if isinstance(v, float) else str(v)))
