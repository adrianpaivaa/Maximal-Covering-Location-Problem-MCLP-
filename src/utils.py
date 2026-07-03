"""
Funções utilitárias para o projeto MCLP.

Inclui funções para:
- Leitura dos dados de nós e distâncias
- Cálculo da matriz de distâncias
- Visualização de resultados
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def get_data_dir():
    """Retorna o caminho do diretório de dados."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def get_results_dir():
    """Retorna o caminho do diretório de resultados, criando-o se necessário."""
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def load_nodes(filepath=None):
    """
    Carrega os dados dos nós a partir de um arquivo CSV.

    Parâmetros:
        filepath: Caminho para o arquivo CSV dos nós.
                  Se None, usa o arquivo padrão (swain55_nodes.csv).

    Retorna:
        DataFrame com colunas: node_id, x, y, population
    """
    if filepath is None:
        filepath = os.path.join(get_data_dir(), "swain55_nodes.csv")
    return pd.read_csv(filepath)


def load_distance_matrix(filepath=None):
    """
    Carrega a matriz de distâncias a partir de um arquivo CSV.

    Parâmetros:
        filepath: Caminho para o arquivo CSV da matriz de distâncias.
                  Se None, usa o arquivo padrão (swain55_distances.csv).

    Retorna:
        numpy array 2D com as distâncias entre nós
    """
    if filepath is None:
        filepath = os.path.join(get_data_dir(), "swain55_distances.csv")
    return np.loadtxt(filepath, delimiter=",")


def compute_distance_matrix(nodes_df):
    """
    Calcula a matriz de distâncias euclidianas a partir das coordenadas.

    Parâmetros:
        nodes_df: DataFrame com colunas 'x' e 'y'

    Retorna:
        numpy array 2D com as distâncias euclidianas
    """
    coords = nodes_df[["x", "y"]].values
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i][j] = np.sqrt(
                (coords[i][0] - coords[j][0]) ** 2
                + (coords[i][1] - coords[j][1]) ** 2
            )
    return np.round(dist, 2)


def compute_coverage_sets(dist_matrix, S):
    """
    Calcula os conjuntos de cobertura N_i para cada nó de demanda.

    N_i = {j ∈ J | d_ij ≤ S} — conjunto de locais candidatos que
    podem cobrir o nó de demanda i dentro da distância S.

    Parâmetros:
        dist_matrix: Matriz de distâncias (numpy array)
        S: Distância máxima de serviço

    Retorna:
        Dicionário {i: [lista de j's que cobrem i]}
    """
    n = dist_matrix.shape[0]
    N = {}
    for i in range(n):
        N[i] = [j for j in range(n) if dist_matrix[i][j] <= S]
    return N


def plot_cost_effectiveness_curve(p_values, coverage_values, total_pop,
                                   S, title=None, save_path=None):
    """
    Plota a curva de custo-efetividade: cobertura vs número de instalações.

    Parâmetros:
        p_values: Lista de valores de p (número de instalações)
        coverage_values: Lista de populações cobertas correspondentes
        total_pop: População total
        S: Distância de serviço usada
        title: Título do gráfico
        save_path: Caminho para salvar o gráfico (opcional)
    """
    pct_coverage = [c / total_pop * 100 for c in coverage_values]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Eixo esquerdo: população coberta (absoluta)
    color1 = "#2196F3"
    ax1.set_xlabel("Número de Instalações (p)", fontsize=12)
    ax1.set_ylabel("População Coberta", fontsize=12, color=color1)
    ax1.plot(p_values, coverage_values, "o-", color=color1, linewidth=2,
             markersize=8, label="População coberta")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, total_pop * 1.1)

    # Eixo direito: percentual de cobertura
    ax2 = ax1.twinx()
    color2 = "#FF5722"
    ax2.set_ylabel("Cobertura (%)", fontsize=12, color=color2)
    ax2.plot(p_values, pct_coverage, "s--", color=color2, linewidth=1.5,
             markersize=6, alpha=0.7, label="Cobertura (%)")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 110)

    # Linha horizontal em 100%
    ax2.axhline(y=100, color="gray", linestyle=":", alpha=0.5)

    if title is None:
        title = f"Curva de Custo-Efetividade (S = {S})"
    plt.title(title, fontsize=14, fontweight="bold")

    ax1.set_xticks(p_values)
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gráfico salvo em: {save_path}")

    plt.close()


def plot_network_solution(nodes_df, dist_matrix, facility_sites, S,
                          T=None, title=None, save_path=None):
    """
    Visualiza a rede com a solução de localização.

    Parâmetros:
        nodes_df: DataFrame com dados dos nós
        dist_matrix: Matriz de distâncias
        facility_sites: Lista de índices (0-based) dos locais com instalações
        S: Distância de serviço desejável
        T: Distância de cobertura obrigatória (opcional)
        title: Título do gráfico
        save_path: Caminho para salvar (opcional)
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    x = nodes_df["x"].values
    y = nodes_df["y"].values
    pop = nodes_df["population"].values

    # Classificar nós
    covered_S = set()
    covered_T = set()
    uncovered = set()

    for i in range(len(nodes_df)):
        dist_to_nearest = min(dist_matrix[i][j] for j in facility_sites)
        if dist_to_nearest <= S:
            covered_S.add(i)
        elif T is not None and dist_to_nearest <= T:
            covered_T.add(i)
        else:
            uncovered.add(i)

    # Plotar nós cobertos dentro de S
    if covered_S:
        idx = list(covered_S)
        ax.scatter(x[idx], y[idx], s=pop[idx] * 10, c="#4CAF50", alpha=0.7,
                   edgecolors="black", linewidth=0.5,
                   label=f"Coberto (d ≤ {S})", zorder=3)

    # Plotar nós cobertos dentro de T (mas fora de S)
    if covered_T:
        idx = list(covered_T)
        ax.scatter(x[idx], y[idx], s=pop[idx] * 10, c="#FFC107", alpha=0.7,
                   edgecolors="black", linewidth=0.5,
                   label=f"Coberto ({S} < d ≤ {T})", zorder=3)

    # Plotar nós não cobertos
    if uncovered:
        idx = list(uncovered)
        ax.scatter(x[idx], y[idx], s=pop[idx] * 10, c="#F44336", alpha=0.7,
                   edgecolors="black", linewidth=0.5,
                   label="Não coberto", zorder=3)

    # Plotar instalações
    ax.scatter(x[facility_sites], y[facility_sites], s=200, c="blue",
               marker="*", edgecolors="black", linewidth=1,
               label="Instalação", zorder=5)

    # Círculos de cobertura S
    for j in facility_sites:
        circle_S = plt.Circle((x[j], y[j]), S, fill=False, color="#4CAF50",
                              linestyle="--", alpha=0.4, linewidth=1.5)
        ax.add_patch(circle_S)

    # Círculos de cobertura T
    if T is not None:
        for j in facility_sites:
            circle_T = plt.Circle((x[j], y[j]), T, fill=False, color="#FFC107",
                                  linestyle=":", alpha=0.3, linewidth=1)
            ax.add_patch(circle_T)

    # Labels dos nós
    for i in range(len(nodes_df)):
        ax.annotate(str(i + 1), (x[i], y[i]), fontsize=6,
                    ha="center", va="bottom", xytext=(0, 5),
                    textcoords="offset points")

    # Estatísticas
    pop_covered_S = sum(pop[i] for i in covered_S)
    pop_covered_T = sum(pop[i] for i in covered_T)
    pop_uncovered = sum(pop[i] for i in uncovered)
    total_pop = sum(pop)

    stats_text = (
        f"Pop. coberta (S≤{S}): {pop_covered_S}/{total_pop} "
        f"({pop_covered_S/total_pop*100:.1f}%)"
    )
    if T is not None:
        stats_text += (
            f"\nPop. coberta ({S}<d≤{T}): {pop_covered_T}"
            f"\nPop. não coberta: {pop_uncovered}"
        )
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if title is None:
        title = f"Solução MCLP — {len(facility_sites)} instalações, S={S}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gráfico salvo em: {save_path}")

    plt.close()
