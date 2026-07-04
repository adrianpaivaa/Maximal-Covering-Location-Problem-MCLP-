"""
Funções auxiliares para o projeto MCLP.

Contém funções para:
    - Leitura dos CSVs com dados da rede
    - Cálculo da matriz de distâncias euclidianas
    - Geração dos gráficos de custo-efetividade
    - Plotagem da rede com solução de localização
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def diretorio_dados():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def diretorio_resultados():
    caminho = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resultados")
    os.makedirs(caminho, exist_ok=True)
    return caminho


def carregar_nos(caminho=None):
    """Carrega os dados dos nós a partir do CSV."""
    if caminho is None:
        caminho = os.path.join(diretorio_dados(), "swain55_nodes.csv")
    return pd.read_csv(caminho)


def carregar_matriz_distancias(caminho=None):
    """Carrega a matriz de distâncias a partir do CSV."""
    if caminho is None:
        caminho = os.path.join(diretorio_dados(), "swain55_distances.csv")
    return np.loadtxt(caminho, delimiter=",")


def calcular_matriz_distancias(df_nos):
    """Calcula a matriz de distâncias euclidianas entre os nós."""
    coords = df_nos[["x", "y"]].values
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i][j] = np.sqrt(
                (coords[i][0] - coords[j][0]) ** 2
                + (coords[i][1] - coords[j][1]) ** 2
            )
    return np.round(dist, 2)


def calcular_conjuntos_cobertura(mat_dist, S):
    """Calcula Ni = {j | d_ij <= S} para cada nó i."""
    n = mat_dist.shape[0]
    Ni = {}
    for i in range(n):
        Ni[i] = [j for j in range(n) if mat_dist[i][j] <= S]
    return Ni


def plotar_curva_custo_efetividade(valores_p, valores_cobertura, pop_total,
                                    S, titulo=None, caminho_salvar=None):
    """Plota a curva de custo-efetividade (cobertura x número de instalações)."""
    pct_cobertura = [c / pop_total * 100 for c in valores_cobertura]

    fig, eixo1 = plt.subplots(figsize=(10, 6))

    cor1 = "#2196F3"
    eixo1.set_xlabel("Número de Instalações (p)", fontsize=12)
    eixo1.set_ylabel("População Coberta", fontsize=12, color=cor1)
    eixo1.plot(valores_p, valores_cobertura, "o-", color=cor1, linewidth=2,
               markersize=8, label="População coberta")
    eixo1.tick_params(axis="y", labelcolor=cor1)
    eixo1.set_ylim(0, pop_total * 1.1)

    eixo2 = eixo1.twinx()
    cor2 = "#FF5722"
    eixo2.set_ylabel("Cobertura (%)", fontsize=12, color=cor2)
    eixo2.plot(valores_p, pct_cobertura, "s--", color=cor2, linewidth=1.5,
               markersize=6, alpha=0.7, label="Cobertura (%)")
    eixo2.tick_params(axis="y", labelcolor=cor2)
    eixo2.set_ylim(0, 110)

    eixo2.axhline(y=100, color="gray", linestyle=":", alpha=0.5)

    if titulo is None:
        titulo = f"Curva de Custo-Efetividade (S = {S})"
    plt.title(titulo, fontsize=14, fontweight="bold")

    eixo1.set_xticks(valores_p)
    eixo1.grid(True, alpha=0.3)
    fig.tight_layout()

    if caminho_salvar:
        plt.savefig(caminho_salvar, dpi=150, bbox_inches="tight")
        print(f"Gráfico salvo em: {caminho_salvar}")

    plt.close()


def plotar_rede(df_nos, mat_dist, locais_instalacao, S,
                T=None, titulo=None, caminho_salvar=None):
    """Plota a rede com a solução de localização encontrada."""
    fig, ax = plt.subplots(figsize=(12, 10))

    cx = df_nos["x"].values
    cy = df_nos["y"].values
    pop = df_nos["population"].values

    # Classificar cada nó
    cobertos_S = set()
    cobertos_T = set()
    descobertos = set()

    for i in range(len(df_nos)):
        dist_min = min(mat_dist[i][j] for j in locais_instalacao)
        if dist_min <= S:
            cobertos_S.add(i)
        elif T is not None and dist_min <= T:
            cobertos_T.add(i)
        else:
            descobertos.add(i)

    # Nós cobertos em S
    if cobertos_S:
        idx = list(cobertos_S)
        ax.scatter(cx[idx], cy[idx], s=pop[idx] * 10, c="#4CAF50", alpha=0.7,
                   edgecolors="black", linewidth=0.5,
                   label=f"Coberto (d ≤ {S})", zorder=3)

    # Nós cobertos entre S e T
    if cobertos_T:
        idx = list(cobertos_T)
        ax.scatter(cx[idx], cy[idx], s=pop[idx] * 10, c="#FFC107", alpha=0.7,
                   edgecolors="black", linewidth=0.5,
                   label=f"Coberto ({S} < d ≤ {T})", zorder=3)

    # Nós descobertos
    if descobertos:
        idx = list(descobertos)
        ax.scatter(cx[idx], cy[idx], s=pop[idx] * 10, c="#F44336", alpha=0.7,
                   edgecolors="black", linewidth=0.5,
                   label="Não coberto", zorder=3)

    # Instalações
    ax.scatter(cx[locais_instalacao], cy[locais_instalacao], s=200, c="blue",
               marker="*", edgecolors="black", linewidth=1,
               label="Instalação", zorder=5)

    # Círculos de cobertura S
    for j in locais_instalacao:
        circulo = plt.Circle((cx[j], cy[j]), S, fill=False, color="#4CAF50",
                              linestyle="--", alpha=0.4, linewidth=1.5)
        ax.add_patch(circulo)

    # Círculos de cobertura T
    if T is not None:
        for j in locais_instalacao:
            circulo = plt.Circle((cx[j], cy[j]), T, fill=False, color="#FFC107",
                                  linestyle=":", alpha=0.3, linewidth=1)
            ax.add_patch(circulo)

    # Identificar nós
    for i in range(len(df_nos)):
        ax.annotate(str(i + 1), (cx[i], cy[i]), fontsize=6,
                    ha="center", va="bottom", xytext=(0, 5),
                    textcoords="offset points")

    # Estatísticas no gráfico
    pop_cob_S = sum(pop[i] for i in cobertos_S)
    pop_cob_T = sum(pop[i] for i in cobertos_T)
    pop_desc = sum(pop[i] for i in descobertos)
    pop_total = sum(pop)

    texto = (
        f"Pop. coberta (S≤{S}): {pop_cob_S}/{pop_total} "
        f"({pop_cob_S/pop_total*100:.1f}%)"
    )
    if T is not None:
        texto += (
            f"\nPop. coberta ({S}<d≤{T}): {pop_cob_T}"
            f"\nPop. não coberta: {pop_desc}"
        )
    ax.text(0.02, 0.02, texto, transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if titulo is None:
        titulo = f"Solução MCLP — {len(locais_instalacao)} instalações, S={S}"
    ax.set_title(titulo, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    if caminho_salvar:
        plt.savefig(caminho_salvar, dpi=150, bbox_inches="tight")
        print(f"Gráfico salvo em: {caminho_salvar}")

    plt.close()
