"""
Execução dos experimentos computacionais para o trabalho de PO.

Reproduz os experimentos do artigo de Church & ReVelle (1974):
    1. Tabela 1: Comparação de algoritmos (55 nós, vários S e p)
    2. Figura 1: Curva de custo-efetividade (30 nós, S=2.0)
    3. Figuras 2-4: Soluções com mandatory closeness constraints (55 nós)

Autor: Adrian
Disciplina: Pesquisa Operacional
"""

import sys
import os
import time

# Adicionar diretório pai ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from src.mclp_model import (
    solve_mclp,
    solve_mclp_mandatory_closeness,
    solve_mclp_range,
    print_solution,
)
from src.utils import (
    load_nodes,
    load_distance_matrix,
    get_results_dir,
    plot_cost_effectiveness_curve,
    plot_network_solution,
)


def experiment_table1(nodes_df, dist_matrix):
    """
    Reproduz a Tabela 1 do artigo.

    Resolve o MCLP para a rede de 55 nós com diferentes valores de S
    e faixas de p, comparando com os resultados do artigo original.

    No artigo, os valores de S testados são: 10, 12, 15, 19, 20
    com faixas de p variáveis.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTO 1: Reprodução da Tabela 1 (Rede de 55 nós)")
    print("=" * 70)

    populations = nodes_df["population"].values
    total_pop = sum(populations)
    print(f"População total: {total_pop}")
    print(f"Número de nós: {len(nodes_df)}")

    # Configurações dos experimentos conforme a Tabela 1 do artigo
    # (S, p_range) — baseado nos valores mencionados no artigo
    experiments = [
        (10, range(1, 10)),    # S=10, p=1 a 9
        (12, range(1, 8)),     # S=12, p=1 a 7
        (15, range(1, 6)),     # S=15, p=1 a 5
        (19, range(1, 4)),     # S=19, p=1 a 3
        (20, range(1, 4)),     # S=20, p=1 a 3
    ]

    all_results = []
    lp_integer_count = 0
    lp_fractional_count = 0

    for S, p_range in experiments:
        print(f"\n--- S = {S} ---")
        results = solve_mclp_range(dist_matrix, populations, p_range, S)

        for r in results:
            r["total_pop"] = total_pop
            r["pop_uncovered"] = total_pop - r["objective"]
            all_results.append(r)

            if r["is_integer_lp"]:
                lp_integer_count += 1
            else:
                lp_fractional_count += 1

    # Resumo estatístico
    total_problems = lp_integer_count + lp_fractional_count
    pct_integer = lp_integer_count / total_problems * 100
    print(f"\n{'=' * 70}")
    print(f"RESUMO - Relaxação LP:")
    print(f"  Total de problemas resolvidos: {total_problems}")
    print(f"  Soluções inteiras do LP: {lp_integer_count} ({pct_integer:.1f}%)")
    print(f"  Soluções fracionárias do LP: {lp_fractional_count} "
          f"({100 - pct_integer:.1f}%)")
    print(f"  (Artigo original reporta ~80% inteiras)")

    # Salvar resultados
    results_dir = get_results_dir()
    df = pd.DataFrame([{
        "S": r["S"],
        "p": r["p"],
        "pop_coberta": int(r["objective"]),
        "pop_nao_coberta": int(r["pop_uncovered"]),
        "cobertura_pct": round(r["coverage_pct"], 1),
        "lp_inteiro": r["is_integer_lp"],
        "lp_obj": round(r["lp_objective"], 2) if r["lp_objective"] else None,
        "tempo_s": round(r["solve_time"], 4),
        "instalacoes": str([f+1 for f in r["facilities"]]),
    } for r in all_results])

    csv_path = os.path.join(results_dir, "tabela1_resultados.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResultados salvos em: {csv_path}")

    # Exibir tabela formatada
    print(f"\n{'S':>4} {'p':>3} {'Pop.Coberta':>12} {'%Cobertura':>12} "
          f"{'LP Inteiro':>10} {'Tempo(s)':>10}")
    print("-" * 60)
    for _, row in df.iterrows():
        print(f"{row['S']:>4} {row['p']:>3} {row['pop_coberta']:>12} "
              f"{row['cobertura_pct']:>11.1f}% "
              f"{'Sim' if row['lp_inteiro'] else 'Não':>10} "
              f"{row['tempo_s']:>10.4f}")

    return all_results


def experiment_cost_effectiveness(nodes_df, dist_matrix):
    """
    Reproduz a Figura 1 do artigo: curva de custo-efetividade.

    No artigo, a curva é gerada para a rede de 30 nós com S=2.0.
    Como usamos a rede de 55 nós, geramos curvas para S=10 e S=15.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTO 2: Curvas de Custo-Efetividade (55 nós)")
    print("=" * 70)

    populations = nodes_df["population"].values
    total_pop = sum(populations)
    results_dir = get_results_dir()

    # Gerar curvas para diferentes valores de S
    for S in [10, 15]:
        print(f"\n--- Curva para S = {S} ---")

        # Determinar faixa de p: de 1 até cobertura total
        max_p = len(nodes_df)
        p_values = []
        coverage_values = []

        for p in range(1, max_p + 1):
            result = solve_mclp(dist_matrix, populations, p, S)
            p_values.append(p)
            coverage_values.append(result["objective"])
            print(f"  p={p}: cobertura = {result['objective']:.0f} "
                  f"({result['coverage_pct']:.1f}%)")

            # Parar quando atingir cobertura total
            if result["objective"] >= total_pop:
                print(f"  >> Cobertura total atingida com p={p}!")
                break

        # Plotar curva
        save_path = os.path.join(
            results_dir, f"curva_custo_efetividade_S{S}.png"
        )
        plot_cost_effectiveness_curve(
            p_values, coverage_values, total_pop, S,
            title=f"Curva de Custo-Efetividade — Rede 55 nós, S={S}",
            save_path=save_path,
        )


def experiment_mandatory_closeness(nodes_df, dist_matrix):
    """
    Reproduz as Figuras 2-4 do artigo: soluções com mandatory closeness.

    O artigo compara:
    - Figura 2: Location Set Covering para S=15 (5 instalações)
    - Figura 3: MCLP com S=10, T=15, p=5 → cobertura dentro de 10 = 354
    - Figura 4: MCLP para S=10, p=5 (sem mandatory) → cobertura dentro de 10 = 609
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTO 3: MCLP com Mandatory Closeness Constraints (55 nós)")
    print("=" * 70)

    populations = nodes_df["population"].values
    total_pop = sum(populations)
    results_dir = get_results_dir()

    # --- Determinar p mínimo para cobertura total dentro de T=15 ---
    print("\n--- Determinando p mínimo para cobertura total (S=15) ---")
    for p in range(1, len(nodes_df) + 1):
        result = solve_mclp(dist_matrix, populations, p, S=15)
        if result["objective"] >= total_pop:
            p_min_T15 = p
            print(f"  p mínimo para cobertura total com S=15: {p_min_T15}")
            break

    # --- Figura 2: Location Set Covering para S=15 ---
    print(f"\n--- Figura 2: Set Covering com p={p_min_T15}, S=15 ---")
    result_fig2 = solve_mclp(dist_matrix, populations, p_min_T15, S=15)
    print_solution(result_fig2, populations)

    # Calcular quantos estão cobertos dentro de S=10 nesta solução
    pop_within_10 = sum(
        populations[i] for i in range(len(populations))
        if min(dist_matrix[i][j] for j in result_fig2["facilities"]) <= 10
    )
    print(f"  População coberta dentro de S=10: {pop_within_10}/{total_pop}")

    save_path = os.path.join(results_dir, "figura2_set_covering_S15.png")
    plot_network_solution(
        nodes_df, dist_matrix, result_fig2["facilities"], S=15, T=None,
        title=f"Set Covering: p={p_min_T15}, S=15 "
              f"(Pop. dentro de 10: {pop_within_10})",
        save_path=save_path,
    )

    # --- Figura 3: MCLP com Mandatory Closeness S=10, T=15 ---
    print(f"\n--- Figura 3: MCLP Mandatory Closeness S=10, T=15, "
          f"p={p_min_T15} ---")
    result_fig3 = solve_mclp_mandatory_closeness(
        dist_matrix, populations, p_min_T15, S=10, T=15
    )
    print_solution(result_fig3, populations)
    print(f"  (Artigo reporta: cobertura dentro de 10 = 354)")

    save_path = os.path.join(results_dir, "figura3_mandatory_S10_T15.png")
    plot_network_solution(
        nodes_df, dist_matrix, result_fig3["facilities"], S=10, T=15,
        title=f"MCLP Mandatory: p={p_min_T15}, S=10, T=15 "
              f"(Pop. coberta S=10: {result_fig3['objective']:.0f})",
        save_path=save_path,
    )

    # --- Figura 4: MCLP sem mandatory, S=10 ---
    print(f"\n--- Figura 4: MCLP sem mandatory, S=10, p={p_min_T15} ---")
    result_fig4 = solve_mclp(dist_matrix, populations, p_min_T15, S=10)
    print_solution(result_fig4, populations)
    print(f"  (Artigo reporta: cobertura dentro de 10 = 609)")

    # Calcular quantos ficaram fora de T=15
    pop_outside_15 = sum(
        populations[i] for i in range(len(populations))
        if min(dist_matrix[i][j] for j in result_fig4["facilities"]) > 15
    )
    print(f"  População fora de T=15: {pop_outside_15}")

    save_path = os.path.join(results_dir, "figura4_mclp_S10_sem_mandatory.png")
    plot_network_solution(
        nodes_df, dist_matrix, result_fig4["facilities"], S=10, T=15,
        title=f"MCLP: p={p_min_T15}, S=10, sem mandatory "
              f"(Pop. coberta: {result_fig4['objective']:.0f})",
        save_path=save_path,
    )

    # --- Comparação ---
    print("\n" + "=" * 70)
    print("COMPARAÇÃO DOS RESULTADOS (Figuras 2-4)")
    print("=" * 70)
    print(f"{'Cenario':<40} {'Pop. S<=10':<12} {'Pop. S<=15':<12}")
    print("-" * 64)
    print(f"{'Fig.2: Set Covering S=15':<40} {pop_within_10:<12} "
          f"{int(result_fig2['objective']):<12}")
    print(f"{'Fig.3: MCLP S=10, T=15 (mandatory)':<40} "
          f"{int(result_fig3['objective']):<12} {total_pop:<12}")
    print(f"{'Fig.4: MCLP S=10 (sem mandatory)':<40} "
          f"{int(result_fig4['objective']):<12} "
          f"{total_pop - pop_outside_15:<12}")
    print()
    print(f"Artigo (valores esperados):")
    print(f"  Fig.2: Pop. dentro de 10 = 201")
    print(f"  Fig.3: Pop. dentro de 10 = 354 (com mandatory T=15)")
    print(f"  Fig.4: Pop. dentro de 10 = 609 (sem mandatory)")


def main():
    """Executa todos os experimentos."""
    print("=" * 70)
    print("TRABALHO FINAL - PESQUISA OPERACIONAL")
    print("Maximal Covering Location Problem (MCLP)")
    print("Church & ReVelle, 1974")
    print("=" * 70)

    # Carregar dados
    print("\nCarregando dados da rede de 55 nós (Swain, 1971)...")
    nodes_df = load_nodes()
    dist_matrix = load_distance_matrix()
    print(f"  Nós: {len(nodes_df)}")
    print(f"  População total: {nodes_df['population'].sum()}")
    print(f"  Distância máxima: {dist_matrix.max():.2f}")

    # Verificar se Gurobi está disponível
    try:
        import gurobipy as gp
        env = gp.Env()
        print(f"  Gurobi versão: {gp.gurobi.version()}")
        env.dispose()
    except Exception as e:
        print(f"  ERRO: Gurobi não disponível: {e}")
        return

    start_total = time.time()

    # Experimento 1: Tabela 1
    experiment_table1(nodes_df, dist_matrix)

    # Experimento 2: Curvas de custo-efetividade
    experiment_cost_effectiveness(nodes_df, dist_matrix)

    # Experimento 3: Mandatory closeness constraints
    experiment_mandatory_closeness(nodes_df, dist_matrix)

    elapsed = time.time() - start_total
    print(f"\n{'=' * 70}")
    print(f"Tempo total de execução: {elapsed:.2f} segundos")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
