"""
    Experimento 1: Tabela 1 — soluções ótimas para vários S e p
    Experimento 2: Curvas de custo-efetividade (S=10 e S=15)
    Experimento 3: Comparativo com Mandatory Closeness Constraints (Figs. 2-4)

Como rodar:
    python src/testes.py

Pré-requisitos:
    - Python 3.x com gurobipy, numpy, pandas, matplotlib
    - Licença do Gurobi (acadêmica)
    - Arquivos CSV na pasta data/ (já incluídos)

Os resultados (gráficos e tabela CSV) são salvos na pasta resultados/.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from src.modelo_mclp import (
    resolver_mclp,
    resolver_mclp_mandatory,
    resolver_mclp_faixa,
    exibir_solucao,
)
from src.auxiliares import (
    carregar_nos,
    carregar_matriz_distancias,
    diretorio_resultados,
    plotar_curva_custo_efetividade,
    plotar_rede,
)


def experimento_tabela1(df_nos, mat_dist):
    """Reproduz a Tabela 1 do artigo."""
    print("\n" + "=" * 70)
    print("EXPERIMENTO 1: Reprodução da Tabela 1 (Rede de 55 nós)")
    print("=" * 70)

    populacoes = df_nos["population"].values
    pop_total = sum(populacoes)
    print(f"População total: {pop_total}")
    print(f"Número de nós: {len(df_nos)}")

    # Configurações conforme a Tabela 1 do artigo
    configs = [
        (10, range(1, 10)),
        (12, range(1, 8)),
        (15, range(1, 6)),
        (19, range(1, 4)),
        (20, range(1, 4)),
    ]

    todos_resultados = []
    cont_lp_inteiro = 0
    cont_lp_fracionario = 0

    for S, faixa_p in configs:
        print(f"\n--- S = {S} ---")
        resultados = resolver_mclp_faixa(mat_dist, populacoes, faixa_p, S)

        for r in resultados:
            r["pop_total"] = pop_total
            r["pop_descoberta"] = pop_total - r["objetivo"]
            todos_resultados.append(r)

            if r["lp_inteiro"]:
                cont_lp_inteiro += 1
            else:
                cont_lp_fracionario += 1

    # Resumo da relaxação LP
    total_problemas = cont_lp_inteiro + cont_lp_fracionario
    pct_inteiro = cont_lp_inteiro / total_problemas * 100
    print(f"\n{'=' * 70}")
    print(f"RESUMO - Relaxação LP:")
    print(f"  Total de problemas: {total_problemas}")
    print(f"  Soluções inteiras: {cont_lp_inteiro} ({pct_inteiro:.1f}%)")
    print(f"  Soluções fracionárias: {cont_lp_fracionario} ({100 - pct_inteiro:.1f}%)")
    print(f"  (Artigo original reporta ~80% inteiras)")

    # Salvar CSV
    dir_res = diretorio_resultados()
    df = pd.DataFrame([{
        "S": r["S"],
        "p": r["p"],
        "pop_coberta": int(r["objetivo"]),
        "pop_descoberta": int(r["pop_descoberta"]),
        "cobertura_pct": round(r["cobertura_pct"], 1),
        "lp_inteiro": r["lp_inteiro"],
        "obj_lp": round(r["obj_lp"], 2) if r["obj_lp"] else None,
        "tempo_s": round(r["tempo"], 4),
        "instalacoes": str([f+1 for f in r["instalacoes"]]),
    } for r in todos_resultados])

    csv_path = os.path.join(dir_res, "tabela1_resultados.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResultados salvos em: {csv_path}")

    # Tabela formatada no terminal
    print(f"\n{'S':>4} {'p':>3} {'Pop.Coberta':>12} {'%Cobertura':>12} "
          f"{'LP Inteiro':>10} {'Tempo(s)':>10}")
    print("-" * 60)
    for _, linha in df.iterrows():
        print(f"{linha['S']:>4} {linha['p']:>3} {linha['pop_coberta']:>12} "
              f"{linha['cobertura_pct']:>11.1f}% "
              f"{'Sim' if linha['lp_inteiro'] else 'Não':>10} "
              f"{linha['tempo_s']:>10.4f}")

    return todos_resultados


def experimento_curvas(df_nos, mat_dist):
    """Gera as curvas de custo-efetividade para S=10 e S=15."""
    print("\n" + "=" * 70)
    print("EXPERIMENTO 2: Curvas de Custo-Efetividade (55 nós)")
    print("=" * 70)

    populacoes = df_nos["population"].values
    pop_total = sum(populacoes)
    dir_res = diretorio_resultados()

    for S in [10, 15]:
        print(f"\n--- Curva para S = {S} ---")
        valores_p = []
        valores_cobertura = []

        for p in range(1, len(df_nos) + 1):
            res = resolver_mclp(mat_dist, populacoes, p, S)
            valores_p.append(p)
            valores_cobertura.append(res["objetivo"])
            print(f"  p={p}: cobertura = {res['objetivo']:.0f} "
                  f"({res['cobertura_pct']:.1f}%)")

            if res["objetivo"] >= pop_total:
                print(f"  >> Cobertura total atingida com p={p}!")
                break

        caminho = os.path.join(dir_res, f"curva_custo_efetividade_S{S}.png")
        plotar_curva_custo_efetividade(
            valores_p, valores_cobertura, pop_total, S,
            titulo=f"Curva de Custo-Efetividade — Rede 55 nós, S={S}",
            caminho_salvar=caminho,
        )


def experimento_mandatory(df_nos, mat_dist):
    """Reproduz as Figuras 2-4: comparação com mandatory closeness."""
    print("\n" + "=" * 70)
    print("EXPERIMENTO 3: MCLP com Mandatory Closeness Constraints (55 nós)")
    print("=" * 70)

    populacoes = df_nos["population"].values
    pop_total = sum(populacoes)
    dir_res = diretorio_resultados()

    # Determinar p mínimo para cobertura total em S=15
    print("\n--- Determinando p mínimo para cobertura total (S=15) ---")
    for p in range(1, len(df_nos) + 1):
        res = resolver_mclp(mat_dist, populacoes, p, S=15)
        if res["objetivo"] >= pop_total:
            p_min = p
            print(f"  p mínimo para cobertura total com S=15: {p_min}")
            break

    # Figura 2: Set Covering para S=15
    print(f"\n--- Figura 2: Set Covering com p={p_min}, S=15 ---")
    res_fig2 = resolver_mclp(mat_dist, populacoes, p_min, S=15)
    exibir_solucao(res_fig2, populacoes)

    pop_em_10 = sum(
        populacoes[i] for i in range(len(populacoes))
        if min(mat_dist[i][j] for j in res_fig2["instalacoes"]) <= 10
    )
    print(f"  População coberta dentro de S=10: {pop_em_10}/{pop_total}")

    caminho = os.path.join(dir_res, "figura2_set_covering_S15.png")
    plotar_rede(
        df_nos, mat_dist, res_fig2["instalacoes"], S=15, T=None,
        titulo=f"Set Covering: p={p_min}, S=15 (Pop. dentro de 10: {pop_em_10})",
        caminho_salvar=caminho,
    )

    # Figura 3: MCLP com Mandatory Closeness S=10, T=15
    print(f"\n--- Figura 3: MCLP Mandatory S=10, T=15, p={p_min} ---")
    res_fig3 = resolver_mclp_mandatory(mat_dist, populacoes, p_min, S=10, T=15)
    exibir_solucao(res_fig3, populacoes)
    print(f"  (Artigo reporta: cobertura dentro de 10 = 354)")

    caminho = os.path.join(dir_res, "figura3_mandatory_S10_T15.png")
    plotar_rede(
        df_nos, mat_dist, res_fig3["instalacoes"], S=10, T=15,
        titulo=f"MCLP Mandatory: p={p_min}, S=10, T=15 "
              f"(Pop. coberta S=10: {res_fig3['objetivo']:.0f})",
        caminho_salvar=caminho,
    )

    # Figura 4: MCLP sem mandatory, S=10
    print(f"\n--- Figura 4: MCLP sem mandatory, S=10, p={p_min} ---")
    res_fig4 = resolver_mclp(mat_dist, populacoes, p_min, S=10)
    exibir_solucao(res_fig4, populacoes)
    print(f"  (Artigo reporta: cobertura dentro de 10 = 609)")

    pop_fora_15 = sum(
        populacoes[i] for i in range(len(populacoes))
        if min(mat_dist[i][j] for j in res_fig4["instalacoes"]) > 15
    )
    print(f"  População fora de T=15: {pop_fora_15}")

    caminho = os.path.join(dir_res, "figura4_mclp_S10_sem_mandatory.png")
    plotar_rede(
        df_nos, mat_dist, res_fig4["instalacoes"], S=10, T=15,
        titulo=f"MCLP: p={p_min}, S=10, sem mandatory "
              f"(Pop. coberta: {res_fig4['objetivo']:.0f})",
        caminho_salvar=caminho,
    )

    # Comparação final
    print("\n" + "=" * 70)
    print("COMPARAÇÃO DOS RESULTADOS (Figuras 2-4)")
    print("=" * 70)
    print(f"{'Cenario':<40} {'Pop. S<=10':<12} {'Pop. S<=15':<12}")
    print("-" * 64)
    print(f"{'Fig.2: Set Covering S=15':<40} {pop_em_10:<12} "
          f"{int(res_fig2['objetivo']):<12}")
    print(f"{'Fig.3: MCLP S=10, T=15 (mandatory)':<40} "
          f"{int(res_fig3['objetivo']):<12} {pop_total:<12}")
    print(f"{'Fig.4: MCLP S=10 (sem mandatory)':<40} "
          f"{int(res_fig4['objetivo']):<12} "
          f"{pop_total - pop_fora_15:<12}")
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
    df_nos = carregar_nos()
    mat_dist = carregar_matriz_distancias()
    print(f"  Nós: {len(df_nos)}")
    print(f"  População total: {df_nos['population'].sum()}")
    print(f"  Distância máxima: {mat_dist.max():.2f}")

    # Verificar Gurobi
    try:
        import gurobipy as gp
        env = gp.Env()
        print(f"  Gurobi versão: {gp.gurobi.version()}")
        env.dispose()
    except Exception as e:
        print(f"  ERRO: Gurobi não disponível: {e}")
        return

    t_inicio = time.time()

    experimento_tabela1(df_nos, mat_dist)
    experimento_curvas(df_nos, mat_dist)
    experimento_mandatory(df_nos, mat_dist)

    t_total = time.time() - t_inicio
    print(f"\n{'=' * 70}")
    print(f"Tempo total de execução: {t_total:.2f} segundos")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
