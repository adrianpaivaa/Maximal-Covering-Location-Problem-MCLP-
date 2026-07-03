"""
Implementação do Maximal Covering Location Problem (MCLP) usando Gurobi.

Baseado no artigo:
    Church, R. & ReVelle, C. (1974). "The Maximal Covering Location Problem."
    Papers of the Regional Science Association, 32, 101-118.

Implementa duas formulações:
    1. MCLP básico (Formulação I / Maximização)
    2. MCLP com Mandatory Closeness Constraints

Autor: Adrian
Disciplina: Pesquisa Operacional
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time


def solve_mclp(dist_matrix, populations, p, S, verbose=False):
    """
    Resolve o Maximal Covering Location Problem (MCLP).

    Formulação (Maximização - Formulação I do artigo):
        Maximizar  Σᵢ aᵢ · yᵢ
        Sujeito a:
            yᵢ ≤ Σⱼ∈Nᵢ xⱼ          ∀ i ∈ I    (cobertura)
            Σⱼ xⱼ = p                            (número de instalações)
            xⱼ ∈ {0, 1}              ∀ j ∈ J
            yᵢ ∈ {0, 1}              ∀ i ∈ I

    Parâmetros:
        dist_matrix: numpy array (n x n) - matriz de distâncias
        populations: lista/array de populações de cada nó
        p: número de instalações a localizar
        S: distância máxima de serviço (maximal service distance)
        verbose: se True, mostra saída detalhada do Gurobi

    Retorna:
        dict com:
            - 'status': status da solução ('optimal', 'infeasible', etc.)
            - 'objective': valor da função objetivo (população coberta)
            - 'facilities': lista de índices (0-based) das instalações
            - 'covered_nodes': lista de índices dos nós cobertos
            - 'uncovered_nodes': lista de índices dos nós não cobertos
            - 'coverage_pct': percentual de cobertura
            - 'solve_time': tempo de resolução (segundos)
            - 'is_integer': se a relaxação LP deu solução inteira
            - 'lp_objective': valor da relaxação LP
    """
    n = len(populations)
    I = range(n)  # Conjunto de nós de demanda
    J = range(n)  # Conjunto de locais candidatos (mesmo que I)

    # Conjuntos de cobertura: N_i = {j ∈ J | d_ij ≤ S}
    N = {i: [j for j in J if dist_matrix[i][j] <= S] for i in I}

    # --- Criar modelo Gurobi ---
    model = gp.Model("MCLP")
    model.setParam("OutputFlag", 1 if verbose else 0)

    # Variáveis de decisão
    x = model.addVars(J, vtype=GRB.BINARY, name="x")  # x_j = 1 se instalação em j
    y = model.addVars(I, vtype=GRB.BINARY, name="y")   # y_i = 1 se nó i coberto

    # Função objetivo: Maximizar população coberta
    model.setObjective(
        gp.quicksum(populations[i] * y[i] for i in I),
        GRB.MAXIMIZE
    )

    # Restrição (1): Cobertura — y_i ≤ Σ_{j∈N_i} x_j
    for i in I:
        if N[i]:  # Se existe ao menos um candidato que cobre i
            model.addConstr(
                y[i] <= gp.quicksum(x[j] for j in N[i]),
                name=f"coverage_{i}"
            )
        else:
            # Se nenhum candidato pode cobrir o nó i, forçar y_i = 0
            model.addConstr(y[i] == 0, name=f"no_coverage_{i}")

    # Restrição (2): Número de instalações = p
    model.addConstr(
        gp.quicksum(x[j] for j in J) == p,
        name="num_facilities"
    )

    # --- Resolver primeiro a relaxação LP ---
    model_lp = model.relax()
    model_lp.optimize()

    lp_objective = None
    is_integer_lp = False
    if model_lp.status == GRB.OPTIMAL:
        lp_objective = model_lp.objVal
        # Verificar se a solução LP é inteira
        is_integer_lp = all(
            abs(v.x - round(v.x)) < 1e-6 for v in model_lp.getVars()
        )

    # --- Resolver o modelo inteiro ---
    start_time = time.time()
    model.optimize()
    solve_time = time.time() - start_time

    # Processar resultados
    result = {
        "status": "unknown",
        "objective": 0,
        "facilities": [],
        "covered_nodes": [],
        "uncovered_nodes": [],
        "coverage_pct": 0.0,
        "solve_time": solve_time,
        "is_integer_lp": is_integer_lp,
        "lp_objective": lp_objective,
    }

    if model.status == GRB.OPTIMAL:
        result["status"] = "optimal"
        result["objective"] = model.objVal

        # Extrair locais das instalações
        result["facilities"] = [j for j in J if x[j].x > 0.5]

        # Extrair nós cobertos e não cobertos
        result["covered_nodes"] = [i for i in I if y[i].x > 0.5]
        result["uncovered_nodes"] = [i for i in I if y[i].x < 0.5]

        total_pop = sum(populations)
        result["coverage_pct"] = (result["objective"] / total_pop * 100
                                  if total_pop > 0 else 0)

    elif model.status == GRB.INFEASIBLE:
        result["status"] = "infeasible"
    elif model.status == GRB.UNBOUNDED:
        result["status"] = "unbounded"
    else:
        result["status"] = f"status_{model.status}"

    return result


def solve_mclp_mandatory_closeness(dist_matrix, populations, p, S, T,
                                    verbose=False):
    """
    Resolve o MCLP com Mandatory Closeness Constraints.

    Formulação (do artigo, seção "Maximal Covering with Mandatory Closeness
    Constraints"):
        Maximizar  Σᵢ aᵢ · yᵢ
        Sujeito a:
            yᵢ ≤ Σⱼ∈Nᵢ xⱼ          ∀ i ∈ I    (cobertura dentro de S)
            Σⱼ∈Mᵢ xⱼ ≥ 1            ∀ i ∈ I    (proximidade obrigatória T)
            Σⱼ xⱼ = p                            (número de instalações)
            xⱼ ∈ {0, 1}              ∀ j ∈ J
            yᵢ ∈ {0, 1}              ∀ i ∈ I

    Onde:
        N_i = {j ∈ J | d_ij ≤ S} — locais que cobrem i dentro de S
        M_i = {j ∈ J | d_ij ≤ T} — locais que cobrem i dentro de T (T > S)

    Parâmetros:
        dist_matrix: numpy array (n x n) - matriz de distâncias
        populations: lista/array de populações de cada nó
        p: número de instalações a localizar
        S: distância de serviço desejável
        T: distância máxima obrigatória (T > S)
        verbose: se True, mostra saída detalhada do Gurobi

    Retorna:
        dict semelhante ao solve_mclp, com campos adicionais:
            - 'covered_T_nodes': nós cobertos entre S e T
            - 'uncovered_T_nodes': nós não cobertos nem dentro de T
    """
    assert T > S, f"T ({T}) deve ser maior que S ({S})"

    n = len(populations)
    I = range(n)
    J = range(n)

    # Conjuntos de cobertura
    N = {i: [j for j in J if dist_matrix[i][j] <= S] for i in I}  # Dentro de S
    M = {i: [j for j in J if dist_matrix[i][j] <= T] for i in I}  # Dentro de T

    # --- Criar modelo Gurobi ---
    model = gp.Model("MCLP_MandatoryCloseness")
    model.setParam("OutputFlag", 1 if verbose else 0)

    # Variáveis de decisão
    x = model.addVars(J, vtype=GRB.BINARY, name="x")
    y = model.addVars(I, vtype=GRB.BINARY, name="y")

    # Função objetivo: Maximizar população coberta dentro de S
    model.setObjective(
        gp.quicksum(populations[i] * y[i] for i in I),
        GRB.MAXIMIZE
    )

    # Restrição (9): Cobertura dentro de S — y_i ≤ Σ_{j∈N_i} x_j
    for i in I:
        if N[i]:
            model.addConstr(
                y[i] <= gp.quicksum(x[j] for j in N[i]),
                name=f"coverage_S_{i}"
            )
        else:
            model.addConstr(y[i] == 0, name=f"no_coverage_S_{i}")

    # Restrição (10): Mandatory Closeness — Σ_{j∈M_i} x_j ≥ 1
    for i in I:
        if M[i]:
            model.addConstr(
                gp.quicksum(x[j] for j in M[i]) >= 1,
                name=f"mandatory_T_{i}"
            )
        else:
            # Se nenhum candidato pode cobrir o nó i dentro de T,
            # o problema é inviável
            print(f"AVISO: Nó {i} não pode ser coberto dentro de T={T}. "
                  f"Problema inviável.")
            model.addConstr(
                gp.quicksum(x[j] for j in J) >= n + 1,
                name=f"force_infeasible_{i}"
            )

    # Restrição: Número de instalações = p
    model.addConstr(
        gp.quicksum(x[j] for j in J) == p,
        name="num_facilities"
    )

    # --- Resolver a relaxação LP ---
    model_lp = model.relax()
    model_lp.optimize()

    lp_objective = None
    is_integer_lp = False
    if model_lp.status == GRB.OPTIMAL:
        lp_objective = model_lp.objVal
        is_integer_lp = all(
            abs(v.x - round(v.x)) < 1e-6 for v in model_lp.getVars()
        )

    # --- Resolver o modelo inteiro ---
    start_time = time.time()
    model.optimize()
    solve_time = time.time() - start_time

    # Processar resultados
    result = {
        "status": "unknown",
        "objective": 0,
        "facilities": [],
        "covered_nodes": [],
        "uncovered_nodes": [],
        "covered_T_nodes": [],
        "uncovered_T_nodes": [],
        "coverage_pct": 0.0,
        "solve_time": solve_time,
        "is_integer_lp": is_integer_lp,
        "lp_objective": lp_objective,
        "S": S,
        "T": T,
    }

    if model.status == GRB.OPTIMAL:
        result["status"] = "optimal"
        result["objective"] = model.objVal
        result["facilities"] = [j for j in J if x[j].x > 0.5]
        result["covered_nodes"] = [i for i in I if y[i].x > 0.5]
        result["uncovered_nodes"] = [i for i in I if y[i].x < 0.5]

        # Classificar nós não cobertos dentro de S em: cobertos em T ou não
        for i in result["uncovered_nodes"]:
            dist_to_nearest = min(
                dist_matrix[i][j] for j in result["facilities"]
            )
            if dist_to_nearest <= T:
                result["covered_T_nodes"].append(i)
            else:
                result["uncovered_T_nodes"].append(i)

        total_pop = sum(populations)
        result["coverage_pct"] = (result["objective"] / total_pop * 100
                                  if total_pop > 0 else 0)

    elif model.status == GRB.INFEASIBLE:
        result["status"] = "infeasible"
    else:
        result["status"] = f"status_{model.status}"

    return result


def solve_mclp_range(dist_matrix, populations, p_range, S, verbose=False):
    """
    Resolve o MCLP para uma faixa de valores de p.

    Útil para gerar curvas de custo-efetividade.

    Parâmetros:
        dist_matrix: numpy array (n x n)
        populations: lista/array de populações
        p_range: lista/range de valores de p
        S: distância máxima de serviço
        verbose: se True, mostra saída do Gurobi

    Retorna:
        Lista de dicts com os resultados para cada p
    """
    results = []
    for p in p_range:
        print(f"  Resolvendo MCLP para p={p}, S={S}...", end=" ")
        result = solve_mclp(dist_matrix, populations, p, S, verbose)
        result["p"] = p
        result["S"] = S
        print(f"Cobertura: {result['objective']:.0f}/{sum(populations)} "
              f"({result['coverage_pct']:.1f}%) — "
              f"LP inteiro: {result['is_integer_lp']} — "
              f"Tempo: {result['solve_time']:.3f}s")
        results.append(result)
    return results


def print_solution(result, populations=None):
    """
    Imprime os resultados de uma solução MCLP de forma formatada.

    Parâmetros:
        result: Dicionário retornado por solve_mclp ou
                solve_mclp_mandatory_closeness
        populations: Lista de populações (opcional, para detalhes)
    """
    print("=" * 60)
    print("RESULTADO DA SOLUÇÃO MCLP")
    print("=" * 60)
    print(f"Status: {result['status']}")
    print(f"Função Objetivo (pop. coberta): {result['objective']:.0f}")
    print(f"Cobertura: {result['coverage_pct']:.1f}%")
    print(f"Instalações (nós, 1-based): "
          f"{[f+1 for f in result['facilities']]}")
    print(f"Número de nós cobertos: {len(result['covered_nodes'])}")
    print(f"Número de nós não cobertos: {len(result['uncovered_nodes'])}")
    print(f"Relaxação LP inteira: {result['is_integer_lp']}")
    if result['lp_objective'] is not None:
        print(f"Valor da relaxação LP: {result['lp_objective']:.2f}")
    print(f"Tempo de resolução: {result['solve_time']:.4f} s")

    if "T" in result:
        print(f"\nRestrições de proximidade obrigatória:")
        print(f"  S (desejável) = {result['S']}")
        print(f"  T (obrigatório) = {result['T']}")
        print(f"  Nós entre S e T: {len(result.get('covered_T_nodes', []))}")
        print(f"  Nós fora de T: {len(result.get('uncovered_T_nodes', []))}")

    if populations is not None and result["uncovered_nodes"]:
        print(f"\nNós não cobertos (1-based):")
        for i in result["uncovered_nodes"]:
            print(f"  Nó {i+1}: população = {populations[i]}")

    print("=" * 60)
