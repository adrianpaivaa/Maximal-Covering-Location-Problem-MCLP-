"""
Implementação do MCLP (Maximal Covering Location Problem) usando Gurobi.

Contém três formulações:
    - resolver_mclp(): MCLP clássico (Formulação I - Maximização)
    - resolver_mclp_mandatory(): MCLP com Mandatory Closeness Constraints
    - resolver_mclp_faixa(): resolve o MCLP para vários valores de p

"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time


def resolver_mclp(mat_dist, populacoes, p, S, verbose=False):
    """
    MCLP clássico (Formulação I - Maximização).
    """
    n = len(populacoes)
    nos = range(n)

    # Conjuntos de cobertura: Ni = {j | d_ij <= S}
    Ni = {i: [j for j in nos if mat_dist[i][j] <= S] for i in nos}

    # Modelo Gurobi
    modelo = gp.Model("MCLP")
    modelo.setParam("OutputFlag", 1 if verbose else 0)

    # Variáveis: x_j (instalar em j), y_i (nó i coberto)
    x = modelo.addVars(nos, vtype=GRB.BINARY, name="x")
    y = modelo.addVars(nos, vtype=GRB.BINARY, name="y")

    # Maximizar população coberta
    modelo.setObjective(
        gp.quicksum(populacoes[i] * y[i] for i in nos),
        GRB.MAXIMIZE
    )

    # Restrição de cobertura: y_i <= soma(x_j) para j em Ni
    for i in nos:
        if Ni[i]:
            modelo.addConstr(
                y[i] <= gp.quicksum(x[j] for j in Ni[i]),
                name=f"cob_{i}"
            )
        else:
            modelo.addConstr(y[i] == 0, name=f"sem_cob_{i}")

    # Restrição: exatamente p instalações
    modelo.addConstr(
        gp.quicksum(x[j] for j in nos) == p,
        name="num_instalacoes"
    )

    # Relaxação LP
    modelo_lp = modelo.relax()
    modelo_lp.optimize()

    obj_lp = None
    lp_inteiro = False
    if modelo_lp.status == GRB.OPTIMAL:
        obj_lp = modelo_lp.objVal
        lp_inteiro = all(
            abs(v.x - round(v.x)) < 1e-6 for v in modelo_lp.getVars()
        )

    # Resolver modelo inteiro
    t_inicio = time.time()
    modelo.optimize()
    t_resolucao = time.time() - t_inicio

    resultado = {
        "status": "desconhecido",
        "objetivo": 0,
        "instalacoes": [],
        "nos_cobertos": [],
        "nos_descobertos": [],
        "cobertura_pct": 0.0,
        "tempo": t_resolucao,
        "lp_inteiro": lp_inteiro,
        "obj_lp": obj_lp,
    }

    if modelo.status == GRB.OPTIMAL:
        resultado["status"] = "otimo"
        resultado["objetivo"] = modelo.objVal
        resultado["instalacoes"] = [j for j in nos if x[j].x > 0.5]
        resultado["nos_cobertos"] = [i for i in nos if y[i].x > 0.5]
        resultado["nos_descobertos"] = [i for i in nos if y[i].x < 0.5]
        pop_total = sum(populacoes)
        resultado["cobertura_pct"] = (
            resultado["objetivo"] / pop_total * 100 if pop_total > 0 else 0
        )
    elif modelo.status == GRB.INFEASIBLE:
        resultado["status"] = "inviavel"
    elif modelo.status == GRB.UNBOUNDED:
        resultado["status"] = "ilimitado"

    return resultado


def resolver_mclp_mandatory(mat_dist, populacoes, p, S, T, verbose=False):
    """
    MCLP com Mandatory Closeness Constraints.
    """
    assert T > S, f"T ({T}) deve ser maior que S ({S})"

    n = len(populacoes)
    nos = range(n)

    Ni = {i: [j for j in nos if mat_dist[i][j] <= S] for i in nos}
    Mi = {i: [j for j in nos if mat_dist[i][j] <= T] for i in nos}

    modelo = gp.Model("MCLP_Mandatory")
    modelo.setParam("OutputFlag", 1 if verbose else 0)

    x = modelo.addVars(nos, vtype=GRB.BINARY, name="x")
    y = modelo.addVars(nos, vtype=GRB.BINARY, name="y")

    # Maximizar cobertura dentro de S
    modelo.setObjective(
        gp.quicksum(populacoes[i] * y[i] for i in nos),
        GRB.MAXIMIZE
    )

    # Cobertura dentro de S
    for i in nos:
        if Ni[i]:
            modelo.addConstr(
                y[i] <= gp.quicksum(x[j] for j in Ni[i]),
                name=f"cob_S_{i}"
            )
        else:
            modelo.addConstr(y[i] == 0, name=f"sem_cob_S_{i}")

    # Proximidade obrigatória dentro de T
    for i in nos:
        if Mi[i]:
            modelo.addConstr(
                gp.quicksum(x[j] for j in Mi[i]) >= 1,
                name=f"prox_T_{i}"
            )
        else:
            print(f"AVISO: Nó {i} sem cobertura em T={T}. Problema inviável.")
            modelo.addConstr(
                gp.quicksum(x[j] for j in nos) >= n + 1,
                name=f"forcar_inviavel_{i}"
            )

    # Exatamente p instalações
    modelo.addConstr(
        gp.quicksum(x[j] for j in nos) == p,
        name="num_instalacoes"
    )

    # Relaxação LP
    modelo_lp = modelo.relax()
    modelo_lp.optimize()

    obj_lp = None
    lp_inteiro = False
    if modelo_lp.status == GRB.OPTIMAL:
        obj_lp = modelo_lp.objVal
        lp_inteiro = all(
            abs(v.x - round(v.x)) < 1e-6 for v in modelo_lp.getVars()
        )

    # Resolver modelo inteiro
    t_inicio = time.time()
    modelo.optimize()
    t_resolucao = time.time() - t_inicio

    resultado = {
        "status": "desconhecido",
        "objetivo": 0,
        "instalacoes": [],
        "nos_cobertos": [],
        "nos_descobertos": [],
        "nos_entre_S_T": [],
        "nos_fora_T": [],
        "cobertura_pct": 0.0,
        "tempo": t_resolucao,
        "lp_inteiro": lp_inteiro,
        "obj_lp": obj_lp,
        "S": S,
        "T": T,
    }

    if modelo.status == GRB.OPTIMAL:
        resultado["status"] = "otimo"
        resultado["objetivo"] = modelo.objVal
        resultado["instalacoes"] = [j for j in nos if x[j].x > 0.5]
        resultado["nos_cobertos"] = [i for i in nos if y[i].x > 0.5]
        resultado["nos_descobertos"] = [i for i in nos if y[i].x < 0.5]

        # Classificar nós fora de S: entre S e T, ou fora de T
        for i in resultado["nos_descobertos"]:
            dist_min = min(mat_dist[i][j] for j in resultado["instalacoes"])
            if dist_min <= T:
                resultado["nos_entre_S_T"].append(i)
            else:
                resultado["nos_fora_T"].append(i)

        pop_total = sum(populacoes)
        resultado["cobertura_pct"] = (
            resultado["objetivo"] / pop_total * 100 if pop_total > 0 else 0
        )
    elif modelo.status == GRB.INFEASIBLE:
        resultado["status"] = "inviavel"

    return resultado


def resolver_mclp_faixa(mat_dist, populacoes, faixa_p, S, verbose=False):
    """
    Resolve o MCLP para vários valores de p.
    """
    resultados = []
    for p in faixa_p:
        print(f"  Resolvendo MCLP para p={p}, S={S}...", end=" ")
        res = resolver_mclp(mat_dist, populacoes, p, S, verbose)
        res["p"] = p
        res["S"] = S
        print(f"Cobertura: {res['objetivo']:.0f}/{sum(populacoes)} "
              f"({res['cobertura_pct']:.1f}%) — "
              f"LP inteiro: {res['lp_inteiro']} — "
              f"Tempo: {res['tempo']:.3f}s")
        resultados.append(res)
    return resultados


def exibir_solucao(resultado, populacoes=None):
    """
    Exibe os resultados no terminal.
    """
    print("=" * 60)
    print("RESULTADO DA SOLUÇÃO MCLP")
    print("=" * 60)
    print(f"Status: {resultado['status']}")
    print(f"Função Objetivo (pop. coberta): {resultado['objetivo']:.0f}")
    print(f"Cobertura: {resultado['cobertura_pct']:.1f}%")
    print(f"Instalações (nós): "
          f"{[f+1 for f in resultado['instalacoes']]}")
    print(f"Nós cobertos: {len(resultado['nos_cobertos'])}")
    print(f"Nós descobertos: {len(resultado['nos_descobertos'])}")
    print(f"Relaxação LP inteira: {resultado['lp_inteiro']}")
    if resultado['obj_lp'] is not None:
        print(f"Valor da relaxação LP: {resultado['obj_lp']:.2f}")
    print(f"Tempo de resolução: {resultado['tempo']:.4f} s")

    if "T" in resultado:
        print(f"\nRestrições de proximidade obrigatória:")
        print(f"  S (desejável) = {resultado['S']}")
        print(f"  T (obrigatório) = {resultado['T']}")
        print(f"  Nós entre S e T: {len(resultado.get('nos_entre_S_T', []))}")
        print(f"  Nós fora de T: {len(resultado.get('nos_fora_T', []))}")

    if populacoes is not None and resultado["nos_descobertos"]:
        print(f"\nNós não cobertos:")
        for i in resultado["nos_descobertos"]:
            print(f"  Nó {i+1}: população = {populacoes[i]}")

    print("=" * 60)
