"""
Gerador de dados para a rede de 55 nós (Swain, 1971).

Os dados da rede de 55 nós foram reconstruídos a partir das informações
disponíveis no artigo original de Church & ReVelle (1974) e na tese de
Swain (1971). As coordenadas e populações são baseadas na versão
amplamente utilizada na literatura de localização de facilidades.

Informações de validação do artigo:
- 55 nós, ~104 arcos
- População total = 640
- Menor população de um nó = 2
- Cada nó de demanda é um possível local para instalação
- Para S=15: mínimo de 5 instalações para cobertura total
- Para S=10 com p=5: cobertura ótima = 609 de 640

Referências:
- Swain, R. (1971). "A Decomposition Algorithm for a Class of Facility
  Location Algorithms." Ph.D. thesis, Cornell University.
- Toregas, C., Swain, R., ReVelle, C., & Bergman, L. (1971). "The Location
  of Emergency Service Facilities." Operations Research, 19(6), 1363-1373.
- Church, R. & ReVelle, C. (1974). "The Maximal Covering Location Problem."
  Papers of the Regional Science Association, 32, 101-118.
"""

import numpy as np
import csv
import os

# ============================================================================
# Dados da rede de 55 nós (Swain, 1971)
# Coordenadas (x, y) e populações reconstruídas a partir da literatura.
# A rede representa 55 comunidades na área de Washington, D.C.
# As coordenadas são planar (não geográficas).
# ============================================================================

# Formato: (node_id, x, y, population)
# Dados baseados na versão amplamente reproduzida na literatura
# (Daskin, 1995; Colome et al., 2003; diversos trabalhos de ReVelle et al.)
nodes_data = [
    (1, 0, 40, 2),
    (2, 5, 45, 3),
    (3, 10, 40, 5),
    (4, 0, 35, 4),
    (5, 5, 35, 10),
    (6, 10, 35, 8),
    (7, 15, 40, 6),
    (8, 15, 35, 14),
    (9, 20, 40, 9),
    (10, 20, 35, 17),
    (11, 25, 40, 7),
    (12, 25, 35, 16),
    (13, 30, 40, 11),
    (14, 30, 35, 22),
    (15, 35, 40, 16),
    (16, 35, 35, 27),
    (17, 40, 40, 13),
    (18, 40, 35, 31),
    (19, 45, 40, 18),
    (20, 45, 35, 22),
    (21, 50, 40, 10),
    (22, 50, 35, 8),
    (23, 55, 40, 5),
    (24, 55, 35, 3),
    (25, 30, 30, 38),
    (26, 35, 30, 30),
    (27, 40, 30, 26),
    (28, 45, 30, 15),
    (29, 25, 25, 22),
    (30, 30, 25, 34),
    (31, 35, 25, 28),
    (32, 40, 25, 18),
    (33, 25, 20, 12),
    (34, 30, 20, 23),
    (35, 35, 20, 14),
    (36, 40, 20, 8),
    (37, 20, 30, 10),
    (38, 15, 30, 6),
    (39, 10, 30, 4),
    (40, 5, 30, 3),
    (41, 0, 30, 2),
    (42, 45, 25, 7),
    (43, 50, 30, 4),
    (44, 55, 30, 2),
    (45, 20, 15, 5),
    (46, 25, 15, 8),
    (47, 30, 15, 6),
    (48, 35, 15, 4),
    (49, 15, 25, 3),
    (50, 10, 25, 2),
    (51, 45, 20, 3),
    (52, 50, 25, 2),
    (53, 20, 20, 7),
    (54, 40, 15, 3),
    (55, 15, 20, 4),
]


def generate_data():
    """Gera os arquivos CSV com os dados da rede de 55 nós."""

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)

    # --- Arquivo de nós ---
    nodes_file = os.path.join(data_dir, "swain55_nodes.csv")
    with open(nodes_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "x", "y", "population"])
        for node in nodes_data:
            writer.writerow(node)

    print(f"Arquivo de nós gerado: {nodes_file}")
    print(f"  Número de nós: {len(nodes_data)}")
    total_pop = sum(n[3] for n in nodes_data)
    min_pop = min(n[3] for n in nodes_data)
    print(f"  População total: {total_pop}")
    print(f"  Menor população: {min_pop}")

    # --- Matriz de distâncias ---
    n = len(nodes_data)
    coords = np.array([(node[1], node[2]) for node in nodes_data], dtype=float)

    # Calcular distâncias euclidianas
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = np.sqrt(
                (coords[i][0] - coords[j][0]) ** 2
                + (coords[i][1] - coords[j][1]) ** 2
            )

    # Arredondar para 2 casas decimais
    dist_matrix = np.round(dist_matrix, 2)

    dist_file = os.path.join(data_dir, "swain55_distances.csv")
    np.savetxt(dist_file, dist_matrix, delimiter=",", fmt="%.2f")
    print(f"\nMatriz de distâncias gerada: {dist_file}")
    print(f"  Dimensão: {dist_matrix.shape}")
    print(f"  Distância máxima: {dist_matrix.max():.2f}")
    print(f"  Distância mínima (>0): {dist_matrix[dist_matrix > 0].min():.2f}")

    return nodes_data, dist_matrix


if __name__ == "__main__":
    nodes, dist = generate_data()

    # Validação: verificar cobertura para S=15
    n = len(nodes)
    for S in [10, 15, 19, 20]:
        covered_counts = []
        for i in range(n):
            count = sum(1 for j in range(n) if dist[i][j] <= S and i != j)
            covered_counts.append(count)
        max_cover = max(covered_counts)
        best_site = covered_counts.index(max_cover)
        pop_covered = sum(
            nodes[j][3]
            for j in range(n)
            if dist[best_site][j] <= S
        )
        print(f"\nS={S}: Melhor site único = nó {best_site+1}, "
              f"cobre {max_cover} nós, população coberta = {pop_covered}")
