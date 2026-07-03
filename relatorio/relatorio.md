# Relatório Técnico — Maximal Covering Location Problem (MCLP)

## Trabalho Final de Pesquisa Operacional

**Aluno:** Adrian  
**Artigo base:** Church, R. & ReVelle, C. (1974). "The Maximal Covering Location Problem." *Papers of the Regional Science Association*, 32, 101-118.

---

## 1. Introdução ao Problema

O problema de localização de facilidades públicas (como estações de bombeiros, ambulâncias e postos de saúde) requer a escolha de locais que otimizem o atendimento à população. Diferentemente de facilidades privadas, cujo objetivo é tipicamente minimizar custos, facilidades públicas buscam **maximizar a acessibilidade do serviço**.

Dois critérios clássicos são usados como medidas de qualidade de uma solução de localização:
1. **Distância total ponderada** — minimizar a soma das distâncias que os usuários percorrem.
2. **Distância máxima de serviço** — garantir que nenhum usuário esteja além de uma distância S de uma facilidade.

O segundo critério levou ao **Location Set Covering Problem** (Toregas et al., 1971), que busca o menor número de facilidades necessário para cobrir todos os pontos de demanda dentro de uma distância S.

Porém, em muitas situações reais, os recursos disponíveis (número de facilidades) são **insuficientes** para cobrir toda a população. Nesse cenário, Church e ReVelle (1974) propuseram o **Maximal Covering Location Problem (MCLP)**: dado um número fixo **p** de facilidades, localizar essas facilidades de modo a **maximizar a população coberta** dentro da distância de serviço S.

---

## 2. Resumo do Artigo

O artigo de Church e ReVelle (1974) introduz o MCLP como uma extensão natural do Location Set Covering Problem. As contribuições principais são:

### 2.1 Formulações Matemáticas
- **Formulação I (Maximização):** Maximizar a população coberta dentro da distância S, sujeito a restrições de cobertura e número de facilidades.
- **Formulação II (Minimização equivalente):** Por substituição de variáveis, o problema pode ser reformulado como minimização da população **não coberta**.

### 2.2 Extensão com Mandatory Closeness Constraints
Uma extensão importante onde se maximiza a cobertura dentro de S, mas garante-se que **nenhum ponto de demanda** fique a mais de T (T > S) da facilidade mais próxima. Isso provê equidade no atendimento.

### 2.3 Métodos de Solução
- **Greedy Adding (GA):** Adiciona facilidades uma a uma no melhor local.
- **Greedy Adding with Substitution (GAS):** Como GA, mas tenta substituir facilidades existentes para melhorar.
- **Programação Linear (LP):** ~80% das vezes a relaxação LP produz solução inteira automaticamente.
- **Branch and Bound:** Para resolver os ~20% de soluções fracionárias.

### 2.4 Resultados Experimentais
Testado em redes de 30 e 55 nós, com diferentes valores de S e p. O artigo demonstra:
- Curvas de custo-efetividade que auxiliam a tomada de decisão.
- Que a mandatory closeness encontra soluções mais desejáveis dentre as alternativas ótimas do set covering.

---

## 3. Descrição Detalhada da Formulação Matemática

### 3.1 Formulação I — MCLP (Maximização)

**Conjuntos:**
- $I$ = conjunto de nós de demanda
- $J$ = conjunto de locais candidatos para facilidades

**Parâmetros:**
- $a_i$ = população no nó de demanda $i$
- $d_{ij}$ = distância do nó $i$ ao local $j$
- $S$ = distância máxima de serviço (maximal service distance)
- $p$ = número de facilidades a localizar
- $N_i = \{j \in J \mid d_{ij} \leq S\}$ = conjunto de locais que cobrem o nó $i$

**Variáveis de decisão:**
- $x_j \in \{0, 1\}$ — 1 se uma facilidade é alocada no local $j$, 0 caso contrário
- $y_i \in \{0, 1\}$ — 1 se o nó de demanda $i$ é coberto, 0 caso contrário

**Modelo:**

$$\max \sum_{i \in I} a_i \cdot y_i$$

Sujeito a:

$$y_i \leq \sum_{j \in N_i} x_j \quad \forall\, i \in I \quad \text{(1)}$$

$$\sum_{j \in J} x_j = p \quad \text{(2)}$$

$$x_j \in \{0, 1\} \quad \forall\, j \in J \quad \text{(3)}$$

$$y_i \in \{0, 1\} \quad \forall\, i \in I \quad \text{(4)}$$

**Interpretação:**
- A restrição (1) garante que um nó $i$ só pode ser considerado coberto ($y_i = 1$) se ao menos uma facilidade estiver alocada dentro da distância $S$.
- A restrição (2) fixa o número de facilidades em $p$.

### 3.2 Formulação com Mandatory Closeness Constraints

Adiciona-se uma distância obrigatória $T > S$, garantindo que todos os nós tenham ao menos uma facilidade dentro de $T$:

$$\max \sum_{i \in I} a_i \cdot y_i$$

Sujeito a:

$$y_i \leq \sum_{j \in N_i} x_j \quad \forall\, i \in I \quad \text{(cobertura dentro de S)}$$

$$\sum_{j \in M_i} x_j \geq 1 \quad \forall\, i \in I \quad \text{(proximidade obrigatória dentro de T)}$$

$$\sum_{j \in J} x_j = p$$

$$x_j \in \{0, 1\}, \quad y_i \in \{0, 1\}$$

Onde $M_i = \{j \in J \mid d_{ij} \leq T\}$ e $T > S$.

---

## 4. Descrição da Implementação

### 4.1 Arquitetura do Código

A implementação foi organizada em módulos Python:

- **`src/mclp_model.py`** — Implementação das formulações MCLP usando a API do Gurobi:
  - `solve_mclp()` — Resolve o MCLP básico (Formulação I)
  - `solve_mclp_mandatory_closeness()` — Resolve o MCLP com mandatory closeness
  - `solve_mclp_range()` — Resolve para uma faixa de valores de p (curvas de custo-efetividade)

- **`src/utils.py`** — Funções auxiliares:
  - Leitura de dados (nós e matriz de distâncias)
  - Cálculo de conjuntos de cobertura
  - Visualização (curvas e mapas da rede)

- **`src/experiments.py`** — Script principal com três experimentos:
  - Experimento 1: Reprodução da Tabela 1 (27 problemas)
  - Experimento 2: Curvas de custo-efetividade
  - Experimento 3: Comparação com mandatory closeness

### 4.2 Decisões de Implementação

1. **Formulação usada:** Implementou-se a Formulação I (maximização) diretamente, sem a transformação para minimização. Ambas são matematicamente equivalentes, mas a maximização 3. **Dados:** A rede de 55 nós de Swain (1971) foi obtida a partir de fontes na literatura. As coordenadas (x, y) dos nós e as populações (total = 640) correspondem aos dados originais utilizados por Church e ReVelle. A matriz de distâncias euclidianas foi calculada a partir das coordenadas.

---

## 5. Ambiente Computacional

| Item | Valor |
|------|-------|
| Sistema Operacional | Windows |
| Linguagem | Python 3.13 |
| Solver | Gurobi (licença acadêmica) |
| Bibliotecas | gurobipy, numpy, pandas, matplotlib |

*Nota: O artigo original (1974) usou um IBM 360 Model 91 com o sistema MPS. A comparação de tempos de execução é naturalmente incompatível.*

---

## 6. Resultados Obtidos

### 6.1 Tabela 1 — Soluções Ótimas (55 nós)

| S | p | Pop. Coberta | Cobertura (%) | LP Inteiro | Tempo (s) |
|---|---|-------------|---------------|------------|-----------|
| 10 | 1 | 425 | 66.4% | Sim | 0.029 |
| 10 | 2 | 502 | 78.4% | Sim | 0.004 |
| 10 | 3 | 548 | 85.6% | Sim | 0.003 |
| 10 | 4 | 581 | 90.8% | Sim | 0.004 |
| 10 | 5 | 609 | 95.2% | Sim | 0.003 |
| 10 | 6 | 625 | 97.7% | Sim | 0.003 |
| 10 | 7 | 633 | 98.9% | Sim | 0.003 |
| 10 | 8 | 638 | 99.7% | Sim | 0.014 |
| 10 | 9 | 640 | 100.0% | Sim | 0.003 |
| 12 | 1 | 457 | 71.4% | Sim | 0.005 |
| 12 | 2 | 533 | 83.3% | Sim | 0.003 |
| 12 | 3 | 574 | 89.7% | Sim | 0.007 |
| 12 | 4 | 610 | 95.3% | Sim | 0.007 |
| 12 | 5 | 632 | 98.8% | Sim | 0.003 |
| 12 | 6 | 638 | 99.7% | Sim | 0.003 |
| 12 | 7 | 640 | 100.0% | Sim | 0.003 |
| 15 | 1 | 515 | 80.5% | Sim | 0.006 |
| 15 | 2 | 593 | 92.7% | Sim | 0.003 |
| 15 | 3 | 625 | 97.7% | Sim | 0.003 |
| 15 | 4 | 638 | 99.7% | Sim | 0.003 |
| 15 | 5 | 640 | 100.0% | Sim | 0.004 |
| 19 | 1 | 567 | 88.6% | Sim | 0.004 |
| 19 | 2 | 629 | 98.3% | Sim | 0.003 |
| 19 | 3 | 640 | 100.0% | Sim | 0.006 |
| 20 | 1 | 573 | 89.5% | Sim | 0.003 |
| 20 | 2 | 631 | 98.6% | Sim | 0.003 |
| 20 | 3 | 640 | 100.0% | Sim | 0.003 |

**Observação sobre a relaxação LP:** 100% das soluções LP foram inteiras nesta instância. O artigo reporta ~80%. Isso ocorre porque o artigo utilizava a rede como um **grafo com arcos** (e distâncias de caminho mínimo), enquanto nesta implementação utilizamos **distâncias euclidianas** entre coordenadas. A estrutura ligeiramente diferente da matriz de distâncias pode influenciar a integralidade da relaxação LP.

### 6.2 Curvas de Custo-Efetividade

As curvas mostram o padrão característico de rendimentos decrescentes descrito no artigo: o incremento marginal na cobertura diminui conforme mais facilidades são adicionadas. Para S=10, a cobertura total é atingida com p=9; para S=15, com p=5.

### 6.3 Mandatory Closeness Constraints

| Cenário | Pop. coberta (d ≤ 10) | Pop. coberta (d ≤ 15) |
|---------|:---:|:---:|
| Set Covering S=15, p=5 | 224 | 640 |
| MCLP S=10, T=15, p=5 (mandatory) | **354** | 640 |
| MCLP S=10, p=5 (sem mandatory) | **609** | 628 |

Os resultados do MCLP (Figuras 3 e 4) correspondem **exatamente** aos valores reportados no artigo:
- **MCLP com mandatory closeness (S=10, T=15, p=5): cobertura = 354** — valor idêntico ao artigo.
- **MCLP sem mandatory (S=10, p=5): cobertura = 609** — valor idêntico ao artigo.

O valor do Set Covering (Fig. 2) difere (224 vs. 201 do artigo) porque o set covering possui múltiplas soluções ótimas — qualquer combinação de 5 facilidades que cubra todos os nós dentro de 15 é ótima. A solução encontrada pelo Gurobi é uma alternativa ótima diferente da apresentada no artigo.

---

## 7. Comparação com os Resultados do Artigo

### 7.1 Resultados Reproduzidos com Exatidão

| Resultado | Artigo (1974) | Nossa Implementação | Correspondência |
|-----------|:---:|:---:|:---:|
| MCLP S=10, T=15, p=5 (Fig. 3) | 354 | **354** | ✅ Exato |
| MCLP S=10, p=5 (Fig. 4) | 609 | **609** | ✅ Exato |
| p mínimo cobertura total S=15 | 5 | **5** | ✅ Exato |
| Pop. fora de T=15 (Fig. 4) | 12 | **12** | ✅ Exato |

### 7.2 Diferenças Esperadas

A diferença na cobertura dentro de S=10 para o Set Covering (224 vs. 201) é esperada e se explica pela existência de múltiplas soluções ótimas alternativas. Ambas as soluções cobrem 100% da população dentro de S=15 (objetivo do set covering), mas localizam as facilidades em posições diferentes, resultando em diferentes coberturas dentro da distância menor S=10.

### 7.3 Padrões Qualitativos Reproduzidos

Todos os padrões qualitativos descritos no artigo foram confirmados:

1. **Monotonia:** A cobertura cresce monotonicamente com p ✓
2. **Rendimentos decrescentes:** O incremento marginal diminui com p ✓
3. **Mandatory closeness:** Cobre menos dentro de S que o MCLP puro, mas garante cobertura total dentro de T ✓
4. **Superioridade do MCLP sobre Set Covering:** O MCLP com mandatory closeness encontra soluções mais desejáveis dentre as alternativas do set covering ✓
5. **Trade-off claro:** Para cobrir todos dentro de T=15, a cobertura dentro de S=10 cai de 609 para 354 — uma redução de 255, que é o "custo" da equidade ✓
6. **LP relaxation:** A relaxação LP tende a produzir soluções inteiras automaticamente ✓

### 7.4 Comparação de Desempenho Computacional

O artigo reporta tempos na ordem de segundos no IBM 360/91 (1974). Nossa implementação resolve cada instância em **milissegundos** (0.003-0.03s), demonstrando o avanço de 50 anos em hardware e software de otimização.

---

## 8. Conclusões

1. O MCLP de Church e ReVelle (1974) foi **implementado com sucesso** usando a API do Gurobi em Python, contemplando ambas as formulações: MCLP básico e MCLP com mandatory closeness constraints.

2. Os resultados numéricos do artigo foram **reproduzidos com exatidão**: cobertura de 354 (MCLP mandatory) e 609 (MCLP sem mandatory) para S=10, p=5, confirmando a correção da implementação e dos dados.

3. As **curvas de custo-efetividade** demonstram sua utilidade como ferramenta de apoio à decisão, ilustrando o trade-off entre investimento (número de facilidades) e qualidade do serviço (cobertura).

4. O MCLP com mandatory closeness constraints mostrou-se uma ferramenta poderosa para encontrar soluções mais desejáveis dentre as alternativas ótimas do Location Set Covering Problem, exatamente como descrito pelos autores.

5. A **relaxação LP** produziu soluções inteiras em 100% dos casos, superando os ~80% do artigo. A diferença é atribuída ao uso de distâncias euclidianas em vez de distâncias de caminho mínimo na rede original.

---

## Referências

1. Church, R. & ReVelle, C. (1974). "The Maximal Covering Location Problem." *Papers of the Regional Science Association*, 32, 101-118.

2. Toregas, C., Swain, R., ReVelle, C., & Bergman, L. (1971). "The Location of Emergency Service Facilities." *Operations Research*, 19(6), 1363-1373.

3. Swain, R. (1971). "A Decomposition Algorithm for a Class of Facility Location Algorithms." Ph.D. thesis, Cornell University.

4. Daskin, M. S. (1995). *Network and Discrete Location: Models, Algorithms, and Applications*. Wiley.man, L. (1971). "The Location of Emergency Service Facilities." *Operations Research*, 19(6), 1363-1373.

3. Swain, R. (1971). "A Decomposition Algorithm for a Class of Facility Location Algorithms." Ph.D. thesis, Cornell University.

4. Daskin, M. S. (1995). *Network and Discrete Location: Models, Algorithms, and Applications*. Wiley.
