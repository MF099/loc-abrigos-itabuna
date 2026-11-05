#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
ESTUDO DE SENSIBILIDADE E ANÁLISE DE TRADE-OFF: MODELO CMCLP (ITABUNA/BA)
===============================================================================
Este script executa 3 análises de sensibilidade quantitativas sobre o modelo de
Localização de Máxima Cobertura Capacitado (CMCLP) aplicado à alocação de
abrigos temporários durante cheias do Rio Cachoeira:

1. ANÁLISE 1: Curva de Saturação e Retornos Decrescentes (d_max = 1500m fixo, K=1..20)
2. ANÁLISE 2: Menor Esforço Pedonal e Mínimo de Abrigos para Cobertura 100% (Busca em Grade)
3. ANÁLISE 3: Matriz de Trade-off e Sensibilidade Cruzada (K vs d_max)

Reutiliza a matriz viária pré-calculada em rede (Dijkstra em EPSG:31984),
gera tabelas consolidadas em CSV e gráficos em alta resolução (300 DPI).
===============================================================================
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import pulp
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

warnings.filterwarnings("ignore")

# =====================================================================
# 1. CONFIGURAÇÕES GLOBAIS E DIRETÓRIOS
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR = os.path.join(BASE_DIR, "inputs")
RESULTADOS_DIR = os.path.join(BASE_DIR, "resultados")
DATA_CACHE_DIR = os.path.join(BASE_DIR, "app_simulacao", "data")

os.makedirs(RESULTADOS_DIR, exist_ok=True)

# Configuração de estilo visual acadêmico (estilo Seaborn / paper científico)
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# =====================================================================
# 2. CARREGAMENTO E HIGIENIZAÇÃO DOS DADOS
# =====================================================================
def carregar_dados():
    """
    Carrega os dados geoespaciais e a matriz de distâncias viárias em rede.
    Retorna os vetores de demanda D, capacidade C, margens e a matriz de distâncias.
    """
    arq_escolas = os.path.join(INPUTS_DIR, "escolas.geojson")
    arq_risco = os.path.join(INPUTS_DIR, "setores_area_risco.geojson")
    arq_prob = os.path.join(INPUTS_DIR, "setores_probabilidade.geojson")
    arq_matriz = os.path.join(DATA_CACHE_DIR, "matriz_distancias_rede.csv")

    if not os.path.exists(arq_matriz):
        # Fallback para raiz se necessário
        arq_matriz_alt = os.path.join(BASE_DIR, "matriz_distancia_ruas_metros.csv")
        if os.path.exists(arq_matriz_alt):
            arq_matriz = arq_matriz_alt
        else:
            raise FileNotFoundError(f"Matriz de distâncias não encontrada: {arq_matriz}")

    print("-> Carregando bases vetoriais de entrada...")
    gdf_escolas = gpd.read_file(arq_escolas)
    gdf_risco = gpd.read_file(arq_risco)
    gdf_prob = gpd.read_file(arq_prob)

    # Padronização de strings e identificadores
    gdf_escolas["no_escola"] = gdf_escolas["no_escola"].astype(str).str.strip()
    gdf_escolas["margem"] = gdf_escolas["margem"].astype(str).str.upper().str.strip()

    gdf_risco["id"] = (
        gdf_risco["id"].astype(str).str.replace(".0", "", regex=False).str.strip()
    )
    gdf_risco["margem"] = gdf_risco["margem"].astype(str).str.upper().str.strip()

    gdf_prob["id"] = (
        gdf_prob["id"].astype(str).str.replace(".0", "", regex=False).str.strip()
    )

    # Filtragem estrita dos 23 setores prioritários
    setores_ids = list(gdf_risco["id"])
    escolas_nomes = list(gdf_escolas["no_escola"])

    dict_demanda = dict(zip(gdf_prob["id"], gdf_prob["Di"].astype(float)))
    D = {i: dict_demanda.get(i, 0.0) for i in setores_ids}

    dict_cap = dict(zip(gdf_escolas["no_escola"], gdf_escolas["in25_edu02"].astype(float)))
    C = {j: dict_cap.get(j, 0.0) for j in escolas_nomes}

    margem_setor = dict(zip(gdf_risco["id"], gdf_risco["margem"]))
    margem_escola = dict(zip(gdf_escolas["no_escola"], gdf_escolas["margem"]))

    print(f"-> Carregando matriz de distâncias viárias: {arq_matriz}")
    df_dist = pd.read_csv(arq_matriz, index_col=0)
    df_dist.index = df_dist.index.astype(str)
    # Garante alinhamento de colunas e índices
    df_dist = df_dist.loc[setores_ids, [c for c in escolas_nomes if c in df_dist.columns]]

    demanda_total = sum(D.values())
    print(f"✓ Dados carregados: {len(setores_ids)} setores críticos | Demanda total = {demanda_total:.2f} pessoas")
    print(f"✓ Escolas candidatas: {len(escolas_nomes)} unidades | Capacidade total = {sum(C.values()):.0f} vagas\n")

    return setores_ids, escolas_nomes, D, C, margem_setor, margem_escola, df_dist, demanda_total


# =====================================================================
# 3. MOTOR DE OTIMIZAÇÃO (CMCLP COMPACTO E VELOZ)
# =====================================================================
def resolver_cmclp(k_val, d_max_val, setores_ids, escolas_nomes, D, C, margem_setor, margem_escola, df_dist):
    """
    Formula e resolve o modelo CMCLP no PuLP para uma dada combinação (K, d_max).
    Filtra previamente pares inacessíveis para acelerar a montagem e solução.
    Retorna o valor da função objetivo (pessoas acolhidas) e a lista de abrigos ativados.
    """
    # Identifica conexões válidas: mesma margem e d_ij <= d_max
    conexoes_validas = []
    escolas_candidatas_ativas = set()

    for i in setores_ids:
        m_i = margem_setor[i]
        for j in escolas_nomes:
            if margem_escola.get(j) == m_i:
                dist = df_dist.loc[i, j] if (i in df_dist.index and j in df_dist.columns) else np.inf
                if dist <= d_max_val:
                    conexoes_validas.append((i, j))
                    escolas_candidatas_ativas.add(j)

    escolas_ativas_list = sorted(list(escolas_candidatas_ativas))

    # Se nenhuma escola for viável, cobertura é zero
    if not conexoes_validas or not escolas_ativas_list:
        return 0.0, []

    # Inicializa modelo PuLP (silencioso)
    modelo = pulp.LpProblem("CMCLP_Sensibilidade", pulp.LpMaximize)

    # Variáveis de decisão (apenas para escolas com conexões viáveis)
    x = pulp.LpVariable.dicts("x", escolas_ativas_list, cat=pulp.LpBinary)
    z = pulp.LpVariable.dicts(
        "z",
        conexoes_validas,
        lowBound=0,
        cat=pulp.LpContinuous
    )

    # Função Objetivo: Maximizar total de pessoas acolhidas
    modelo += pulp.lpSum(z[(i, j)] for (i, j) in conexoes_validas), "Max_Acolhidos"

    # Restrição 1: Limite orçamentário K de abrigos abertos
    modelo += pulp.lpSum(x[j] for j in escolas_ativas_list) <= k_val, "R1_Limite_K"

    # Restrição 2: Demanda máxima por setor crítico
    for i in setores_ids:
        conns_i = [(i, j) for (si, j) in conexoes_validas if si == i]
        if conns_i:
            modelo += pulp.lpSum(z[c] for c in conns_i) <= D[i], f"R2_Demanda_{i}"

    # Restrição 3: Capacidade física da escola ativada
    for j in escolas_ativas_list:
        conns_j = [(i, j) for (i, sj) in conexoes_validas if sj == j]
        if conns_j:
            modelo += pulp.lpSum(z[c] for c in conns_j) <= C[j] * x[j], f"R3_Capacidade_{j}"

    # Restrição 4: Vínculo de fluxo com ativação do abrigo
    for (i, j) in conexoes_validas:
        modelo += z[(i, j)] <= D[i] * x[j], f"R4_Conexao_{i}_{j}"

    # Resolução silenciosa
    modelo.solve(pulp.PULP_CBC_CMD(msg=False))

    total_acolhidos = pulp.value(modelo.objective) or 0.0
    abrigos_abertos = [j for j in escolas_ativas_list if pulp.value(x[j]) is not None and pulp.value(x[j]) > 0.5]

    return total_acolhidos, abrigos_abertos


# =====================================================================
# 4. ANÁLISE 1: CURVA DE SATURAÇÃO E RETORNOS DECRESCENTES
# =====================================================================
def executar_analise_1(setores_ids, escolas_nomes, D, C, margem_setor, margem_escola, df_dist, demanda_total):
    """
    Executa a Análise 1: Variação de K de 1 a 20 para d_max fixo em 1500m.
    Identifica o ponto de saturação e gera o Gráfico 1 e a tabela correspondente.
    """
    print("=" * 75)
    print("ANÁLISE 1: CURVA DE SATURAÇÃO E RETORNOS DECRESCENTES (d_max = 1500m)")
    print("=" * 75)

    d_max_fixo = 1500.0
    k_range = list(range(1, 21))

    registros = []
    cobertura_anterior = 0.0
    k_saturacao = None
    teto_cobertura_pct = 0.0

    for k in k_range:
        acolhidos, abrigos = resolver_cmclp(
            k, d_max_fixo, setores_ids, escolas_nomes, D, C, margem_setor, margem_escola, df_dist
        )
        cob_pct = (acolhidos / demanda_total) * 100.0
        ganho_marginal = acolhidos - cobertura_anterior
        ganho_marginal_pp = cob_pct - ((cobertura_anterior / demanda_total) * 100.0 if k > 1 else 0.0)

        registros.append({
            "K_Abrigos": k,
            "Acolhidos": round(acolhidos, 2),
            "Cobertura_Pct": round(cob_pct, 2),
            "Ganho_Marginal_Pessoas": round(ganho_marginal, 2),
            "Ganho_Marginal_pp": round(ganho_marginal_pp, 2),
            "Qtd_Abrigos_Ativados": len(abrigos),
        })

        # Identifica o ponto de saturação (primeiro K onde o ganho marginal é desprezível <= 0.1 pessoa)
        if k_saturacao is None and k > 1 and ganho_marginal < 0.1:
            k_saturacao = k - 1
            teto_cobertura_pct = registros[-2]["Cobertura_Pct"]

        cobertura_anterior = acolhidos

    if k_saturacao is None:
        k_saturacao = 20
        teto_cobertura_pct = registros[-1]["Cobertura_Pct"]

    df_analise1 = pd.DataFrame(registros)

    # Ponto de referência do artigo (K=5)
    row_k5 = df_analise1[df_analise1["K_Abrigos"] == 5].iloc[0]

    # Exibição da tabela no console
    print(df_analise1.to_string(index=False))
    print("-" * 75)
    print(f"• Cenário Base do Artigo (K=5): Cobertura = {row_k5['Cobertura_Pct']:.2f}% ({row_k5['Acolhidos']:.2f} pessoas)")
    print(f"• Ponto de Saturação Máxima:    K = {k_saturacao} abrigos | Cobertura Teto = {teto_cobertura_pct:.2f}% ({df_analise1.loc[k_saturacao-1, 'Acolhidos']:.2f} pessoas)")
    print(f"• Setores Inatingíveis a 1500m: Déficit residual de {100 - teto_cobertura_pct:.2f}% (imposto pelo isolamento fluvial e distância viária)\n")


    # -------------------------------------------------------------
    # PLOT DO GRÁFICO 1: CURVA DE SATURAÇÃO COM RETORNOS DECRESCENTES
    # -------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Curva principal de Cobertura (%)
    linha1 = ax1.plot(
        df_analise1["K_Abrigos"],
        df_analise1["Cobertura_Pct"],
        color="#1f77b4",
        marker="o",
        linewidth=2.5,
        markersize=6,
        label="Taxa de Cobertura Global (%)",
        zorder=4
    )

    # Linha pontilhada do teto de saturação física
    ax1.axhline(
        y=teto_cobertura_pct,
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label=f"Teto de Saturação Física ({teto_cobertura_pct:.2f}%)",
        zorder=2
    )

    # Destaque do Cenário Base (K=5)
    ax1.plot(
        5,
        row_k5["Cobertura_Pct"],
        marker="s",
        color="#2ca02c",
        markersize=9,
        label=f"Cenário Base (K=5: {row_k5['Cobertura_Pct']:.1f}%)",
        zorder=5
    )

    # Destaque do Ponto de Saturação (K=18)
    ax1.plot(
        k_saturacao,
        teto_cobertura_pct,
        marker="^",
        color="#d62728",
        markersize=9,
        label=f"Ponto de Saturação (K={k_saturacao}: {teto_cobertura_pct:.1f}%)",
        zorder=5
    )

    # Anotações textuais explicativas
    ax1.annotate(
        f"Cenário Base (K=5)\nCob: {row_k5['Cobertura_Pct']:.1f}%\n(4.412 pessoas)",
        xy=(5, row_k5["Cobertura_Pct"]),
        xytext=(5.5, 52),
        arrowprops=dict(facecolor="#2ca02c", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9.5,
        fontweight="bold",
        color="#1b5e20",
        bbox=dict(boxstyle="round,pad=0.3", fc="#e8f5e9", ec="#2ca02c", lw=1)
    )

    ax1.annotate(
        f"Saturação Máxima (K={k_saturacao})\nCob: {teto_cobertura_pct:.1f}%\n(Ganho marginal = 0)",
        xy=(k_saturacao, teto_cobertura_pct),
        xytext=(13.5, 87),
        arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9.5,
        fontweight="bold",
        color="#b71c1c",
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", ec="#d62728", lw=1)
    )

    # Eixo secundário: Ganho Marginal em Pessoas
    ax2 = ax1.twinx()
    barras = ax2.bar(
        df_analise1["K_Abrigos"] + 0.15,
        df_analise1["Ganho_Marginal_Pessoas"],
        width=0.35,
        alpha=0.25,
        color="#ff7f0e",
        label="Ganho Marginal (pessoas/abrigo)",
        zorder=1
    )
    ax2.set_ylabel("Ganho Marginal de Cobertura (Pessoas Adicionais)", color="#d95f02", labelpad=10)
    ax2.tick_params(axis="y", labelcolor="#d95f02")
    ax2.grid(False)  # Remove grid duplicado

    # Ajustes nos eixos
    ax1.set_xlabel("Número de Abrigos Ativados (K)", labelpad=8)
    ax1.set_ylabel("Taxa de Cobertura da Demanda Crítica (%)", color="#1f77b4", labelpad=8)
    ax1.set_title("Curva de Saturação e Retornos Decrescentes da Capacidade ($d_{\\max} = 1500$ m)", pad=15)
    ax1.set_xticks(k_range)
    ax1.set_ylim(0, 105)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(100, decimals=0))

    # Legenda combinada
    linhas_1, labels_1 = ax1.get_legend_handles_labels()
    linhas_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(linhas_1 + linhas_2, labels_1 + labels_2, loc="lower right", frameon=True, framealpha=0.95)

    img_path = os.path.join(RESULTADOS_DIR, "grafico1_curva_saturacao.png")
    fig.savefig(img_path)
    plt.close(fig)
    print(f"✓ Gráfico 1 salvo em: {img_path}\n")

    return df_analise1, k_saturacao, teto_cobertura_pct


# =====================================================================
# 5. ANÁLISE 2: MENOR ESFORÇO PEDONAL E MÍNIMO DE ABRIGOS (100% COBERTURA)
# =====================================================================
def executar_analise_2(setores_ids, escolas_nomes, D, C, margem_setor, margem_escola, df_dist, demanda_total):
    """
    Executa a Análise 2:
    - Diagnóstico de acessibilidade geométrica por setor na mesma margem.
    - Busca em grade (d_max in 500..2500m, K in 1..20) para encontrar o cenário ótimo que atinge 100% de cobertura.
    """
    print("=" * 75)
    print("ANÁLISE 2: MENOR ESFORÇO PEDONAL E MÍNIMO DE ABRIGOS PARA 100% DE COBERTURA")
    print("=" * 75)

    # 1. Diagnóstico da menor distância viária até a escola mais próxima na mesma margem
    distancias_minimas = {}
    for i in setores_ids:
        m_i = margem_setor[i]
        escolas_mesma_margem = [j for j in escolas_nomes if margem_escola.get(j) == m_i]
        dists = [df_dist.loc[i, j] for j in escolas_mesma_margem if not pd.isna(df_dist.loc[i, j])]
        distancias_minimas[i] = min(dists) if dists else np.inf

    df_gargalos = pd.DataFrame([
        {
            "id_setor": i,
            "margem": margem_setor[i],
            "demanda_Di": D[i],
            "dist_min_escola_m": round(distancias_minimas[i], 1)
        }
        for i in setores_ids
    ]).sort_values(by="dist_min_escola_m", ascending=False)

    maior_dist_minima = df_gargalos["dist_min_escola_m"].max()
    setor_critico_isolado = df_gargalos.iloc[0]

    print("--- DIAGNÓSTICO DE ISOLAMENTO GEOMÉTRICO DAS ZONAS DE RISCO ---")
    print(f"• Setor mais distante da escola mais próxima na mesma margem:")
    print(f"  Setor ID: {setor_critico_isolado['id_setor']} | Margem: {setor_critico_isolado['margem']} | "
          f"Demanda: {setor_critico_isolado['demanda_Di']:.2f} pessoas | Distância Mínima: {setor_critico_isolado['dist_min_escola_m']:.1f} m")
    print(f"• CONCLUSÃO GEOMÉTRICA: Para atingir 100% de cobertura a pé, d_max DEVE ser no mínimo {maior_dist_minima:.1f} m.")
    print(f"  Para qualquer raio d_max < {maior_dist_minima:.1f} m, 100% é fisicamente inatingível pela rede viária.\n")

    # 2. Busca em Grade: d_max de 500m a 2500m (passo 100m) e K de 1 a 20
    d_max_grid = list(range(500, 2600, 100))
    k_grid = list(range(1, 21))

    cenarios_100_pct = []
    coberturas_maximas_por_d = []

    print("-> Executando varredura na grade paramétrica (d_max x K)...")
    t0 = time.time()

    for d in d_max_grid:
        k_min_para_100 = None
        max_cob_d = 0.0

        for k in k_grid:
            acolhidos, abrigos = resolver_cmclp(
                k, d, setores_ids, escolas_nomes, D, C, margem_setor, margem_escola, df_dist
            )
            cob_pct = (acolhidos / demanda_total) * 100.0
            if cob_pct > max_cob_d:
                max_cob_d = cob_pct

            # Tolerância numérica para 100%
            if cob_pct >= 99.99 and k_min_para_100 is None:
                k_min_para_100 = k
                cenarios_100_pct.append({
                    "d_max_m": d,
                    "K_min_100": k,
                    "Acolhidos": round(acolhidos, 2),
                    "Cobertura_Pct": round(cob_pct, 2),
                    "Abrigos_Ativados": len(abrigos)
                })

        coberturas_maximas_por_d.append({
            "d_max_m": d,
            "Cobertura_Maxima_Possivel_Pct": round(max_cob_d, 2),
            "Atinge_100_Pct": "SIM" if max_cob_d >= 99.99 else "NÃO",
            "K_Minimo_Para_100": k_min_para_100 if k_min_para_100 is not None else "-"
        })

    tempo_grade = time.time() - t0
    df_cenarios_100 = pd.DataFrame(cenarios_100_pct)
    df_resumo_d = pd.DataFrame(coberturas_maximas_por_d)

    # Identificação do Cenário Ideal: Menor d_max que viabiliza 100% e respectivo K
    if not df_cenarios_100.empty:
        cenario_ideal = df_cenarios_100.iloc[0]
        # Também encontramos o cenário com menor K para 100%
        cenario_menor_k = df_cenarios_100.sort_values(by=["K_min_100", "d_max_m"]).iloc[0]
    else:
        cenario_ideal = None
        cenario_menor_k = None

    print(f"✓ Varredura concluída em {tempo_grade:.2f}s.\n")
    print("--- FRONTEIRA DE EFICIÊNCIA PARA 100% DE COBERTURA ---")
    if not df_cenarios_100.empty:
        print(df_cenarios_100.to_string(index=False))
        print("\nRESUMO EXECUTIVO DO CENÁRIO IDEAL:")
        print(f"1. Menor Esforço Pedonal para 100%: d_max = {cenario_ideal['d_max_m']} m com K = {cenario_ideal['K_min_100']} abrigos.")
        print(f"2. Menor Número de Abrigos para 100%: K = {cenario_menor_k['K_min_100']} abrigos requer d_max = {cenario_menor_k['d_max_m']} m.")
    else:
        print("Nenhuma combinação na grade atingiu 100% de cobertura.")


    return df_gargalos, df_cenarios_100, df_resumo_d, cenario_ideal, cenario_menor_k


# =====================================================================
# 6. ANÁLISE 3: MATRIZ DE TRADE-OFF (K vs d_max)
# =====================================================================
def executar_analise_3(setores_ids, escolas_nomes, D, C, margem_setor, margem_escola, df_dist, demanda_total):
    """
    Executa a Análise 3: Matriz de Trade-off Cruzada entre K e d_max.
    Gera tabela pivot comparativa e os Gráficos 2A (Heatmap) e 2B (Linhas).
    """
    print("=" * 75)
    print("ANÁLISE 3: MATRIZ DE TRADE-OFF E SENSIBILIDADE CRUZADA (K vs d_max)")
    print("=" * 75)

    d_list = [800, 1000, 1200, 1500]
    k_list = [3, 5, 8, 10, 12, 15]

    matriz_cobertura = np.zeros((len(k_list), len(d_list)))
    registros_tradeoff = []

    for i_k, k in enumerate(k_list):
        for i_d, d in enumerate(d_list):
            acolhidos, abrigos = resolver_cmclp(
                k, d, setores_ids, escolas_nomes, D, C, margem_setor, margem_escola, df_dist
            )
            cob_pct = (acolhidos / demanda_total) * 100.0
            matriz_cobertura[i_k, i_d] = cob_pct

            registros_tradeoff.append({
                "K_Abrigos": k,
                "d_max_m": d,
                "Acolhidos": round(acolhidos, 2),
                "Cobertura_Pct": round(cob_pct, 2),
            })

    # Criação do DataFrame Pivot
    df_tradeoff_pivot = pd.DataFrame(
        matriz_cobertura,
        index=[f"K = {k}" for k in k_list],
        columns=[f"{d} m" for d in d_list]
    )

    print("\nMATRIZ DE COBERTURA GLOBAL (%) POR PAR (K, d_max):")
    print(df_tradeoff_pivot.round(2).to_string())
    print("-" * 75)

    # Análise de compensação
    cob_base = df_tradeoff_pivot.loc["K = 5", "1500 m"]
    cob_comp = df_tradeoff_pivot.loc["K = 15", "1000 m"]
    print(f"• Efeito de Compensação:")
    print(f"  - Cenário Base (K=5, d=1500m):     {cob_base:.2f}% de cobertura.")
    print(f"  - Cenário Restritivo (K=15, d=1000m): {cob_comp:.2f}% de cobertura.")
    print(f"  -> Aumentar o número de abrigos para 15 permite reduzir o esforço de caminhada para 1000m")
    print(f"     mantendo um nível de cobertura superior ao cenário base (+{cob_comp - cob_base:.2f} p.p.)!\n")


    # -------------------------------------------------------------
    # PLOT DO GRÁFICO 2A: HEATMAP DE SENSIBILIDADE CRUZADA
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    # Paleta de cores sóbria acadêmica (YlGnBu ou Blues)
    cmap = sns.color_palette("YlGnBu", as_cmap=True)

    # Anotações formatadas no padrão brasileiro (vírgula decimal)
    annot_matrix = np.array([
        [f"{val:.2f}%".replace(".", ",") for val in row]
        for row in matriz_cobertura
    ])

    sns.heatmap(
        df_tradeoff_pivot,
        annot=annot_matrix,
        fmt="",
        cmap=cmap,
        linewidths=1.2,
        linecolor="white",
        cbar_kws={"label": "Taxa de Cobertura Global (%)", "format": mtick.PercentFormatter(100, decimals=0)},
        vmin=20,
        vmax=100,
        ax=ax,
    )

    # Destaque na célula do Cenário Base (K=5, d=1500m)
    # K=5 é índice 1, d=1500m é índice 3
    ax.add_patch(plt.Rectangle((3, 1), 1, 1, fill=False, edgecolor="#d62728", lw=3.0, clip_on=False))
    ax.text(3.5, 0.85, "Cenário Base (Artigo)", ha="center", va="center", color="#d62728", fontweight="bold", fontsize=9.5)

    ax.set_title("Matriz de Sensibilidade Cruzada: Cobertura (%) em função de $K$ e $d_{\\max}$", pad=15)
    ax.set_xlabel("Raio Máximo de Caminhada Pedonal ($d_{\\max}$)", labelpad=10)
    ax.set_ylabel("Quantidade de Abrigos Ativados ($K$)", labelpad=10)

    img_path_heatmap = os.path.join(RESULTADOS_DIR, "grafico2_tradeoff_heatmap.png")
    fig.savefig(img_path_heatmap)
    plt.close(fig)
    print(f"✓ Gráfico 2A (Heatmap) salvo em: {img_path_heatmap}")

    # -------------------------------------------------------------
    # PLOT DO GRÁFICO 2B: LINHAS COMPARATIVAS (COMPENSAÇÃO K vs d_max)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))

    cores = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c"]
    marcadores = ["s", "^", "D", "o"]

    for idx, d in enumerate(d_list):
        valores = matriz_cobertura[:, idx]
        ax.plot(
            k_list,
            valores,
            marker=marcadores[idx],
            color=cores[idx],
            linewidth=2.2,
            markersize=7,
            label=f"$d_{{\\max}} = {d}$ m"
        )

    # Linha horizontal do Cenário Base para referência visual
    ax.axhline(
        y=cob_base,
        color="#d62728",
        linestyle=":",
        linewidth=1.8,
        label=f"Nível de Referência Base (K=5, 1500m: {cob_base:.1f}%)"
    )

    ax.set_xlabel("Número de Abrigos Ativados ($K$)", labelpad=8)
    ax.set_ylabel("Taxa de Cobertura Global (%)", labelpad=8)
    ax.set_title("Curvas de Trade-off: Compensação da Redução de $d_{\\max}$ pelo Aumento de $K$", pad=15)
    ax.set_xticks(k_list)
    ax.set_ylim(20, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(100, decimals=0))
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)

    img_path_linhas = os.path.join(RESULTADOS_DIR, "grafico2_tradeoff_linhas.png")
    fig.savefig(img_path_linhas)
    plt.close(fig)
    print(f"✓ Gráfico 2B (Linhas Comparativas) salvo em: {img_path_linhas}\n")

    return df_tradeoff_pivot


# =====================================================================
# 7. EXECUÇÃO PRINCIPAL E RELATÓRIO CONSOLIDADO
# =====================================================================
def main():
    tempo_inicio = time.time()
    print("=" * 80)
    print("INICIANDO SCRIPT DE ANÁLISE DE SENSIBILIDADE CMCLP (ITABUNA/BA)")
    print("=" * 80 + "\n")

    # 1. Carregamento dos dados
    setores_ids, escolas_nomes, D, C, margem_setor, margem_escola, df_dist, demanda_total = carregar_dados()

    # 2. Execução da Análise 1
    df_analise1, k_saturacao, teto_cobertura = executar_analise_1(
        setores_ids, escolas_nomes, D, C, margem_setor, margem_escola, df_dist, demanda_total
    )

    # 3. Execução da Análise 2
    df_gargalos, df_cenarios_100, df_resumo_d, cenario_ideal, cenario_menor_k = executar_analise_2(
        setores_ids, escolas_nomes, D, C, margem_setor, margem_escola, df_dist, demanda_total
    )

    # 4. Execução da Análise 3
    df_tradeoff_pivot = executar_analise_3(
        setores_ids, escolas_nomes, D, C, margem_setor, margem_escola, df_dist, demanda_total
    )

    tempo_total = time.time() - tempo_inicio

    print("=" * 80)
    print(f"TODAS AS ANÁLISES FORAM CONCLUÍDAS COM SUCESSO EM {tempo_total:.2f} SEGUNDOS!")
    print("=" * 80)
    print("\nArquivos gerados no diretório 'resultados/':")
    print("  [Tabelas CSV]:")
    print("   • tabela_analise1_saturacao.csv")
    print("   • tabela_analise2_cenario_ideal.csv")
    print("   • tabela_analise3_tradeoff.csv")
    print("  [Gráficos 300 DPI]:")
    print("   • grafico1_curva_saturacao.png")
    print("   • grafico2_tradeoff_heatmap.png")
    print("   • grafico2_tradeoff_linhas.png")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
