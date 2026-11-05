#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Gerador de Cenários de Otimização (MCLP Capacitado).
Este script realiza uma varredura paramétrica de cenários variando:
- K: Número máximo de abrigos autorizados (1 a 10)
- D_MAX: Distância máxima viária a pé (ex: 500, 1000, 1500, 2000 metros)

Para cada combinação (K, D_MAX), o modelo de Programação Linear Inteira
é resolvido via PuLP e os resultados espaciais são exportados em GeoJSON
(projetados em WGS84 / EPSG:4326 para máxima compatibilidade web/Folium)
na pasta: app_simulacao/data/K{k}_D{dmax}/
"""

import os
import json
import warnings
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import pulp
from shapely.geometry import LineString

warnings.filterwarnings("ignore")

# =====================================================================
# 1. CONFIGURAÇÕES DE CAMINHOS E SISTEMAS DE COORDENADAS
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUTS_DIR = os.path.join(DATA_DIR, "inputs")

# Fallback para diretório de inputs na raiz do projeto, se necessário
if not os.path.exists(INPUTS_DIR):
    ROOT_INPUTS = os.path.abspath(os.path.join(BASE_DIR, "..", "inputs"))
    if os.path.exists(ROOT_INPUTS):
        INPUTS_DIR = ROOT_INPUTS

EPSG_METRICO = "EPSG:31984"   # SIRGAS 2000 / UTM 24S (cálculos de distância métrica)
EPSG_WEB = "EPSG:4326"        # WGS 84 (coordenadas geográficas para Folium e GeoJSON Web)
CIDADE = "Itabuna, Bahia, Brazil"
NETWORK_TYPE = "walk"


def obter_caminho_input(nome_arquivo):
    """Retorna o caminho completo de um arquivo de entrada."""
    caminho = os.path.join(INPUTS_DIR, nome_arquivo)
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {caminho}")
    return caminho


# =====================================================================
# 2. CARREGAMENTO E HIGIENIZAÇÃO DOS DADOS
# =====================================================================
def carregar_dados():
    """
    Carrega e padroniza as bases vetoriais de escolas, setores críticos e probabilidades.
    """
    arq_escolas = obter_caminho_input("escolas.geojson")
    arq_risco = obter_caminho_input("setores_area_risco.geojson")
    arq_prob = obter_caminho_input("setores_probabilidade.geojson")

    gdf_escolas = gpd.read_file(arq_escolas)
    gdf_centroides_risco = gpd.read_file(arq_risco)
    gdf_setores_prob = gpd.read_file(arq_prob)

    # Padronização de identificadores e strings
    gdf_escolas["no_escola"] = gdf_escolas["no_escola"].astype(str).str.strip()
    gdf_escolas["margem"] = gdf_escolas["margem"].astype(str).str.upper().str.strip()

    gdf_centroides_risco["id"] = (
        gdf_centroides_risco["id"].astype(str).str.replace(".0", "", regex=False).str.strip()
    )
    gdf_centroides_risco["margem"] = (
        gdf_centroides_risco["margem"].astype(str).str.upper().str.strip()
    )

    gdf_setores_prob["id"] = (
        gdf_setores_prob["id"].astype(str).str.replace(".0", "", regex=False).str.strip()
    )

    # Filtragem estrita dos 23 setores prioritários
    setores_ids = list(gdf_centroides_risco["id"])
    escolas_nomes = list(gdf_escolas["no_escola"])

    gdf_setores_criticos = gdf_setores_prob[gdf_setores_prob["id"].isin(setores_ids)].copy()

    # Mapeamento de demanda (D_i), capacidades (C_j) e margens
    dict_demanda_prob = dict(zip(gdf_setores_prob["id"], gdf_setores_prob["Di"].astype(float)))
    D = {i: dict_demanda_prob.get(i, 0.0) for i in setores_ids}

    dict_capacidade_escolas = dict(
        zip(gdf_escolas["no_escola"], gdf_escolas["in25_edu02"].astype(float))
    )
    C = {j: dict_capacidade_escolas.get(j, 0.0) for j in escolas_nomes}

    margem_setor = dict(zip(gdf_centroides_risco["id"], gdf_centroides_risco["margem"]))
    margem_escola = dict(zip(gdf_escolas["no_escola"], gdf_escolas["margem"]))

    return (
        gdf_escolas,
        gdf_centroides_risco,
        gdf_setores_criticos,
        setores_ids,
        escolas_nomes,
        D,
        C,
        margem_setor,
        margem_escola,
    )


# =====================================================================
# 3. CÁLCULO E CACHE DA MATRIZ DE DISTÂNCIAS REAIS NA REDE VIÁRIA
# =====================================================================
def obter_matriz_distancias(gdf_centroides_risco, gdf_escolas, setores_ids, escolas_nomes):
    """
    Calcula a matriz de distâncias reais na malha viária de pedestres (Dijkstra).
    Utiliza cache persistente em parquet/csv para evitar requisições repetidas ao OSM.
    """
    cache_path = os.path.join(DATA_DIR, "matriz_distancias_rede.csv")

    if os.path.exists(cache_path):
        print(f"-> Carregando matriz de distâncias viárias do cache: {cache_path}")
        df_dist = pd.read_csv(cache_path, index_col=0)
        df_dist.index = df_dist.index.astype(str)
        return df_dist

    print(f"-> Baixando rede viária de pedestres ({NETWORK_TYPE}) para {CIDADE} via OSMnx...")
    G = ox.graph_from_place(CIDADE, network_type=NETWORK_TYPE)
    G_proj = ox.project_graph(G, to_crs=EPSG_METRICO)

    gdf_centroides_proj = gdf_centroides_risco.to_crs(EPSG_METRICO)
    gdf_escolas_proj = gdf_escolas.to_crs(EPSG_METRICO)

    print("-> Identificando nós mais próximos no grafo...")
    origens_nodes = ox.nearest_nodes(
        G_proj,
        X=gdf_centroides_proj.geometry.x,
        Y=gdf_centroides_proj.geometry.y,
    )
    destinos_nodes = ox.nearest_nodes(
        G_proj,
        X=gdf_escolas_proj.geometry.x,
        Y=gdf_escolas_proj.geometry.y,
    )

    dict_nos_origem = dict(zip(setores_ids, origens_nodes))
    dict_nos_destino = dict(zip(escolas_nomes, destinos_nodes))

    print("-> Executando Dijkstra para todos os 23 setores críticos...")
    matriz_dist = []
    for origem_id in setores_ids:
        no_origem = dict_nos_origem[origem_id]
        distancias_origem = nx.single_source_dijkstra_path_length(
            G_proj, source=no_origem, weight="length"
        )
        linha = {}
        for escola_nome in escolas_nomes:
            no_destino = dict_nos_destino[escola_nome]
            linha[escola_nome] = distancias_origem.get(no_destino, np.inf)
        matriz_dist.append(linha)

    df_dist = pd.DataFrame(matriz_dist, index=setores_ids)
    os.makedirs(DATA_DIR, exist_ok=True)
    df_dist.to_csv(cache_path)
    print(f"-> Matriz de distâncias salva no cache: {cache_path}")
    return df_dist


# =====================================================================
# 4. SOLVER DO MODELO DE OTIMIZAÇÃO (MCLP CAPACITADO COM BARREIRA FLUVIAL)
# =====================================================================
def otimizar_cenario(
    k_abrigos,
    d_max,
    df_distancias,
    setores_ids,
    escolas_nomes,
    D,
    C,
    margem_setor,
    margem_escola,
    gdf_escolas,
    gdf_centroides_risco,
    gdf_setores_criticos,
):
    """
    Configura e resolve o modelo PuLP para um par de parâmetros (K, D_MAX).
    Retorna os 3 GeoDataFrames resultantes e um dicionário com KPIs consolidados.
    """
    # Matriz binária de acessibilidade com barreira fluvial
    matriz_acessivel = (df_distancias <= d_max).astype(int)
    a = {}
    for i in setores_ids:
        for j in escolas_nomes:
            if margem_setor.get(i) != margem_escola.get(j):
                matriz_acessivel.loc[i, j] = 0
            a[(i, j)] = int(matriz_acessivel.loc[i, j])

    # Inicialização do modelo PuLP
    nome_modelo = f"MCLP_K{k_abrigos}_D{int(d_max)}"
    modelo = pulp.LpProblem(nome_modelo, pulp.LpMaximize)

    # Variáveis de decisão
    x = pulp.LpVariable.dicts("Abrigo_Aberto", escolas_nomes, cat=pulp.LpBinary)
    z = pulp.LpVariable.dicts(
        "Alocacao_Pessoas",
        (setores_ids, escolas_nomes),
        lowBound=0,
        cat=pulp.LpContinuous,
    )

    # Função Objetivo: Maximizar o acolhimento das zonas críticas
    modelo += pulp.lpSum(z[i][j] for i in setores_ids for j in escolas_nomes)

    # R1: Limite máximo de K abrigos
    modelo += pulp.lpSum(x[j] for j in escolas_nomes) <= k_abrigos, "R1_Limite_K_Abrigos"

    # R2: Limite pela demanda do setor crítico D_i
    for i in setores_ids:
        modelo += pulp.lpSum(z[i][j] for j in escolas_nomes) <= D[i], f"R2_Demanda_{i}"

    # R3: Limite pela capacidade física da escola C_j * x_j
    for j in escolas_nomes:
        modelo += pulp.lpSum(z[i][j] for i in setores_ids) <= C[j] * x[j], f"R3_Capacidade_{j}"

    # R4: Acessibilidade viária e restrição do Rio Cachoeira
    for i in setores_ids:
        for j in escolas_nomes:
            if a.get((i, j), 0) == 0:
                modelo += z[i][j] == 0, f"R4_Inacessivel_{i}_{j}"
            else:
                modelo += z[i][j] <= D[i] * x[j], f"R4_Conexao_Ativa_{i}_{j}"

    # Resolução via CBC Solver
    modelo.solve(pulp.PULP_CBC_CMD(msg=False))

    # Indicadores consolidados
    demanda_total = float(sum(D.values()))
    total_acolhidos = float(pulp.value(modelo.objective) or 0.0)
    deficit_total = max(0.0, demanda_total - total_acolhidos)
    taxa_cobertura = (total_acolhidos / demanda_total * 100.0) if demanda_total > 0 else 0.0

    abrigos_ativados = [
        j for j in escolas_nomes
        if pulp.value(x[j]) is not None and round(pulp.value(x[j])) == 1
    ]

    # 1. GeoDataFrame: Abrigos Selecionados
    gdf_abrigos_out = gdf_escolas[gdf_escolas["no_escola"].isin(abrigos_ativados)].copy()
    ocupacao_map = {}
    for j in abrigos_ativados:
        tot_aloc = sum(
            pulp.value(z[i][j]) for i in setores_ids if pulp.value(z[i][j]) is not None
        )
        ocupacao_map[j] = round(tot_aloc, 2)

    gdf_abrigos_out["pessoas_abrigadas"] = gdf_abrigos_out["no_escola"].map(ocupacao_map).fillna(0.0)
    gdf_abrigos_out["capacidade_vagas"] = gdf_abrigos_out["in25_edu02"].astype(float)
    gdf_abrigos_out["taxa_ocupacao_pct"] = np.where(
        gdf_abrigos_out["capacidade_vagas"] > 0,
        (gdf_abrigos_out["pessoas_abrigadas"] / gdf_abrigos_out["capacidade_vagas"] * 100.0).round(2),
        0.0,
    )
    gdf_abrigos_out["status_aberto"] = 1

    # 2. GeoDataFrame: Fluxos de Alocação
    gdf_centr_metr = gdf_centroides_risco.to_crs(EPSG_METRICO)
    gdf_esc_metr = gdf_escolas.to_crs(EPSG_METRICO)
    geom_centr_dict = dict(zip(gdf_centr_metr["id"], gdf_centr_metr.geometry))
    geom_esc_dict = dict(zip(gdf_esc_metr["no_escola"], gdf_esc_metr.geometry))

    linhas_fluxo = []
    for i in setores_ids:
        for j in escolas_nomes:
            val_z = pulp.value(z[i][j])
            if val_z is not None and round(val_z, 4) > 0:
                pt_orig = geom_centr_dict.get(i)
                pt_dest = geom_esc_dict.get(j)
                if pt_orig is not None and pt_dest is not None:
                    linha = LineString([pt_orig, pt_dest])
                    dist_real = float(df_distancias.loc[i, j]) if (i in df_distancias.index and j in df_distancias.columns) else round(float(linha.length), 2)
                    linhas_fluxo.append({
                        "id_setor": i,
                        "no_escola": j,
                        "margem": margem_setor.get(i),
                        "pessoas_alocadas": round(float(val_z), 2),
                        "distancia_caminhada_m": round(dist_real, 2),
                        "distancia_reta_m": round(float(linha.length), 2),
                        "geometry": linha,
                    })

    if linhas_fluxo:
        gdf_fluxos_out = gpd.GeoDataFrame(linhas_fluxo, crs=EPSG_METRICO)
    else:
        # GeoDataFrame vazio estruturado
        gdf_fluxos_out = gpd.GeoDataFrame(
            columns=["id_setor", "no_escola", "margem", "pessoas_alocadas", "distancia_caminhada_m", "distancia_reta_m", "geometry"],
            crs=EPSG_METRICO,
        )

    # 3. GeoDataFrame: Setores Críticos e Déficit
    dados_deficit = []
    for i in setores_ids:
        dem_i = float(D.get(i, 0.0))
        atend_i = sum(
            pulp.value(z[i][j]) for j in escolas_nomes if pulp.value(z[i][j]) is not None
        )
        def_i = max(0.0, dem_i - atend_i)
        cob_i = (atend_i / dem_i * 100.0) if dem_i > 0 else 100.0
        dados_deficit.append({
            "id_setor": i,
            "demanda_Di": round(dem_i, 2),
            "atendidos": round(float(atend_i), 2),
            "deficit": round(float(def_i), 2),
            "cobertura_pct": round(float(cob_i), 2),
        })

    df_def = pd.DataFrame(dados_deficit)
    gdf_deficit_out = gdf_setores_criticos[["id", "geometry"]].merge(
        df_def, left_on="id", right_on="id_setor", how="inner"
    )
    gdf_deficit_out = gdf_deficit_out[
        ["id", "demanda_Di", "atendidos", "deficit", "cobertura_pct", "geometry"]
    ]

    # Conversão de todas as camadas para WGS84 (EPSG:4326) para Folium e GeoJSON padrão
    gdf_abrigos_wgs84 = gdf_abrigos_out.to_crs(EPSG_WEB)
    gdf_fluxos_wgs84 = gdf_fluxos_out.to_crs(EPSG_WEB)
    gdf_deficit_wgs84 = gdf_deficit_out.to_crs(EPSG_WEB)

    kpis = {
        "k": k_abrigos,
        "d_max": d_max,
        "status": pulp.LpStatus[modelo.status],
        "demanda_total": round(demanda_total, 2),
        "total_acolhidos": round(total_acolhidos, 2),
        "deficit_total": round(deficit_total, 2),
        "taxa_cobertura_pct": round(taxa_cobertura, 2),
        "abrigos_ativados_qtd": len(abrigos_ativados),
        "abrigos_ativados_nomes": abrigos_ativados,
    }

    return gdf_deficit_wgs84, gdf_abrigos_wgs84, gdf_fluxos_wgs84, kpis


# =====================================================================
# 5. LOOP DE GERAÇÃO EM MASSA DE CENÁRIOS
# =====================================================================
def executar_geracao(lista_k=None, lista_dmax=None):
    """Executa a geração de cenários para a grade completa de K e D_MAX."""
    if lista_k is None:
        lista_k = list(range(1, 91))  # 1 a 10
    if lista_dmax is None:
        lista_dmax = [500, 750, 1000, 1250, 1500, 2000, 3000, 4000, 5000]

    print("=" * 70)
    print(" INICIANDO GERAÇÃO DE CENÁRIOS DE OTIMIZAÇÃO (MCLP)")
    print(f" • Cenários de K ({len(lista_k)} valores): {lista_k}")
    print(f" • Cenários de D_max ({len(lista_dmax)} valores): {lista_dmax}")
    print(f" • Total de combinações a calcular: {len(lista_k) * len(lista_dmax)}")
    print("=" * 70)

    # Carrega dados
    (
        gdf_escolas,
        gdf_centroides_risco,
        gdf_setores_criticos,
        setores_ids,
        escolas_nomes,
        D,
        C,
        margem_setor,
        margem_escola,
    ) = carregar_dados()

    # Obtém matriz de distâncias viárias (calcula ou busca em cache)
    df_distancias = obter_matriz_distancias(
        gdf_centroides_risco, gdf_escolas, setores_ids, escolas_nomes
    )

    cenario_idx = 0
    total_cenarios = len(lista_k) * len(lista_dmax)

    for k in lista_k:
        for dmax in lista_dmax:
            cenario_idx += 1
            nome_pasta = f"K{k}_D{int(dmax)}"
            dir_cenario = os.path.join(DATA_DIR, nome_pasta)
            os.makedirs(dir_cenario, exist_ok=True)

            print(f"[{cenario_idx}/{total_cenarios}] Processando {nome_pasta}...", end=" ")

            gdf_def, gdf_abr, gdf_flux, kpis = otimizar_cenario(
                k_abrigos=k,
                d_max=float(dmax),
                df_distancias=df_distancias,
                setores_ids=setores_ids,
                escolas_nomes=escolas_nomes,
                D=D,
                C=C,
                margem_setor=margem_setor,
                margem_escola=margem_escola,
                gdf_escolas=gdf_escolas,
                gdf_centroides_risco=gdf_centroides_risco,
                gdf_setores_criticos=gdf_setores_criticos,
            )

            # Exportação dos GeoJSONs
            arq_deficit = os.path.join(dir_cenario, "setores_criticos_deficit.geojson")
            arq_abrigos = os.path.join(dir_cenario, "abrigos_selecionados.geojson")
            arq_fluxos = os.path.join(dir_cenario, "fluxos_alocacao.geojson")
            arq_kpis = os.path.join(dir_cenario, "resumo_cenario.json")

            gdf_def.to_file(arq_deficit, driver="GeoJSON")
            gdf_abr.to_file(arq_abrigos, driver="GeoJSON")
            gdf_flux.to_file(arq_fluxos, driver="GeoJSON")

            with open(arq_kpis, "w", encoding="utf-8") as f:
                json.dump(kpis, f, indent=2, ensure_ascii=False)

            print(
                f"OK! Acolhidos: {kpis['total_acolhidos']} / {kpis['demanda_total']} "
                f"({kpis['taxa_cobertura_pct']}%) | Abrigos: {kpis['abrigos_ativados_qtd']}"
            )

    print("\n" + "=" * 70)
    print("✓ GERAÇÃO DE TODOS OS CENÁRIOS CONCLUÍDA COM SUCESSO!")
    print(f"✓ Dados salvos em: {DATA_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gerador de Cenários MCLP")
    parser.add_argument("--k", nargs="+", type=int, default=None, help="Lista de valores de K (ex: --k 1 2 3)")
    parser.add_argument("--dmax", nargs="+", type=int, default=None, help="Lista de valores de D_max (ex: --dmax 500 1000)")
    args = parser.parse_args()

    executar_geracao(lista_k=args.k, lista_dmax=args.dmax)
