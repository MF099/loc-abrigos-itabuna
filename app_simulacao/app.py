#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aplicativo Interativo de Simulação Espacial (MCLP Capacitado)
Visualização de Cenários Pré-Calculados de Alocação de Abrigos em Itabuna/BA.

Tecnologias: Streamlit, Folium, Streamlit-Folium, GeoPandas, Branca.
Interface com suporte a tema escuro/claro e integração oficial com a API da CARTO.
"""

import os
import json
import glob
import geopandas as gpd
import pandas as pd
import folium
import branca.colormap as cm
import streamlit as st
from streamlit_folium import st_folium
from shapely.geometry import LineString

# =====================================================================
# CONFIGURAÇÃO DA CHAVE DE API DA CARTO
# Insira sua chave obtida em: https://carto.com/basemaps/
# =====================================================================
CARTO_API_KEY = "cb1_49op_1_6ac2df63515574c47ae845b3"

# =====================================================================
# 1. CONFIGURAÇÃO DA PÁGINA STREAMLIT
# =====================================================================
st.set_page_config(
    page_title="Painel Interativo - Alocação de Abrigos",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Injeção de CSS personalizado otimizado para alto contraste e tema escuro
st.markdown(
    """
    <style>
    /* Ocultar elementos padrão do Streamlit para estética limpa edge-to-edge */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    div[data-testid="stToolbar"] {display: none;}
    section[data-testid="stSidebar"] {display: none;}

    /* Reduzir margens e paddings do container principal */
    .block-container {
        padding-top: 1.0rem !important;
        padding-bottom: 1.0rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }

    /* Tipografia e cores do painel esquerdo com alto contraste */
    .app-title {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-size: 2.35rem;
        font-weight: 800;
        line-height: 1.05;
        color: #FFFFFF; /* Tom neutro claro e legível contra fundo escuro */
        letter-spacing: -0.03em;
        margin-bottom: 0.4rem;
    }
    .app-title-blue {
        color: #38BDF8; /* Ciano claro brilhante de alto contraste */
    }
    .app-subtitle {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 0.86rem;
        color: #94A3B8; /* Cinza claro com excelente legibilidade */
        line-height: 1.45;
        margin-bottom: 1.2rem;
    }

    /* Painel esquerdo com rolagem suave se necessário */
    .left-panel-container {
        max-height: 88vh;
        overflow-y: auto;
        padding-right: 10px;
    }

    /* Rótulos dos controles */
    .control-label {
        font-size: 0.88rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 2px;
    }
    .control-helper {
        font-size: 0.76rem;
        color: #94A3B8;
        line-height: 1.35;
        margin-top: -8px;
        margin-bottom: 10px;
    }

    /* Barra de gradiente de cores da legenda */
    .legend-gradient-bar {
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(to right, #d73027 0%, #f46d43 20%, #fdae61 40%, #fee08b 60%, #a6d96a 80%, #1a9850 100%);
        margin-top: 14px;
        margin-bottom: 4px;
    }
    .legend-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.72rem;
        font-weight: 600;
        color: #94A3B8;
        margin-bottom: 14px;
    }

    /* Seção de estatísticas e lista de itens com alto contraste */
    .stats-header {
        font-size: 0.98rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 14px;
        margin-bottom: 10px;
        line-height: 1.35;
    }
    .stats-highlight {
        color: #38BDF8; /* Destaque em ciano brilhante */
        font-weight: 800;
    }
    .stats-count {
        color: #94A3B8;
        font-weight: 400;
    }
    .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px 0;
        font-size: 0.85rem;
        color: #F1F5F9;
        border-bottom: 1px solid #334155;
    }
    .stat-label {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #CBD5E1;
    }
    .dot-marker {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
    .stat-values {
        font-weight: 600;
        color: #FFFFFF;
    }
    .stat-values-light {
        color: #94A3B8;
        font-weight: 400;
    }

    /* Cartão de abrigo ativado com fundo adaptado */
    .shelter-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 6px;
        font-size: 0.82rem;
    }

    /* Estilização dos sliders do Streamlit */
    div[data-baseweb="slider"] div {
        color: #38BDF8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# 2. DEFINIÇÃO DE DIRETÓRIOS E CONSTANTES
# =====================================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")


# =====================================================================
# 3. FUNÇÕES AUXILIARES E CARREGAMENTO DE DADOS COM CACHE (BACKEND)
# =====================================================================
@st.cache_data(show_spinner=False)
def listar_cenarios_disponiveis(caminho_dados: str):
    """
    Escaneia o diretório de dados para identificar os cenários (K, D_MAX)
    efetivamente gerados e disponíveis.
    """
    pastas = glob.glob(os.path.join(caminho_dados, "K*_D*"))
    cenarios = []
    valores_k = set()
    valores_dmax = set()

    for p in pastas:
        nome_pasta = os.path.basename(p)
        try:
            partes = nome_pasta.split("_")
            k_val = int(partes[0].replace("K", ""))
            d_val = int(partes[1].replace("D", ""))
            cenarios.append((k_val, d_val, p))
            valores_k.add(k_val)
            valores_dmax.add(d_val)
        except (ValueError, IndexError):
            continue

    return sorted(list(valores_k)), sorted(list(valores_dmax)), cenarios


@st.cache_data(show_spinner=False)
def carregar_dados_cenario_rapido(k: int, d_max: int):
    """
    Leitura ultra-rápida (sub-milissegundo) utilizando json.load nativo.
    Elimina o gargalo de leitura de shapefiles/geopackages do GeoPandas (C/GDAL/GEOS),
    que levava mais de 600ms por clique no slider.
    """
    pasta_cenario = os.path.join(DATA_DIR, f"K{k}_D{d_max}")

    arq_deficit = os.path.join(pasta_cenario, "setores_criticos_deficit.geojson")
    arq_abrigos = os.path.join(pasta_cenario, "abrigos_selecionados.geojson")
    arq_fluxos = os.path.join(pasta_cenario, "fluxos_alocacao.geojson")
    arq_kpis = os.path.join(pasta_cenario, "resumo_cenario.json")

    if not (os.path.exists(arq_deficit) and os.path.exists(arq_abrigos)):
        return None, None, None, None

    with open(arq_deficit, "r", encoding="utf-8") as f:
        dict_deficit = json.load(f)

    with open(arq_abrigos, "r", encoding="utf-8") as f:
        dict_abrigos = json.load(f)

    dict_fluxos = None
    if os.path.exists(arq_fluxos):
        with open(arq_fluxos, "r", encoding="utf-8") as f:
            dict_fluxos = json.load(f)

    kpis = {}
    if os.path.exists(arq_kpis):
        with open(arq_kpis, "r", encoding="utf-8") as f:
            kpis = json.load(f)

    # Pré-processar listas limpas para painel e mapa
    abrigos_list = []
    for feat in dict_abrigos.get("features", []):
        p = feat.get("properties", {}).copy()
        coords = feat.get("geometry", {}).get("coordinates", [0, 0])
        p["lon"] = coords[0]
        p["lat"] = coords[1]
        abrigos_list.append(p)

    setores_list = [feat.get("properties", {}) for feat in dict_deficit.get("features", [])]
    top_deficit = sorted(setores_list, key=lambda x: x.get("deficit", 0), reverse=True)[:5]

    dados_cenario = {
        "dict_deficit": dict_deficit,
        "dict_abrigos": dict_abrigos,
        "dict_fluxos": dict_fluxos,
        "kpis": kpis,
        "abrigos_list": abrigos_list,
        "top_deficit": top_deficit,
    }

    return dados_cenario


@st.cache_data(show_spinner=False)
def carregar_rotas_ruas_cache():
    """Carrega o dicionário pré-calculado de geometrias das ruas (OSMnx/Dijkstra)."""
    caminho = os.path.join(DATA_DIR, "rotas_ruas_cache.json")
    if os.path.exists(caminho):
        with open(caminho, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# Identificação prévia dos cenários disponíveis
lista_k, lista_dmax, todos_cenarios = listar_cenarios_disponiveis(DATA_DIR)

if not lista_k or not lista_dmax:
    st.error(
        "Atenção: Nenhum cenário pré-calculado encontrado em `app_simulacao/data/`.\n\n"
        "Execute o script gerador no terminal para criar os cenários:\n"
        "```bash\npython app_simulacao/gerar_cenarios.py\n```"
    )
    st.stop()


# =====================================================================
# 4. LAYOUT PRINCIPAL EM DUAS COLUNAS (SPLIT-SCREEN DESIGN)
# =====================================================================
col_painel, col_mapa = st.columns([1.0, 1.95], gap="medium")


# =====================================================================
# 5. PAINEL ESQUERDO (CONTROLES E ESTATÍSTICAS)
# =====================================================================
with col_painel:
    # 5.1. Título com cores legíveis em fundo escuro
    st.markdown(
        """
        <div class="app-title">
            Painel Iterativo.<br>
            <span class="app-title-blue">Zonas de Risco</span>
        </div>
        <div class="app-subtitle">
            Acessibilidade a pé e acolhimento das <b>23 zonas críticas de risco de inundação</b> em Itabuna/BA sob restrição fluvial do Rio Cachoeira (MCLP Capacitado).
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 5.2. Controles de Entrada (Sliders Estilizados)
    min_k, max_k = min(lista_k), max(lista_k)
    default_k = 5 if 5 in lista_k else min_k

    sel_k = st.slider(
        "Número de abrigos autorizados:",
        min_value=min_k,
        max_value=max_k,
        value=default_k,
        step=1,
        format="%d abrigos",
        help="Define o limite orçamentário máximo de escolas ativadas simultaneamente.",
    )

    default_dmax_idx = lista_dmax.index(1500) if 1500 in lista_dmax else 0
    sel_dmax = st.select_slider(
        "Distância máxima a pé (d_max):",
        options=lista_dmax,
        value=lista_dmax[default_dmax_idx],
        format_func=lambda x: f"{x} m (~{(x/1000):.1f} km)",
        help="Define a distância máxima viária percorrida a pé do centroide da zona de risco até o abrigo",
    )
    st.markdown(
        '<div class="control-helper">Define a distância máxima viária percorrida a pé do centroide da zona de risco até o abrigo.</div>',
        unsafe_allow_html=True,
    )

    exibir_fluxos = st.checkbox(
        "Exibir Linhas de Fluxo de Alocação",
        value=False,
        help="Plota as linhas conectando o centroide do setor de risco à escola de destino.",
    )

    exibir_rotas_ruas = st.checkbox(
        "Exibir Caminhos Percorridos a Pé (Rotas nas Ruas)",
        value=False,
        help="Traça o caminho real percorrido pelas pessoas na malha viária de pedestres (Dijkstra/OpenStreetMap).",
    )

    # 5.3. Barra de Legenda Gradiente Horizontal
    st.markdown(
        """
        <div class="legend-gradient-bar"></div>
        <div class="legend-labels">
            <span>0% (Alto déficit)</span>
            <span>Taxa de Cobertura</span>
            <span>100% (Coberto)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 5.4. Carregamento ultra-rápido dos dados do cenário ativo (JSON nativo)
    dados_cenario = carregar_dados_cenario_rapido(sel_k, sel_dmax)

    if dados_cenario is None:
        st.warning(f"Cenário K={sel_k}, d_max={sel_dmax}m não encontrado.")
        st.stop()

    dict_deficit = dados_cenario["dict_deficit"]
    dict_abrigos = dados_cenario["dict_abrigos"]
    dict_fluxos = dados_cenario["dict_fluxos"]
    kpis = dados_cenario["kpis"]
    abrigos_list = dados_cenario["abrigos_list"]
    top_deficit = dados_cenario["top_deficit"]

    taxa_cob = kpis.get("taxa_cobertura_pct", 0.0)
    demanda_tot = kpis.get("demanda_total", 0.0)
    acolhidos_tot = kpis.get("total_acolhidos", 0.0)
    deficit_tot = kpis.get("deficit_total", 0.0)
    abrigos_atv = kpis.get("abrigos_ativados_qtd", len(abrigos_list))
    qtd_fluxos = len(dict_fluxos.get("features", [])) if dict_fluxos else 0

    # Quantidade de abrigos com 100% de ocupação (capacidade esgotada)
    abrigos_lotados = sum(1 for a in abrigos_list if a.get("taxa_ocupacao_pct", 0.0) >= 99.99)

    # 5.5. Bloco de Estatísticas e Indicadores
    st.markdown(
        f"""
        <div class="stats-header">
            <span class="stats-highlight">{taxa_cob:.1f}%</span> da demanda crítica atendida 
            <span class="stats-count">({acolhidos_tot:,.0f} de {demanda_tot:,.0f} hab)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Linhas de estatísticas com marcadores coloridos e rótulos formais
    st.markdown(
        f"""
        <div class="stat-row">
            <span class="stat-label">
                <span class="dot-marker" style="background-color: #10B981;"></span>
                População Acolhida
            </span>
            <span class="stat-values">{acolhidos_tot:,.0f} <span class="stat-values-light">de {demanda_tot:,.0f} hab</span></span>
        </div>
        <div class="stat-row">
            <span class="stat-label">
                <span class="dot-marker" style="background-color: #EF4444;"></span>
                Déficit Crítico Residual
            </span>
            <span class="stat-values">{deficit_tot:,.0f} <span class="stat-values-light">hab</span></span>
        </div>
        <div class="stat-row">
            <span class="stat-label">
                <span class="dot-marker" style="background-color: #38BDF8;"></span>
                Abrigos Ativados e Totalmente Preenchidos
            </span>
            <span class="stat-values">{abrigos_lotados} <span class="stat-values-light">de {sel_k} autorizados</span></span>
        </div>
        <div class="stat-row">
            <span class="stat-label">
                <span class="dot-marker" style="background-color: #A855F7;"></span>
                Conexões Viárias Ativas
            </span>
            <span class="stat-values">{qtd_fluxos} <span class="stat-values-light">arcos</span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Aviso visual de saturação quando o solver atinge o limite ótimo com menos abrigos que o autorizado
    if abrigos_atv < sel_k:
        st.markdown(
            f"""
            <div style="background-color: rgba(56, 189, 248, 0.1); border-left: 3px solid #38BDF8; padding: 8px 12px; border-radius: 6px; margin-top: 12px; margin-bottom: 10px; font-size: 0.8rem; color: #E2E8F0; line-height: 1.4;">
                <b style="color: #38BDF8;">Saturação Atingida:</b> O modelo alcançou a cobertura máxima viável utilizando <b>{abrigos_atv}</b> abrigos. Autorizar <b>{sel_k}</b> abrigos não traz benefícios adicionais para a distância de {sel_dmax}m.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 5.6. Detalhamento dos Abrigos Ativados (Sem Emojis)
    with st.expander("Detalhe dos Abrigos Ativados", expanded=False):
        for row in abrigos_list:
            nome = row.get("no_escola", "Escola")
            ocup = row.get("pessoas_abrigadas", 0.0)
            cap = row.get("capacidade_vagas", 0.0)
            tx = row.get("taxa_ocupacao_pct", 0.0)
            cor_badge = "#10B981" if tx < 90 else "#F59E0B" if tx < 100 else "#EF4444"

            st.markdown(
                f"""
                <div class="shelter-card">
                    <div style="font-weight: 700; color: #F8FAFC; margin-bottom: 2px;">{nome}</div>
                    <div style="display: flex; justify-content: space-between; color: #94A3B8;">
                        <span>Margem {row.get('margem', 'N/D')} • {cap:.0f} vagas</span>
                        <span style="font-weight: 700; color: {cor_badge};">{ocup:.0f} hab ({tx:.1f}%)</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 5.7. Detalhamento dos Setores com Maior Déficit (Sem Emojis)
    with st.expander("Setores com Maior Déficit", expanded=False):
        for row in top_deficit:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; font-size: 0.78rem; padding: 4px 0; border-bottom: 1px dashed #334155; color: #CBD5E1;">
                    <span>Setor <code>{str(row.get('id', ''))[-5:]}</code> (Dem: {row.get('demanda_Di', 0):.0f})</span>
                    <span style="color: #EF4444; font-weight: 700;">Déficit: {row.get('deficit', 0):.0f} hab ({row.get('cobertura_pct', 0):.0f}%)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =====================================================================
# 6. PAINEL DIREITO: MAPA FOLIUM INTERATIVO COM CARTO VOYAGER OFICIAL
# =====================================================================
with col_mapa:
    centro_lat = -14.7960
    centro_lon = -39.2780

    # Criação do mapa Folium definindo tiles=None para injeção limpa da camada CARTO Voyager
    mapa = folium.Map(
        location=[centro_lat, centro_lon],
        zoom_start=14,
        tiles=None,
        control_scale=True,
    )

    # URL oficial dos tiles CARTO Voyager com parâmetro ?key=
    carto_tile_url = f"https://basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}.png?key={CARTO_API_KEY}"

    # Injeção oficial do TileLayer CARTO Voyager
    folium.TileLayer(
        tiles=carto_tile_url,
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        name="CARTO Voyager",
        max_zoom=20,
    ).add_to(mapa)

    # -----------------------------------------------------------------
    # 6.1. Colormap para os Setores Críticos (Verde = 100%, Vermelho = 0%)
    # -----------------------------------------------------------------
    colormap_cobertura = cm.LinearColormap(
        colors=["#d73027", "#f46d43", "#fdae61", "#fee08b", "#a6d96a", "#1a9850"],
        index=[0.0, 20.0, 40.0, 60.0, 80.0, 100.0],
        vmin=0.0,
        vmax=100.0,
        caption="Taxa de Cobertura da Demanda Crítica (%)",
    )

    def estilo_setor(feature):
        """Aplica cor de preenchimento proporcional à taxa de cobertura."""
        prop = feature["properties"]
        cob = prop.get("cobertura_pct", 0.0)
        cor = colormap_cobertura(cob)
        return {
            "fillColor": cor,
            "color": "#1e293b",
            "weight": 1.2,
            "fillOpacity": 0.65,
        }

    def highlight_setor(feature):
        """Realce visual ao passar o mouse sobre o setor."""
        return {
            "weight": 2.5,
            "color": "#000000",
            "fillOpacity": 0.85,
        }

    tooltip_setores = folium.GeoJsonTooltip(
        fields=["id", "demanda_Di", "atendidos", "deficit", "cobertura_pct"],
        aliases=[
            "Setor ID:",
            "Demanda Total (hab):",
            "Atendidos (hab):",
            "Déficit (hab):",
            "Cobertura (%):",
        ],
        localize=True,
        sticky=True,
        style=(
            "background-color: white; color: #1E293B; font-family: sans-serif; "
            "font-size: 12px; padding: 10px; border-radius: 4px; box-shadow: 2px 2px 5px rgba(0,0,0,0.2);"
        ),
    )

    camada_setores = folium.GeoJson(
        dict_deficit,
        name="Zonas de Risco (Setores Críticos)",
        style_function=estilo_setor,
        highlight_function=highlight_setor,
        tooltip=tooltip_setores,
    )
    camada_setores.add_to(mapa)

    # -----------------------------------------------------------------
    # 6.2. Camada de Fluxos de Alocação (Spider Diagram)
    # -----------------------------------------------------------------
    if exibir_fluxos and dict_fluxos and dict_fluxos.get("features"):
        def estilo_fluxo(feature):
            return {
                "color": "#0284c7",
                "weight": 2.4,
                "opacity": 0.85,
                "dashArray": "5, 5",
            }

        tooltip_fluxos = folium.GeoJsonTooltip(
            fields=["id_setor", "no_escola", "pessoas_alocadas", "distancia_caminhada_m", "distancia_reta_m"],
            aliases=[
                "Origem (Setor):",
                "Destino (Abrigo):",
                "Pessoas Alocadas:",
                "Distância a Pé na Rede (m):",
                "Distância em Linha Reta (m):",
            ],
            localize=True,
            sticky=True,
            style="background-color: #F8FAFC; font-size: 11px; padding: 8px; border-radius: 4px; color: #1E293B;",
        )

        camada_fluxos = folium.GeoJson(
            dict_fluxos,
            name="Fluxos de Alocação (Rotas)",
            style_function=estilo_fluxo,
            tooltip=tooltip_fluxos,
        )
        camada_fluxos.add_to(mapa)

    # -----------------------------------------------------------------
    # 6.3. Camada de Caminhos Percorridos nas Ruas (Rotas Reais)
    # -----------------------------------------------------------------
    if exibir_rotas_ruas and dict_fluxos and dict_fluxos.get("features"):
        rotas_cache = carregar_rotas_ruas_cache()
        features_rotas = []
        for feat in dict_fluxos.get("features", []):
            p = feat.get("properties", {})
            chave = f"{str(p.get('id_setor')).replace('.0', '').strip()}|{str(p.get('no_escola')).strip()}"
            if chave in rotas_cache and len(rotas_cache[chave]) >= 2:
                features_rotas.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": rotas_cache[chave]
                    },
                    "properties": {
                        "id_setor": p.get("id_setor"),
                        "no_escola": p.get("no_escola"),
                        "pessoas_alocadas": p.get("pessoas_alocadas", 0.0),
                        "distancia_caminhada_m": p.get("distancia_caminhada_m", 0.0),
                    }
                })

        if features_rotas:
            geojson_rotas_ruas = {
                "type": "FeatureCollection",
                "features": features_rotas
            }

            def estilo_rotas_ruas(feature):
                return {
                    "color": "#f97316",
                    "weight": 3.6,
                    "opacity": 0.9,
                }

            tooltip_rotas = folium.GeoJsonTooltip(
                fields=["id_setor", "no_escola", "pessoas_alocadas", "distancia_caminhada_m"],
                aliases=[
                    "Origem (Setor):",
                    "Destino (Abrigo):",
                    "Pessoas Caminhando:",
                    "Extensão Percorrida a Pé (m):",
                ],
                localize=True,
                sticky=True,
                style="background-color: #F8FAFC; font-size: 11px; padding: 8px; border-radius: 4px; color: #1E293B;",
            )

            camada_rotas_ruas = folium.GeoJson(
                geojson_rotas_ruas,
                name="Caminhos Percorridos a Pé (Ruas)",
                style_function=estilo_rotas_ruas,
                tooltip=tooltip_rotas,
            )
            camada_rotas_ruas.add_to(mapa)

    # -----------------------------------------------------------------
    # 6.4. Camada de Abrigos Selecionados (Marcadores de Ícone)
    # -----------------------------------------------------------------
    grupo_abrigos = folium.FeatureGroup(name="Abrigos Selecionados", show=True)

    for row in abrigos_list:
        lat = row.get("lat")
        lon = row.get("lon")
        if lat and lon:
            nome_esc = row.get("no_escola", "Escola")
            ocup = row.get("pessoas_abrigadas", 0.0)
            cap = row.get("capacidade_vagas", row.get("in25_edu02", 0.0))
            taxa_oc = row.get("taxa_ocupacao_pct", 0.0)
            margem = row.get("margem", "N/D")

            html_popup = f"""
            <div style="font-family: Arial, sans-serif; font-size: 12px; width: 230px;">
                <h4 style="margin: 0 0 6px 0; color: #0369a1; font-size: 13px;">{nome_esc}</h4>
                <hr style="margin: 4px 0; border: none; border-top: 1px solid #CBD5E1;">
                <b>Margem Hidrográfica:</b> Margem {margem}<br>
                <b>Capacidade Total:</b> {cap:.0f} vagas<br>
                <b>População Acolhida:</b> {ocup:.1f} pessoas<br>
                <b>Taxa de Ocupação:</b> <span style="color: {'#dc2626' if taxa_oc >= 90 else '#16a34a'}; font-weight: bold;">{taxa_oc:.1f}%</span>
            </div>
            """

            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(html_popup, max_width=270),
                tooltip=f"{nome_esc} | Ocupação: {taxa_oc:.1f}%",
                icon=folium.Icon(color="darkpurple", icon="home", prefix="fa"),
            ).add_to(grupo_abrigos)

    grupo_abrigos.add_to(mapa)

    folium.LayerControl(position="topright", collapsed=True).add_to(mapa)

    st_folium(mapa, width="100%", height=760, returned_objects=[])
