# 🏫 Simulador Web Interativo - Alocação Ótima de Abrigos (MCLP)

Aplicação Web Interativa desenvolvida com **Streamlit**, **Folium** e **GeoPandas** para simulação espacial da alocação de abrigos temporários de emergência em **Itabuna/BA**, utilizando o modelo matemático **MCLP Capacitado** (*Maximal Covering Location Problem*) com restrição de barreira fluvial no Rio Cachoeira.

---

## 📁 Arquitetura Autocontida da Pasta `app_simulacao/`

```text
app_simulacao/
├── app.py                      # Arquivo principal da aplicação Streamlit
├── gerar_cenarios.py           # Script para pré-cálculo e exportação em massa de cenários
├── requirements.txt            # Dependências Python do projeto
├── README.md                   # Documentação de arquitetura e uso
└── data/                       # Diretório autocontido de dados
    ├── inputs/                 # Camadas vetoriais originais (escolas, áreas de risco, probabilidades)
    ├── matriz_distancias_rede.csv # Cache de distâncias reais na malha viária (Dijkstra)
    ├── K1_D500/                # Cenário com K=1 e d_max=500m
    ├── K5_D1500/               # Cenário com K=5 e d_max=1500m
    │   ├── abrigos_selecionados.geojson      # Escolas ativadas (pontos)
    │   ├── fluxos_alocacao.geojson          # Desejo de linhas / conexões (linhas)
    │   ├── setores_criticos_deficit.geojson  # Zonas críticas com cobertura e déficit (polígonos)
    │   └── resumo_cenario.json              # Indicadores consolidados (KPIs)
    └── ... (40 cenários calculados)
```

---

## 🚀 Como Executar o Aplicativo

### 1. Iniciar o Web App Streamlit
A partir da raiz do projeto:

```bash
streamlit run app_simulacao/app.py
```

Ou entrando diretamente no diretório do aplicativo:

```bash
cd app_simulacao
streamlit run app.py
```

---

## ⚙️ Como Gerar Novos Cenários

O script `gerar_cenarios.py` utiliza o solver **COIN-OR CBC** via **PuLP** e a malha viária do **OSMnx**:

- **Grade completa padrão** (K de 1 a 10, d_max de 500m a 2000m = 40 cenários):
  ```bash
  python app_simulacao/gerar_cenarios.py
  ```

- **Cenários específicos**:
  ```bash
  python app_simulacao/gerar_cenarios.py --k 3 5 8 --dmax 1000 1500
  ```

---

## 🗺️ Camadas Espaciais no Mapa Folium

1. **Zonas de Risco (Setores Críticos)**:
   - Polígonos dos 23 setores censitários de alto risco de inundação.
   - Gradiente de cores baseado na coluna `cobertura_pct`:
     - **Verde**: 100% de cobertura da demanda crítica.
     - **Amarelo / Laranja**: Cobertura parcial.
     - **Vermelho**: 0% de cobertura (alto déficit residual).
   - Tooltips detalhando Demanda ($D_i$), Moradores Atendidos, Déficit e Cobertura.

2. **Abrigos Selecionados (Marcadores)**:
   - Escolas municipais e estaduais ativadas pelo modelo de otimização ($x_j = 1$).
   - Popups interativos contendo capacidade total instalada ($C_j$), vagas ocupadas, margem do rio e taxa de ocupação física.

3. **Fluxos de Alocação (Spider Diagram)**:
   - Linhas pontilhadas ligando o centroide da zona crítica até o abrigo onde a população foi alocada.
   - Ativado/desativado via botão de alternância (*Toggle*) na barra lateral.
