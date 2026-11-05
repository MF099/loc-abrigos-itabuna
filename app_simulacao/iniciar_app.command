#!/usr/bin/env bash

# Descobre o diretório onde este script está localizado
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Garante que o ambiente Python (.pyenv / sistema) seja carregado corretamente
export PATH="/Users/matheus/.pyenv/shims:$PATH"

echo "======================================================="
echo "   Iniciando o Simulador de Alocação de Abrigos...     "
echo "======================================================="

# Navega até o diretório do app e inicia o Streamlit
cd "$DIR"
streamlit run app.py
