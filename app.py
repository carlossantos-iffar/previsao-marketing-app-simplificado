import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Carregar artefatos
modelo = joblib.load('modelo_random_forest.joblib')
encoders = joblib.load('encoders.joblib')
scaler = joblib.load('scaler.joblib')
colunas = joblib.load('colunas_modelo.joblib')

st.title("🧠 Previsão de Aceitação de Campanha Bancária")

# Inputs do usuário
entrada = {}

for coluna in colunas:
    if coluna in encoders:
        opcoes = encoders[coluna].classes_.tolist()
        entrada[coluna] = st.selectbox(coluna, opcoes)
    else:
        entrada[coluna] = st.number_input(coluna, step=1.0)

# Previsão
if st.button("🔍 Prever"):
    entrada_df = pd.DataFrame([entrada])

    # Aplicar os encoders apenas nas colunas categóricas
    for col in entrada_df.columns:
        if col in encoders:
            entrada_df[col] = encoders[col].transform([entrada_df[col][0]])

    # Escalar os dados
    entrada_scaled = scaler.transform(entrada_df)

    # Previsão
    pred = modelo.predict(entrada_scaled)[0]
    prob = modelo.predict_proba(entrada_scaled)[0][int(pred)]

    st.success(f"Resultado: {'Sim' if pred == 1 else 'Não'} (probabilidade: {prob:.2%})")
