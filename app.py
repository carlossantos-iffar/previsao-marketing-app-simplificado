import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

st.set_page_config(page_title="Campanha Bancária", page_icon="💰")

st.title("🧠 Previsão de Aceitação de Campanha Bancária")

# === 1. Carregamento seguro dos artefatos ===
try:
    modelo = joblib.load("modelo_random_forest.joblib")
    encoders = joblib.load("encoders.joblib")
    scaler = joblib.load("scaler.joblib")
    colunas = joblib.load("colunas_modelo.joblib")
except FileNotFoundError as e:
    st.error(f"Erro ao carregar artefatos: {e}")
    st.stop()

# === 2. Coleta de dados do usuário ===
st.subheader("📥 Preencha os dados do cliente")

entrada = {}

for coluna in colunas:
    if coluna in encoders:
        opcoes = encoders[coluna].classes_.tolist()
        entrada[coluna] = st.selectbox(f"{coluna.capitalize()}:", opcoes)
    else:
        entrada[coluna] = st.number_input(f"{coluna.capitalize()}:", step=1.0)

# === 3. Previsão ===
if st.button("🔍 Prever"):
    entrada_df = pd.DataFrame([entrada])

    # Codificação de variáveis categóricas
    for col in entrada_df.columns:
        if col in encoders:
            entrada_df[col] = [encoders[col].transform([entrada_df[col][0]])[0]]

    # Conversão para float antes de aplicar o scaler
    entrada_df = entrada_df.astype(float)

    # Normalização dos dados
    entrada_scaled = scaler.transform(entrada_df)

    # Predição
    pred = modelo.predict(entrada_scaled)[0]
    prob = modelo.predict_proba(entrada_scaled)[0][int(pred)]

    # Resultado
    st.subheader("📊 Resultado")
    st.success(f"Previsão: {'✅ Sim' if pred == 1 else '❌ Não'}")
    st.info(f"Probabilidade: {prob:.2%}")

    # (Opcional) Mostrar entrada processada
    with st.expander("🔍 Ver dados processados"):
        st.write(pd.DataFrame(entrada_scaled, columns=colunas))
