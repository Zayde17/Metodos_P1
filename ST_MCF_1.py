import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kurtosis, skew, shapiro, norm, t
import altair as alt

st.cache_data.clear()

st.title("Cálculo de Value-At-Risk y de Expected Shortfall.")

#######################################---BACKEND---##################################################

@st.cache_data
def obtener_datos(stocks):
    df = yf.download(stocks, period="1y")['Close']
    return df

@st.cache_data
def calcular_rendimientos(df):
    return df.pct_change().dropna()

# Lista de acciones de ejemplo
stocks_lista = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']

with st.spinner("Descargando datos..."):
    df_precios = obtener_datos(stocks_lista)
    df_rendimientos = calcular_rendimientos(df_precios)








#######################################---FRONTEND---##################################################


st.header("Selección de Acción")
st.text("Selecciona una acción de la lista ya que a partir de ella se calculará todo lo que se indica en cada ejercicio")

stock_seleccionado = st.selectbox("Selecciona una acción", stocks_lista)

# Definir niveles de confianza
alphas = [0.95, 0.975, 0.99]

if stock_seleccionado:

    # ------------------- Métricas descriptivas -------------------
    st.subheader(f"Métricas de Rendimiento: {stock_seleccionado}")
    
    rendimiento_medio = df_rendimientos[stock_seleccionado].mean()
    Kurtosis = kurtosis(df_rendimientos[stock_seleccionado])
    Skewness = skew(df_rendimientos[stock_seleccionado])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Rendimiento Medio Diario", f"{rendimiento_medio:.4%}")
    col2.metric("Kurtosis", f"{Kurtosis:.4}")
    col3.metric("Skew", f"{Skewness:.2}")

    # ------------------- Cálculo de VaR y ES -------------------
    std_dev = np.std(df_rendimientos[stock_seleccionado])
    df_size = df_rendimientos[stock_seleccionado].size
    df_t = df_size - 1
    resultados = []

    for alpha in alphas:
        hVaR = df_rendimientos[stock_seleccionado].quantile(1 - alpha)
        ES_hist = df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= hVaR].mean()
        
        VaR_norm = norm.ppf(1 - alpha, rendimiento_medio, std_dev)
        ES_norm = df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= VaR_norm].mean()
        
        t_ppf = t.ppf(1 - alpha, df_t)
        VaR_t = rendimiento_medio + std_dev * t_ppf * np.sqrt((df_t - 2) / df_t)
        ES_t = df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= VaR_t].mean()
        
        simulaciones = np.random.normal(rendimiento_medio, std_dev, 10000)
        VaR_mc = np.percentile(simulaciones, (1 - alpha) * 100)
        ES_mc = df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= VaR_mc].mean()
        
        resultados.append([alpha, hVaR, ES_hist, VaR_norm, ES_norm, VaR_t, ES_t, VaR_mc, ES_mc])

    df_resultados = pd.DataFrame(resultados, columns=["Alpha", "hVaR", "ES_hist", "VaR_Norm", "ES_Norm", "VaR_t", "ES_t", "VaR_MC", "ES_MC"])

    # ------------------- Visualización tabla VaR y ES -------------------
    st.subheader("Tabla comparativa de VaR y ES")
    st.text("Esta tabla muestra los resultados de los diferentes métodos de cálculo de VaR y ES")

    st.dataframe(
        df_resultados.set_index("Alpha").style.format("{:.4%}")
        .applymap(lambda _: "background-color: #FFDDC1; color: black;", subset=["hVaR"])  # Durazno 
        .applymap(lambda _: "background-color: #C1E1FF; color: black;", subset=["ES_hist"])  # Azul 
        .applymap(lambda _: "background-color: #B5EAD7; color: black;", subset=["VaR_Norm"])  # Verde 
        .applymap(lambda _: "background-color: #FFB3BA; color: black;", subset=["ES_Norm"])  # Rosa 
        .applymap(lambda _: "background-color: #FFDAC1; color: black;", subset=["VaR_t"])  # Naranja 
        .applymap(lambda _: "background-color: #E2F0CB; color: black;", subset=["ES_t"])  # Verde 
        .applymap(lambda _: "background-color: #D4A5A5; color: black;", subset=["VaR_MC"])  # Rojo 
        .applymap(lambda _: "background-color: #CBAACB; color: black;", subset=["ES_MC"])  # Lila 
    )

    st.subheader("Gráfico de comparación de VaR y ES")
    st.text("Este gráfico muestra la comparación de los diferentes métodos de cálculo de VaR y ES")
    st.bar_chart(df_resultados.set_index("Alpha").T)

    # ------------------- Cálculo de violaciones -------------------
    st.subheader("Evaluación de Violaciones")

    window = 253
    rendimientos = df_rendimientos[stock_seleccionado].values
    violaciones_data = []

if len(rendimientos) <= window:
    st.warning("No hay suficientes datos históricos para evaluar violaciones (mínimo 253 días requeridos).")
else:
    # aquí va todo el código del for alpha in alphas

    for alpha in alphas:
        violaciones = {
            "Alpha": alpha,
            "hVaR": 0,
            "VaR_Norm": 0,
            "VaR_t": 0,
            "VaR_MC": 0
        }
        total = 0

        for i in range(window, len(rendimientos)):
            muestra = rendimientos[i - window:i]
            real = rendimientos[i]
            media = np.mean(muestra)
            std = np.std(muestra)
            df_t = window - 1

            hVaR = np.quantile(muestra, 1 - alpha)
            if real < hVaR:
                violaciones["hVaR"] += 1

            var_norm = norm.ppf(1 - alpha, media, std)
            if real < var_norm:
                violaciones["VaR_Norm"] += 1

            t_ppf = t.ppf(1 - alpha, df_t)
            var_t = media + std * t_ppf * np.sqrt((df_t - 2) / df_t)
            if real < var_t:
                violaciones["VaR_t"] += 1

            simulaciones = np.random.normal(media, std, 10000)
            var_mc = np.percentile(simulaciones, (1 - alpha) * 100)
            if real < var_mc:
                violaciones["VaR_MC"] += 1

            total += 1



        if total > 0:
            for key in violaciones:
        
             if key != "Alpha":
                violaciones[key] = (violaciones[key] / total) * 100
             else:

              for key in violaciones:
                if key != "Alpha":
                     violaciones[key] = np.nan  # o 0 si prefieres

        violaciones_data.append(violaciones)

    df_violaciones = pd.DataFrame(violaciones_data)

    st.markdown("""
    Este cuadro muestra el porcentaje de veces que el retorno real fue menor al VaR estimado para cada método y nivel de confianza.
    Una buena estimación debe generar un **porcentaje de violaciones menor al 2.5%**.
    """)

    st.dataframe(
        df_violaciones.set_index("Alpha").style.format("{:.2f}%")
        .applymap(lambda val: "background-color: #FFB3BA; color: black" if val > 2.5 else "background-color: #B5EAD7; color: black")
    )

