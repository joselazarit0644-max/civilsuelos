import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Configuración básica de la pestaña del navegador
st.set_page_config(page_title="Cálculo de Esfuerzos", layout="wide")

st.title("🏗️ Módulo de Mecánica de Suelos")
st.markdown("Cálculo de esfuerzo cortante mediante interpolación matricial (Ahlvin y Ulery).")


# Barra lateral para inputs del usuario
with st.sidebar:
    st.header("Parámetros del Sistema")
    # st.number_input crea una caja numérica interactiva
    P = st.number_input("Presión de contacto P (kg/cm²)", value=6.3, step=0.1)
    a = st.number_input("Radio de la carga a (cm)", value=13.0, step=1.0)
    z = st.number_input("Profundidad z (cm)", value=35.0, step=1.0)
    r = st.number_input("Distancia radial r (cm)", value=9.0, step=1.0)

# Mostramos los valores adimensionales en la pantalla principal
R_val = r / a
Z_val = z / a
st.write(f"**Valores Adimensionales:** $R = {R_val:.2f}$, $Z = {Z_val:.2f}$")

def calcular_funcion_G(R, Z, matriz, limites_R, limites_Z):
    R1, R2 = limites_R
    Z1, Z2 = limites_Z
    
    # 1. Vectores de distancia
    vec_R = np.array([R2 - R, R - R1])
    vec_Z = np.array([[Z2 - Z], [Z - Z1]])
    
    # 2. Factor escalar (el área del rectángulo de interpolación)
    area = (R2 - R1) * (Z2 - Z1)
    
    # 3. Multiplicación matricial: (1/Area) * (vec_R * Matriz * vec_Z)
    G = (1 / area) * np.dot(np.dot(vec_R, matriz), vec_Z)
    
    return float(G[0])

# Definimos los límites y la matriz extraída de tus tablas
lim_R = (0.6, 0.8)
lim_Z = (2.5, 3.0)
matriz_G = np.array([
    [0.03611, 0.02376],
    [0.04484, 0.02994]
])

# Ejecutamos el cálculo si los valores están dentro del rango permitido
if 0.6 <= R_val <= 0.8 and 2.5 <= Z_val <= 3.0:
    G_exacto = calcular_funcion_G(R_val, Z_val, matriz_G, lim_R, lim_Z)
    tau_rz = P * G_exacto
    
    st.success(f"### Esfuerzo Cortante ($\\tau_{{rz}}$): {tau_rz:.3f} kg/cm²")
else:
    st.warning("⚠️ Los valores R y Z están fuera del rango de la matriz precargada (R: 0.6-0.8, Z: 2.5-3.0).")

if 'tau_rz' in locals():
    st.subheader("Representación Visual")
    
    # Creamos un gráfico vacío
    fig = go.Figure()
    
    # Agregamos el punto exacto que acabamos de calcular con matrices
    fig.add_trace(go.Scatter(
        x=[tau_rz], 
        y=[z], 
        mode='markers+text', 
        name='Punto de Análisis',
        text=[f"{tau_rz:.3f} kg/cm²"],
        textposition="top right",
        marker=dict(color='red', size=12, symbol='cross')
    ))

    # Configuramos el diseño (invertimos el eje Y porque es profundidad)
    fig.update_layout(
        title="Ubicación del Esfuerzo Cortante Calculado",
        xaxis_title="Esfuerzo (kg/cm²)",
        yaxis_title="Profundidad Z (cm)",
        yaxis=dict(autorange="reversed"), 
        template="plotly_white"
    )
    
    # Mostramos el gráfico en la web
    st.plotly_chart(fig, use_container_width=True)