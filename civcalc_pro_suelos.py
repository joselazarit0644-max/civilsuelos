import streamlit as st
import numpy as np
import pandas as pd
import os
from fpdf import FPDF
from datetime import datetime

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA Y BRANDING
# ==========================================
st.set_page_config(page_title="Mecánica de Suelos | CivCalc Pro", layout="wide", page_icon="🏛️")

st.markdown("""
    <style>
    .stMetric { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #1f4e79; }
    .titulo-seccion { color: #1f4e79; border-bottom: 2px solid #1f4e79; padding-bottom: 5px; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

col_logo, col_titulo = st.columns([1, 6])
with col_logo:
    try:
        st.image("logo_cutlajo.png", width=100)
    except:
        st.info("Logo pendiente")

with col_titulo:
    st.title("🏛️ CivCalc Pro: Esfuerzos en Masa Elástica")
    st.markdown("**Desarrollado por:** Lazarit Lopez Jose Carlos | **Ingeniería Civil - CUTLAJOMULCO**")

st.divider()

# ==========================================
# 2. GENERADOR Y LECTOR DE BASE DE DATOS (EXCEL)
# ==========================================
ARCHIVO_BD = "base_datos_suelos.xlsx"
ENCABEZADOS_R = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0]
ENCABEZADOS_Z = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

@st.cache_data
def cargar_base_datos():
    """Si el Excel no existe, lo crea. Luego carga las tablas en memoria."""
    if not os.path.exists(ARCHIVO_BD):
        with pd.ExcelWriter(ARCHIVO_BD) as writer:
            matrices_iniciales = {
                "A": [[0.06698, 0.04886], [0.06373, 0.04707]],
                "B": [[0.11327, 0.08635], [0.10298, 0.08033]],
                "C": [[-0.05259, -0.04089], [-0.04522, -0.03642]],
                "D": [[0.06068, 0.04548], [0.05777, 0.04391]],
                "E": [[0.03435, 0.02491], [0.03360, 0.02444]],
                "F": [[0.03263, 0.02395], [0.03014, 0.02263]],
                "G": [[0.03611, 0.02376], [0.04484, 0.02994]]
            }
            # Generar hojas en blanco e inyectar los datos que ya conocemos
            for letra in ["A", "B", "C", "D", "E", "F", "G"]:
                df = pd.DataFrame(0.0, index=ENCABEZADOS_Z, columns=ENCABEZADOS_R)
                df.loc[2.5, 0.6], df.loc[2.5, 0.8] = matrices_iniciales[letra][0][0], matrices_iniciales[letra][0][1]
                df.loc[3.0, 0.6], df.loc[3.0, 0.8] = matrices_iniciales[letra][1][0], matrices_iniciales[letra][1][1]
                df.to_excel(writer, sheet_name=letra)
    
    # Leer el Excel completo en un diccionario de Pandas
    xls = pd.ExcelFile(ARCHIVO_BD)
    dfs = {}
    for letra in ["A", "B", "C", "D", "E", "F", "G"]:
        df = pd.read_excel(xls, sheet_name=letra, index_col=0)
        # Asegurar que filas y columnas se traten matemáticamente
        df.index = np.round(df.index.astype(float), 2)
        df.columns = np.round(df.columns.astype(float), 2)
        dfs[letra] = df
    return dfs

# Cargamos los datos
BD_SUELOS = cargar_base_datos()

# ==========================================
# 3. PANEL LATERAL (INPUTS)
# ==========================================
with st.sidebar:
    st.header("⚙️ Parámetros de Entrada")
    P = st.number_input("Presión de contacto P (kg/cm²)", value=6.3, step=0.1)
    a = st.number_input("Radio de la zapata a (cm)", value=13.0, step=1.0)
    z = st.number_input("Profundidad z (cm)", value=35.0, step=1.0)
    r = st.number_input("Distancia radial r (cm)", value=9.0, step=1.0)
    mu = st.number_input("Relación de Poisson μ", value=0.34, step=0.01)

R_val = r / a
Z_val = z / a

st.markdown("<h3 class='titulo-seccion'>Análisis Adimensional</h3>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
c1.metric("Radio Adimensional (R = r/a)", f"{R_val:.3f}")
c2.metric("Profundidad Adim. (Z = z/a)", f"{Z_val:.3f}")

# ==========================================
# 4. MOTOR MATEMÁTICO UNIVERSAL
# ==========================================
def encontrar_limites(valor, ejes):
    if valor < min(ejes) or valor > max(ejes):
        return None, None
    for i in range(len(ejes) - 1):
        if ejes[i] <= valor <= ejes[i+1]:
            if ejes[i] == ejes[i+1]: continue 
            return round(ejes[i], 2), round(ejes[i+1], 2)
    return None, None

R1, R2 = encontrar_limites(R_val, ENCABEZADOS_R)
Z1, Z2 = encontrar_limites(Z_val, ENCABEZADOS_Z)

if R1 is None or Z1 is None:
    st.error("⚠️ Parámetros fuera de rango. Revisa los valores de R y Z.")
    st.stop()

# Vectores y Factor Escalar
area = (R2 - R1) * (Z2 - Z1)
factor = 1 / area
c3.metric("Área de Interpolación", f"{area:.2f}")

v_R = np.array([R2 - R_val, R_val - R1])
v_Z = np.array([[Z2 - Z_val], [Z_val - Z1]])

valores_F = {}
st.markdown("<h3 class='titulo-seccion'>1. Interpolación Dinámica (Ahlvin y Ulery)</h3>", unsafe_allow_html=True)

tabs = st.tabs([f"Función {letra}" for letra in ["A", "B", "C", "D", "E", "F", "G"]])

for tab, letra in zip(tabs, ["A", "B", "C", "D", "E", "F", "G"]):
    df = BD_SUELOS[letra]
    
    # Extracción dinámica de la matriz de la base de datos
    try:
        matriz = df.loc[[Z1, Z2], [R1, R2]].values
    except KeyError:
        st.error(f"Error extrayendo coordenadas Z:[{Z1},{Z2}] y R:[{R1},{R2}] de la BD.")
        st.stop()
        
    # Multiplicación matricial
    resultado = factor * np.dot(np.dot(v_R, matriz), v_Z)
    val = float(resultado[0])
    valores_F[letra] = val
    
    with tab:
        # Alerta inteligente si la celda del Excel está vacía (0.0)
        if np.all(matriz == 0.0):
            st.warning(f"⚠️ ¡Atención! Los datos para el cuadrante $R \in [{R1}, {R2}]$ y $Z \in [{Z1}, {Z2}]$ no han sido capturados en el archivo Excel. Abre 'base_datos_suelos.xlsx' y rellena los valores desde las tablas.")
        
        st.markdown(f"**Matriz extraída de la Base de Datos:**")
        st.latex(f"M_{letra} = \\begin{{bmatrix}} {matriz[0,0]} & {matriz[0,1]} \\\\ {matriz[1,0]} & {matriz[1,1]} \\end{{bmatrix}}")
        st.markdown("**Cálculo matricial:**")
        st.latex(f"{letra} = {factor:.2f} \\cdot \\begin{{bmatrix}} {v_R[0]:.2f} & {v_R[1]:.2f} \\end{{bmatrix}} \\begin{{bmatrix}} {matriz[0,0]} & {matriz[0,1]} \\\\ {matriz[1,0]} & {matriz[1,1]} \\end{{bmatrix}} \\begin{{bmatrix}} {v_Z[0][0]:.2f} \\\\ {v_Z[1][0]:.2f} \\end{{bmatrix}} = \mathbf{{{val:.5f}}}")

st.divider()

# ==========================================
# 5. CÁLCULO DE ESFUERZOS FINALES
# ==========================================
A, B, C, D, E, F_val, G = [valores_F[k] for k in ["A", "B", "C", "D", "E", "F", "G"]]

sigma_z = P * (A + B)
sigma_r = P * (2 * mu * A + C + (1 - 2 * mu) * F_val)
sigma_t = P * (2 * mu * A - D + (1 - 2 * mu) * E)
tau_rz = P * G

st.markdown("<h3 class='titulo-seccion'>2. Resumen de Esfuerzos Finales</h3>", unsafe_allow_html=True)
r1, r2, r3, r4 = st.columns(4)
r1.metric("Esfuerzo Vertical (σ_z)", f"{sigma_z:.4f} kg/cm²")
r2.metric("Esfuerzo Radial (σ_r)", f"{sigma_r:.4f} kg/cm²")
r3.metric("Esfuerzo Tangencial (σ_t)", f"{sigma_t:.4f} kg/cm²")
r4.metric("Esfuerzo Cortante (τ_rz)", f"{tau_rz:.4f} kg/cm²")

st.divider()

# ==========================================
# 6. GENERADOR DE PDF CORPORATIVO
# ==========================================
st.markdown("<h3 class='titulo-seccion'>📄 Generación de Reportes</h3>", unsafe_allow_html=True)

class PDFMemoria(FPDF):
    def header(self):
        self.set_fill_color(31, 78, 121)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 15, "MEMORIA DE CALCULO ESTRUCTURAL", border=False, ln=True, align="C", fill=True)
        
        self.set_text_color(0, 0, 0)
        self.set_font("Helvetica", "B", 10)
        self.ln(5)
        self.cell(0, 6, "PROYECTO: Analisis de Esfuerzos en Masa Elastica (Una Capa)", ln=True)
        self.set_font("Helvetica", "", 10)
        self.cell(0, 6, "SOFTWARE: CivCalc Pro v1.0", ln=True)
        self.cell(0, 6, "CALCULO POR: Lazarit Lopez Jose Carlos | CUTLAJOMULCO", ln=True)
        self.cell(0, 6, f"FECHA: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True)
        self.set_draw_color(31, 78, 121)
        self.line(10, 52, 200, 52)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Pagina {self.page_no()} | CivCalc Pro - Ingenieria Civil", align="C")

def generar_pdf_pro():
    pdf = PDFMemoria()
    pdf.add_page()
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(230, 240, 250)
    pdf.cell(0, 8, " 1. PARAMETROS DE ENTRADA", border=1, ln=True, fill=True)
    pdf.set_font("Helvetica", "", 11)
    
    ancho_col = 95
    pdf.cell(ancho_col, 8, f" Presion de contacto (P): {P} kg/cm2", border="L")
    pdf.cell(ancho_col, 8, f" Profundidad (z): {z} cm", border="R", ln=True)
    pdf.cell(ancho_col, 8, f" Radio de carga (a): {a} cm", border="L")
    pdf.cell(ancho_col, 8, f" Distancia radial (r): {r} cm", border="R", ln=True)
    pdf.cell(ancho_col, 8, f" Relacion de Poisson (mu): {mu}", border="L B")
    pdf.cell(ancho_col, 8, "", border="R B", ln=True)
    pdf.ln(8)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, " 2. VALORES ADIMENSIONALES", border=1, ln=True, fill=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(ancho_col, 8, f" Radio adim. (R = r/a): {R_val:.3f}", border="L B")
    pdf.cell(ancho_col, 8, f" Profundidad adim. (Z = z/a): {Z_val:.3f}", border="R B", ln=True)
    pdf.ln(8)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, " 3. FACTORES DE INFLUENCIA (Ahlvin y Ulery)", border=1, ln=True, fill=True)
    pdf.set_font("Helvetica", "", 11)
    texto_funciones = f" A={A:.5f}  |  B={B:.5f}  |  C={C:.5f}  |  D={D:.5f}\n E={E:.5f}  |  F={F_val:.5f}  |  G={G:.5f}"
    pdf.multi_cell(0, 8, texto_funciones, border=1, align="C")
    pdf.ln(8)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, " 4. RESUMEN DE ESFUERZOS CALCULADOS", border=1, ln=True, fill=True)
    pdf.set_font("Helvetica", "", 11)
    
    pdf.cell(120, 10, " Esfuerzo Vertical (sigma_z)", border="L B")
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(70, 10, f"{sigma_z:.4f} kg/cm2", border="R B", align="R", ln=True)
    
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(120, 10, " Esfuerzo Radial Horizontal (sigma_r)", border="L B")
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(70, 10, f"{sigma_r:.4f} kg/cm2", border="R B", align="R", ln=True)
    
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(120, 10, " Esfuerzo Tangencial Horizontal (sigma_t)", border="L B")
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(70, 10, f"{sigma_t:.4f} kg/cm2", border="R B", align="R", ln=True)
    
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(120, 10, " Esfuerzo Cortante (tau_rz)", border="L B")
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(70, 10, f"{tau_rz:.4f} kg/cm2", border="R B", align="R", ln=True)

    return bytes(pdf.output())

pdf_bytes_pro = generar_pdf_pro()
st.download_button(
    label="📥 Descargar Memoria de Cálculo Ejecutiva (PDF)",
    data=pdf_bytes_pro,
    file_name=f"Memoria_CivCalcPro_{datetime.now().strftime('%Y%m%d')}.pdf",
    mime="application/pdf"
)