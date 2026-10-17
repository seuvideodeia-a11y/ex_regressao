import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Previsão de Consumo de Energia Industrial",
    page_icon="⚡",
    layout="wide"
)

# 1. Geração do Dataset Sintético
@st.cache_data
def carregar_dados():
    np.random.seed(42)
    n_registros = 100  # Mais de 20 registros conforme solicitado
    
    velocidade = np.random.uniform(10, 100, n_registros)      # m/min
    maquinas = np.random.randint(1, 21, n_registros)           # Unidades ativas
    temp_amb = np.random.uniform(15, 40, n_registros)          # °C
    
    # Criando uma relação lógica com ruído para o consumo alvo (kWh)
    consumo = (
        50 
        + (2.5 * velocidade) 
        + (18.0 * maquinas) 
        + (1.5 * temp_amb) 
        + np.random.normal(0, 15, n_registros)
    )
    
    df = pd.DataFrame({
        'Velocidade_Esteira': velocidade,
        'Maquinas_Ativas': maquinas,
        'Temp_Ambiente': temp_amb,
        'Consumo_kWh': consumo
    })
    return df

df = carregar_dados()

# 2. Treinamento do Modelo de Regressão
X = df[['Velocidade_Esteira', 'Maquinas_Ativas', 'Temp_Ambiente']]
y = df['Consumo_kWh']

modelo = LinearRegression()
modelo.fit(X, y)

# 3. Barra Lateral (Sidebar) com Controles Numéricos
st.sidebar.header("⚙️ Parâmetros Operacionais")
st.sidebar.markdown("Insira os valores atuais para prever o consumo:")

vel_input = st.sidebar.number_input(
    "Velocidade da Esteira (m/min)", 
    min_value=10.0, max_value=100.0, value=50.0, step=1.0
)
maq_input = st.sidebar.number_input(
    "Máquinas Ativas", 
    min_value=1, max_value=20, value=8, step=1
)
temp_input = st.sidebar.number_input(
    "Temperatura Ambiente (°C)", 
    min_value=15.0, max_value=40.0, value=25.0, step=0.5
)

# 4. Predição com base nos inputs do usuário
entrada_usuario = pd.DataFrame([[vel_input, maq_input, temp_input]], 
                               columns=['Velocidade_Esteira', 'Maquinas_Ativas', 'Temp_Ambiente'])
consumo_previsto = modelo.predict(entrada_usuario)[0]

# 5. Tela Principal
st.title("⚡ Painel de Previsão de Consumo de Energia Industrial")
st.markdown("Este painel utiliza Machine Learning (**Regressão Linear**) para estimar o consumo energético com base em parâmetros operacionais em tempo real.")

st.markdown("---")

# Exibição do Consumo Previsto em Destaque
col1, col2 = st.columns([1, 2])

with col1:
    st.metric(
        label="Consumo Energético Previsto", 
        value=f"{consumo_previsto:.2f} kWh"
    )

with col2:
    st.info(
        f"**Status do Sistema:**\n"
        f"- Velocidade: {vel_input} m/min\n"
        f"- Máquinas Ativas: {maq_input} un.\n"
        f"- Temperatura: {temp_input} °C"
    )

st.markdown("---")

# 6. Gráfico de Contribuição das Variáveis (Matplotlib)
st.subheader("📊 Contribuição de Cada Variável no Modelo")
st.markdown("O gráfico abaixo ilustra o impacto proporcional de cada parâmetro de entrada na predição atual (Coeficiente do Modelo $\times$ Valor Inserido):")

# Cálculo da contribuição (Coeficiente * Valor de Entrada)
nomes_features = ['Velocidade da Esteira', 'Máquinas Ativas', 'Temperatura Ambiente']
valores_entrada = [vel_input, maq_input, temp_input]
contribuicoes = modelo.coef_ * valores_entrada

fig, ax = plt.subplots(figsize=(8, 4))
cores = ['#1f77b4', '#ff7f0e', '#2ca02c']
barras = ax.bar(nomes_features, contribuicoes, color=cores)

ax.set_ylabel("Impacto no Consumo (kWh)")
ax.set_title("Contribuição Relativa dos Parâmetros para a Previsão")
ax.grid(axis='y', linestyle='--', alpha=0.6)

# Adicionar os valores exatos em cima de cada barra
for barra in barras:
    altura = barra.get_height()
    ax.annotate(f'{altura:.1f} kWh',
                xy=(barra.get_x() + barra.get_width() / 2, altura),
                xytext=(0, 4 if altura >= 0 else -12),
                textcoords="offset points",
                ha='center', va='bottom' if altura >= 0 else 'top',
                fontsize=10, fontweight='bold')

st.pyplot(fig)
