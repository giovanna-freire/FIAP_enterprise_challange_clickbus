import streamlit as st
import pandas as pd
import pickle
import datetime
import shap
import random
import xgboost as xgb
from io import BytesIO
from faker import Faker
from matplotlib import pyplot as plt
from azure.storage.blob import BlobServiceClient

st.title("Previsão de Próxima Compra por Cliente")

@st.cache_data(ttl=3600)  # cache por 1h
def carregar_dados():
    df_compras = pd.read_parquet("Dados/dataframe.parquet", engine="pyarrow")
    features_dia = pd.read_parquet("Dados/cb_previsao_data.parquet", engine="pyarrow")
    features_trecho = pd.read_parquet("Dados/cb_previsao_trecho.parquet", engine="pyarrow")
    classes = pd.read_parquet("Dados/classes.parquet", engine="pyarrow")
    modelo_dia = xgb.Booster()
    modelo_destino = xgb.Booster()
    modelo_dia.load_model(f"Modelos/xgboost_model_dia_exato.json")
    modelo_destino.load_model(f"Modelos/xgboost_model_trecho.json")
    return df_compras, features_dia, features_trecho, classes, modelo_dia, modelo_destino

with st.spinner("🔄 Carregando dados..."):
    # Carregar modelos treinados
    # Carregar base de clientes
    df_compras_cliente, features_dia, features_trecho, classes, modelo_dia, modelo_destino = carregar_dados()
    st.write("✅ Dados carregados com sucesso.")

# Gerar nomes fictícios com Faker
Faker.seed(42)
fake = Faker('pt_BR')
unique_ids = features_trecho['id_cliente'].unique()
fake_names = [fake.name() for _ in unique_ids]
id_to_name = dict(zip(unique_ids, fake_names))
name_to_id = dict(zip(fake_names, unique_ids))

# Selecionar cliente pelo nome fictício
selected_fake_name = st.selectbox("Selecione o cliente", fake_names)
id_cliente = name_to_id[selected_fake_name]

# Previsão de dia
input_dia = features_dia[features_dia['id_cliente'] == id_cliente].drop(columns=["id_cliente"])
data_prevista = modelo_dia.predict(input_dia)[0]

# Previsão de destino
input_trecho = features_trecho[features_trecho['id_cliente'] == id_cliente].drop(columns=["id_cliente"])
destino_pred = modelo_destino.predict(input_trecho)[0]

# Classes
todos_ids = set()
df_compras_cliente['Trechos'] = df_compras_cliente["origem_ida"] + "_" + df_compras_cliente["destino_ida"]
for item in df_compras_cliente['Trechos']:
    origem, destino = item.split('_')
    todos_ids.update([origem, destino])

Faker.seed(42)  
def gerar_cidade_fake(id_unico):
    random.seed(hash(id_unico))
    cidade = fake.city()
    return f"{cidade}"

# Criar dicionário de mapeamento
id_para_cidade = {
    id_: gerar_cidade_fake(id_)
    for id_ in todos_ids
}

# Mapear cada par para a cidade correspondente, mantendo a ordem
def mapear_para_cidades(par):
    origem, destino = par.split('_')
    cidade_origem = id_para_cidade[origem]
    cidade_destino = id_para_cidade[destino]
    return f"{cidade_origem} -> {cidade_destino}"

classes['trecho_fake'] = classes['Trechos'].apply(mapear_para_cidades)
cliente_data = df_compras_cliente[df_compras_cliente['id_cliente'] == id_cliente]

# Mostrar resultado
data_final = datetime.date.today() + datetime.timedelta(days=int(data_prevista))
st.write(f"📅 Data provável da próxima compra: **{data_final.strftime('%Y-%m-%d')}**")
st.write(f"✈️ Trecho provável da próxima compra: **{classes.iloc[destino_pred][['trecho_fake']][0]}**")

col1, col2, col3 = st.columns(3)
col1.metric("🛒 Quantidade total de compras", int(cliente_data['qtd_total_compras'].iloc[0]))
col2.metric("📊 Intervalo médio (dias)", int(cliente_data['intervalo_medio_dias'].iloc[0]))
col3.metric("📈 Valor médio ticket (reais)", int(cliente_data['vl_medio_compra'].iloc[0]))

st.metric("Cluster: ", str(cliente_data['cluster_name'].iloc[0]))

# historico de compras
st.subheader("🛒 Histórico de compras do cliente")
cliente_data["trecho_fake"] = cliente_data["Trechos"].apply(mapear_para_cidades)
cliente_data = cliente_data.sort_values("data_compra", ascending=False)
# converter data_compra para string no formato YYYY-MM-DD
cliente_data["data_compra"] = cliente_data["data_compra"].dt.strftime("%Y-%m-%d")
cliente_data = cliente_data.rename(columns={
    "data_compra": "Data",
    "trecho_fake": "Trecho",
    "qnt_passageiros": "Quantidade de Passageiros",
    "vl_total_compra": "Valor do Ticket (R$)"
})
st.dataframe(
    cliente_data[["Data", "Trecho", "Quantidade de Passageiros", "Valor do Ticket (R$)"]],
    use_container_width=True
)

# Criar explainer e calcular SHAP values
explainer = shap.Explainer(modelo_dia)
shap_values = explainer(input_dia)

# Mostrar gráfico waterfall no app
st.subheader("🔍 Explicação da previsão da data (impacto das variáveis)")
fig, ax = plt.subplots()
shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(fig)

# Criar explainer e calcular SHAP values
explainer = shap.Explainer(modelo_destino)
shap_values = explainer(input_trecho)
shap_value_classe = shap_values[0, :, destino_pred]

# Mostrar gráfico waterfall no app
st.subheader("🔍 Explicação da previsão do trecho (impacto das variáveis)")
fig, ax = plt.subplots()
shap.plots.waterfall(shap_value_classe, show=False)
st.pyplot(fig)


