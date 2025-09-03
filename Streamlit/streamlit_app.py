import streamlit as st
import pandas as pd
import pickle
import datetime
from faker import Faker
Faker.seed(42)

# Carregar modelos treinados
modelo_dia = pickle.load(open("../Modelos/xgboost_model_dia_exato.pkl", "rb"))
modelo_destino = pickle.load(open("../Modelos/xgboost_model_trecho.pkl", "rb"))

# Carregar base de clientes
st.title("Previsão de Próxima Compra por C" \
"liente")

# Obter features do cliente para modelo de previsão do dia
features_dia = pd.read_csv("../Dados/cb_previsao_data.csv")

# Obter features do cliente para modelo de previsão do trecho
features_trecho = pd.read_csv("../Dados/cb_previsao_trecho.csv")

# Gerar nomes fictícios com Faker
fake = Faker('pt_BR')
unique_ids = features_trecho['id_cliente'].unique()
fake_names = [fake.name() for _ in unique_ids]
id_to_name = dict(zip(unique_ids, fake_names))
name_to_id = dict(zip(fake_names, unique_ids))

# Selecionar cliente pelo nome fictício
selected_fake_name = st.selectbox("Selecione o cliente", fake_names)
id_cliente = name_to_id[selected_fake_name]

# Previsão de dia
data_prevista = modelo_dia.predict(features_dia[features_dia['id_cliente'] == id_cliente].drop(columns=["id_cliente"]))[0]

# Previsão de destino
destino_pred = modelo_destino.predict(features_trecho[features_trecho['id_cliente'] == id_cliente].drop(columns=["id_cliente"]))[0]

# Classes
classes = pd.read_csv('../Dados/classes.csv')
todos_ids = set()
for item in classes['Trechos']:
    origem, destino = item.split('_')
    todos_ids.update([origem, destino])

# Criar mapeamento ID -> cidade fake
id_para_cidade = {id_: fake.administrative_unit() for id_ in todos_ids}

# Mapear cada par para a cidade correspondente, mantendo a ordem
def mapear_para_cidades(par):
    origem, destino = par.split('_')
    cidade_origem = id_para_cidade[origem]
    cidade_destino = id_para_cidade[destino]
    return f"{cidade_origem} -> {cidade_destino}"

classes['trecho_fake'] = classes['Trechos'].apply(mapear_para_cidades)

# Mostrar resultado
data_final = datetime.date.today() + datetime.timedelta(days=int(data_prevista))
st.write(f"📅 Data provável da próxima compra: **{data_final.strftime('%d/%m/%Y')}**")
st.write(f"✈️ Trecho provável da próxima compra: **{classes.iloc[destino_pred][['trecho_fake']][0]}**")