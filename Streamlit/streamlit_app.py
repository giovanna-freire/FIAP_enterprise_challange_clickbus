import streamlit as st
import pandas as pd
import pickle
import datetime
import shap
import random
import xgboost as xgb
import gzip
import joblib
from io import BytesIO
from faker import Faker
from matplotlib import pyplot as plt

st.title("Previsão de Próxima Compra por Cliente")

# Configuração de página para usar menos memória
st.set_page_config(page_title="Previsão de Compra", layout="wide")

@st.cache_data(ttl=3600, max_entries=3)  # Limitar entradas do cache
def carregar_dados_leves():
    """Carrega apenas os dados essenciais primeiro"""
    try:
        # Carregar apenas as colunas necessárias e com tipos otimizados
        df_compras = pd.read_parquet(
            "Dados/dataframe.parquet", 
            engine="pyarrow",
            columns=["id_cliente", "origem_ida", "destino_ida", "qtd_total_compras", 
                     "qnt_passageiros", "data_compra", "vl_total_compra", "vl_medio_compra",
                     "intervalo_medio_dias"]
        )
        
        # Otimizar tipos de dados
        df_compras = otimizar_tipos_dados(df_compras)
        
        classes = pd.read_parquet("Dados/classes.parquet", engine="pyarrow")
        
        return df_compras, classes
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None, None

@st.cache_data(ttl=3600, max_entries=2)
def carregar_features(tipo_feature):
    """Carrega features sob demanda"""
    if tipo_feature == "dia":
        return pd.read_parquet("Dados/cb_previsao_data.parquet", engine="pyarrow")
    elif tipo_feature == "trecho":
        return pd.read_parquet("Dados/cb_previsao_trecho.parquet", engine="pyarrow")

@st.cache_resource  # Use cache_resource para objetos que não são serializáveis
def carregar_modelo(tipo_modelo):
    """Carrega modelos sob demanda"""
    try:
        modelo = xgb.Booster()
        if tipo_modelo == "dia":
            modelo.load_model("Modelos/xgboost_model_dia_exato.json")
        elif tipo_modelo == "destino":
            modelo.load_model("Modelos/xgboost_model_trecho.json")
        return modelo
    except Exception as e:
        st.error(f"Erro ao carregar modelo {tipo_modelo}: {e}")
        return None

def otimizar_tipos_dados(df):
    """Otimiza tipos de dados para economizar memória"""
    for col in df.select_dtypes(include=['int64']).columns:
        if df[col].max() < 32767 and df[col].min() > -32768:
            df[col] = df[col].astype('int16')
        elif df[col].max() < 2147483647 and df[col].min() > -2147483648:
            df[col] = df[col].astype('int32')
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    return df

# Carregamento inicial apenas dos dados essenciais
with st.spinner("🔄 Carregando dados essenciais..."):
    df_compras_cliente, classes = carregar_dados_leves()
    if df_compras_cliente is None:
        st.stop()
    st.success("✅ Dados essenciais carregados.")

# Gerar nomes fictícios (apenas uma vez)
@st.cache_data
def gerar_nomes_ficticios():
    Faker.seed(42)
    fake = Faker('pt_BR')
    
    # Carrega features de trecho apenas para obter IDs únicos
    features_trecho_temp = carregar_features("trecho")
    unique_ids = features_trecho_temp['id_cliente'].unique()
    
    fake_names = [fake.name() for _ in unique_ids]
    id_to_name = dict(zip(unique_ids, fake_names))
    name_to_id = dict(zip(fake_names, unique_ids))
    
    return fake_names, name_to_id, id_to_name

fake_names, name_to_id, id_to_name = gerar_nomes_ficticios()

# Selecionar cliente
selected_fake_name = st.selectbox("Selecione o cliente", fake_names)
id_cliente = name_to_id[selected_fake_name]

# Interface com abas para reduzir carregamento simultâneo
tab1, tab2, tab3 = st.tabs(["📊 Previsões", "🛒 Histórico", "🔍 Explicações"])

with tab1:
    st.subheader("Previsões do Cliente")
    
    if st.button("🔮 Gerar Previsões"):
        with st.spinner("Carregando modelos e fazendo previsões..."):
            # Carregar dados necessários sob demanda
            features_dia = carregar_features("dia")
            features_trecho = carregar_features("trecho")
            
            # Carregar modelos sob demanda
            modelo_dia = carregar_modelo("dia")
            modelo_destino = carregar_modelo("destino")
            
            if modelo_dia and modelo_destino:
                # Previsão de dia
                input_dia = features_dia[features_dia['id_cliente'] == id_cliente].drop(columns=["id_cliente"])
                data_prevista = modelo_dia.predict(input_dia)[0]
                
                # Previsão de destino
                input_trecho = features_trecho[features_trecho['id_cliente'] == id_cliente].drop(columns=["id_cliente"])
                destino_pred = modelo_destino.predict(input_trecho)[0]
                
                # Processar cidades (otimizado)
                todos_ids = set()
                df_compras_cliente['Trechos'] = df_compras_cliente["origem_ida"] + "_" + df_compras_cliente["destino_ida"]
                for item in df_compras_cliente['Trechos']:
                    origem, destino = item.split('_')
                    todos_ids.update([origem, destino])
                
                @st.cache_data
                def gerar_mapeamento_cidades(todos_ids_tuple):
                    Faker.seed(42)
                    fake = Faker('pt_BR')
                    
                    def gerar_cidade_fake(id_unico):
                        random.seed(hash(id_unico))
                        return fake.city()
                    
                    return {id_: gerar_cidade_fake(id_) for id_ in todos_ids_tuple}
                
                id_para_cidade = gerar_mapeamento_cidades(tuple(todos_ids))
                
                def mapear_para_cidades(par):
                    origem, destino = par.split('_')
                    cidade_origem = id_para_cidade[origem]
                    cidade_destino = id_para_cidade[destino]
                    return f"{cidade_origem} -> {cidade_destino}"
                
                classes_temp = classes.copy()
                classes_temp['trecho_fake'] = classes_temp['Trechos'].apply(mapear_para_cidades)
                
                # Mostrar resultados
                data_final = datetime.date.today() + datetime.timedelta(days=int(data_prevista))
                st.success(f"📅 Data provável da próxima compra: **{data_final.strftime('%Y-%m-%d')}**")
                st.success(f"✈️ Trecho provável da próxima compra: **{classes_temp.iloc[destino_pred]['trecho_fake']}**")
                
                # Métricas do cliente
                cliente_data = df_compras_cliente[df_compras_cliente['id_cliente'] == id_cliente]
                
                col1, col2, col3 = st.columns(3)
                col1.metric("🛒 Total de compras", int(cliente_data['qtd_total_compras'].iloc[0]))
                col2.metric("📊 Intervalo médio (dias)", int(cliente_data['intervalo_medio_dias'].iloc[0]))
                col3.metric("📈 Valor médio (R$)", int(cliente_data['vl_medio_compra'].iloc[0]))
                
                st.info(f"🏷️ Cluster: {cliente_data['cluster_name'].iloc[0]}")

with tab2:
    st.subheader("🛒 Histórico de Compras")
    df_compras_cliente["Trechos"] = df_compras_cliente["origem_ida"] + "_" + df_compras_cliente["destino_ida"]
    cliente_data = df_compras_cliente[df_compras_cliente['id_cliente'] == id_cliente].copy()
    
    if not cliente_data.empty:
        # Processar dados do histórico
        cliente_data["trecho_fake"] = cliente_data["Trechos"].apply(
            lambda x: f"Origem -> Destino"  # Simplificado para economizar processamento
        )
        cliente_data = cliente_data.sort_values("data_compra", ascending=False)
        cliente_data["data_compra"] = cliente_data["data_compra"].dt.strftime("%Y-%m-%d")
        
        cliente_data_display = cliente_data.rename(columns={
            "data_compra": "Data",
            "trecho_fake": "Trecho",
            "qnt_passageiros": "Qtd Passageiros",
            "vl_total_compra": "Valor (R$)"
        })
        
        st.dataframe(
            cliente_data_display[["Data", "Trecho", "Qtd Passageiros", "Valor (R$)"]].head(20),  # Limitar a 20 registros
            use_container_width=True
        )
    else:
        st.warning("Nenhum histórico encontrado para este cliente.")

with tab3:
    st.subheader("🔍 Explicações SHAP")
    st.info("⚠️ As explicações SHAP consomem muita memória. Clique no botão abaixo apenas se necessário.")
    
    if st.button("🧮 Gerar Explicações SHAP"):
        with st.spinner("Gerando explicações... Isso pode demorar um pouco."):
            try:
                # Carregar apenas o que é necessário para SHAP
                features_dia = carregar_features("dia")
                modelo_dia = carregar_modelo("dia")
                
                input_dia = features_dia[features_dia['id_cliente'] == id_cliente].drop(columns=["id_cliente"])
                
                # Usar uma amostra menor para SHAP se necessário
                explainer = shap.Explainer(modelo_dia)
                shap_values = explainer(input_dia.iloc[:1])  # Apenas primeira linha
                
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(fig)
                plt.close()  # Importante: fechar figura para liberar memória
                
            except Exception as e:
                st.error(f"Erro ao gerar explicações SHAP: {e}")
                st.info("Tente selecionar outro cliente ou reinicie a aplicação.")