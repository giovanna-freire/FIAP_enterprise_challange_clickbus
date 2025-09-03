# 🚍 FIAP Enterprise Challenge - Clickbus

Este projeto foi desenvolvido como parte do **Enterprise Challenge da FIAP**, em parceria com a **Clickbus**, com o objetivo de aplicar técnicas de ciência de dados para **prever o comportamento de compra de clientes** de passagens rodoviárias.

## 🧠 Objetivo

Desenvolver soluções analíticas preditivas e segmentações de clientes para apoiar decisões estratégicas, com foco em:

- Previsão da próxima compra (**data**)
- Previsão do próximo **trecho (origem-destino)**
- Segmentação comportamental via **clusterização**
- Criação de **dashboard analítico** para clusters
- Aplicativo interativo para **previsões individuais**

---

## 📁 Estrutura do Repositório

```
├── Analise e Modelos/          # Análises exploratórias e notebooks de modelagem
├── Modelos/                    # Modelos preditivos prontos para uso
├── Streamlit/                  # Aplicativo Streamlit para previsões individuais
├── .gitignore                  # Arquivos ignorados (ex: CSVs grandes)
```


---

## 📊 Modelos Preditivos

### 1. Previsão da Data da Próxima Compra
- **Modelo**: XGBoost
- **Entradas**: histórico de compras, frequência, periodicidade etc.
- **Saída**: número de dias até a próxima compra

### 2. Previsão do Próximo Trecho (origem-destino)
- **Modelo**: XGBoost (multiclass)
- **Saída**: par de origem-destino mais provável da próxima compra

---

## 🧬 Clusterização de Clientes

Utilizando o algoritmo **K-Means**, os clientes foram segmentados com base em seu comportamento de compra. Os **4 clusters principais** identificados foram:

1. Clientes que viajam **sozinhos em feriados**
2. Clientes que viajam **sozinhos fora de feriados**
3. Clientes que viajam **infrequentemente**
4. Clientes que viajam **em grupos**

---

## 📈 Dashboard Power BI

Os resultados da clusterização foram salvos no **Azure Blob Storage** e visualizados em um **dashboard no Power BI**, contendo:

- Valor médio por compra
- Intervalo médio entre compras
- Volume de compras por data
- Distribuição dos clientes por cluster

---

## 💻 Aplicativo Streamlit

Um aplicativo interativo foi desenvolvido em **Streamlit** para prever a **data** e o **trecho** da próxima compra de um cliente, com base em seu histórico.

---

## 🧾 Tecnologias Utilizadas

- `Python` (pandas, scikit-learn, XGBoost, Faker, etc.)
- `Streamlit` – app web interativo
- `Power BI` – visualização dos clusters
- `Azure Blob Storage` – armazenamento final dos dados
- `Jupyter Notebooks` – desenvolvimento dos modelos

---

