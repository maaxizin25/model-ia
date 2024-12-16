from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Inicialização da API
app = FastAPI(title="API de Previsão de Categoria D5")

# Carregar o modelo e o scaler
modelo_dsa = joblib.load('modeloD5.pkl')
scaler = joblib.load('padronizador.pkl')

# Modelo de dados para entrada
class InputData(BaseModel):
    first_deposit_amount: float
    amount_7_days: float
    qtd_depositos_7_days: int

# Função de pré-processamento
def preprocess_input(data: InputData):
    try:
        # Criar DataFrame com os nomes usados no treinamento
        df = pd.DataFrame({
            'first_deposit_amount': [data.first_deposit_amount],
            'amount_7_days': [data.amount_7_days],
            'qtd_depositos_7_days': [data.qtd_depositos_7_days]
        })
        
        # Aplicar o scaler
        df[['first_deposit_amount', 'amount_7_days', 'qtd_depositos_7_days']] = scaler.transform(
            df[['first_deposit_amount', 'amount_7_days', 'qtd_depositos_7_days']]
        )
        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro no pré-processamento: {str(e)}")

# Rota principal de previsão
@app.post("/predict")
async def predict(data: InputData):
    try:
        # Pré-processar os dados
        input_data = preprocess_input(data)
        
        # Fazer a previsão
        prediction = modelo_dsa.predict(input_data)
        probabilities = modelo_dsa.predict_proba(input_data).tolist()
        categoria = "D5" if int(prediction[0]) == 1 else "Não D5"
        return {
            "categoria_prevista": categoria,
            "probabilidades": probabilities
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao fazer a previsão: {str(e)}")

# Rota para verificar a saúde da API
@app.get("/")
async def health_check():
    return {"status": "API ativa e funcionando!"}
