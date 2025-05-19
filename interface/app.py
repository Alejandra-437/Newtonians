import gradio as gr
import numpy as np
import pandas as pd
import joblib
from logistic_model import OneVsRestLogisticRegression, LogisticRegressionGD

_ = OneVsRestLogisticRegression
_ = LogisticRegressionGD

# carga los archivos kt que contienen el model, scale y feature previamente entrenado en el notebook
modelo = joblib.load("interface/modelo_parkinson.pkl")
scaler = joblib.load("interface/scaler_parkinson.pkl")
features = joblib.load("interface/features_parkinson.pkl")

#realiza la prediccion y se conecta a la interfaz
def predecir_parkinson(*inputs):
    datos = np.array([inputs])
    datos_df = pd.DataFrame(datos, columns=features)
    datos_escalados = scaler.transform(datos_df)
    pred = modelo.predict(datos_escalados)
    return "Parkinson" if pred[0] == 1 else "Sano"

#crea las entradas de tipo numerico para cada caracteristica
inputs = [gr.Number(label=feat) for feat in features]

#configuracion de la interfaz de gradio
demo = gr.Interface(
    fn=predecir_parkinson,
    inputs=inputs,
    outputs=gr.Text(label="Resultado"),
    title="Clasificador de Parkinson"
)

if __name__ == "__main__":
    demo.launch()
