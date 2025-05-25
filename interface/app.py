import gradio as gr
import numpy as np
import pandas as pd
import joblib
from logistic_model import OneVsRestLogisticRegression, LogisticRegressionGD
from linear_model import LaptopPricePredictor

# carga el modelo Parkinson
_ = OneVsRestLogisticRegression
_ = LogisticRegressionGD
modelo = joblib.load("interface/modelo_parkinson.pkl")
scaler = joblib.load("interface/scaler_parkinson.pkl")
features = joblib.load("interface/features_parkinson.pkl")

# carga el modelo de laptop
predictor = LaptopPricePredictor()
predictor.fit()

#funcion de predicion de parkinson
def predecir_parkinson(*inputs):
    datos = np.array([inputs])
    df = pd.DataFrame(datos, columns=features)
    datos_escalados = scaler.transform(df)
    pred = modelo.predict(datos_escalados)
    return "Persona con Parkinson" if pred[0] == 1 else "Persona Sana"

# funcion de predicción Laptop
def predecir_laptop(
    Company, Product, TypeName, Inches, ScreenResolution, OpSys,
    Weight, Ram, Cpu, Gpu, Memory
):
    input_data = {
        "Company": Company,
        "Product": Product,
        "TypeName": TypeName,
        "Inches": Inches,
        "ScreenResolution": ScreenResolution,
        "OpSys": OpSys,
        "Weight": Weight,
        "Ram": Ram,
        "Cpu": Cpu,
        "Gpu": Gpu,
        "Memory": Memory
    }
    price = predictor.predict_price_usd(input_data)
    return f"${price:,.2f} USD"

interface_parkinson = gr.Interface(
    fn=predecir_parkinson,
    inputs=[gr.Number(label=feat) for feat in features],
    outputs=gr.Textbox(label="Resultado"),
    title="Clasificador de Parkinson",
    theme="soft"
)

interface_laptop = gr.Interface(
    fn=predecir_laptop,
    inputs=[
        gr.Textbox(label="Company"),
        gr.Textbox(label="Product"),
        gr.Textbox(label="TypeName"),
        gr.Number(label="Inches", precision=2),
        gr.Textbox(label="ScreenResolution"),
        gr.Textbox(label="OpSys"),
        gr.Textbox(label="Weight"),
        gr.Textbox(label="RAM"),
        gr.Textbox(label="CPU"),
        gr.Textbox(label="GPU"),
        gr.Textbox(label="Memory"),
    ],
    outputs=gr.Textbox(label="Precio estimado (USD)"),
    title="Predicción de Precio de Laptop",
    theme="soft"
)

#tab con las dos interfaces

demo = gr.TabbedInterface(
    [interface_parkinson, interface_laptop],
    tab_names=["Clasificador de Parkinson \n(Modelo de regresión logística)", "Predicción de Laptop \n(Modelo de regresión lineal)"],
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()
