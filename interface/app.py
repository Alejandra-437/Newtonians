import gradio as gr
import numpy as np
import pandas as pd
import joblib
from logistic_model import OneVsRestLogisticRegression, ManualScaler, OneVsRestLogisticRegression, ParkinsonPredictor
from linear_model import LaptopPricePredictor

# Cargar datos
df = pd.read_csv("datasources/parkinsons/parkinsons.data").drop_duplicates()
X = df.drop(['name', 'status'], axis=1)
y = df['status']
feature_names = X.columns.tolist()

# Entrenar modelo
scaler = ManualScaler()
X_scaled = scaler.fit_transform(X)

model = OneVsRestLogisticRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X_scaled, y)

# Crear instancia del predictor
predictor_parkinson = ParkinsonPredictor(model, scaler, feature_names)
predictor_laptop = LaptopPricePredictor()
predictor_laptop.fit()

# Función para la interfaz
def predecir_parkinson(*inputs):
    raw_line = ",".join(str(x) for x in inputs)
    resultado = predictor_parkinson.predict(raw_line)
    return "Persona con Parkinson" if resultado == 1 else "Persona Sana"

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
    price = predictor_laptop.predict_price_usd(input_data)
    return f"${price:,.2f} USD"

interface_parkinson = gr.Interface(
    fn=predecir_parkinson,
    inputs=[gr.Number(label=feat) for feat in feature_names],
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
