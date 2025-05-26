import numpy as np
import pandas as pd
import re
from typing import Dict, Any
import os

class LaptopPricePredictor:
    def __init__(self):
        # parametros del modelo
        self.theta = None
        self.min_values = None
        self.max_values = None
        self.target_encodings = {}
        self.is_trained = False
        self.feature_columns = []
        
    def load_and_preprocess_data(self, csv_path: str = './datasources/laptops/laptop_price.csv'):
      
        # Leer datos
        df = pd.read_csv(csv_path, encoding='latin-1')
        
        # Eliminar laptop_ID
        df.drop('laptop_ID', axis=1, inplace=True)
        
        # Preprocesar Weight
        df['Weight'] = df['Weight'].str.replace('kg','')
        df['Weight'] = df['Weight'].astype('float64')
        
        # Preprocesar Ram
        df['Ram'] = df['Ram'].str.replace('GB','')
        df['Ram'] = df['Ram'].astype('int64')
        
        # Convertir precio de euros a dólares
        df.rename(columns={'Price_euros':'Price'}, inplace=True)
        df['Price'] *= 1.13
        
        # Eliminar duplicados
        df = df.drop_duplicates()
        
        # Procesar CPU - extraer frecuencia
        def split_str_freq(x):
            x = x.split(' ')
            return x[-1] 
        
        df['CpuFrequency'] = df['Cpu'].apply(split_str_freq)
        df['CpuFrequency'] = df['CpuFrequency'].str.replace('GHz', '')
        df['CpuFrequency'] = df['CpuFrequency'].astype('float64')
        
        # Quitar frecuencia del modelo CPU
        df['Cpu'] = df['Cpu'].str.replace(r'(\d+(?:\.\d+)?GHz)', '', regex=True)
        df['CpuModel'] = df['Cpu']
        df.drop('Cpu', axis=1, inplace=True)
        
        # Procesar GPU - extraer marca
        def split_str_brand(x):
            x = x.split(' ')
            return x[0]
        
        df['GpuBrand'] = df['Gpu'].apply(split_str_brand)
        
        # Extraer modelo GPU
        def split_model(x):
            x = x.split(' ')
            if len(x) == 2:
                model = x[-1]
            elif len(x) == 3:
                model = x[-2:]  
            elif len(x) == 4:
                model = x[-3:]  
            elif len(x) == 5:
                model = x[-3:]  
            return ' '.join(model) if isinstance(model, list) else model
        
        df['GpuModel'] = df['Gpu'].apply(split_model)
        df.drop('Gpu', axis=1, inplace=True)
        
        # Procesar Memory (almacenamiento)
        df[['Storage', 'SSD', 'HDD', 'Flash Storage', 'Hybrid']] = 0
        
        def convert_size(size_text):
            size_text = size_text.replace(" ", "")
            if "TB" in size_text:
                return int(float(size_text.replace("TB", "").replace("GB", "")) * 1000)
            elif "GB" in size_text:
                return int(float(size_text.replace("GB", "")))
            return 0
        
        for i, row in df.iterrows():
            memory = row['Memory']
            devices = memory.split('+')
            total = ssd = hdd = flash = hybrid = 0
            
            for device in devices:
                device = device.strip()
                parts = device.split()
                if len(parts) >= 2:
                    size_text = parts[0]
                    dtype = " ".join(parts[1:])
                    size = convert_size(size_text)
                    total += size
                    if 'SSD' in dtype:
                        ssd += size
                    elif 'HDD' in dtype:
                        hdd += size
                    elif 'Flash Storage' in dtype:
                        flash += size
                    elif 'Hybrid' in dtype:
                        hybrid += size
            
            df.loc[i, ['Storage', 'SSD', 'HDD', 'Flash Storage', 'Hybrid']] = [total, ssd, hdd, flash, hybrid]
        
        df.drop('Memory', axis=1, inplace=True)
        
        # Procesar ScreenResolution
        df['Resolution'] = df['ScreenResolution'].str.extract(r'(\d+x\d+)')
        df[['Width', 'Height']] = df['Resolution'].str.split('x', expand=True)
        df['Width'] = df['Width'].astype('int64')
        df['Height'] = df['Height'].astype('int64')
        df.drop('Resolution', axis=1, inplace=True)
        
        # Extraer tipo de pantalla
        df['Screen'] = df['ScreenResolution'].str.replace(r'(\d+x\d+)','',regex=True)
        df['Screen'] = df['Screen'].replace(r'(/)','',regex=True)
        
        # Detectar touchscreen
        df['Touchscreen'] = df['Screen'].str.extract(r'(Touchscreen)')
        df['Screen'] = df['Screen'].str.replace(r'(Touchscreen)','',regex=True)
        
        # Convertir touchscreen a binario
        df['Touchscreen'] = df['Touchscreen'].replace('Touchscreen', 1)
        df['Touchscreen'] = df['Touchscreen'].fillna(0)
        df['Touchscreen'] = df['Touchscreen'].astype('int64')
        
        df.drop('ScreenResolution', axis=1, inplace=True)
        
        return df
    
    def target_encode_and_save(self, df, by, on, m=300):
       
        mean = df[on].mean()
        agg = df.groupby(by)[on].agg(['count', 'mean'])
        counts = agg['count']
        means = agg['mean']
        smooth = (counts * means + m * mean) / (counts + m)
        
        # Guardar encoding
        self.target_encodings[by] = smooth.to_dict()
        
        return df[by].map(smooth)
    
    def fit(self, csv_path: str = './datasources/laptops/laptop_price.csv', alpha: float = 0.1, iterations: int = 5000):
      
        print("Cargando y preprocesando datos...")
        df = self.load_and_preprocess_data(csv_path)
        
        print("Aplicando target encoding...")
        # Aplicar target encoding a columnas categóricas
        categorical_columns = ['Company', 'Product', 'TypeName', 'Screen', 'CpuModel', 'GpuBrand', 'GpuModel', 'OpSys']
        
        for col in categorical_columns:
            if col in df.columns:
                df[col] = self.target_encode_and_save(df, col, 'Price')
        
        print("Normalizando datos...")
        # guardar valores para normalización
        self.min_values = df.min()
        self.max_values = df.max()
        
        # normalizar
        df_normalized = (df - self.min_values) / (self.max_values - self.min_values)
        
        # preparar datos para entrenamiento
        X = df_normalized.drop('Price', axis=1).values
        y = df_normalized['Price'].values.reshape(-1, 1)
        
        # guardar nombres de columnas para referencia
        self.feature_columns = df_normalized.drop('Price', axis=1).columns.tolist()
        
        # agregar columna de bias
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        
        print(f"Entrenando modelo con {len(df)} muestras...")
        # entrenando usando gradiente descendente
        self.theta, cost_history = self._gradient_descent(X, y, alpha, iterations)
        
        self.is_trained = True
        print(f"Entrenamiento completado. Costo final: {cost_history[-1]:.6f}")
        
        return cost_history
    
    def _gradient_descent(self, X, y, alpha, iterations):
      
        m, n = X.shape
        theta = np.zeros((n, 1))
        cost_history = []
        
        for i in range(iterations):
            predictions = X @ theta
            errors = predictions - y
            gradient = (1/m) * X.T @ errors
            theta = theta - alpha * gradient
            
            # calcular costo
            cost = (1/(2*m)) * np.sum(errors**2)
            cost_history.append(cost)
            
            if i % 1000 == 0:
                print(f"Iteración {i}: Costo = {cost:.6f}")
        
        return theta, cost_history
    
    def preprocess_single_input(self, laptop_data: Dict[str, Any]) -> np.ndarray:
       
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado. Use fit() primero.")
        
        # creando DataFrame temporal
        df = pd.DataFrame([laptop_data])
        
        
        # Weight
        df['Weight'] = df['Weight'].str.replace('kg', '')
        df['Weight'] = df['Weight'].astype('float64')
        
        # Ram
        df['Ram'] = df['Ram'].str.replace('GB', '')
        df['Ram'] = df['Ram'].astype('int64')
        
        # CPU
        def split_str_freq(x):
            x = x.split(' ')
            return x[-1]
        
        df['CpuFrequency'] = df['Cpu'].apply(split_str_freq)
        df['CpuFrequency'] = df['CpuFrequency'].str.replace('GHz', '')
        df['CpuFrequency'] = df['CpuFrequency'].astype('float64')
        
        df['Cpu'] = df['Cpu'].str.replace(r'(\d+(?:\.\d+)?GHz)', '', regex=True)
        df['CpuModel'] = df['Cpu']
        df.drop('Cpu', axis=1, inplace=True)
        
        # GPU
        def split_str_brand(x):
            x = x.split(' ')
            return x[0]
        
        def split_model(x):
            x = x.split(' ')
            if len(x) == 2:
                model = x[-1]
            elif len(x) == 3:
                model = x[-2:]  
            elif len(x) == 4:
                model = x[-3:]  
            elif len(x) == 5:
                model = x[-3:]  
            return ' '.join(model) if isinstance(model, list) else model
        
        df['GpuBrand'] = df['Gpu'].apply(split_str_brand)
        df['GpuModel'] = df['Gpu'].apply(split_model)
        df.drop('Gpu', axis=1, inplace=True)
        
        # Memory
        df[['Storage', 'SSD', 'HDD', 'Flash Storage', 'Hybrid']] = 0
        
        def convert_size(size_text):
            size_text = size_text.replace(" ", "")
            if "TB" in size_text:
                return int(float(size_text.replace("TB", "").replace("GB", "")) * 1000)
            elif "GB" in size_text:
                return int(float(size_text.replace("GB", "")))
            return 0
        
        for i, row in df.iterrows():
            memory = row['Memory']
            devices = memory.split('+')
            total = ssd = hdd = flash = hybrid = 0
            
            for device in devices:
                device = device.strip()
                parts = device.split()
                if len(parts) >= 2:
                    size_text = parts[0]
                    dtype = " ".join(parts[1:])
                    size = convert_size(size_text)
                    total += size
                    if 'SSD' in dtype:
                        ssd += size
                    elif 'HDD' in dtype:
                        hdd += size
                    elif 'Flash Storage' in dtype:
                        flash += size
                    elif 'Hybrid' in dtype:
                        hybrid += size
            
            df.loc[i, ['Storage', 'SSD', 'HDD', 'Flash Storage', 'Hybrid']] = [total, ssd, hdd, flash, hybrid]
        
        df.drop('Memory', axis=1, inplace=True)
        
        # ScreenResolution
        df['Resolution'] = df['ScreenResolution'].str.extract(r'(\d+x\d+)')
        df[['Width', 'Height']] = df['Resolution'].str.split('x', expand=True)
        df['Width'] = df['Width'].astype('int64')
        df['Height'] = df['Height'].astype('int64')
        df.drop('Resolution', axis=1, inplace=True)
        
        df['Screen'] = df['ScreenResolution'].str.replace(r'(\d+x\d+)', '', regex=True)
        df['Screen'] = df['Screen'].replace(r'(/)', '', regex=True)
        
        df['Touchscreen'] = df['Screen'].str.extract(r'(Touchscreen)')
        df['Screen'] = df['Screen'].str.replace(r'(Touchscreen)', '', regex=True)
        
        df['Touchscreen'] = df['Touchscreen'].replace('Touchscreen', 1)
        df['Touchscreen'] = df['Touchscreen'].fillna(0)
        df['Touchscreen'] = df['Touchscreen'].astype('int64')
        
        df.drop('ScreenResolution', axis=1, inplace=True)
        
        # target encoding usando valores guardados
        categorical_columns = ['Company', 'Product', 'TypeName', 'Screen', 'CpuModel', 'GpuBrand', 'GpuModel', 'OpSys']
        
        for col in categorical_columns:
            if col in df.columns:
                value = df[col].iloc[0]
                if col in self.target_encodings and value in self.target_encodings[col]:
                    df[col] = self.target_encodings[col][value]
                else:
                    if col in self.target_encodings:
                        df[col] = np.mean(list(self.target_encodings[col].values()))
                    else:
                        df[col] = 0.5  # Valor neutral
        
        # normalizar usando valores guardados
        df_normalized = (df - self.min_values.drop('Price')) / (self.max_values.drop('Price') - self.min_values.drop('Price'))
        
        df_normalized = df_normalized.fillna(0)
        
        df_normalized = df_normalized.reindex(columns=self.feature_columns, fill_value=0)
        
        return df_normalized.values
    
    def predict_price_usd(self, laptop_data: Dict[str, Any]) -> float:
       
        # preprocesar datos de entrada
        X = self.preprocess_single_input(laptop_data)
        
        # agregaar columna de bias
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        
        # hacer prediccion normalizada
        prediction_normalized = X @ self.theta
        
        # denormalizar resultado
        price_min = self.min_values['Price']
        price_max = self.max_values['Price']
        
        prediction_usd = prediction_normalized * (price_max - price_min) + price_min
        
        return float(prediction_usd[0, 0])
    
    def get_model_info(self):
       
        if not self.is_trained:
            return "Modelo no entrenado"
        
        info = {
            "estado": "Entrenado",
            "num_features": len(self.feature_columns),
            "features": self.feature_columns,
            "target_encodings_disponibles": list(self.target_encodings.keys())
        }
        
        return info

