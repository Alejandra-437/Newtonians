import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("./datasources/parkinsons/parkinsons.data")
df = df.drop_duplicates()

X = df.drop(['name', 'status'], axis=1)
y = df['status']

feature_names = X.columns.tolist()

class ManualScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.fitted = False
    
    def fit(self, X):
        X = np.array(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0, ddof=0)
        self.std_ = np.where(self.std_ == 0, 1, self.std_)
        self.fitted = True
        return self
    
    def transform(self, X):
        if not self.fitted:
            raise ValueError("Scaler no ajustado.")
        X = np.array(X)
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)


class LogisticRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000, tol=1e-4):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol
        self.weights = None
        self.bias = None
        self.losses = []
    
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, X, y):
        m = X.shape[0]
        z = np.dot(X, self.weights) + self.bias
        h = self.sigmoid(z)
        epsilon = 1e-15
        h = np.clip(h, epsilon, 1 - epsilon)
        return (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    
    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        self.losses = []

        for i in range(self.n_iterations):
            z = np.dot(X, self.weights) + self.bias
            h = self.sigmoid(z)
            dw = (1/m) * np.dot(X.T, (h - y))
            db = (1/m) * np.sum(h - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            loss = self.compute_loss(X, y)
            self.losses.append(loss)
            if i > 0 and abs(self.losses[i] - self.losses[i-1]) < self.tol:
                break
        return self
    
    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

class OneVsRestLogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, tol=1e-4):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol
        self.classifiers = {}
        self.classes = None
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            y_binary = (y == c).astype(int)
            clf = LogisticRegressionGD(
                learning_rate=self.learning_rate,
                n_iterations=self.n_iterations,
                tol=self.tol
            )
            clf.fit(X, y_binary)
            self.classifiers[c] = clf
        return self
    
    def predict_proba(self, X):
        proba = np.zeros((X.shape[0], len(self.classes)))
        for i, c in enumerate(self.classes):
            proba[:, i] = self.classifiers[c].predict_proba(X)
        return proba
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes[np.argmax(proba, axis=1)]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = ManualScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = OneVsRestLogisticRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Exactitud en test: {accuracy:.4f}")

class ParkinsonPredictor:
    def __init__(self, model, scaler, feature_names):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names

    def preprocess_input(self, raw_csv_line):
        raw_values = raw_csv_line.strip().split(",")
        values = [float(val) for val in raw_values if val != '']
        if len(values) != len(self.feature_names):
            raise ValueError(f"Esperado {len(self.feature_names)} columnas, recibido {len(values)}.")
        X = np.array(values).reshape(1, -1)
        return self.scaler.transform(X)

    def predict(self, raw_csv_line):
        X_scaled = self.preprocess_input(raw_csv_line)
        prediction = self.model.predict(X_scaled)[0]
        return prediction


