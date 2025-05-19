import numpy as np

class LogisticRegressionGD:
    """
    Implementamos Regresión Logística con Descenso de Gradiente
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000, tol=1e-4):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol
        self.weights = None
        self.bias = None
        self.losses = []
        
    def sigmoid(self, z):
        """Función sigmoid para transformar valores a probabilidades"""
        # Evitamos el desbordamiento numérico
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, X, y):
        """Función de pérdida logística (cross-entropy)"""
        m = X.shape[0]
        z = np.dot(X, self.weights) + self.bias
        h = self.sigmoid(z)
        
        # Evitar log(0) agregando un epsilon pequeño
        epsilon = 1e-15
        h = np.clip(h, epsilon, 1 - epsilon)
        
        # Cálculo de la pérdida logística (cross-entropy)
        loss = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return loss
    
    def fit(self, X, y):
        """Entrenamos el modelo usando descenso de gradiente"""
        # Inicializamos parámetros
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        self.losses = []
        
        # Ejecución del Descenso de gradiente
        for i in range(self.n_iterations):
            # Obtenemos Paso forward
            z = np.dot(X, self.weights) + self.bias
            h = self.sigmoid(z)
            
            # Calculamos gradientes
            dw = (1/m) * np.dot(X.T, (h - y))
            db = (1/m) * np.sum(h - y)
            
            # Actualización de parámetros
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Calculamos y almacenamos la pérdida
            loss = self.compute_loss(X, y)
            self.losses.append(loss)
            
            # Verificamos convergencia temprana
            if i > 0 and abs(self.losses[i] - self.losses[i-1]) < self.tol:
                print(f"Convergencia alcanzada en la iteración {i}")
                break
            
            # Imprimimos el progreso cada 100 iteraciones
            if i % 100 == 0:
                print(f"Iteración {i}, Pérdida: {loss:.6f}")
                
        return self
    
    def predict_proba(self, X):
        """Predecimos probabilidades de clase"""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """Predecimos clase (0 o 1)"""
        return (self.predict_proba(X) >= threshold).astype(int)
    

class OneVsRestLogisticRegression:
    """
    Implementamos la estrategia One-vs-Rest para clasificación multiclase
    usando regresión logística con descenso de gradiente
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000, tol=1e-4):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol
        self.classifiers = {}
        self.classes = None
        
    def fit(self, X, y):
        """Entrenamos un clasificador binario para cada clase"""
        self.classes = np.unique(y)
        
        # Para cada clase, entrenar un clasificador binario
        for c in self.classes:
            print(f"\nEntrenando clasificador para clase {c}")
            
            # Crear etiquetas binarias (1 para la clase actual, 0 para el resto)
            y_binary = (y == c).astype(int)
            
            # Crear y entrenar un clasificador para esta clase
            clf = LogisticRegressionGD(
                learning_rate=self.learning_rate,
                n_iterations=self.n_iterations,
                tol=self.tol
            )
            clf.fit(X, y_binary)
            
            # Almacenar el clasificador
            self.classifiers[c] = clf
        
        return self
    
    def predict_proba(self, X):
        """PRedecimos probabilidades para cada clase"""
        # Creamos una matriz de probabilidades (muestras x clases)
        proba = np.zeros((X.shape[0], len(self.classes)))
        
        # Predecimos la probabilidad para cada clase
        for i, c in enumerate(self.classes):
            proba[:, i] = self.classifiers[c].predict_proba(X)
            
        return proba
    
    def predict(self, X):
        """PRedecimos la clase con mayor probabilidad"""
        proba = self.predict_proba(X)
        return self.classes[np.argmax(proba, axis=1)]