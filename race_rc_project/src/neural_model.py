class NeuralModel:
    def __init__(self):
        self.is_fitted = False

    def fit(self, x, y):
        self.is_fitted = True
        return self

    def predict(self, x):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        return [0 for _ in range(len(x))]
