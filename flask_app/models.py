import mlflow.pyfunc
import os

class ModelHandler:
    """Handles loading and predicting with the model."""
    def __init__(self, use_mlflow=True):
        self.model = None
        self.use_mlflow = use_mlflow
        self.mlflow_model_uri = os.getenv("MLFLOW_MODEL_URI", "models:/24-nn-1/1")
        self.local_model_path = os.getenv("LOCAL_MODEL_PATH", "model.pkl")

    def load_model(self):
        """Load the model, either from MLflow or locally."""
        if self.use_mlflow:
            print(f"Loading model from MLflow: {self.mlflow_model_uri}")
            self.model = mlflow.pyfunc.load_model(self.mlflow_model_uri)
        else:
            import joblib
            print(f"Loading local model from: {self.local_model_path}")
            self.model = joblib.load(self.local_model_path)

    def predict(self, input_data):
        """Make predictions with the loaded model."""
        if not self.model:
            raise ValueError("Model is not loaded. Call `load_model()` first.")
        
        return self.model.predict(input_data)