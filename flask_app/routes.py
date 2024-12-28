from flask import Blueprint, request, jsonify
from flask_app.models import ModelHandler
import traceback

# Create Flask Blueprint
api_blueprint = Blueprint("api", __name__)

# Initialize model handler (modify `use_mlflow` as needed)
model_handler = ModelHandler(use_mlflow=True)
model_handler.load_model()

@api_blueprint.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data
        input_data = request.get_json()
        print("Input Data:", input_data)  # Log the input for debugging

        # Convert input to a tensor (if necessary)
        import torch
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        print("Converted to Tensor:", input_tensor)

        # Perform prediction using MLflow model
        predictions = model_handler.model.predict(input_tensor.numpy())
        print("Predictions:", predictions)

        # Return predictions as JSON
        return jsonify({"predictions": predictions.tolist()})

    except Exception as e:
        # Log the full traceback for debugging
        traceback.print_exc()

        # Return an error response
        return jsonify({"error": str(e)}), 500