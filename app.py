from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pickle
import cv2
from utils import pre_process_image

app = Flask(__name__)

# Load TensorFlow models
cnn_model = tf.keras.models.load_model("models/cnn_model.h5")
mlp_model = tf.keras.models.load_model("models/mlp_model.h5")

# Load pickled models
with open("models/log_reg_model.pkl", "rb") as file:
    log_reg_model = pickle.load(file)

with open("models/svm_model.pkl", "rb") as file:
    svm_model = pickle.load(file)

with open("models/knn_model.pkl", "rb") as file:
    knn_model = pickle.load(file)

# Load PCA for dimensionality reduction
with open("models/pca_transform.pkl", "rb") as file:
    pca = pickle.load(file)

# Model dictionary for prediction routing
models = {
    "cnn": cnn_model,
    "mlp": mlp_model,
    "log_reg": log_reg_model,
    "svm": svm_model,
    "knn": knn_model,
}


@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        # Read image from request
        file = request.files["image"]
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (28, 28))  # Resize to (28, 28)

        try:
            # Preprocess the image
            processed_image = pre_process_image(image)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        # Flattened and 4D images for different model requirements
        processed_image_flat = processed_image.flatten().reshape(
            1, -1
        )  # For MLP and traditional ML models
        processed_image_4d = processed_image.reshape(1, 28, 28, 1)  # For CNN model

        # Apply PCA transformation for non-CNN models
        processed_image_pca = pca.transform(
            processed_image_flat
        )  # Transform to 50 features

        predictions = {}

        # Iterate through models to generate predictions
        for model_name, model in models.items():
            if model_name == "cnn":
                # Use the 4D input shape for CNN
                prediction = model.predict(processed_image_4d)
                confidence_score = float(np.max(prediction))
                predicted_class = int(np.argmax(prediction))
            elif model_name == "mlp":
                # Use the flattened input shape for MLP
                prediction = model.predict(processed_image_flat)
                confidence_score = float(np.max(prediction))
                predicted_class = int(np.argmax(prediction))
            else:
                # Use PCA-transformed input for traditional ML models
                prediction = (
                    model.predict_proba(processed_image_pca)
                    if hasattr(model, "predict_proba")
                    else model.predict(processed_image_pca)
                )
                confidence_score = (
                    float(np.max(prediction))
                    if hasattr(model, "predict_proba")
                    else 1.0
                )
                predicted_class = (
                    int(np.argmax(prediction))
                    if hasattr(model, "predict_proba")
                    else int(prediction[0])
                )

            predictions[model_name] = {
                "predicted_class": predicted_class,
                "predicted_digit": str(predicted_class),
                "confidence_score": confidence_score,
            }

        # Identify the best model based on confidence score
        best_model = max(predictions, key=lambda k: predictions[k]["confidence_score"])
        best_prediction = predictions[best_model]
        best_prediction["model"] = best_model

        return jsonify(
            {"all_predictions": predictions, "best_prediction": best_prediction}
        )
    else:
        return jsonify("Welcome to Alkeema Image Classifcation API")


if __name__ == "__main__":
    app.run(debug=True, port=7070)
