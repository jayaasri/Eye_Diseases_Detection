from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import matplotlib
import torch
from torchvision import transforms
from PIL import Image
import io
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os
from lime import lime_image
from skimage.segmentation import mark_boundaries
from model1 import HybridModel  # Import your model class

app = Flask(__name__)
CORS(app)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridModel().to(device)
model.load_state_dict(torch.load('glaucoma_hybrid_model.pth', map_location=device))
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Class labels
classes = ['Glaucoma', 'Normal']

# Directory to save LIME explanations
lime_dir = "lime_explanations"
os.makedirs(lime_dir, exist_ok=True)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'detail': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'detail': 'No selected file'}), 400

    # Read and preprocess the image
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction using the trained model
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = probabilities.argmax(dim=1).item()
        confidence = probabilities.max(dim=1).values.item() * 100  # Convert to percentage

    # Generate LIME explanation
    explainer = lime_image.LimeImageExplainer()

    def batch_predict(images):
        """Predict function for LIME, processes batch of images."""
        batch = torch.stack([transform(Image.fromarray(img)) for img in images], dim=0).to(device)
        with torch.no_grad():
            outputs = model(batch)
            return F.softmax(outputs, dim=1).cpu().numpy()

    image_np = np.array(image.resize((224, 224)))
    explanation = explainer.explain_instance(
        image_np,
        batch_predict,
        top_labels=2,
        hide_color=0,
        num_samples=50  # Faster execution
    )

    # Get the LIME mask for the predicted class
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,  # Focusing on fewer important features
        hide_rest=False
    )

    # Overlay LIME mask on original image
    lime_image_path = os.path.join(lime_dir, f'lime_{file.filename}.png')
    
    plt.imshow(mark_boundaries(temp, mask))
    plt.axis('off')
    plt.savefig(lime_image_path, bbox_inches='tight')
    plt.close()

    # **Generate LIME interpretation statement based on classification**
    if predicted_class == 0:  # Glaucoma detected
        interpretation = (
            "The highlighted areas in the heatmap indicate regions influencing the classification. "
            "For Glaucoma, attention is drawn to the optic nerve head, suggesting abnormalities."
        )
    else:  # Normal detected
        interpretation = (
            "The LIME visualization shows no abnormal patterns, indicating a healthy optic nerve. "
            "The model found no significant glaucoma-related features."
        )

    # Response
    response = {
        'diagnosis': classes[predicted_class],
        'confidence': confidence,
        'finding': "Potential glaucoma detected" if predicted_class == 0 else "No signs of glaucoma",
        'heatmap_url': f'/lime/{file.filename}',  # URL for LIME visualization
        'interpretation': interpretation  # LIME Interpretation statement
    }
    return jsonify(response)



@app.route('/lime/<filename>')
def get_lime_explanation(filename):
    """Serve the generated LIME explanation image."""
    lime_image_path = os.path.join(lime_dir, f'lime_{filename}.png')
    if os.path.exists(lime_image_path):
        return send_file(lime_image_path, mimetype='image/png')
    return jsonify({'detail': 'LIME image not found'}), 404

@app.route('/')
def home():
    return '''
        <h1 style="text-align:center; color:green;">✅ Glaucoma Prediction Flask App is Running!</h1>
    '''

if __name__ == '__main__':
    print("✅ Glaucoma Prediction Flask App is running on http://127.0.0.1:5679/")
    app.run(port=5679, debug=True)
