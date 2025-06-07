from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import io
import os
import numpy as np
import torch.nn.functional as F
import cv2
from model import MultiScaleCNNSwinTransformer  # Import the model
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiScaleCNNSwinTransformer().to(device)
model.load_state_dict(torch.load('multi_scale_cnn_swin_transformer1.pth', map_location=device))
model.eval()  # Set to evaluation mode

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Class labels
classes = ['Immature', 'Mature', 'Normal']

# Directory to save heatmaps
heatmap_dir = "heatmaps"
os.makedirs(heatmap_dir, exist_ok=True)


def generate_gradcam_heatmap(model, image, target_layer, file_name):
    image_np = np.array(image).astype(np.float32) / 255.0
    transformed_image = transform(image).unsqueeze(0).to(device)

    cam = GradCAM(model=model.cnn, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=transformed_image)[0]
    heatmap = cv2.resize(grayscale_cam, (image_np.shape[1], image_np.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed_image = cv2.addWeighted(image_np, 0.6, heatmap.astype(np.float32) / 255.0, 0.4, 0)

    heatmap_filename = f"gradcam_{file_name}.jpg"
    heatmap_path = os.path.join(heatmap_dir, heatmap_filename)
    cv2.imwrite(heatmap_path, cv2.cvtColor(overlayed_image * 255, cv2.COLOR_RGB2BGR))
    return heatmap_filename


def generate_gradcam_interpretation(pred_class, probability, heatmap_focus):
    """
    Generates an interpretation of a Grad-CAM heatmap for cataract classification.

    Parameters:
    pred_class (str): The predicted class label ("Mature", "Immature", "Normal").
    probability (float): Model confidence score.
    heatmap_focus (str): The primary region of the eye the heatmap highlights.

    Returns:
    str: A refined explanation of the Grad-CAM heatmap.
    """
    interpretation = [f"Interpretation of the Grad-CAM Heatmap ({pred_class}):"]

    if pred_class == "Normal":
        interpretation.append("Model Focus Areas (Red/Yellow):")
        if pred_class == "Normal" and heatmap_focus != "lens":
            interpretation.append(f"⚠️ Warning: Heatmap focus is '{heatmap_focus}', but prediction is 'Normal'.")
            interpretation.append("Possible issue: Model may not be focusing on relevant eye regions.")
    
        if heatmap_focus == "lens":
            interpretation.append("- The model is verifying the clarity of the lens, which is a key feature of a normal eye.")
            interpretation.append("- The attention on the lens confirms that no significant opacity or clouding is detected.")
        else:
            interpretation.append("- The model is inspecting other parts of the eye, which might indicate some normal variations.")
    
    elif pred_class == "Mature":
        interpretation.append("Red and Yellow Areas (High Attention):")
        interpretation.append("- The model focuses on the lens, detecting high opacity and cloudiness.")
        interpretation.append("- This is characteristic of a mature cataract, where vision is significantly obstructed.")
    
    elif pred_class == "Immature":
        interpretation.append("Red and Yellow Areas (High Attention):")
        interpretation.append("- The heatmap shows partial lens opacity, suggesting early-stage cataract formation.")
        interpretation.append("- The model detects mild clouding, but the eye retains some transparency.")

    

    return "\n".join(interpretation)



@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'detail': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'detail': 'No selected file'}), 400

    image = Image.open(io.BytesIO(file.read())).convert('RGB')
    transformed_image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(transformed_image)
        probabilities = F.softmax(output, dim=1)
        predicted_class = probabilities.argmax(dim=1).item()
        confidence = probabilities.max(dim=1).values.item() * 100

   
    heatmap_filename = generate_gradcam_heatmap(model, image, target_layer=model.cnn.layer4[-1], file_name=file.filename)


    # Determine heatmap focus (for interpretation)
    heatmap_focus = "lens"  # You may need a more dynamic way to determine this

    # Generate Grad-CAM interpretation
    interpretation = generate_gradcam_interpretation(classes[predicted_class], confidence, heatmap_focus)

    response = {
    'diagnosis': classes[predicted_class],
    'confidence': confidence,
    'heatmap_url': f'/heatmaps/{heatmap_filename}',
    'interpretation': interpretation
}

    return jsonify(response)

@app.route('/')
def home():
    return '''
        <h1 style="text-align:center; color:green;">✅ Cataract Prediction Flask App is Running!</h1>
    '''

@app.route('/heatmaps/<filename>')
def serve_heatmap(filename):
    return send_file(os.path.join(heatmap_dir, filename), mimetype='image/jpg')

if __name__ == '__main__':
    app.run(port=5673, debug=True)


