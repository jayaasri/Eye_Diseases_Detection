
import os
import matplotlib
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, render_template_string,send_file
from transformers import ViTForImageClassification
from torchvision import transforms
import shap
import matplotlib.pyplot as plt
matplotlib.use('Agg') 
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import os
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class labels
class_labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# Model path
MODEL_PATH = "model_epoch_9.pth"

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load model
def load_model():
    print("Loading ViT model...")
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k', num_labels=len(class_labels)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    return model

model = load_model()

# Preprocess uploaded image
def preprocess_image(image):
    image = transform(image).unsqueeze(0)
    return image.to(device)

#def generate_shap_explanation(img_tensor):  
    import shap
    import numpy as np
    import torch
    from io import BytesIO
    import base64
    import cv2
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    img_tensor = img_tensor.to(device)

    def model_forward(x): 
        if isinstance(x, np.ndarray): 
            x = torch.from_numpy(x).permute(0, 3, 1, 2).float().to(device)
        return model(x).logits

    # Convert tensor to numpy
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    if img_np.dtype != np.uint8:
        img_np = (img_np * 255).astype(np.uint8)
    img_np_batch = np.expand_dims(img_np, axis=0)

    # Setup SHAP
    masker = shap.maskers.Image("inpaint_telea", img_np.shape)
    explainer = shap.Explainer(
        model_forward,
        masker,
        output_names=["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    )

    # SHAP values
    shap_values = explainer(img_np_batch, max_evals=50)
    shap_img = shap_values[0].values

    # Smoothen SHAP mask
    if shap_img.ndim == 4:
        shap_mask_gray = shap_img.mean(axis=(2, 3))  # Take the mean across channels
    elif shap_img.ndim == 3:
        shap_mask_gray = shap_img.mean(axis=-1)  # Take the mean across the last axis (channels)
    else:
        shap_mask_gray = shap_img

    shap_mask_gray = (shap_mask_gray - shap_mask_gray.min()) / (shap_mask_gray.max() - shap_mask_gray.min() + 1e-8)

    # 📏 Resize to match original image size
    shap_mask_resized = cv2.resize(shap_mask_gray, (img_np.shape[1], img_np.shape[0]))

    # Blur the SHAP mask
    shap_mask_blurred = gaussian_filter(shap_mask_resized, sigma=2)

    # Create a circular mask
    height, width = img_np.shape[:2]
    Y, X = np.ogrid[:height, :width]
    center_x, center_y = width // 2, height // 2
    radius = min(center_x, center_y)
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    mask = dist_from_center <= radius

    # Apply mask to blurred SHAP values
    shap_mask_blurred_masked = shap_mask_blurred * mask

    # Plot the image with SHAP overlay
    fig, ax = plt.subplots()
    ax.imshow(img_np)
    ax.imshow(shap_mask_blurred_masked, cmap='plasma', alpha=0.5)
    ax.set_title('SHAP Interpretation (Smoothed and Masked)')
    ax.axis('off')
    plt.show()

    # Convert the plot to base64 for web display
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode()
    plt.close()

    return image_base64



#working new code with old-final
def generate_shap_explanation(img_tensor):  
    import shap
    import numpy as np
    import torch
    from io import BytesIO
    import base64
    import cv2
    import matplotlib.pyplot as plt

    img_tensor = img_tensor.to(device)

    # 🔁 Model forward pass for SHAP
    def model_forward(x): 
        if isinstance(x, np.ndarray): 
            x = torch.from_numpy(x).permute(0, 3, 1, 2).float().to(device)
        return model(x).logits

    # 🖼️ Convert tensor to image
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    if img_np.dtype != np.uint8:
        img_np = (img_np * 255).astype(np.uint8)
    img_np_batch = np.expand_dims(img_np, axis=0)

    # 🔍 SHAP explainer setup
    masker = shap.maskers.Image("inpaint_telea", img_np.shape)
    explainer = shap.Explainer(
        model_forward,
        masker,
        output_names=["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    )

    # 🔥 Get SHAP values
    shap_values = explainer(img_np_batch, max_evals=50)
    shap_img = shap_values[0].values

    # 🩻 Normalize SHAP shape to grayscale
    if shap_img.ndim == 4:
        # shape: (H, W, C, Classes) → reduce to (H, W)
        shap_mask_gray = shap_img.mean(axis=(2, 3))
    elif shap_img.ndim == 3:
        shap_mask_gray = shap_img.mean(axis=-1)
    else:
        shap_mask_gray = shap_img


    gray = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    _, eye_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(eye_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        eye_contour = max(contours, key=cv2.contourArea)
        eye_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(eye_mask, [eye_contour], -1, 255, thickness=cv2.FILLED)
        eye_mask = eye_mask.astype(np.float32) / 255.0
    else:
        eye_mask = np.ones_like(gray, dtype=np.float32)


    # 🖼️ Plot SHAP overlay image
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img_np)
   

    ax.set_title("SHAP Interpretation", fontsize=15)
    ax.axis('off')

    # 📦 Convert to base64
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode()
    plt.close()

    
    return image_base64








def generate_explanation(pred_class):
    explanation = f"Explanation for {pred_class}:\n\n"

    # SHAP general interpretation for any class
    explanation += (
                    "- The regions highlighted in purple have the highest contribution towards the model's prediction.\n"
                    "- The green areas also contribute but to a lesser extent.\n"
                    "- Other regions have minimal or no significant impact on the prediction.\n\n")

    if pred_class == "No DR":
        explanation += "No diabetic retinopathy detected.\nThe retina appears normal and healthy."

    elif pred_class == "Mild":
        explanation += "Early signs of diabetic retinopathy.\nThere may be the presence of small, localized microaneurysms."

    elif pred_class == "Moderate":
        explanation += "Moderate signs of vessel damage.\nThere may be increased blood leakage and signs of vascular stress."

    elif pred_class == "Severe":
        explanation += "Severe diabetic retinopathy detected.\nThere may be widespread retinal damage with significant thickening."

    elif pred_class == "Proliferative DR":
        explanation += "There may be uncontrolled new vessel growth posing a high risk of blindness."


    return explanation

from flask import jsonify

STATIC_FOLDER = 'static'
os.makedirs(STATIC_FOLDER, exist_ok=True)


@app.route('/predict', methods=['POST'])
def predict():
    print("🔍 Predict endpoint was hit!")
    # Inside your predict function, after generating the SHAP image
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Load image and preprocess
        image = Image.open(file).convert('RGB')
        img_tensor = preprocess_image(image)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            predicted_class = predicted.item()
            confidence_score = confidence.item() * 100

        # Generate SHAP image and explanation
        shap_img= generate_shap_explanation(img_tensor)
        explanation = generate_explanation(class_labels[predicted_class])

        a=f'static_{file.filename}'
        shap_image_path = os.path.join(STATIC_FOLDER, a)
        full_path = os.path.abspath(shap_image_path)
        print(f"SHAP image saved to: {full_path}")
        print("Full path:", full_path)
        heatmap_url = f'/static/{a}'
        
        with open(shap_image_path, "wb") as f:
            f.write(base64.b64decode(shap_img))
        
        result = {
        'diagnosis': class_labels[predicted_class],
        'confidence': round(confidence_score, 2),
        'heatmap_url': heatmap_url,
        'findings': explanation,
       
        
            }
        print(f"File exists: {os.path.exists(full_path)}")

        return jsonify(result)

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500





@app.route('/', methods=['GET', 'POST'])
def index():
    with open("home.html", "r", encoding="utf-8") as f:
        html_template = f.read()

    if request.method == 'POST':
        file = request.files['image']
        if file:
            image = Image.open(file).convert('RGB')
            img_tensor = preprocess_image(image)

            with torch.no_grad():
                outputs = model(img_tensor)
                _, predicted = torch.max(outputs.logits, 1)
                pred_class = class_labels[predicted.item()]

            shap_img = generate_shap_explanation(img_tensor)
            explanation = generate_explanation(pred_class)
        
            return render_template_string(html_template,
                                          prediction=pred_class,
                                          shap_image=shap_img,
                                          explanation=explanation)

    return render_template_string(html_template)




@app.route('/static/<path:filename>')
def get_shap_explanation(filename):
    
    shap_image_path = os.path.join(STATIC_FOLDER, f'static_{filename}.png')
    if os.path.exists(shap_image_path):
        return send_file(shap_image_path, mimetype='image/png')
    return jsonify({'detail': 'SHAP image not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)