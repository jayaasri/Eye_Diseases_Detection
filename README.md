# Eye_Diseases_Detection
# 🧠 Diagnosis of Eye Diseases using Deep Learning and Explainable AI

This project aims to diagnose three major eye diseases — **Diabetic Retinopathy**, **Cataracts**, and **Glaucoma** — using Deep Learning models combined with Explainable AI (XAI) techniques.

> 👩‍💻 Developed by Jayaa Sri , Prasheeba and Mahalakshmi

---

## 🩺 Problem Statement

Blindness caused by Diabetic Retinopathy, Cataracts, and Glaucoma accounts for over **90% of irreversible vision loss** globally. Early detection plays a vital role in preserving vision.

We propose a solution using **Deep Learning** for accurate diagnosis from retinal images and **Explainable AI** to bring transparency into predictions — a critical need in the medical field.

---

## 🎯 Objectives

- Accurately detect and classify stages of Diabetic Retinopathy, Cataracts, and Glaucoma.
- Integrate Explainable AI for visualizing and understanding predictions.
- Build trust and transparency for clinical decision-making using XAI.

---


## 🛠️ Methodologies

### 1. Diabetic Retinopathy (Jayaa Sri)

- **Dataset:** 5,000 images from APTOS 2019, EyePACS, and Messidor-2.
- **Model:** Vision Transformer (ViT)
- **Classes:** No_DR, Mild, Moderate, Severe, Proliferative
- **Input Size:** 224×224 pixels
- **Key Steps:**
  - Patch splitting, embedding, positional encoding
  - 12 Transformer encoder blocks
  - Final classification head

### 2. Cataract (Prasheeba)

[Add Cataract methodology summary here.]

### 3. Glaucoma (Maha)

- **Dataset:** EYEPACS AIROGS (2,500 images)
- **Model:** Vision Transformer
- **Classes:** Glaucoma, Normal
- **Split:** 80% training, 20% testing
- **Preprocessing:** Resize to 224×224, normalization

---

## 📊 Results

### ✅ Accuracy Comparison

| Disease             | Our Model | Previous Best |
|---------------------|-----------|----------------|
| Diabetic Retinopathy | 92.57%    | 82.5%          |
| Cataract             | 99.86%    | 95.00%         |
| Glaucoma             | 98.00%    | 77.97%         |

### 📈 Classification Reports

- **DR Model:** No DR class – Precision: 0.966, Recall: 0.995, F1: 0.98+
- **Cataract Model:** Accuracy: 99.86%
- **Glaucoma Model:** Precision: 98%, Recall: 96%, F1-Score: 97%

---

## 📌 Explainability with XAI

- Integrated **Grad-CAM** to visualize key regions influencing predictions.
- Helps clinicians understand and trust the decision-making process.

---

## 📸 Screenshots

[Insert screenshots of the model outputs, heatmaps, and Grad-CAM results.]

---

## 🧾 Conclusion

Our system demonstrates the power of combining **Vision Transformers** with **Explainable AI** for real-world medical diagnostics. It provides:
- High diagnostic accuracy
- Transparent predictions
- Clinician-friendly visualization

---

## 📂 Project Structure


