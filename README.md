# 🚦 Traffic Sign Detection & Recognition using ResNet34

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet34-red?logo=pytorch)
![Gradio](https://img.shields.io/badge/GUI-Gradio-orange?logo=gradio)
![License](https://img.shields.io/badge/License-MIT-green)

A deep learning project that **detects and recognizes traffic signs** from images using **ResNet34 (PyTorch)**.  
The model is trained on the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset and achieves **~98% test accuracy**.  
An **interactive web GUI** is provided with **Gradio** for quick testing and visualization.

---

## ✨ Features
- **Deep Learning Model:** ResNet34 with transfer learning + fine-tuning  
- **Dataset:** [GTSRB](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) (43 classes, 50,000+ images)  
- **Data Processing:** preprocessing, normalization, and augmentation for robustness  
- **Evaluation:** classification report + confusion matrix  
- **Interactive Demo:** Gradio GUI for image upload & prediction  
- **High Accuracy:** ~98% on test set  

---

## 🛠️ Tech Stack
- **Programming Language:** Python 3.8+  
- **Deep Learning:** PyTorch, Torchvision  
- **Data Handling:** NumPy, Pandas  
- **Visualization:** Matplotlib, Seaborn  
- **Image Processing:** OpenCV, Pillow  
- **Web GUI:** Gradio  
- **Environment:** Jupyter Notebook / Google Colab  

---

## 📊 Dataset: GTSRB
- **Name:** German Traffic Sign Recognition Benchmark  
- **Classes:** 43 categories  
- **Size:** 50,000+ images with variations in lighting, rotation, scale, and occlusion  
- **Preprocessing Applied:**  
  - Resize & normalization  
  - Augmentation: rotation, zoom, shift, flips  
  - Train/validation/test split  

📌 Official link: [GTSRB Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)  

---

## 🧠 Model & Training
- **Base Model:** ResNet34 (pretrained on ImageNet)  
- **Transfer Learning:** frozen base + fine-tuned layers  
- **Optimizer:** Adam  
- **Loss Function:** CrossEntropyLoss  
- **Training Strategy:** early stopping, learning rate scheduling  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score  

✅ **Final Test Accuracy:** ~98%  

---

## 📈 Results
- High performance across all 43 classes  
- Minimal misclassification between visually similar signs  
- Generated **confusion matrix** & **classification report** for analysis  

---

## 📂 Project Structure
