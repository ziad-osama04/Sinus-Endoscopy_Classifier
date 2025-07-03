
# 🧠 Sinus Endoscopy Image Classifier – PyQt5 + ResNet18

An interactive **desktop application** for classifying sinus endoscopy images using a fine-tuned **ResNet18 deep learning model**. Built with PyQt5, this tool enables real-time image prediction, model training with data augmentation, and progress visualization—designed for medical imaging research and education. 🩺📊

---

## 📷 Project Overview

This project provides a hands-on platform to explore deep learning in medical diagnostics. With a clean GUI, users can:

🔹 Load and classify sinus endoscopy images
🔹 Train a CNN model with augmentation and real-time training logs
🔹 Save and load models for future use
🔹 Visualize predictions, confidence scores, and class labels

---

## 💻 Features

✅ **GUI-Based Image Classification** – Select and classify medical images from disk
✅ **Fine-Tuned CNN** – Uses transfer learning with pretrained ResNet18
✅ **Real-Time Training Progress** – Visual feedback during training (loss, accuracy, epochs)
✅ **Model Persistence** – Save and reload model checkpoints
✅ **Augmentation Pipeline** – Applies rotation, flip, normalization, etc., to enhance data
✅ **Fast & Lightweight** – Runs locally with minimal resource consumption

---

## 🧠 Deep Learning Details

* **Architecture**: ResNet18 (pretrained on ImageNet, fine-tuned for binary/multiclass sinus classification)
* **Loss Function**: CrossEntropyLoss
* **Optimizer**: Adam
* **Training Input**: Preprocessed endoscopic images (.jpg, .png)

---

## 🛠️ Tech Stack

| Technology   | Purpose                        |
| ------------ | ------------------------------ |
| PyTorch      | Model definition & training    |
| PyQt5        | GUI development                |
| torchvision  | Data augmentation & transforms |
| PIL / OpenCV | Image loading and handling     |
| matplotlib   | Training visualization         |

---

## 📁 Project Structure

```bash
├── main.py                  # PyQt5 GUI app
├── model.py                 # ResNet18 model definition
├── train.py                 # Training logic
├── predict.py               # Inference pipeline
├── utils/                   # Image processing, helpers
├── saved_models/            # Trained model checkpoints
└── dataset/                 # Training/testing images
```

---

## 🚀 How to Run

### 1️⃣ Install Dependencies

```bash
pip install torch torchvision pyqt5 matplotlib pillow opencv-python
```

### 2️⃣ Launch the App

```bash
python main.py
```

### 3️⃣ Use the GUI

* Load an image from your system
* Click “Classify” to see the predicted class
* Use the “Train” tab to start training on new data

---

## 🔬 Use Cases

This tool is ideal for:

🧪 **Medical AI Prototyping**
🏥 **Clinical Education**
🧠 **Image Classification Research**
🖼️ **Dataset Visualization & Testing**

---

## 🙌 Contributors

Special thanks to our highly talented team for their expertise across software and hardware development:  
**Ward Selkini, Hassan Badawy, Ziad Osama, Mostafa Hany.**  
This simulator exists thanks to your creativity, commitment, and technical excellence! 👏

---

## 🔗 Contact

📧 **Email**: [anas.bayoumi05@eng-st.cu.edu.eg](mailto:anas.bayoumi05@eng-st.cu.edu.eg)
🔗 **LinkedIn**: [Anas Mohamed](https://www.linkedin.com/in/anas-mohamed-716959313/)
