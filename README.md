# Remote Sensing Satellite 🛰️

## Overview
The **Remote Sensing Satellite Project** focuses on improving classification of remote sensing images using **advanced convolutional neural network (CNN) architectures combined with attention mechanisms**. This study integrates attention modules to enhance feature extraction, improving accuracy and efficiency in satellite image classification.

## Tech Stack 🛠
### **Deep Learning & AI**
- Python
- TensorFlow / PyTorch
- OpenCV
- Scikit-learn

### **Data Processing & Visualization**
- NumPy & Pandas
- Matplotlib & Seaborn
- GDAL (Geospatial Data Abstraction Library)

## Dataset 📂
- **Source**: Aerial Image Dataset (AID) – 30 scene classes (airports, forests, urban areas, etc.)
- **Formats**: JPEG, PNG
- **Preprocessing**: Data augmentation, normalization, resizing (600x600 → 224x224 pixels)

## Installation 🚀
### **Prerequisites**
- Python 3.8+
- Virtual Environment (`venv` or `conda`)
- GPU (Optional for faster training)

### **Setup Instructions**
1. **Clone the repository**
   ```bash
   git clone https://github.com/Jothika1526/Remote_Sensing_Satellite.git
   cd Remote_Sensing_Satellite
   ```

2. **Create a virtual environment & install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the available Python scripts**
   ```bash
   python cbam_mobilenetv2_code.ipynb
   python densenet_attention_mech.ipynb
   python ResNet50.ipynb
   ```

## How It Works 🎯
1. **Data Preprocessing**
   - Images resized to **224x224 pixels**.
   - Applied data augmentation: rotation, shifting, shearing, flipping.
   - Standardized pixel values for numerical stability.

2. **Model Architectures**
   - **MobileNetV2 + CBAM**: Uses depthwise separable convolutions + attention mechanism.
   - **DenseNet-121 + SEBlock**: Improves feature reuse and enhances channel importance.
   - **ResNet50 + SEBlock**: Incorporates skip connections for better gradient flow.

3. **Training & Optimization**
   - Adam optimizer with **learning rate 0.0001**.
   - **Batch size 32**, early stopping, learning rate reduction.
   - Models trained for **30-40 epochs**.

4. **Evaluation Metrics**
   - **Accuracy, Precision, Recall, F1-score, Mean Average Precision**.
   - **Best performance**:
     - **MobileNetV2 + CBAM**: **95.39% test accuracy**.
     - **ResNet50 + SEBlock**: **93.69% test accuracy**.
     - **DenseNet-121 + SEBlock**: **89.52% test accuracy**.

## Future Enhancements 🔮
- **🌍 Multi-Spectral Analysis** – Integrate spectral band processing.
- **🛰️ Automated Change Detection** – Detect land-use changes over time.
- **⚡ GIS Integration** – Connect with **Google Earth Engine** for real-time insights.

## License 📜
This project is **open-source** under the [MIT License](LICENSE).

---
> *Advancing satellite image classification with AI-powered attention mechanisms!* 🚀

