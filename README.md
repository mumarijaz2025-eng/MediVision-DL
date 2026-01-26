# MediVision-DL: AI-Assisted Skin Lesion Analysis

## Project Overview
MediVision-DL is a deep learning prototype designed for the classification of skin lesions. Leveraging state-of-the-art computer vision techniques, this project aims to provide a preliminary analysis of skin conditions to assist in dermatological screening.

The system is built using **PyTorch** and utilizes a fine-tuned **MobileNetV3** architecture, chosen for its balance between high accuracy and computational efficiency, making it suitable for potential deployment on mobile or edge devices.

## Key Features
- **Multi-Class Classification:** Identifies 7 distinct types of skin lesions based on the HAM10000 dataset schema.
- **Efficient Architecture:** Uses MobileNetV3 Large for optimized performance.
- **Interactive Interface:** Features a user-friendly web interface built with Gradio for real-time image uploads and predictions.
- **Transfer Learning:** Employs pre-trained weights from ImageNet to achieve robust feature extraction.

## Supported Lesion Types
The model is trained to recognize the following categories:
1. Actinic keratoses
2. Basal cell carcinoma
3. Benign keratosis
4. Dermatofibroma
5. Melanoma
6. Nevus
7. Vascular lesions

## Technical Stack
| Component | Technology |
| :--- | :--- |
| **Framework** | PyTorch |
| **Model Architecture** | MobileNetV3 Large |
| **UI Framework** | Gradio |
| **Image Processing** | Torchvision, PIL |
| **Language** | Python 3.x |

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- Torchvision
- Gradio
- Pillow

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/mumarijaz2025-eng/MediVision-DL.git
   cd MediVision-DL
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
Launch the interactive interface by running:
```bash
python app.py
```
Once started, the Gradio interface will be available at `http://127.0.0.1:7860`.

## Future Roadmap
- [ ] Integration of a larger dataset for improved accuracy.
- [ ] Deployment as a mobile application.
- [ ] Implementation of Explainable AI (XAI) features like Grad-CAM to highlight areas of concern.

## Disclaimer
*This project is for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider with any questions regarding a medical condition.*
