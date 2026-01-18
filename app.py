import torch
from torchvision import transforms
from PIL import Image
import gradio as gr
from model import get_skin_model

# Note: Using these specific labels from the HAM10000 dataset schema
LABELS = ["Actinic keratoses", "Basal cell carcinoma", "Benign keratosis", 
          "Dermatofibroma", "Melanoma", "Nevus", "Vascular lesions"]

# Initializing the model. 
# I went with MobileNetV3 because I wanted something that could 
# potentially run on a mobile device later (Track 4 interest).
print("[System] Loading weights... this might take a second.")
model = get_skin_model()
model.eval()

# Transformation pipeline
# Decided to stick with standard ImageNet normalization for now
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(img):
    # Had to convert to PIL because Gradio sometimes sends numpy arrays
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img.astype('uint8'), 'RGB')
    
    img_tensor = preprocess(img).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.nn.functional.softmax(logits[0], dim=0)
        
        # Creating a dictionary for Gradio's Label component
        results = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
        return results

# Simple UI setup
demo = gr.Interface(
    fn=predict, 
    inputs=gr.Image(label="Upload Skin Image"), 
    outputs=gr.Label(num_top_classes=3),
    title="MediVision: Skin Lesion Analysis (Beta)",
    description="A prototype for AI-assisted dermatology. Built as part of my deep learning portfolio."
)

if __name__ == "__main__":
    print("[Success] Interface is live. Copy the link below to your browser.")
    demo.launch()