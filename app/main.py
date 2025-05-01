#main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import transforms
import io
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.combined_model import CombinedModel


app = FastAPI()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = CombinedModel()
model.load_state_dict(torch.load("model/model.pth", map_location=device))
model.to(device)
model.eval()

# Class names
condition_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Image transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

@app.get("/")
def root():
    return {"message": "Brain Tumor Classification API is running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        prediction = condition_names[predicted.item()]

    return JSONResponse(content={"prediction": prediction})
