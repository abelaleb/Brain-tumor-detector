from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from torchvision import transforms
import io
import os
import gdown
from dotenv import load_dotenv

# Import your model classes
from app.models.brain_tumor_model import BrainTumorModel
from app.models.stroke_model import StrokeModel
from app.models.parkinson_model import ParkinsonModel
from app.models.hemorrhagic_model import HemorrhagicModel

load_dotenv()

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Loading ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_class, model_path_env_var):
    MODEL_PATH = os.getenv(model_path_env_var)
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading {model_path_env_var} model...")
        gdown.download(os.getenv(f"{model_path_env_var}_URL"), MODEL_PATH, quiet=False)
    
    model = model_class()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

models = {
    "brain-tumor": load_model(BrainTumorModel, "BRAIN_TUMOR_MODEL_PATH"),
    "stroke": load_model(StrokeModel, "STROKE_MODEL_PATH"),
    "parkinson": load_model(ParkinsonModel, "PARKINSON_MODEL_PATH"),
    "hemorrhagic": load_model(HemorrhagicModel, "HEMORRHAGIC_MODEL_PATH"),
}

class_names = {
    "brain-tumor": ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'],
    "stroke": ['Bleeding', 'Ischemia', 'Normal'],
    "parkinson": ["Parkinson's", "Normal"],
    "hemorrhagic": ["Hemorrhagic", "Normal"],
}

# Image Transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

@app.get("/")
def root():
    return {"message": "NeuroScopeAI Diagnostics API is running"}

@app.post("/predict/{model_type}")
async def predict(model_type: str, file: UploadFile = File(...)):
    if model_type not in models:
        return JSONResponse(status_code=404, content={"message": "Model not found"})

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        model = models[model_type]
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        prediction = class_names[model_type][predicted_idx.item()]
        
    return JSONResponse(content={
        "prediction": prediction,
        "confidence": round(confidence.item() * 100, 2)
    })