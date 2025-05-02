# 🧠 Brain Tumor Classification API

This is the **backend API** built with **FastAPI + PyTorch** that classifies brain MRI scans into four categories using a deep learning model.

## 🧬 Prediction Classes

- Glioma
- Meningioma
- No Tumor
- Pituitary

## 🧠 Model

The model is a custom `CombinedModel` using pretrained CNNs. It is automatically downloaded from Google Drive (if not found locally) at runtime.

## 🛠️ Requirements

- Python 3.9+
- pip
- torch
- torchvision
- Pillow
- fastapi
- uvicorn
- gdown

### Install dependencies

```bash
pip install -r requirements.txt
requirements.txt
txt
Copy
Edit
fastapi
uvicorn
torch
torchvision
pillow
gdown
```

📁 Project Structure
```bash
backend/
│
├── app/
│   ├── main.py                # FastAPI server logic
│   └── model/
│       └── combined_model.py  # PyTorch model class
├── model/
│   └── model.pth              # Will be downloaded if not found
└── .env                       # Contains MODEL_URL
```

🧪 Running the Server Locally
```bash
uvicorn app.main:app --reload
```

The server will run on http://localhost:8000

