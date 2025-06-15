# 🧠 Neurological Disease Classification API

This is the **backend API** built with **FastAPI + PyTorch** that classifies various brain conditions using MRI scans and deep learning models. It supports prediction for brain tumors, stroke, Parkinson’s disease, and hemorrhagic conditions.

---

# 🧬 Supported Disease Categories

### 1. Brain Tumor (4 classes)
- Glioma
- Meningioma
- Pituitary
- No Tumor

### 2. Stroke (3 classes)
- Ischemia  
- Bleeding  
- Normal  

### 3. Hemorrhagic (2 classes)
- Hemorrhagic  
- Normal  

### 4. Parkinson’s Disease (2 classes)
- Parkinson’s  
- Normal  

---

## 🚀 Features

- Multi-disease classification using pretrained CNNs
- Automatic model weight loading (via Google Drive if not local)
- REST API endpoints for each model
- JWT-based user authentication with roles (user/admin)
- Feedback submission with optional image upload
- Admin-only endpoints to manage users and feedback
- PostgreSQL/Supabase integration

---

## 🛠️ Requirements

- Python 3.9+
- pip
- torch
- torchvision
- pillow
- fastapi
- uvicorn
- gdown
- sqlalchemy
- asyncpg
- python-multipart
- python-jose[cryptography]
- passlib[bcrypt]

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Model

The model is a custom `CombinedModel` using pretrained CNNs. It is automatically downloaded from Google Drive (if not found locally) at runtime.

---

📁 Project Structure
```bash
backend/
│
├── app/
│   ├── main.py
│   └── models/
│       ├── __init__.py
│       ├── brain_tumor_model.py
│       ├── stroke_model.py
│       ├── parkinson_model.py
│       └── hemorrhagic_model.py
├── model_weights/
│   ├── brain_tumor.pth
│   ├── stroke.pth
│   ├── parkinson.pth
│   └── hemorrhagic.pth
└── .env
```

---

## 🔌 API Endpoints
### 🔍 Prediction
- POST /predict/brain-tumor
- POST /predict/stroke
- POST /predict/hemorrhagic
- POST /predict/parkinson

### 🔐 Authentication
- POST /auth/register – User registration
- POST /auth/login – User login (returns JWT)

### 📝 Feedback System
- POST /feedback – Submit feedback (auth required)
- GET /feedback – View all feedback (admin only)

### 🛠 Admin Controls
- GET /admin/users – List all users (admin only)
- POST /admin/set-role – Assign/revoke admin role

---
---

## 🗃️ Database Schema (PostgreSQL / Supabase Compatible)

### `users` Table

| Column           | Type    | Description                        |
|------------------|---------|------------------------------------|
| `id`             | UUID    | Primary key                        |
| `email`          | TEXT    | User's email address               |
| `hashed_password`| TEXT    | Encrypted user password            |
| `role`           | TEXT    | Role assigned (`user`, `admin`)   |
| `created_at`     | TIMESTAMP | Auto-generated on creation       |

---

### `feedback` Table

| Column           | Type     | Description                             |
|------------------|----------|-----------------------------------------|
| `id`             | UUID     | Primary key                             |
| `user_id`        | UUID     | Foreign key referencing `users.id`      |
| `model_type`     | TEXT     | e.g., `brain_tumor`, `stroke`, etc.     |
| `predicted_class`| TEXT     | Model's prediction                      |
| `actual_class`   | TEXT     | User-provided correction (optional)     |
| `comments`       | TEXT     | Optional text feedback                  |
| `image_url`      | TEXT     | Optional re-uploaded image link         |
| `created_at`     | TIMESTAMP| Auto-generated timestamp                |

---

## 🔐 Environment Variables

Add a `.env` file in the backend root with the following keys:

```env
DATABASE_URL=postgresql+asyncpg://username:password@host:port/db_name
JWT_SECRET_KEY=your_super_secret_key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

---
🧪 Running the Server Locally
```bash
uvicorn app.main:app --reload
```

The server will run on http://localhost:8000

