# Vision Backend

FastAPI backend for the Vision Diabetic Retinopathy Detection System.

## Setup

```bash
cd Backend
pip install -r requirements.txt
```

## Run locally

```bash
uvicorn main:app --reload --port 8000
```

## Run on Google Colab

```python
!pip install -r requirements.txt
!nohup uvicorn main:app --host 0.0.0.0 --port 8000 &

# Then expose with ngrok:
!pip install pyngrok
from pyngrok import ngrok
public_url = ngrok.connect(8000)
print("Backend URL:", public_url)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/api/predict/` | Trigger prediction for a scan UUID |
| GET | `/api/predict/status/{uuid}` | Check prediction status |
| GET | `/api/scans/{uuid}` | Get single scan result |
| GET | `/api/scans/user/{user_id}` | Get all scans for a user |

## Environment Variables

Set in root `.env`:
```
SUPABASE_DATABASE_PASSWORD=your_password
NEXT_PUBLIC_SUPABASE_URL=https://tjinvmtramxjijxbtnld.supabase.co
NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY=sb_publishable_...
```
