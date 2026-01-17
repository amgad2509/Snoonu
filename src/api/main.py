from pathlib import Path
import sys

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

load_dotenv()

from api.routes.extract import router as extract_router
from api.routes.livekit import router as livekit_router

app = FastAPI()

app.include_router(extract_router, prefix="/api")
app.include_router(livekit_router, prefix="/api")

app.mount("/", StaticFiles(directory=str(ROOT / "dashboard"), html=True), name="dashboard")
