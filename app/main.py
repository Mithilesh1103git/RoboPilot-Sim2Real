"""FastAPI entry point for RoboPilot's starter scaffold."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.tasks import router as tasks_router
from app.api.telemetry import router as telemetry_router

app = FastAPI(title="RoboPilot", version="0.1.0", description="Simulation-first VLA manipulation agent scaffold")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tasks_router, prefix="/api")
app.include_router(telemetry_router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok", "project": "RoboPilot"}

app.mount("/web", StaticFiles(directory="web", html=True), name="web")
