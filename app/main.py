"""FastAPI application entry point for RoboPilot-Sim2Real.

Configuration is read from the process environment. This module deliberately
leaves `.env` loading to the caller so local secrets/config files are not read
implicitly and planner credentials are never required for this baseline.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.datasets import router as datasets_router
from app.api.tasks import environment
from app.api.tasks import router as tasks_router
from app.api.telemetry import router as telemetry_router
from app.config import settings

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Simulation-first robotics manipulation baseline with deterministic Panda control.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allow_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(tasks_router, prefix="/api/v1")
app.include_router(telemetry_router, prefix="/api/v1/telemetry")
app.include_router(datasets_router, prefix="/api/v1")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "project": settings.app_name,
        "simulation_connected": environment.model is not None
        and environment.data is not None,
        "robot": settings.robot_name,
    }


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/web/", status_code=307)


web_dir = Path(__file__).resolve().parents[1] / "web"
if web_dir.is_dir():
    app.mount("/web", StaticFiles(directory=str(web_dir), html=True), name="web")
