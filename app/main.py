"""FastAPI entrypoint."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.datasets import router as datasets_router
from app.api.robots import router as robots_router
from app.api.tasks import router as tasks_router
from app.api.telemetry import router as telemetry_router
from app.settings import settings
from app.schemas import HealthStatus

print(f"Nebius token factory base url: {settings.nebius_base_url}")
print(f"Nebius Nemotron model id: {settings.nebius_model}")
print(f"Nebius token factory base url: {settings.nebius_base_url}")

app = FastAPI(
    title=settings.app_name,
    version="2.0.0",
    description="Planner-in-the-loop simulation-first robot manipulation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allow_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(tasks_router, prefix="/api/v1")
app.include_router(datasets_router, prefix="/api/v1")
app.include_router(telemetry_router, prefix="/api/v1")
app.include_router(robots_router, prefix="/api/v1")


@app.get("/health", response_model=HealthStatus, tags=["system"])
def health():
    return HealthStatus(
        status="ok",
        project=settings.app_name,
        planner_configured=bool(settings.nebius_api_key and settings.nebius_model),
    )


BASE_DIR = Path(__file__).resolve().parent
_WEB_DIR = BASE_DIR / "web"
if _WEB_DIR.is_dir():
    app.mount(
        "/web",
        StaticFiles(directory=str(_WEB_DIR), html=True),
        name="web",
    )


@app.get("/web", include_in_schema=False)
def web_home():
    index = _WEB_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {"project": settings.app_name, "docs": "/docs", "health": "/health"}


@app.post("/simulation/reset", tags=["simulation"])
def reset_simulation():
    return {
        "status": "not_connected",
        "message": "No persistent simulation is attached. Each task creates and resets its own simulation session.",
    }
