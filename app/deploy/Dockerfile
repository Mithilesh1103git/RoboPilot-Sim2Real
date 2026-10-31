FROM python:3.11-slim
LABEL authors="Mithilesh"

ENV APP_NAME=RoboPilot-Sim2Real
ENV NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1/
ENV NEBIUS_MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B
ENV NEBIUS_TIMEOUT_SECONDS=45
ENV NEBIUS_TEMPERATURE=0.1
ENV TAVILY_ENABLED=false
ENV TAVILY_TIMEOUT_SECONDS=15
ENV TAVILY_MAX_RESULTS=5
ENV PLANNER_CONTEXT_CACHE_TTL_SECONDS=86400
ENV ROBOPILOT_ASSETS_DIR=assets/robots
ENV ROBOPILOT_DATA_DIR=data/adapter
ENV ROBOT_NAME=panda
ENV SIMULATION_HEADLESS=true
ENV SIMULATION_REALTIME=false
ENV MAX_TASK_RETRIES=2
ENV TASK_WORKERS=1
ENV CORS_ALLOW_ORIGINS=http://localhost:8000,http://127.0.0.1:8000

WORKDIR /app

RUN python -m venv appvenv

ENV PATH="/app/appvenv/bin:$PATH"

ENV PYTHONPATH=":/"

ENV MUJOCO_PATH=/root/.mujoco

COPY requirements-simapp.txt /opt/requirements-simapp.txt

RUN pip install --no-cache-dir -r /opt/requirements-simapp.txt

RUN rm /opt/requirements-simapp.txt

RUN groupadd appgroup

RUN useradd -r -g appgroup appuser

USER appuser

COPY . .

EXPOSE 8080

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8080", "--app-dir", "/app/", "main:app"]