from fastapi import FastAPI
from app.routers.health import router as health_router


app = FastAPI(
    title="colorstrip-backend",
    description="FastAPI backend for colorimetric strip analysis.",
    version="0.1.0",
)

# Routers
app.include_router(health_router)