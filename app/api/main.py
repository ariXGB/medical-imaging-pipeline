from fastapi import FastAPI

from api.routers.predict import router 


app = FastAPI(
    title="Chest X-Ray Diagnostic API",
    version="1.0.0"
)

app.include_router(
    router,
    prefix="/predict"
)