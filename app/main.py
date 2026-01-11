from fastapi import FastAPI 
from .api_routes import router as api_router 
from .logging_config import setup_logging

# Activate logging once at app startup
setup_logging()

app = FastAPI(title="RedBus traveller count Prediction API",version="1.0.0")
app.include_router(api_router,prefix="/api")

