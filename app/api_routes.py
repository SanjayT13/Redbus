from pathlib import Path 
import sys
from fastapi import APIRouter,HTTPException
import logging

current_path = Path(__file__) 
root_path1 = current_path.parent.parent
print("root_path1 : ",root_path1)
sys.path.append(str(root_path1))
from app.schemas import input_schema 
from redbus.src.inference.predict import Predictor 


router = APIRouter() 
logger = logging.getLogger(__name__)
predictor = Predictor() 

@router.post("/predict") 
def predict_endpoint(request_data: input_schema):
    """
    Endpoint to get predictions based on input data.
    """
    logger.info("Received /predict request.")
    try: 
        request_dict = request_data.model_dump() 
        
        result = predictor.predict(request_dict)

        # for k,v in result.items(): 
        #     result[k] = v.tolist()
        return {"status": "success", "predictions": result} 
    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e)) 
    
@router.get("/health")
def health_check():
    """
    Health check endpoint.
    """
    try : 
        if predictor.model is None : 
            raise RuntimeError("Model not loaded properly.") 
        return {"status": "healthy", "model_loaded": True} 
    except Exception as e : 
        logger.error(f"Health check failed: {e}") 
        return {"status": "unhealthy", "model_loaded": False, "error": str(e)}

@router.get("/version")
def version_info():
    """
    Endpoint to get version information.
    """
    return {
        "api_version": "1.0.0",
        "model_version": predictor.model_version
    }
