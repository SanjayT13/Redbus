import logging
import sys 
import os 

def setup_logging():
    """
    Central logging configuration.
    Logs go to stdout so Docker / AWS can capture them.
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper() 
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Reduce noisy logs from libraries
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING) 
    