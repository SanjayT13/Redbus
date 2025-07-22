from pathlib import Path  
print("filepath is :",Path(__file__))
print("filepath parent :",Path(__file__).parent)
print("filepath parent parent is :",Path(__file__).parent.parent)
from src.logger import create_log_path, CustomLogger 
