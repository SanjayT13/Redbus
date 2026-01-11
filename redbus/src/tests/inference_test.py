import sys
from pathlib import Path
current_path = Path(__file__)
root_path1  = current_path.parent.parent
root_path2 = current_path.parent.parent.parent
print("root_path2 : ",root_path2)
sys.path.append(str(root_path2))
from src.inference.predict import Predictor 
pred = Predictor()  
data_sample = {"route_key": ["2025-02-11_46_45","2025-01-20_17_23"],
               "doj": ["2025-02-11","2025-01-20"],
               "srcid": [46,17],
               "destid": [45,23]}
result = pred.predict(data_sample)
print(result)