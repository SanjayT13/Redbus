import sys 
from pathlib import Path 
import datetime as dt

current_path = Path(__file__)
root_path  = current_path.parent.parent.parent
root_path1  = current_path.parent.parent
print("root path :",root_path1)
sys.path.append(str(root_path1))

import logging 
from logger import create_log_path, CustomLogger
from yaml import safe_load  

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import pickle
from lightgbm import LGBMRegressor

print('All imports done') 
pd.set_option('display.max_columns', None)

log_file_path = create_log_path('train model') 
model_train_logger = CustomLogger('Train_Model',log_file_path)
model_train_logger.set_log_level(level=logging.INFO) 

# DV_var = 'final_seatcount'

def create_df(train_data_path) :
    try :   
        Train_Data = pd.read_csv(train_data_path) 
        model_train_logger.save_logs(msg=f'Shape of data used for training model is {Train_Data.shape}',log_level='info')
        return Train_Data
    except Exception as e : 
        model_train_logger.save_logs(msg= f'Exception happened : {e} ',log_level='error') 
        raise 

def create_X_y(df) :  
    X = df.drop(['final_seatcount'],axis = 1) 
    y = df['final_seatcount'] 
    model_train_logger.save_logs(msg=f'X and y split completed',log_level='info')
    return X,y 

def train_model(X,y,model_name,hyperparam_dict) : 
    if model_name == 'xgb_regressor' : 
        model = XGBRegressor(objective = hyperparam_dict['objective'],  # Required for regression
                     n_estimators = hyperparam_dict['n_estimators'],
                     learning_rate = hyperparam_dict['learning_rate'],
                     max_depth = hyperparam_dict['max_depth'],
                     subsample = hyperparam_dict['subsample'],
                     colsample_bytree = hyperparam_dict['colsample_bytree'],
                     random_state = hyperparam_dict['random_state'])
        model = model.fit(X,y)
        model_train_logger.save_logs(msg=f'Model {model_name} is trained',log_level='info')
        return model

    elif model_name == 'LGBMRegressor' : 
        model = LGBMRegressor(n_estimators= hyperparam_dict['n_estimators'],
                              learning_rate = hyperparam_dict['learning_rate'],
                              max_depth=hyperparam_dict['max_depth'],
                              feature_fraction = hyperparam_dict['feature_fraction'], 
                              bagging_fraction = hyperparam_dict['bagging_fraction'],
                              min_child_samples = hyperparam_dict['min_child_samples'],
                              lambda_l1 = hyperparam_dict['lambda_l1'],
                              lambda_l2 = hyperparam_dict['lambda_l2'],
                              random_state=hyperparam_dict['random_state'])
        model = model.fit(X,y)
        model_train_logger.save_logs(msg=f'Model {model_name} is trained',log_level='info')
        return model
    else : 
        print('Model not found')
        model_train_logger.save_logs(msg=f'Model not found',log_level='error')
        raise

def model_save(model_obj,model_path) : 
    # path = "E:\DS\Analytics_vidhya\June 2025\my_saves\model_lightgbm_5.pkl" 
    try : 
        pickle.dump(model_obj,open(model_path, "wb")) 
        model_train_logger.save_logs(msg=f'Model savedd',log_level='info')

    except Exception as e : 
        model_train_logger.save_logs(msg=f'Exception : {e}',log_level='error') 
        raise 
def get_params(input_file,subsection_name) : 
    try:
        with open(input_file, 'r') as f:
            params_file = safe_load(f)
            # print("✅ YAML loaded:", params_file)  # <-- Debug here

        param_out = params_file['model_train'][subsection_name]
        model_train_logger.save_logs(msg=f'params read successfully for {subsection_name}',log_level='info')
        return param_out
    except Exception as e : 
        model_train_logger.save_logs(msg= f'Error : {e}',log_level='error')
        raise

def main() : 
    # read input dataframe to be trained on 
    train_data_path = sys.argv[1]
    Train_Data = create_df(train_data_path)
    # Split X, y 
    X,y = create_X_y(Train_Data) 
    
    # Train model 
    model_name = get_params(root_path / 'params.yaml','active_model')
    hyperparam_dict = get_params(root_path / 'params.yaml',model_name)
    model_trained = train_model(X,y,model_name,hyperparam_dict)
    # Save model   
    model_save_data_path = sys.argv[2]
    model_save(model_trained,model_save_data_path)
    print('Model Saved')

if __name__ == "__main__" : 
    main()