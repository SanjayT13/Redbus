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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,root_mean_squared_error
import pickle

print('All imports done') 
pd.set_option('display.max_columns', None)

log_file_path = create_log_path('pred model') 
model_pred_logger = CustomLogger('Pred_Model',log_file_path)
model_pred_logger.set_log_level(level=logging.INFO) 

DV_var = 'final_seatcount'

def create_df(pred_data_path) :
    try :   
        pred_Data = pd.read_csv(pred_data_path)  
        flag_eval = False
        if DV_var in pred_Data.columns : 
            flag_eval = True
            y_test = pd.DataFrame(pred_Data[DV_var],index = pred_Data.index)
            pred_Data = pred_Data.drop([DV_var],axis = 1) 
                            
        var_order_list = ['cumsum_searchcount_30','cumsum_seatcount_30','cumsum_searchcount_20','cumsum_seatcount_20','cumsum_searchcount_15','cumsum_seatcount_15',
         'cumsum_searchcount_30_25','cumsum_seatcount_30_25','cumsum_searchcount_25_20','cumsum_seatcount_25_20','cumsum_searchcount_20_14',
         'cumsum_seatcount_20_14','ewm_1_cumsum_searchcount','ewm_1_cumsum_seatcount','ewm_5_cumsum_searchcount','ewm_5_cumsum_seatcount',
         'ewm_9_cumsum_searchcount','ewm_9_cumsum_seatcount','destid_encoded','destid_region_encoded','srcid_encoded','srcid_region_encoded',
         'day_of_week','fri_to_sun','week_of_year','month','monnth_start_end','is_holiday_india','srcid_tier_Tier 1','srcid_tier_Tier 3',
         'srcid_tier_Tier 4','srcid_tier_Tier2','destid_tier_Tier 1','destid_tier_Tier 3','destid_tier_Tier 4','destid_tier_Tier2','day_of_year_sin',
         'day_of_year_cos','slope_lin','slope1_quad','slope2_quad']
        
        pred_Data = pred_Data[var_order_list]
        model_pred_logger.save_logs(msg=f'Data for prediction created with shape {pred_Data.shape}',log_level='info')
        if flag_eval : 
            return pred_Data,y_test 
        else :
            return pred_Data,None
    except Exception as e : 
        model_pred_logger.save_logs(msg= f'Exception happened : {e} ',log_level='error') 
        raise 

def get_preds(df_pred_Data,model_file_name) : 
    try : 
        model_loaded = pickle.load(open(model_file_name, "rb"))
        print(df_pred_Data.columns)
        df_pred = pd.DataFrame(model_loaded.predict(df_pred_Data),index=df_pred_Data.index)
        model_pred_logger.save_logs(msg=f'Prediction dataframe created with shape {df_pred.shape}',log_level='info')
        return df_pred
    except Exception as e : 
        model_pred_logger.save_logs(msg= f'Exception happened : {e} ',log_level='error')
        raise

def func_eval(df_pred,y_test) : 
    Mean_squared_error = mean_squared_error(y_test,df_pred)
    rmse = root_mean_squared_error(y_test,df_pred)
    mean_absolute_error_ = mean_absolute_error(y_test,df_pred)
    r2_score_ = r2_score(y_test,df_pred)
    
    print("Mean squared error : ",Mean_squared_error)
    print("rmse : ",rmse)
    print("mean_absolute_error : ",mean_absolute_error_ )
    print(" r2_score : ",r2_score_) 
    metrics_df = pd.DataFrame({"Column" : ['Mean_squared_error','rmse','mean_absolute_error','r2_score'],
                  "Value" : [Mean_squared_error,rmse,mean_absolute_error_,r2_score_]})
    model_pred_logger.save_logs(msg=f'Mean squared error : {Mean_squared_error}',log_level='info')
    model_pred_logger.save_logs(msg=f'rmse : {rmse}',log_level='info')
    model_pred_logger.save_logs(msg=f'mean_absolute_error : {mean_absolute_error_}',log_level='info')
    model_pred_logger.save_logs(msg=f'r2_score : {r2_score_}',log_level='info')
    return metrics_df
    

def main() : 
    # read dataframe 
    pred_data_path = sys.argv[1]
    df_pred_Data,y_test = create_df(pred_data_path) # get X to be predicted 
    # read model 
    model_file_name = sys.argv[2]
    # predict result 
    df_pred = get_preds(df_pred_Data,model_file_name) 
    file_path_output = sys.argv[3] 
    df_pred.to_csv(file_path_output, index=False)
    model_pred_logger.save_logs(msg = f'File written to path {file_path_output}',log_level='info')
    # evaluate 
    if y_test is not None : 
        if sys.argv[4] == 'train' :  
            run_date = str(dt.date.today())
            file_path_output_eval = f'redbus/data/processed/train_{run_date}.csv' 
        elif sys.argv[4] == 'val' : 
            run_date = str(dt.date.today())
            file_path_output_eval = f'redbus/data/processed/val_{run_date}.csv'
        metrics_df = func_eval(df_pred,y_test) 
        metrics_df.to_csv(file_path_output_eval, index=False)
        model_pred_logger.save_logs(msg = f'Evaluation File written to path {file_path_output_eval}',log_level='info')
        print('Evaluation done') 

if __name__ == '__main__' : 
    main()