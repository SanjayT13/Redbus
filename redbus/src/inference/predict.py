import logging
import pandas as pd
import numpy as np
import sys 
from pathlib import Path 
import datetime as dt 
from yaml import safe_load 
import holidays
import pickle
current_path = Path(__file__)
root_path  = current_path.parent.parent.parent
logger = logging.getLogger(__name__)

# read input function 
def read_input(data : dict):
    df_input = pd.DataFrame(data) 
    return df_input 

def get_params(input_file,subsection_name) : 
    try:
        with open(input_file, 'r') as f:
            params_file = safe_load(f)
        param_out = params_file['inference'][subsection_name]
        return param_out
    except Exception as e : 
        raise    

def create_14_day_data(df_train,df_trans) : 
    df_train_merged = pd.merge(df_train,df_trans[df_trans['dbd'] > 14],on = ['doj','srcid','destid'],how = 'left')
    # print('df_trans 14 days shape : ', print(df_trans[df_trans['dbd'] > 14].shape))
    return df_train_merged 

def lead_var_curr(dbd_days,df_trans,df_train,var_lead) : 
    df_temp = df_trans[df_trans['dbd'] == dbd_days] 
    df_train = pd.merge(df_train,df_temp[['doj','srcid','destid',var_lead]],on = ['doj','srcid','destid'],how = 'inner')
    var_new = var_lead + '_' + str(dbd_days)
    df_train.rename(columns = {var_lead : var_new},inplace=True)
    return df_train 

def avg_func(dbd_days_start,dbd_days_end,df_trans,df_train,var_lead) : 
    df_temp = df_trans[df_trans['dbd'].isin(list(range(dbd_days_start,dbd_days_end,-1)))].groupby(['doj','srcid','destid'])[var_lead].mean().reset_index()
    df_train = pd.merge(df_train,df_temp[['doj','srcid','destid',var_lead]],on = ['doj','srcid','destid'],how = 'inner')
    var_new = var_lead + '_' + str(dbd_days_start) +'_' + str(dbd_days_end)
    df_train.rename(columns = {var_lead : var_new},inplace=True)
    return df_train

def ewma_var_curr(alpha_val,df_trans,df_train,var) : 
    # order dataframe
    df_trans_cp_temp = df_trans.copy()
    df_trans_cp_temp = df_trans_cp_temp.sort_values(by = ['doj','srcid','destid','dbd'],ascending = [True,True,True,False])
    df_trans_cp_temp.reset_index(drop = True)
    
    # calculate ewm
    var_new = 'ewm_' + str(alpha_val)[2:] + '_' + var
    df_trans_cp_temp[var_new] = df_trans_cp_temp.groupby(['doj','srcid','destid'])[var].transform(lambda x : x.ewm(alpha = alpha_val).mean())
    
    # join with df_train
    df_train = pd.merge(df_train,df_trans_cp_temp[df_trans_cp_temp['dbd'] == 15][['doj','srcid','destid',var_new]],on = ['doj','srcid','destid'],how = 'inner')
    
    return df_train


def func_date_time_cols(df,var_name) : 
    df[var_name] = pd.to_datetime(df[var_name])
    in_holidays = holidays.CountryHoliday('IN')
    # day of week 
    df['day_of_week'] = df[var_name].dt.dayofweek
    # for this case, friday, saturday and Sunday are con
    df['fri_to_sun'] = np.where(df['day_of_week'] >=4,1,0)
    # weekofyear 
    df['week_of_year'] = df[var_name].dt.isocalendar().week
    # Month
    df['month'] = df[var_name].dt.month
    # month end or month start
    df['monnth_start_end'] = df[var_name].dt.is_month_start.astype(int) + df[var_name].dt.is_month_end.astype(int)
    # holiday flag 
    df['is_holiday_india'] = df[var_name].apply(lambda x : x in in_holidays)
    df[var_name] = df[var_name].astype(str)
    return df

# Ohe --> srcid_tier,destid_tier,is_holiday_india
# target encoding --> srcid_region,destid_region,srcid,destid
def target_encoding(df,df_train,encode_path,var_name) : 
    df_temp = df.drop_duplicates(subset = ['doj','srcid','destid'])
    temp_vals = pd.read_csv(encode_path,index_col = var_name).squeeze() 
    # temp_vals =df_temp.groupby(var_name)['final_seatcount'].mean()
    df_temp.loc[:,var_name + '_encoded'] = df_temp[var_name].map(temp_vals)
    df_train = pd.merge(df_train,df_temp[['doj','srcid','destid',var_name + '_encoded']],on = ['doj','srcid','destid'],how = 'inner')
    return df_train,temp_vals

def fourier_dayofyear(df) :
    df['doj'] = pd.to_datetime(df['doj']) 
    df['day_of_year'] = df['doj'].dt.dayofyear
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df = df.drop(['day_of_year'],axis = 1)
    df['doj'] = df['doj'].astype(str)
    return df

def fit_poly(grp) : 
    # order 1 
    x = list(range(16))
    y = grp['cumsum_seatcount']
    return pd.Series(np.polyfit(x,y,1),index = ['slope_lin','coefficient_lin'])

def fit_poly2(grp) : 
    # order 1 
    x = list(range(16))
    y = grp['cumsum_seatcount']
    return pd.Series(np.polyfit(x,y,2),index = ['slope1_quad','slope2_quad','coefficient_quad'])

def preprocess_data(data): 
    df_input = read_input(data) 
    # load transaction data path from params.yaml 
    transaction_data_path = get_params(root_path / 'params.yaml', 'trans_data_path')
    df_trans = pd.read_csv(transaction_data_path) 

    print('df_trans shape :', df_trans.shape)
    df_inp_merged = create_14_day_data(df_input,df_trans) 
    print('df_inp_merged shape :', df_inp_merged.shape)
    # feature engineering
    lead_days_lst = get_params(root_path / 'params.yaml','lead_days') 
    for i in lead_days_lst : 
        df_input = lead_var_curr(i,df_inp_merged,df_input,'cumsum_searchcount')
        df_input = lead_var_curr(i,df_inp_merged,df_input,'cumsum_seatcount') 

    avg_func_list = get_params(root_path / 'params.yaml','avg_func_list') 
    for j in avg_func_list : 
        df_input = avg_func(j[0],j[1],df_inp_merged,df_input,'cumsum_searchcount')
        df_input = avg_func(j[0],j[1],df_inp_merged,df_input,'cumsum_seatcount')    
    ewma_alpha_list = get_params(root_path / 'params.yaml','ewma_alpha_list') 
    for k in ewma_alpha_list :  
        df_input = ewma_var_curr(k,df_inp_merged,df_input,'cumsum_searchcount')
        df_input = ewma_var_curr(k,df_inp_merged,df_input,'cumsum_seatcount')

    df_input = func_date_time_cols(df_input,'doj')
    df_input['is_holiday_india'] = np.where(df_input['is_holiday_india'] == True,1,0)

    # target encoding
    target_enc_destid_path = get_params(root_path / 'params.yaml','target_enc_destid_path')
    target_enc_dest_reg_path = get_params(root_path / 'params.yaml','target_enc_dest_reg_path')
    target_enc_srcid_path = get_params(root_path / 'params.yaml','target_enc_srcid_path')
    target_end_src_reg_path = get_params(root_path / 'params.yaml','target_end_src_reg_path')   
    df_input,temp_vals_destid = target_encoding(df_inp_merged,df_input,target_enc_destid_path,'destid')
    df_input,temp_vals_dest_reg = target_encoding(df_inp_merged,df_input,target_enc_dest_reg_path,'destid_region')
    df_input,temp_vals_srcid = target_encoding(df_inp_merged,df_input,target_enc_srcid_path,'srcid')
    df_input,temp_vals_src_reg = target_encoding(df_inp_merged,df_input,target_end_src_reg_path,'srcid_region')
    
    df_inp_merged = pd.get_dummies(df_inp_merged,columns = ['srcid_tier','destid_tier'])
    for i in ['srcid_tier_Tier 1', 'srcid_tier_Tier 3', 'srcid_tier_Tier 4',
    'srcid_tier_Tier2', 'destid_tier_Tier 1', 'destid_tier_Tier 3','destid_tier_Tier 4', 'destid_tier_Tier2']:  
        if i not in df_inp_merged.columns : 
            df_inp_merged[i] = 0

    df_input = pd.merge(df_input,df_inp_merged[df_inp_merged['dbd'] == 15][['doj','srcid','destid','srcid_tier_Tier 1', 'srcid_tier_Tier 3', 'srcid_tier_Tier 4',
    'srcid_tier_Tier2', 'destid_tier_Tier 1', 'destid_tier_Tier 3','destid_tier_Tier 4', 'destid_tier_Tier2']],on = ['doj','srcid','destid'],how = 'inner')
    
    # fourier transformation
    df_input = fourier_dayofyear(df_input)
    # polynomial fitting
    df_inp_merged.to_csv('E:/DS/mlops_dvc_ds_project/redbus/data/test/df_inp_merged.csv',index = False)
    poly_fit_df = df_inp_merged.groupby(['doj','srcid','destid']).apply(fit_poly).reset_index()
    df_input = pd.merge(df_input,poly_fit_df[['doj','srcid','destid','slope_lin']],on = ['doj','srcid','destid'],how = 'inner')
    poly_fit2_df = df_inp_merged.groupby(['doj','srcid','destid']).apply(fit_poly2).reset_index()
    df_input = pd.merge(df_input,poly_fit2_df[['doj','srcid','destid','slope1_quad','slope2_quad']],on = ['doj','srcid','destid'],how = 'inner')

    var_order_list = ['cumsum_searchcount_30','cumsum_seatcount_30','cumsum_searchcount_20','cumsum_seatcount_20','cumsum_searchcount_15','cumsum_seatcount_15',
         'cumsum_searchcount_30_25','cumsum_seatcount_30_25','cumsum_searchcount_25_20','cumsum_seatcount_25_20','cumsum_searchcount_20_14',
         'cumsum_seatcount_20_14','ewm_1_cumsum_searchcount','ewm_1_cumsum_seatcount','ewm_5_cumsum_searchcount','ewm_5_cumsum_seatcount',
         'ewm_9_cumsum_searchcount','ewm_9_cumsum_seatcount','destid_encoded','destid_region_encoded','srcid_encoded','srcid_region_encoded',
         'day_of_week','fri_to_sun','week_of_year','month','monnth_start_end','is_holiday_india','srcid_tier_Tier 1','srcid_tier_Tier 3',
         'srcid_tier_Tier 4','srcid_tier_Tier2','destid_tier_Tier 1','destid_tier_Tier 3','destid_tier_Tier 4','destid_tier_Tier2','day_of_year_sin',
         'day_of_year_cos','slope_lin','slope1_quad','slope2_quad']
    df_input = df_input[var_order_list]

    return df_input 

def load_correct_model(model_path: str):
    try:
        model_loaded = pickle.load(open(model_path, "rb"))
        logger.info("Model loaded successfully.")
        return model_loaded
    except Exception as e:
        logger.exception(f"Error loading model: {e}")
        raise

class Predictor:
    """
    Main inference class used by FastAPI or batch jobs.
    Loads model once and serves predictions.
    """
    def __init__(self, model_path: str = 'redbus/data/models/model.pkl'):
        self.model_path = model_path
        self.model_version = "1.0.0"  # Example versioning
        self.model = load_correct_model(self.model_path )

    def predict(self, data: dict) -> pd.DataFrame:
        try:
            logger.info("Starting prediction process.")
            df_input = preprocess_data(data)
            logger.info(f"Preprocessed data shape: {df_input.shape}") 
            df_pred = self.model.predict(df_input)
            df_pred =[round(n) for n in df_pred]
            result = {"estimate" : df_pred}
            logger.info(f"Prediction completed. Predictions : {result}")
            return result 
        except Exception as e:
            logger.exception(f"Error during prediction: {e}")
            raise


