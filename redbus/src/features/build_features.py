import pandas as pd
import numpy as np
import sys 
from pathlib import Path 
import datetime as dt

current_path = Path(__file__)
root_path = current_path.parent.parent.parent
root_path1  = current_path.parent.parent
print("root path :",root_path1)
sys.path.append(str(root_path1))

from logger import create_log_path, CustomLogger 
import logging  
from yaml import safe_load 
import holidays
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
pd.set_option('display.max_columns', None)


print('All imports done')

# logging things 
log_file_path = create_log_path('build_features')
feature_logger = CustomLogger('Create_Features',log_file_path)
feature_logger.set_log_level(level=logging.INFO)

def create_df(train_data_path,transaction_data_path) :  
    df_train = pd.read_csv(train_data_path)  
    df_trans = pd.read_csv(transaction_data_path)
    feature_logger.save_logs(msg = f'Created Train and Transaction dataframe. Shape of df_train is :{df_train.shape}',log_level='info')
    feature_logger.save_logs(msg = f'Shape of df_train is :{df_trans.shape}',log_level='info')
    return df_train,df_trans

def create_14_day_data(df_train,df_trans) : 
    df_train_merged = pd.merge(df_train,df_trans[df_trans['dbd'] > 14],on = ['doj','srcid','destid'],how = 'left')
    print('df_trans 14 days shape : ', print(df_trans[df_trans['dbd'] > 14].shape))
    feature_logger.save_logs(msg = f'Created dataset with only data older than 14 days. Shape {df_train_merged.shape}',log_level='info')
    return df_train_merged 

def lead_var_curr(dbd_days,df_trans,df_train,var_lead) : 
    df_temp = df_trans[df_trans['dbd'] == dbd_days] 
    df_train = pd.merge(df_train,df_temp[['doj','srcid','destid',var_lead]],on = ['doj','srcid','destid'],how = 'inner')
    var_new = var_lead + '_' + str(dbd_days)
    df_train.rename(columns = {var_lead : var_new},inplace=True)
    feature_logger.save_logs(msg = f'New variable {var_new} created.',log_level='info')
    return df_train 

def get_params(input_file,subsection_name) : 
    try:
        with open(input_file, 'r') as f:
            params_file = safe_load(f)
            print("✅ YAML loaded:", params_file)  # <-- Debug here

        param_out = params_file['build_features'][subsection_name]
        feature_logger.save_logs(msg=f'params read successfully for {subsection_name}',log_level='info')
        return param_out
    except Exception as e : 
        feature_logger.save_logs(msg= f'Error : {e}',log_level='error')
        raise

def avg_func(dbd_days_start,dbd_days_end,df_trans,df_train,var_lead) : 
    df_temp = df_trans[df_trans['dbd'].isin(list(range(dbd_days_start,dbd_days_end,-1)))].groupby(['doj','srcid','destid'])[var_lead].mean().reset_index()
    df_train = pd.merge(df_train,df_temp[['doj','srcid','destid',var_lead]],on = ['doj','srcid','destid'],how = 'inner')
    var_new = var_lead + '_' + str(dbd_days_start) +'_' + str(dbd_days_end)
    df_train.rename(columns = {var_lead : var_new},inplace=True)
    feature_logger.save_logs(msg = f'New variable {var_new} created.',log_level='info')
    return df_train

def ewma_var_curr(alpha_val,df_trans,df_train,var) : 
    # order dataframe
    df_trans_cp_temp = df_trans.copy()
    print(df_trans_cp_temp.shape)
    df_trans_cp_temp = df_trans_cp_temp.sort_values(by = ['doj','srcid','destid','dbd'],ascending = [True,True,True,False])
    df_trans_cp_temp.reset_index(drop = True)
    
    # calculate ewm
    var_new = 'ewm_' + str(alpha_val)[2:] + '_' + var
    print(var_new)
    df_trans_cp_temp[var_new] = df_trans_cp_temp.groupby(['doj','srcid','destid'])[var].transform(lambda x : x.ewm(alpha = alpha_val).mean())
    
    # join with df_train
    df_train = pd.merge(df_train,df_trans_cp_temp[df_trans_cp_temp['dbd'] == 15][['doj','srcid','destid',var_new]],on = ['doj','srcid','destid'],how = 'inner')
    feature_logger.save_logs(msg = f'New variable {var_new} created.',log_level='info')
    
    return df_train


def func_date_time_cols(df,var_name) : 
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
    feature_logger.save_logs(msg = f'New variables related to datetime created.',log_level='info')
    return df

# Ohe --> srcid_tier,destid_tier,is_holiday_india
# target encoding --> srcid_region,destid_region,srcid,destid
def target_encoding(df,df_train,encode_path,var_name) : 
    df_temp = df.drop_duplicates(subset = ['doj','srcid','destid'])
    temp_vals = pd.read_csv(encode_path,index_col = var_name).squeeze() 
    # temp_vals =df_temp.groupby(var_name)['final_seatcount'].mean()
    df_temp[var_name + '_encoded'] = df_temp[var_name].map(temp_vals)
    df_train = pd.merge(df_train,df_temp[['doj','srcid','destid',var_name + '_encoded']],on = ['doj','srcid','destid'],how = 'inner')
    feature_logger.save_logs(msg = f'New variables {var_name + '_encoded'} creatd.',log_level='info')
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

def main() : 
    train_data_path = sys.argv[1]
    transaction_data_path = sys.argv[2] 
    df_train,df_trans = create_df(train_data_path,transaction_data_path)
    df_train_merged = create_14_day_data(df_train,df_trans) 
    print('yaml file path',root_path / 'params.yaml')
    
    lead_days_lst = get_params(root_path / 'params.yaml','lead_days') 
    for i in lead_days_lst : 
        df_train = lead_var_curr(i,df_train_merged,df_train,'cumsum_searchcount')
        df_train = lead_var_curr(i,df_train_merged,df_train,'cumsum_seatcount') 
    print('lead variables created')

    avg_func_list = get_params(root_path / 'params.yaml','avg_func_list') 
    for i,j in avg_func_list : 
        df_train = avg_func(i,j,df_train_merged,df_train,'cumsum_searchcount')
        df_train = avg_func(i,j,df_train_merged,df_train,'cumsum_seatcount')
    print('average variables created')

    ewma_alpha_list = get_params(root_path / 'params.yaml','ewma_alpha_list')  
    for i in ewma_alpha_list : 
        df_train = ewma_var_curr(i,df_train_merged,df_train,'cumsum_searchcount')
        df_train = ewma_var_curr(i,df_train_merged,df_train,'cumsum_seatcount') 
    print('EWMA variables created')

    print('Date time variables created')

    target_enc_destid_path = get_params(root_path / 'params.yaml','target_enc_destid_path')
    df_train,_ = target_encoding(df_train_merged,df_train,target_enc_destid_path,'destid')

    target_enc_dest_reg_path = get_params(root_path / 'params.yaml','target_enc_dest_reg_path')
    df_train,_ = target_encoding(df_train_merged,df_train,target_enc_dest_reg_path,'destid_region')

    target_enc_srcid_path = get_params(root_path / 'params.yaml','target_enc_srcid_path')
    df_train,_ = target_encoding(df_train_merged,df_train,target_enc_srcid_path,'srcid')

    target_end_src_reg_path = get_params(root_path / 'params.yaml','target_end_src_reg_path')
    df_train,_ = target_encoding(df_train_merged,df_train,target_end_src_reg_path,'srcid_region')
    print('target encoding variables created')

    df_train['doj'] = pd.to_datetime(df_train['doj']) 
    df_train = func_date_time_cols(df_train,'doj') 
    df_train['doj'] = df_train['doj'].astype(str)

    df_train_merged = pd.get_dummies(df_train_merged,columns = ['srcid_tier','destid_tier'])
    feature_logger.save_logs(msg = f'One hot encoding done.',log_level='info')

    feature_logger.save_logs(msg = f'Shape before ohe merge {df_train.shape}',log_level='info')
    df_train = pd.merge(df_train,df_train_merged[df_train_merged['dbd'] == 15][['doj','srcid','destid','srcid_tier_Tier 1', 'srcid_tier_Tier 3', 'srcid_tier_Tier 4',
    'srcid_tier_Tier2', 'destid_tier_Tier 1', 'destid_tier_Tier 3','destid_tier_Tier 4', 'destid_tier_Tier2']],on = ['doj','srcid','destid'],how = 'inner')
    df_train['is_holiday_india'] = np.where(df_train['is_holiday_india'] == True,1,0)
    feature_logger.save_logs(msg = f'Shape after ohe merge {df_train.shape}',log_level='info')

    df_train = fourier_dayofyear(df_train)
    feature_logger.save_logs(msg = f'Fourier day of year created',log_level='info')

    feature_logger.save_logs(msg = f'Shape before poly merges {df_train.shape}',log_level='info')
    df_poly_slope = df_train_merged.groupby(['doj','srcid','destid']).apply(fit_poly).reset_index() 
    df_train = pd.merge(df_train,df_poly_slope,on = ['doj','srcid','destid'],how = 'inner') 
    df_poly2_slope = df_train_merged.groupby(['doj','srcid','destid']).apply(fit_poly2).reset_index() 
    df_train = pd.merge(df_train,df_poly2_slope,on = ['doj','srcid','destid'],how = 'inner') 
    feature_logger.save_logs(msg = f'Polynomial codefficient vars added.',log_level='info')
    feature_logger.save_logs(msg = f'Shape after poly merges {df_train.shape}',log_level='info')

    df_train = df_train.drop(['coefficient_lin','coefficient_quad','doj','srcid', 'destid',],axis = 1)
    print(df_train.columns)

    # current_date = dt.date.today()
    # date_format_str = current_date.strftime("%m-%d-%Y")
    # file_path_output = root_path + '/data/processed/' +  'processed_data_' + date_format_str

    file_path_output = sys.argv[3]  
    df_train.to_csv(file_path_output, index=False) 
    feature_logger.save_logs(msg = f'File written to path {file_path_output}',log_level='info')

if __name__ == "__main__" : 
    main()