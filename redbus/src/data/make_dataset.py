# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np 
import sys 
from pathlib import Path 
current_path = Path(__file__)
root_path1  = current_path.parent.parent.parent
print('root path1 :',root_path1)
sys.path.append(str(root_path1))
from src.logger import create_log_path, CustomLogger 
import logging 
from sklearn.model_selection import train_test_split 
import yaml
from yaml import safe_load 

# get logging things 
log_file_path = create_log_path('make_dataset')
dataset_logger = CustomLogger('make_train_test_split',log_file_path) 
dataset_logger.set_log_level(level=logging.INFO)

# read csv files 
def load_raw_data(input_path) : 
    raw_data = pd.read_csv(input_path)
    rows,cols = raw_data.shape
    dataset_logger.save_logs(msg = f'The row count is {rows} and columns count is {cols}',log_level='info') 
    return raw_data

# do train test split
def tran_test_split(data,test_size,random_state) : 
    train_data,val_data = train_test_split(data,test_size=test_size,random_state=random_state)
    dataset_logger.save_logs(msg=f'train data shape is {train_data.shape} and val data shape is {val_data.shape}.',
                             log_level='info') 
    dataset_logger.save_logs(msg=f'Params values are : {test_size}, {random_state}',
                             log_level='info') 
    return train_data,val_data 

# save the split 
def save_df(df,output_path) : 
    df.to_csv(output_path,index = False)
    dataset_logger.save_logs(msg = f'dataframe {df} is saved in location {output_path}') 

def get_params(input_file) : 
    try:
        print('Reading YAML from:', input_file)
        with open(input_file, 'r') as f:
            params_file = safe_load(f)
            print("✅ YAML loaded:", params_file)  # <-- Debug here

        test_size = params_file['make_dataset']['test_size']
        print('test size loaded',test_size)
        random_state = params_file['make_dataset']['random_state']
        print('random state loaded')

        dataset_logger.save_logs(
            msg='params read successfully for train test split',
            log_level='info'
        )
        return test_size, random_state

    except Exception as e:
        print('❌ Exception:', e)
        dataset_logger.save_logs(
            msg=f"ERROR: {e}. Using default parameters.",
            log_level='error'
        )
        return 0.2, 123

def main() : 
    input_file_name = sys.argv[1]
    current_path = Path(__file__) 
    root_path  = current_path.parent.parent.parent
    raw_df_path = root_path /'data'/'raw'/ input_file_name
    raw_df = load_raw_data(raw_df_path)
    print('yaml file path',root_path / 'params.yaml')
    test_size,random_state = get_params(root_path / 'params.yaml') 
    train_df,val_df = tran_test_split(raw_df,test_size,random_state) 
    interim_data_path = root_path /'data'/'interim'
    interim_data_path.mkdir(exist_ok= True)
    save_df(train_df,interim_data_path/'train.csv')
    save_df(val_df,interim_data_path/'val.csv')

if __name__ == '__main__' : 
    main()

# get params for train test split from yaml files. 
 


