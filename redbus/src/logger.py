import logging 
from pathlib import Path 
import datetime as dt

def create_log_path (module_name:str) : 
    # logging folder and file for given modeul and date 
    # get date 
    current_date = dt.date.today()
    # create folder 
    root_path = Path(__file__).parent.parent
    log_folder_path = root_path/'logs'
    log_folder_path.mkdir(exist_ok = True) 

    # folder for module logs 
    module_log_path = log_folder_path /module_name
    module_log_path.mkdir(exist_ok = True) 

    # get log file name 
    date_format_str = current_date.strftime("%m-%d-%Y")
    log_file_name = module_log_path/(date_format_str + '.log')
    return log_file_name 


class CustomLogger : 
    def __init__(self,logger_name,log_filename): 
        self.__logger = logging.getLogger(name = logger_name)
        self.__log_path = log_filename 

        file_handler = logging.FileHandler(filename=self.__log_path,mode = 'w') 
        self.__logger.addHandler(hdlr = file_handler) 
        
        #formatter 
        log_format = '%(asctime)s - %(levelname)s : %(message)s'
        time_format = '%d-%m-%Y %H:%M:%S'
        formatter = logging.Formatter(fmt = log_format,datefmt=time_format) 
        file_handler.setFormatter(fmt = formatter)

    def get_log_path(self): 
        return self.__log_path 
    
    def get_logger(self) : 
        return self.__logger 
    def set_log_level(self,level = logging.DEBUG) : 
        logger =  self.get_logger() 
        logger.setLevel(level= level) 

    def save_logs(self,msg,log_level = 'info') : 
        logger = self.get_logger()
        if log_level == 'debug' : 
            logger.debug(msg = msg) 
        elif log_level == 'info' : 
            logger.info(msg=msg) 
        elif log_level == 'warning' : 
            logger.warning(msg = msg) 
        elif log_level == 'error' : 
            logger.error(msg=msg) 
        elif log_level == 'exception' : 
            logger.exception(msg=msg) 
        elif log_level == 'critical' : 
            logger.critical(msg = msg) 
    
if __name__ == '__main__' : 
    logger = CustomLogger(logger_name='my_logger', log_filename=create_log_path('my_trial') )
    logger.set_log_level()
    logger.save_logs('Just trying it out',log_level = 'info')
    print("code ran as main")
    