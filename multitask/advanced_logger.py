from enum import Enum
from typing import Optional
import json

class AdvancedLoggerType(Enum):
    '''An enum to specify the type of logger to use.'''
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"
    NONE = "none"

    @classmethod
    def from_str(cls, logger_type: str) -> 'AdvancedLoggerType':
        '''Converts a string to an AdvancedLoggerType.

        Args:
            logger_type (str): The string to convert.

        Returns:
            AdvancedLoggerType: The AdvancedLoggerType corresponding to the string.
        '''
        mapping = {advanced_logger_type.value: advanced_logger_type for advanced_logger_type in cls}
        try:
            return mapping[logger_type]
        except KeyError:
            raise ValueError(f"Invalid logger type: '{logger_type}'. Must be one of {list(mapping.keys())}")

class AdvancedLogger:
    '''A class to log data about machine learning training. Can use either Tensorboard or Weights and Biases.'''
    SAVE_JSON_KEY = 'save_json'

    def __init__(self, logger_type: AdvancedLoggerType, params=None):
        '''Initializes a StatLogger object.

        Args:
            logger_type (LoggerType): The type of logger to use.
            params (dict, optional): A dictionary of parameters to pass to the logger. Defaults to None.

        For Tensorboard, params should be a dictionary with the following keys:
            log_dir (str): The directory to save the logs to.
        For Weights and Biases, params should be a dictionary with the following keys:
            project (str): The name of the project.
            name (str): The name of the run.

        For all logger types:
            Params should be a dictionary and have the key 'save_json' (whatever AdvancedLogger.SAVE_JSON_KEY is) with a boolean value.
                This indicates whether to save all the logged data to a dictionary.
                If set to True, you can access the dictionary with the 'json' attribute.
                If set to False, the 'json' attribute will be None.
                If this key is not present, it defaults to False.
        '''
        self.logger_type = logger_type

        if logger_type == AdvancedLoggerType.TENSORBOARD:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError("You requested a tensorboard logger, but tensorboard is not installed.")
            log_dir = params['log_dir']
            self.logger = SummaryWriter(log_dir=log_dir)

        elif logger_type == AdvancedLoggerType.WANDB:
            try:
                import wandb
            except ImportError:
                raise ImportError("You requested a Weights and Biases logger, but wandb is not installed.")
            self.logger = wandb
            project = params['project']
            name = params['name']
            wandb.init(project=project, name=name)
        elif logger_type == AdvancedLoggerType.NONE:
            pass
        else:
            raise ValueError(f"Invalid logger type: {logger_type}")
        
        save_json = False
        if self.SAVE_JSON_KEY in params:
            save_json = params[self.SAVE_JSON_KEY]
        self.save_json = save_json
        self.json: Optional[dict] = None
        if save_json:
            self.json = {}

    def log(self, stat_name, stat_value, step):
        '''Logs a stat.

        Args:
            stat_name (str): The name of the stat.
            stat_value (float): The value of the stat.
            step (int): The step number.
        '''
        if self.logger_type == AdvancedLoggerType.TENSORBOARD:
            self.logger.add_scalar(stat_name, stat_value, step)

        elif self.logger_type == AdvancedLoggerType.WANDB:
            self.logger.log({stat_name: stat_value}, step=step)
        elif self.logger_type == AdvancedLoggerType.NONE:
            pass
        else:
            raise ValueError(f"Invalid logger type: {self.logger_type}")
        
        if self.save_json:
            self.json_log(stat_name, stat_value, step)
        
    def json_log(self, stat_name, stat_value, step):
        '''Logs a stat to a JSON file.

        Args:
            stat_name (str): The name of the stat.
            stat_value (float): The value of the stat.
            step (int): The step number.
        '''
        if step not in self.json:
            self.json[step] = {}

        self.json[step][stat_name] = stat_value
        
    def log_dict(self, data_dict, step):
        '''Logs a dictionary of stats.

        Args:
            data_dict (dict): A dictionary of stats.
            step (int): The step number.
        '''
        if self.logger_type == AdvancedLoggerType.TENSORBOARD:
            for stat_name, stat_value in data_dict.items():
                self.logger.add_scalar(stat_name, stat_value, step)

        elif self.logger_type == AdvancedLoggerType.WANDB:
            self.logger.log(data_dict, step=step)

        elif self.logger_type == AdvancedLoggerType.NONE:
            pass
        else:
            raise ValueError(f"Invalid logger type: {self.logger_type}")
        
        if self.save_json:
            for stat_name, stat_value in data_dict.items():
                self.json_log(stat_name, stat_value, step)
        
if __name__ == "__main__":
    # Example usage of from_str() method
    logger_type_str = "tensorboard"
    logger_type = AdvancedLoggerType.from_str(logger_type_str)
    print(logger_type)
