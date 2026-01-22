import logging
import sys
import time
from functools import wraps
from tqdm import tqdm
from rich.text import Text
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    ProgressColumn,
)
import datetime

class PreciseTimeElapsedColumn(ProgressColumn):
    def render(self, task) -> Text:
        elapsed = task.elapsed
        if elapsed is None:
            return Text("--:--:--", style="progress.elapsed")
        
        delta = datetime.timedelta(seconds=float(elapsed))
        return Text(f"{str(delta)[:13]:>10}", style="progress.elapsed")


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG) 

    if logger.handlers:
        return logger

    # console_handler = logging.StreamHandler(sys.stdout)
    console_handler = RichHandler(rich_tracebacks=True)
    console_handler.setLevel(logging.INFO) 

    file_handler = logging.FileHandler('cs336_global.log', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG) 


    file_formatter = logging.Formatter(
        '%(asctime)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def timer(func):
    logger = get_logger(__name__)
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        logger.info(f"Function '{func.__name__}' duration: {end_time - start_time:.6f} sec")
        return result
    return wrapper

def ProgressBar():
    return Progress(
        TextColumn("[progress.description]{task.description}",style="bold blue"), 
        SpinnerColumn(),    
        PreciseTimeElapsedColumn(),
        BarColumn(),  
        TaskProgressColumn(),                  
        MofNCompleteColumn(),                   
        TimeRemainingColumn(),    
    )