import logging
import sys, os
import time
from typing import BinaryIO
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

def fast_chunk_reader(file_path, chunk_size_bytes=10 * 1024 * 1024):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            chunk = f.read(chunk_size_bytes)
            if not chunk:
                break
            last_newline = chunk.rfind('\n')
            if last_newline != -1 and last_newline != len(chunk) - 1:
                f.seek(f.tell() - (len(chunk) - last_newline - 1))
                chunk = chunk[:last_newline + 1]
            
            yield chunk

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int = 1,
    split_special_token: bytes = b"\n",
    chunk_size: int | None = None
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    # chunk_size has higher priority
    if chunk_size != None: 
        desired_num_chunks = file_size // chunk_size
    else: 
        chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))