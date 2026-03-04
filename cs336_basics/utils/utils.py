import logging
import sys, os
import time
from typing import BinaryIO
from functools import wraps
from tqdm import tqdm
from rich.text import Text
from rich.logging import RichHandler
import rich
import json
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

def resources_accounting(
        vocab_size: int,
        seq_len: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int
) -> None:
    #Parameters
    ## Per Layer
    attn_proj_weights_per_layer =  4 * d_model ** 2
    ln_weights_per_layer = 2 * d_model
    ffn_proj_weights_per_layer =  8 * d_model ** 2

    transformer_weights_per_layer = 2 * d_model + 12 * d_model ** 2

    ## Total
    token_embedding_weights = vocab_size * d_model
    attn_layers_proj_weights = num_layers * attn_proj_weights_per_layer
    ln_layers_weights = num_layers * ln_weights_per_layer
    ffn_layers_proj_weights = num_layers * ffn_proj_weights_per_layer

    transformer_layers_weights = num_layers * transformer_weights_per_layer
    ln_final_weights = 1 * d_model
    lm_head_weights = d_model * vocab_size

    total_weights = token_embedding_weights + transformer_layers_weights + ln_final_weights + lm_head_weights

    parameters = {
        # "attn_proj_weights_per_layer" :  (4 * 2 * seq_len * d_model * d_model),
        # "ln_weights_per_layer" : 2 * d_model,
        # "ffn_proj_weights_per_layer" :  8 * d_model ** 2,

        # "transformer_weights_per_layer" : 2 * d_model + 12 * d_model ** 2,

        "token_embedding_weights" : (vocab_size * d_model) / total_weights * 100,
        "attn_layers_proj_weights" : (num_layers * attn_proj_weights_per_layer) / total_weights * 100,
        "ln_layers_weights" : (num_layers * ln_weights_per_layer) / total_weights * 100,
        "ffn_layers_proj_weights" : (num_layers * ffn_proj_weights_per_layer) / total_weights * 100,
        "lm_head & ln_final": (lm_head_weights + ln_final_weights)  / total_weights * 100 ,

        "transformer_layers_weights" : (num_layers * transformer_weights_per_layer) / total_weights * 100 ,
        "total_BParameters" : total_weights  / 1000000000
    }
    rich.print_json(json.dumps(parameters))

    # Forward Matmuls

    attn_proj_flops_per_layer = 8 * seq_len * d_model**2
    attn_MHA_flops_per_layer =  4 * seq_len ** 2 * d_model
    ffn_proj_flops_per_layer = 16 * seq_len * d_model**2

    attn_proj_flops = num_layers * attn_proj_flops_per_layer
    attn_MHA_flops = num_layers * attn_MHA_flops_per_layer 
    ffn_proj_flops = num_layers * ffn_proj_flops_per_layer 
    lm_head_flops = 2 * seq_len * d_model * vocab_size

    total_flops = attn_proj_flops + attn_MHA_flops + ffn_proj_flops + lm_head_flops

    forward_matmul_flops = {
        # "attn_proj_flops_per_layer" : 8 * seq_len * d_model**2 ,
        # "attn_MHA_flops_per_layer" :  4 * seq_len ** 2 * d_model ,
        # "ffn_proj_flops_per_layer" : 16 * seq_len * d_model**2 ,
        "attn_proj_flops" : (num_layers * attn_proj_flops_per_layer) / total_flops * 100 ,
        "attn_MHA_flops" : (num_layers * attn_MHA_flops_per_layer) / total_flops * 100  ,
        "ffn_proj_flops" : (num_layers * ffn_proj_flops_per_layer) / total_flops * 100  ,
        "lm_head_flops" : (2 * seq_len * d_model * vocab_size) / total_flops * 100 ,
        "total_TFLOPS": total_flops / 1000000000000
    }
   
    rich.print_json(json.dumps(forward_matmul_flops))
    


def cmp_resources() -> None:
    # GPT-2-small (12layers,768d_model,12heads)
    # GPT-2-medium (24layers,1024d_model,16heads)
    # GPT-2-large (36layers,1280d_model,20heads)
   
    rich.print("GPT-2-SMALL")
    GPT_2_small = {
        "vocab_size" : 50257,
        "seq_len" : 1024,
        "num_layers" : 12,
        "d_model" : 768,
        "num_heads" : 12,
        "d_ff" : 4 * 768
    }
    resources_accounting(**GPT_2_small)

    rich.print("GPT-2-MEDIUM")
    GPT_2_medium = {
        "vocab_size" : 50257,
        "seq_len" : 1024,
        "num_layers" : 24,
        "d_model" : 1024,
        "num_heads" : 16,
        "d_ff" : 4 * 1024
    }
    resources_accounting(**GPT_2_medium)

    rich.print("GPT-2-LARGE")
    GPT_2_large = {
        "vocab_size" : 50257,
        "seq_len" : 1024,
        "num_layers" : 36,
        "d_model" : 1280,
        "num_heads" : 20,
        "d_ff" : 4 * 1280
    }
    resources_accounting(**GPT_2_large)

    rich.print("GPT-2-XL")
    GPT_2_xl = {
        "vocab_size" : 50257,
        "seq_len" : 1024,
        "num_layers" : 48,
        "d_model" : 1600,
        "num_heads" : 25,
        "d_ff" : 4 * 1600
    }
    resources_accounting(**GPT_2_xl)

    rich.print("GPT-2-XL-16384L")
    GPT_2_xl_16384l = {
        "vocab_size" : 50257,
        "seq_len" : 16384,
        "num_layers" : 48,
        "d_model" : 1600,
        "num_heads" : 25,
        "d_ff" : 4 * 1600
    }
    resources_accounting(**GPT_2_xl_16384l)

if __name__ == "__main__":
    cmp_resources()