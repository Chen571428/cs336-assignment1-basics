import multiprocessing
import os
from tokenize import Special
from typing import BinaryIO
from collections import Counter
import functools
import regex as re
import tqdm
from tqdm import tqdm
import time
from cs336_basics.utils import get_logger, timer
from rich.progress import track

logger = get_logger(__name__)

BYTES_LOOKUP = [bytes([i]) for i in range(256)]
PRETOKENIZE_PAT =  r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PATTERN = re.compile(PRETOKENIZE_PAT)

@timer
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
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

def pretokenize_single_text(
    text : str,
) -> Counter[tuple[bytes, ...]]:
    """
    Pretokenize the single Special_tokens-removed text
    """
    raw_bytes_gen = (m.group().encode() for m in re.finditer(PATTERN, text))

    pretoken_gen = (
        tuple(BYTES_LOOKUP[i] for i in b) # using Lookup table to optimize
        for b in raw_bytes_gen
    )

    pretoken_freq = Counter(pretoken_gen)

    return pretoken_freq
def pre_tokenize_single_chunk(
    pos : tuple[int, int],
    input_path : str | os.PathLike,
    special_tokens : list[str]
) -> Counter[tuple[bytes, ...]]:
    """
    pretoken a single chunk in a single process
    """
    with open(input_path, "rb") as f:

        pretoken_freq = Counter()
        split_pattern = "|".join(re.escape(t) for t in special_tokens)

        start, end = pos

        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        
        for text in re.split(split_pattern, chunk):
            pretoken_freq.update(pretokenize_single_text(text))

    return pretoken_freq

@timer
def pretokenize(
    input_path : str | os.PathLike = "data/TinyStoriesV2-GPT4-valid.txt",
    special_tokens : list[str] = ["<|endoftext|>"],
    num_processes : int = max(1, (os.cpu_count() or 1) - 1)
) -> Counter[tuple[bytes, ...]]:

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    pretokens_freq = Counter()

    with multiprocessing.Pool(processes= num_processes) as pool:

        logger.info(f"Starting Pretokenization with {num_processes} processes.")
        results = pool.imap_unordered(functools.partial(
                                pre_tokenize_single_chunk,
                                special_tokens= special_tokens,
                                input_path= input_path
                            ),
                            zip(boundaries[:-1], boundaries[1:]))
        # map pretoken task to each process
        logger.info("Multiprocessing completed, Merging results.")
        rmerge_st_time = time.perf_counter()

        for partial_res in track(results, description="[bold blue]Merging Pretokenization results[/]"):
            pretokens_freq.update(partial_res)
            
        rmerge_ed_time = time.perf_counter()

        logger.info(f"Results Merging Duration is {rmerge_ed_time - rmerge_st_time : .6f}")

        return pretokens_freq
    
if __name__ == "__main__":
    x = pretokenize()
    print(dict(x))