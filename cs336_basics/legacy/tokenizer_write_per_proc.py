import os
import json
from cs336_basics.pretokenization import pretokenize, PREPATTERN, BYTES_LOOKUP
from collections import Counter, defaultdict
from rich.progress import track
from typing import Optional, Self, Iterable, Iterator, Sequence, BinaryIO
import heapq
from cs336_basics.utils import get_logger, timer, ProgressBar, find_chunk_boundaries
from cs336_basics.rust_utils import RustBPE
import time
import regex as re
from itertools import zip_longest
import multiprocessing
import multiprocessing.pool
import functools
import numpy as np

logger = get_logger(__name__)

@timer
def train_bpe(
    input_path: str | os.PathLike = "data/TinyStoriesV2-GPT4-train.txt",
    vocab_size: int = 10000,
    special_tokens: list[str] = ["<|endoftext|>"],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    logger.info(f"Starting BPE Training, dataset is {input_path}")
    

    vocab = dict(enumerate(t.encode() for t in  special_tokens))
    vocab.update({x + len(special_tokens) : bytes([x]) for x in range(256)})
    merges = []

    pretokens_freqs = pretokenize(input_path= input_path, special_tokens= special_tokens)

    byte_pairs : list[Optional[tuple[tuple[bytes, bytes], int]]] = []
    byte_pair_counts : dict[tuple[bytes, bytes], int] = defaultdict(int)

    logger.info(f"Finished Pretokenization. Started Preprocessing.")

    preprocess_st_time = time.perf_counter()

    bp_indices = defaultdict(list)

    for pretoken_bytes, freq in pretokens_freqs.items():

        start_idx = len(byte_pairs)

        for i, byte_pair in enumerate(zip(pretoken_bytes, pretoken_bytes[1:])):
            byte_pair_counts[byte_pair] += freq
            byte_pairs.append((byte_pair, freq))
            bp_indices[byte_pair].append(i + start_idx)

        byte_pairs.append(None)
    
    class Byte_Pair_Counts_Obj:
        """
        Wrap the object and rewrite the less then method so that heapq make sense
        """
        __slots__ = ('val',)

        def __init__(self, val: tuple[int, tuple[bytes, bytes]]):
            self.val = val
        
        def __lt__(self, other: Self):
            return self.val > other.val
        
        def __repr__(self):
            return str(self.val)
    
    byte_pair_heap = [Byte_Pair_Counts_Obj((fq, bp)) for bp, fq in byte_pair_counts.items()]
    heapq.heapify(byte_pair_heap)
    # Use Heap to maintain the Most Common Bytes Pair
    
    deleted = set()

    preprocess_ed_time = time.perf_counter()

    logger.info(f"Preprocess completed, duration is {preprocess_ed_time - preprocess_st_time : .6f} sec.")
    logger.info(f"Starting BPE Merging")
    merge_st_time = time.perf_counter()

    with ProgressBar() as pbar:
        task = pbar.add_task(description="BPE Merging...", total=vocab_size - len(vocab))

        while len(vocab) < vocab_size:
            pbar.update(task, advance=1)

            # get the most common byte pair
            new_index = len(vocab)
            most_common_bp: Optional[tuple[bytes, bytes]] = None
            while len(byte_pair_heap):
                most_fq, most_common_bp = heapq.heappop(byte_pair_heap).val
                if byte_pair_counts[most_common_bp] == most_fq:
                    break
            if most_common_bp is None:
                break
                # This is Unreachable

            # update vocab
            merges.append(most_common_bp)
            new_byte = most_common_bp[0] + most_common_bp[1]
            vocab[new_index] = new_byte

            # incrementally update byte_pair_counts
            
            new_byte_pair_counts = defaultdict(int)
            # Mark the updated byte pair counts for heap updateing
            for occ_index in bp_indices[most_common_bp]:
                if occ_index in deleted :
                    continue
                if byte_pairs[occ_index][0] != most_common_bp:
                    continue
                if occ_index > 0:
                    prev_index = occ_index - 1

                    while prev_index in deleted:
                        prev_index -= 1
                    if prev_index >= 0 and byte_pairs[prev_index] != None:

                        bp, freq = byte_pairs[prev_index]
                        nbp = ((byte_pairs[prev_index][0][0], new_byte), freq)
                        bp_indices[nbp[0]].append(prev_index)

                        byte_pair_counts[bp] -= freq
                        byte_pairs[prev_index] = nbp
                        byte_pair_counts[nbp[0]] += freq

                        new_byte_pair_counts[bp] = byte_pair_counts[bp]
                        new_byte_pair_counts[nbp[0]] += freq
                
                if occ_index < len(byte_pairs) - 1:
                    next_index = occ_index + 1

                    while next_index in deleted:
                        next_index += 1
                    if next_index < len(byte_pairs) and byte_pairs[next_index] is not None:

                        bp, freq = byte_pairs[next_index]
                        nbp = ((new_byte, byte_pairs[next_index][0][1]), freq)
                        bp_indices[nbp[0]].append(next_index)

                        byte_pair_counts[bp] -= freq
                        byte_pairs[next_index] = nbp
                        byte_pair_counts[nbp[0]] += freq

                        new_byte_pair_counts[bp] = byte_pair_counts[bp]
                        new_byte_pair_counts[nbp[0]] += freq

                deleted.add(occ_index)
                byte_pair_counts[most_common_bp] -= byte_pairs[occ_index][1]
                new_byte_pair_counts[most_common_bp] = byte_pair_counts[most_common_bp]

            
            # update the heap
            for bp, freq in new_byte_pair_counts.items():
                if freq > 0:
                    heapq.heappush(byte_pair_heap, Byte_Pair_Counts_Obj((freq, bp)))
    
    merge_ed_time = time.perf_counter()
    logger.info(f"Merge Completed, duration is {merge_ed_time - merge_st_time: .6f} sec.")

    return (vocab, merges)


# --------------------------------------------

def worker_task(args) -> tuple[int, int, int, str]:
    chunk_id, pos, file_path, tokenizer_instance = args
    start, end = pos
    
    with open(file_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    
    ids = tokenizer_instance.encode(text)
    
    arr = np.array(ids, dtype=np.uint16)
    
    temp_filename = f"temp_part_{chunk_id:06d}.bin"
    with open(temp_filename, "wb") as f_out:
        f_out.write(arr.tobytes())

    return chunk_id, len(ids), end - start, temp_filename



class Tokenizer:
    """
    The BPE Tokeinizer
    """
    def __init__(
            self,
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None = None,
            use_rust: bool | None = None
    ):
        """Given a vocabulary, a list of merges, and a list of special tokens,
        return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

        Args:
            vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
            special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
                be split into multiple tokens, and will always be kept as a single token.

        Returns:
            A BPE tokenizer that uses the provided vocab, merges, and special tokens.
        """
        self.vocab = vocab
        self.invert_vocab = {value : key for key, value in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens
        self.cache : dict[tuple[bytes, ...], list[int]] = {}
        # implement cache to optimize
        self.merge_rank : dict[tuple[bytes, bytes], int] = dict(zip(merges, range(len(merges))))
        self.use_rust = use_rust
        if use_rust:
            self.merge_rk_nid : dict[tuple[int, int], tuple[int, int]] = {(self.invert_vocab[b1], self.invert_vocab[b2]) : (rk, self.invert_vocab[b1 + b2]) for (b1, b2), rk in self.merge_rank.items()}
            self.rust_bpe_engine = RustBPE(self.merge_rk_nid)
        # implement rank to optimize

        if special_tokens is not None:
            self.special_tokens = sorted(special_tokens,reverse=True, key=lambda x : len(x))
            self.split_pattern = re.compile(f"({"|".join(re.escape(t) for t in self.special_tokens)})")
            
            for t in self.special_tokens:
                t_bytes = t.encode()
                if t_bytes not in self.invert_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = t_bytes
                    self.invert_vocab[t_bytes] = new_id
    
    @classmethod
    def from_files(
            cls,
            vocab_filepath: str,
            merges_filepath: str,
            special_tokens: list[str] | None = None,
            use_rust: bool | None = None
    ) -> Self:
        
        vocab : dict[int, bytes] = {} 
        merges : list[tuple[bytes, bytes]]

        with open(vocab_filepath, "rb") as f:
            vocab = {int(k) : v.encode("latin-1") for k,v in json.load(f).items()}
        
        with open(merges_filepath, "rb") as f:
            merges = [(x[0].encode("latin-1"), x[1].encode("latin-1")) for x in json.load(f) ]

        return cls(vocab, merges, special_tokens, use_rust)
    
    def merge(self, token_bytes: tuple[bytes, ...]) -> list[int]:
        if self.use_rust:
            inital_ids = [self.invert_vocab[b] for b in token_bytes]
            return self.rust_bpe_engine.merge(inital_ids)
        else:
            if token_bytes in self.cache:
                return self.cache[token_bytes]
            
            word = list(token_bytes)
            while len(word) > 1:
                min_rank = float('inf')
                min_pair_i = None

                for i in range(len(word) - 1):
                    pair = (word[i], word[i+1])
                    if self.merge_rank.get(pair, float("inf")) < min_rank:
                        min_rank = self.merge_rank[pair]
                        min_pair_i = i

                if min_pair_i is not None and min_rank != float("inf"):
                    word[min_pair_i] = word[min_pair_i] + word[min_pair_i + 1]
                    del word[min_pair_i + 1]
                else:
                    break
            
            ids = [self.invert_vocab[token] for token in word]
            self.cache[token_bytes] = ids
            return ids
        

    def encode_single_text(self, text: str, pytestflag : bool = True, num_processes: int = 1) -> list[int]:
        raw_bytes_gen = (m.group().encode() for m in re.finditer(PREPATTERN, text, concurrent=True))
        pretoken_gen = (
                tuple(BYTES_LOOKUP[i] for i in b)
                for b in raw_bytes_gen
            )
        if pytestflag:
            pretoken_gen = list(pretoken_gen)

        encoded_str = []
        
        if self.use_rust:
            batch_initial_ids = [
                [self.invert_vocab[b] for b in t] 
                for t in pretoken_gen
            ]
            
            merged_batch = self.rust_bpe_engine.merge_batch(batch_initial_ids)
            
            for ids in merged_batch:
                encoded_str.extend(ids)
        else:
            for t in pretoken_gen:
                encoded_str.extend(self.merge(t))
        return encoded_str
    
    def encode(self, text: str) -> list[int]:
        if self.special_tokens is not None:
            results = re.split(self.split_pattern, text)
            encoded_str = []
            for t, s in zip_longest(results[0::2], results[1::2], fillvalue=None):
                    if t is not None:
                        encoded_str.extend(self.encode_single_text(t))
                    if s is not None:
                        encoded_str.append(self.invert_vocab[s.encode()])
            return encoded_str
        else:
            return self.encode_single_text(text)


    def _encode_batch_worker(self, chunk: Sequence[str]) -> list[int]:
        res = []
        for text in chunk:
            res.extend(self.encode(text))
        return res
    
    def encode_iterable(self, iterable: Iterable[str], pool: multiprocessing.pool.Pool | None = None, chunk_size : int = 8192) -> Iterator[int]:
        """Iterable Encoder,processing encoding by chunks for memory efficient"""
        def batch_generator():
            batch = []
            for line in iterable:
                batch.append(line)
                if len(batch) >= chunk_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

        if pool is not None:
            for batch_ids in pool.imap(self._encode_batch_worker, batch_generator()):
                yield from batch_ids
        else:
            for batch in batch_generator():
                yield from self._encode_batch_worker(batch)
    def _encode_inside_file(
            self,
            pos : tuple[int, int],
            input_path : str | os.PathLike,
    ) -> tuple[np.ndarray, int]:
        with open(input_path, "rb") as f:
            start, end = pos

            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            ids_list = self.encode(chunk)
            chunk_len = sum(len(self.vocab[i]) for i in ids_list)
            
            return np.array(ids_list, dtype=np.uint16), chunk_len

    def encode_from_file(self, input_file: str, output_file: str, num_processes: int = (os.cpu_count() or 2) - 1, chunk_size: int = 10 * 1024 * 1024) -> None:
        boundaries =[]
        with open(input_file, "rb") as f:
            boundaries = find_chunk_boundaries(f, chunk_size = chunk_size)
        logger.info(f"Got file boundaries, chunk_size is {chunk_size / 1024 } KiB , {len(boundaries) - 1} chunks in total")

        tasks = [
            (i, bound, input_file, self) 
            for i, bound in enumerate(zip(boundaries[:-1], boundaries[1:]))
        ]

        total_chunks = len(tasks)
        temp_files: list[None | str] = [None] * total_chunks

        st_time = time.perf_counter()

        with ProgressBar() as pbar:
            task = pbar.add_task(description=f"Encoding {input_file}... ",total= os.path.getsize(input_file))
            with multiprocessing.Pool(processes= num_processes) as pool:
                results = pool.imap_unordered(worker_task, tasks)

                tot_tks = 0
                tot_bts = 0
                for chunk_id, token_num, bytes_num, temp_filename in results:
                    temp_files[chunk_id] = temp_filename

                    tot_tks += token_num
                    tot_bts += bytes_num

                    pbar.update(task, description=f"Encoding {input_file}... Chunk {chunk_id} done. Throughput: { tot_bts / 1048576 / (time.perf_counter() - st_time) :.6f} MiB/s")
                    pbar.advance(task, bytes_num)
                
        logger.info(f"Processed. Total Duration is {time.perf_counter() - st_time:.6f} Tokens: {tot_tks} Size: {tot_bts / 1048576:.2f}MiB")
        
        logger.info("Merging Files.")

        with open(output_file, "wb") as f_out:
            for fname in temp_files:
                if fname and os.path.exists(fname):
                    with open(fname, "rb") as f_in:
                        import shutil
                        shutil.copyfileobj(f_in, f_out)
                    os.remove(fname) 
                else:
                    logger.error(f"Missing chunk file: {fname}")
        logger.info("Merged. Converting to npy.")
        final_ids = np.fromfile(f"{output_file}", dtype=np.uint16)
        np.save(f"{output_file}.npy", final_ids)

    
    def decode(self, ids: list[int]) -> str:
        """Decode the list of tokens into string"""
        decoded_bytes = b"".join(self.vocab[id] for id in ids)
        return decoded_bytes.decode(encoding="UTF-8", errors="replace")

if __name__ == "__main__":
    dataset = "data/owt_train.txt"
    output = "output/owt_train.bin"
    tokenizer = Tokenizer.from_files(
        vocab_filepath="vocab_owt_train.txt.json",
        merges_filepath="merges_owt_train.txt.json",
        use_rust= None
    )

    tokenizer.encode_from_file(
        input_file= dataset,
        output_file= output,
        chunk_size= 50 * 1024 * 1024,
    )