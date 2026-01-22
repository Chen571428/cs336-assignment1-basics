import os
import json
from cs336_basics.pretokenization import pretokenize
from collections import Counter, defaultdict
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn
)
from typing import Optional, Self, Iterable, Iterator
import heapq
from cs336_basics.utils import get_logger, timer, ProgressBar
import time

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

class Tokenizer:
    """
    The BPE Tokeinizer
    """
    def __init__(
            self,
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None = None,
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
        self.merges = merges
        self.special_tokens = special_tokens

    
    @classmethod
    def from_files(
            cls,
            vocab_filepath: str,
            merges_filepath: str,
            special_tokens: list[str] | None = None
    ) -> Self:
        
        vocab : dict[int, bytes] = {} 
        merges : list[tuple[bytes, bytes]]

        with open(vocab_filepath, "rb") as f:
            vocab = {k: v.encode("latin-1") for k,v in json.load(f).items()}
        
        with open(merges_filepath, "rb") as f:
            merges = [ (a.encode("latin-1"), b.encode("latin-1")) for a, b in json.load(f).items()]

        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:


        encoded_str = []
        return encoded_str
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass
    
    def decode(self, ids: list[int]) -> str:


        decoded_str = ""
        return decoded_str
    




if __name__ == "__main__":
    # input_path = "data/owt_train.txt"
    # input_path="data/TinyStoriesV2-GPT4-train.txt"
    # vocab, merges = train_bpe(
    #     input_path="data/TinyStoriesV2-GPT4-train.txt"
    # )
    vocab, merges = train_bpe(
        vocab_size=32000,
        input_path="data/owt_train.txt",
    )
    #
    # with open(f"vocab_{str(input_path)[5:]}.json", "w") as f:
    #     json.dump(vocab, f, ensure_ascii=False, default=lambda x: x.decode('latin-1') if isinstance(x, bytes) else x)

    # with open(f"merges_{str(input_path)[5:]}.json", "w") as f:
    #     json.dump(merges, f, ensure_ascii=False, default=lambda x: x.decode('latin-1') if isinstance(x, bytes) else x)

    # vocab, merges = train_bpe(
    #     input_path="data/TinyStoriesV2-GPT4-train.txt",
    # )
    # print(vocab)
    # print(merges)