import os
from cs336_basics.pretokenization import pretokenize
from collections import Counter, defaultdict
from tqdm import tqdm

def train_bpe(
    input_path: str | os.PathLike = "data/TinyStoriesV2-GPT4-valid.txt",
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

    vocab = dict(enumerate(t.encode() for t in  special_tokens))
    vocab.update({x + len(special_tokens) : bytes([x]) for x in range(256)})
    # print(vocab)

    merges = []

    pretokens_freqs = pretokenize(input_path= input_path, special_tokens= special_tokens)
    byte_pairs : list[tuple[tuple[bytes, bytes], int]] = []
    byte_pair_counts : dict[tuple[bytes, bytes], int] = defaultdict(int)


    for pretoken_bytes, freq in pretokens_freqs.items():
        for i in range(len(pretoken_bytes) - 1):
            byte_pair = (pretoken_bytes[i], pretoken_bytes[i+1])
            if not byte_pair_counts[byte_pair]:
                byte_pair_counts[byte_pair] = freq
            else:
                byte_pair_counts[byte_pair] += freq
            byte_pairs.append((byte_pair, freq))

        byte_pairs.append(None)
    
    bp_indices = defaultdict(list)
    deleted_bp_indicies = defaultdict(set)
    deleted = set()

    for index, item in enumerate(byte_pairs):
        if item is None:
            continue
        bp_indices[item[0]].append(index)


    # Save the occurences of byte_pairs for incrementally update


    with tqdm(total= vocab_size - len(vocab), desc= "Merging") as pbar:
        while len(vocab) < vocab_size:
            pbar.update()

            # get the most common byte pair
            new_index = len(vocab)
            most_common_bp = max(byte_pair_counts, key=lambda x: (byte_pair_counts[x], x))

            # update vocab

            # update vocab
            merges.append(most_common_bp)
            new_byte = most_common_bp[0] + most_common_bp[1]
            vocab[new_index] = new_byte

            # incrementally update byte_pair_counts
            
            for occ_index in bp_indices[most_common_bp]:
                if occ_index in deleted or occ_index in deleted_bp_indicies[most_common_bp]:
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
                        deleted_bp_indicies[bp].add(prev_index)

                        byte_pair_counts[bp] -= freq
                        byte_pairs[prev_index] = nbp
                        byte_pair_counts[nbp[0]] += freq
                
                if occ_index < len(byte_pairs) - 1:
                    next_index = occ_index + 1

                    while next_index in deleted:
                        next_index += 1
                    if next_index < len(byte_pairs) and byte_pairs[next_index] is not None:

                        bp, freq = byte_pairs[next_index]
                        nbp = ((new_byte, byte_pairs[next_index][0][1]), freq)
                        bp_indices[nbp[0]].append(next_index)
                        deleted_bp_indicies[bp].add(next_index)

                        byte_pair_counts[bp] -= freq
                        byte_pairs[next_index] = nbp
                        byte_pair_counts[nbp[0]] += freq

                deleted.add(occ_index)
                byte_pair_counts[most_common_bp] -= byte_pairs[occ_index][1]
            

        # print(vocab)
        # print(merges)

    return (vocab, merges)


if __name__ == "__main__":
    train_bpe()