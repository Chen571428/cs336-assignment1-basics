# dataloader.py
# 实现从tokeinzed Numpy .npy文件读取token ids用于训练
import numpy as np
import numpy.typing as npt 
import torch
from jaxtyping import Float

def get_batch(
        dataset: npt.NDArray, 
        batch_size: int, 
        context_length: int, 
        device: str,
        dtype: torch.dtype = torch.float16
) -> tuple[Float[torch.Tensor, "batch_size seq_len"],
           Float[torch.Tensor, "batch_size seq_len"]]:
    assert dataset.ndim == 1, "dataset must be 1D vector!"
    
    limit = len(dataset) - context_length
    assert limit > 0, "dataset must be longer than context length!"

    rng = np.random.default_rng()
    pos = rng.integers(0, limit, size= batch_size)

    seqs = torch.stack([torch.tensor(dataset[i:i + context_length], dtype= dtype, device= device) for i in pos])
    target_token = torch.stack([torch.tensor(dataset[i + 1:i + context_length + 1], dtype= dtype, device= device) for i in pos])

    return (seqs, target_token)