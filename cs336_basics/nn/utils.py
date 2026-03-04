# nn.utils
# 对标torch.nn.utils
# 实现梯度裁切等操作
from typing import Iterable
import torch
import os
import typing

@torch.no_grad
def GradientClipping(
        parameters: Iterable[torch.nn.Parameter], 
        max_l2_norm: float,
        eps: float = 1e-6
) -> None:
    params_grad = torch.cat([p.grad for p in parameters if p.grad is not None])
    l2_norm = torch.linalg.vector_norm(params_grad)
    # Vector Norm会把忽略任何额外维度把矩阵展平为1D向量再计算
    # 或使用Frobenius 范数 定义为矩阵元素平方和的平方根，即L2 Norm
    if l2_norm > max_l2_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad *= max_l2_norm / (l2_norm + eps)

def save_chectpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
) -> None:
    torch.save(
        {
            "params": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "iteration": iteration,
        },
        out
    )

def load_checkpoint(
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer
) -> int:
    ckpt = torch.load(src)
    model.load_state_dict(ckpt["params"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["iteration"]