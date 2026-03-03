# nn.loss
# 对标torch.nn.loss
# 实现交叉熵损失函数
import einops
import torch
from jaxtyping import Float, Int
from typing import Callable
from cs336_basics.nn.activation import softmax

def CrossEntropyLoss(
            logits: Float[torch.Tensor, "... seq_len vocab_size"],
            target: Int[torch.Tensor, "... seq_len"],
            reduction: Callable = torch.mean
    ) -> Float[torch.Tensor, ""]:
    # l_i = -log softmax(o_i)[x_{i+1}]
    #     = -log (exp(o_i[x_{i+1}])/sum(exp(o_i[a])))
    #     = log (sum(exp(o_i[a]))) - o_i[x_{i+1}]

    logits = (logits - logits.amax(-1, keepdim= True))
    loss = logits.logsumexp(-1, keepdim= True) - logits.gather(-1, index= target.unsqueeze(-1))

    # torch.logsumexp calcs log sum exp(x)
    # torch.gather用index tensor的值替换索引依次取值再拼起来
    # unsqueeze后shape: "... seq_len 1"，与logsumexp(keepdim = True)的dim一致

    # 结果：loss: Float[torch.Tensor, "... seq_len 1"
    # 在 0..seq_len的每一个位置上有一个loss值

    # 考虑每个loss均作为一个example, 于是取global mean

    return reduction(loss)

def perplexity(
        logits: Float[torch.Tensor, "... seq_len vocab_size"],
        target: Int[torch.Tensor, "... seq_len"],
        batch_PLL: bool = True
) -> torch.Tensor:
    if batch_PLL:
        return CrossEntropyLoss(logits, target).exp()
    else:
        return CrossEntropyLoss(logits, target, reduction= lambda x : x).squeeze().mean(-1).exp()
