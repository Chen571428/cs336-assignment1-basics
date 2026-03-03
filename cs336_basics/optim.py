# optim 对标torch.nn.optim
# 实现SGD & AdamW Optimizer
import torch
import einops
import math 
from typing import Callable, Iterable, Optional, Any

class SGD(torch.optim.Optimizer):
    def __init__(
            self,
            params: Iterable,
            lr: float = 1e-3
    ) -> None:
        
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        super().__init__(
            params, 
            {"lr": lr}
        )

    @torch.no_grad()
    def step(
            self,
            closure: Optional[Callable] = None
    ) -> Any:
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                t = state.get("t", 0)

                p -= lr / math.sqrt(t + 1) * p.grad
                state["t"] = t + 1
        
        return loss

class SGDw(torch.optim.Optimizer):
    def __init__(
            self,
            params: Iterable,
            lr: float = 1e-3,
            weight_decay: float = 1e-2
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        super().__init__(
            params, 
            {
                "lr": lr,
                "weight_decay": weight_decay
            }
        )

    @torch.no_grad()
    def step(
            self,
            closure: Optional[Callable] = None
    ) -> Any:
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                t = state.get("t", 0)

                p.data -= lr / math.sqrt(t + 1) * p.grad
                p.data -= lr * weight_decay * p.data
                
                state["t"] = t + 1
        
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
            self,
            params: Iterable,
            lr: float = 1e-3,
            betas: tuple[float, float] = (0.9, 0.95),
            eps: float = 1e-8,
            weight_decay: float = 1e-2
    ) -> None:
        super().__init__(
            params= params,
            defaults={
                "lr" : lr,
                "betas" : betas,
                "eps": eps,
                "weight_decay": weight_decay
            }
        )

    @torch.no_grad()
    def step(
            self,
            closure: Optional[Callable] = None
    ) -> Any:
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
            for p in group["params"]:
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", 0)
                v = state.get("v", 0)

                m = betas[0] * m + (1 - betas[0]) * p.grad
                v = betas[1] * v + (1 - betas[1]) * (p.grad.pow(2))
                
                alpha = lr * math.sqrt(1 - (betas[1] ** t)) / (1 - (betas[0] ** t))

                p -= alpha * m / (v.sqrt() + eps) # Adam Updates
                p -= lr * weight_decay * p # Decoupled Weights Decay

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss

if __name__  == "__main__":
    all_results = []

    for lr in [1, 2, 5, 1e1, 2e1, 5e1, 1e2, 1e3]:
        torch.manual_seed(42)
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGDw([weights], lr=lr, weight_decay = 0.1)
        train_history = {}
        for t in range(20):
            opt.zero_grad() # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean() # Compute a scalar loss value.
            all_results.append({
                "Step": t, 
                "Loss": loss.cpu().item(), 
                "Learning Rate": f"LR = {lr}"
            })
            loss.backward() # Run backward pass, which computes gradients.
            opt.step() # Run optimizer step.

    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(all_results)

    p = sns.lineplot(data=df, x="Step", y="Loss", hue="Learning Rate", linewidth=2)
    p.set_title("Tuning Learning Rate for SGDw")
    p.set_ylim(0,30)

    plt.savefig("test_SGDw.svg")