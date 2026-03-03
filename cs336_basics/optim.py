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
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        
        return loss

class SGDw(torch.optim.Optimizer):
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
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        
        return loss


if __name__  == "__main__":
    all_results = []

    for lr in [1, 2, 5, 1e1, 2e1, 5e1, 1e2, 1e3]:
        torch.manual_seed(42)
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([weights], lr=lr)
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
    p.set_title("Tuning Learning Rate for SGD")
    p.set_ylim(0,30)

    plt.savefig("test_SGD.svg")