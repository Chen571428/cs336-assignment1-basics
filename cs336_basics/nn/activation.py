# nn.activation
import torch
def softmax(
        x: torch.Tensor,
        i: int
) -> torch.Tensor:
    """softmax activation function which scales every elements to (0,1), while the sum is 1.

    Args:
        x (torch.Tensor): input tensor
        i (int): the dimension to apply softmax

    Returns:
        torch.Tensor: The tensor with softmax applied at dim i.
    """
    x = (x - torch.amax(x, i, keepdim= True)).exp()
    x = x / x.sum(i, keepdim= True)
    return x