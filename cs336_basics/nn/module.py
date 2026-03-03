import einops
import torch
import math
from jaxtyping import Float, Bool, Int
from typing import Optional
from cs336_basics.nn.activation import softmax

class Linear(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device: Optional[torch.device]= None,
            dtype: Optional[torch.dtype] = None
    ) -> None:
        """linear transformation module.

        Args:
            in_features (int): dimension of the input
            out_features (int): dimension of the output
            device (torch.device | None, optional): Device to store the parameters on. Defaults to None.
            dtype (torch.dtype | None, optional): Data type of the parameters. Defaults to None.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.nn.Parameter(
            torch.empty(
                (out_features, in_features),
                device= device, 
                dtype= dtype
            )
        )

        self.reset_parameters()

    def forward(
            self,
            x: Float[torch.Tensor, "... in_features"]
    ) -> Float[torch.Tensor, "... out_features"]:
        """Apply the linear transformation to the input

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Transformed tensor
        """
        x = einops.einsum(x, self.weight, "... in_feat, out_feat in_feat -> ... out_feat")

        return x
    
    def reset_parameters(self) -> None:
        std = math.sqrt(2.0 / (self.in_features + self.out_features))

        # truncate at +-3 * sigma
        torch.nn.init.trunc_normal_(
            self.weight,
            mean= 0, 
            std= std, a= -3 * std, b= 3 * std
        )

class Embedding(torch.nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
    ) -> None:
        """The Embedding layer, embedding each token id to an embedding_dim vector.

        Args:
            num_embeddings (int): i.e. vocab_size
            embedding_dim (int): i.e. d_{model}
            device (torch.device | None, optional): Defaults to None.
            dtype (torch.dtype | None, optional): Defaults to None.
        """
        super().__init__()

        self.weight = torch.nn.Parameter(
            torch.empty(
                (num_embeddings, embedding_dim),
                device= device,
                dtype= dtype
            )
        )

        self.reset_parameters()

    def forward(
            self,
            token_ids: Int[torch.Tensor, "... seq_len"]
    ) -> Float[torch.Tensor, "... seq_len embedding_dim"]:
        return self.weight[token_ids]
    
    def reset_parameters(self) -> None:
        torch.nn.init.trunc_normal_(
            self.weight,
            mean = 0,
            std = 1,
            a= -3, b = 3
        )

class RMSNorm(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            eps: float,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
    ) -> None:
        
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.ones(
                d_model,
                device= device,
                dtype= dtype
            )
        )

    def forward(
            self,
            x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        
        in_dtype =  x.dtype
        x = x.to(torch.float32)
        
        x = x * (x.square().mean(-1, keepdim=True) + self.eps).rsqrt() * self.weight

        return x.to(in_dtype)

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

class SiLU(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
            self,
            x: torch.Tensor
    ):
        return x * torch.sigmoid(x)

class ReLU(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
            self,
            x: torch.Tensor
    ):
        return torch.clamp(x, min=0.0)

class SwiGLU(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
    ) -> None:
        super().__init__()
    
        # d_ff approximately 8/3 * d_model and is a multiple of 64
        # d_ff = (8 * d_model) // 3
        # d_ff += (64 - d_ff % 64)

        self.w1 = Linear(d_model, d_ff, device, dtype)
        # up_proj
        self.w2 = Linear(d_ff, d_model, device, dtype)
        # down_proj
        self.w3 = Linear(d_model, d_ff, device, dtype)
        # gate_proj
        self.silu = SiLU()

    def forward(
            self,
            x: Float[torch.Tensor, "... d_model"]    
    ) -> Float[torch.Tensor, "... d_model"]:
        return self.w2(self.silu(self.w1(x)) * self.w3(x))

class RoPE(torch.nn.Module):
    def __init__(
            self,
            theta: float,
            d_k: int,
            max_seq_len: int,
            device: Optional[torch.device] = None
    ) -> None:
        assert d_k % 2 == 0, "d_k must be even."
        super().__init__()

        thetas = einops.einsum(
            torch.arange(
                max_seq_len, 
                dtype=torch.float
            ),
            torch.pow(
                theta,
                -torch.arange(0, d_k, 2, dtype= torch.float) / d_k
                # Theta^{-k / half_d_model}
            ),
            "seq_len, half_d_model -> seq_len half_d_model"
        )


        self.cos: Float[torch.Tensor, "seq_len half_d_model"] = torch.nn.Buffer(
            torch.cos(thetas),
            persistent= False
        )

        self.sin: Float[torch.Tensor, "seq_len half_d_model"] = torch.nn.Buffer(
            torch.sin(thetas),
            persistent= False
        )

    def forward(
            self,
            x: Float[torch.Tensor, "... seq_len d_k"],
            token_positions: Optional[Int[torch.Tensor, "... seq_len"]] = None
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        seq_len = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], dtype= torch.int)
        assert seq_len == token_positions.shape[-1], "seq_len of x and token_positions must match."

        even_x = x[..., 0::2]
        odd_x = x[..., 1::2]

        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        out_even_x = even_x * cos - odd_x * sin
        out_odd_x = odd_x * cos + even_x * sin

        result = torch.empty_like(x)
        result[..., 0::2] = out_even_x
        result[..., 1::2] = out_odd_x

        return result
    

def scaled_dot_product_attention(
        q: Float[torch.Tensor, "... N d_k"],
        k: Float[torch.Tensor, "... M d_k"],
        v: Float[torch.Tensor, "... M d_v"],
        mask: Optional[Bool[torch.Tensor, "N M"]] = None
) -> Float[torch.Tensor, "... N d_v"]:
    d_k = q.shape[-1]
    assert d_k == k.shape[-1], "d_k matches d_k."

    o = einops.einsum(q / math.sqrt(d_k), k, "... N d_k, ... M d_k -> ... N M")
    if mask is not None:
        o = torch.where(mask, o, float("-inf"))
    
    o = softmax(o, -1)

    return einops.einsum(o, v, "... N M,... M d_v -> ... N d_v")

    

class MHA(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_k: Optional[int] = None,
            d_v: Optional[int] = None,
            rope: Optional[RoPE] = None,
            use_pack_proj: bool = True
    ) -> None:
        super().__init__()
        
        self.num_heads = num_heads
        if d_k is None:
            self.d_k = d_model // num_heads
            assert d_model % num_heads == 0, "if set d_k to be d_model // num_heads, then the rem must be zero"
        else:
            self.d_k = d_k
        
        if d_v is None:
            self.d_v = d_model // num_heads
        else:
            self.d_v = d_v
        
        self.use_pack_proj = use_pack_proj
        if self.use_pack_proj:
            self.q_k_v_proj = Linear(d_model, num_heads * (self.d_k * 2 + self.d_v))
        
        else:
            self.q_proj = Linear(d_model, num_heads * self.d_k)
            self.k_proj = Linear(d_model, num_heads * self.d_k)
            self.v_proj = Linear(d_model, num_heads * self.d_v)
        
        self.output_proj = Linear(num_heads * self.d_v, d_model)
        
        self.rope = rope

    def forward(
            self,
            h: Float[torch.Tensor, "... seq_len d_model"],
            is_casual: bool = True,
            mask: Optional[Bool[torch.Tensor, "seq_len seq_len"]] = None,
            token_positions: Optional[torch.Tensor] = None
    ) -> Float[torch.Tensor, "... seq_len d_model"]:
        seq_len = h.shape[-2]

        if is_casual:
            mask = torch.tril(
                torch.ones(seq_len, seq_len)
            ).bool()

        if self.use_pack_proj:
            q, k, v = torch.split(
                self.q_k_v_proj(h), 
                [self.num_heads * self.d_k, 
                 self.num_heads * self.d_k, 
                 self.num_heads * self.d_v],
                 dim= -1
            )
        else:
            q = self.q_proj(h)
            k = self.k_proj(h)
            v = self.v_proj(h)
        
        q = einops.rearrange(q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads= self.num_heads)
        k = einops.rearrange(k, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads= self.num_heads)
        v = einops.rearrange(v, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads= self.num_heads)

        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        return self.output_proj(
            einops.rearrange(
                scaled_dot_product_attention(q, k, v, mask),
                "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)"
            )
        )

class TransformerBlock(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            eps: float = 1e-5,
            rope: Optional[RoPE] = None
    ) -> None:
        super().__init__()

        self.ln1 = RMSNorm(d_model, eps)
        self.attn = MHA(d_model, num_heads, rope= rope)
        self.ln2 = RMSNorm(d_model, eps)
        self.ffn = SwiGLU(d_model, d_ff)
        
    def forward(
            self,
            h: Float[torch.Tensor, "... seq_len d_model"]
    ) -> Float[torch.Tensor, "... seq_len d_model"]:
        h = h + self.attn(self.ln1(h))
        h = h + self.ffn(self.ln2(h))
        return h
    
class TransformerLM(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            vocab_size: int,
            context_length: int,
            num_layers: int,
            rope_theta: float,
            eps: float = 1e-5,

    ) -> None:
        super().__init__()

        self.rope = RoPE(rope_theta, d_model // num_heads, context_length)

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = torch.nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff, eps= eps, rope= self.rope)
            for _ in range(num_layers)
        )
        self.ln_final = RMSNorm(d_model, eps)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(
            self,
            in_indices: Int[torch.Tensor, "batch_size seq_len"]
    ) -> Float[torch.Tensor, "batch_size seq_len vocab_size"]:
        h = self.token_embeddings(in_indices)

        for layer in self.layers:
            h = layer(h)
        h = self.ln_final(h)
        h = self.lm_head(h)

        return h
    
