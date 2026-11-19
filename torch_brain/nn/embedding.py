import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Embedding):
    r"""A simple extension of :class:`torch.nn.Embedding` to allow more control over
    the weights initializer. The learnable weights of the module of shape
    `(num_embeddings, embedding_dim)` are initialized from
    :math:`\mathcal{N}(0, \text{init_scale})`.

    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        init_scale (float, optional): standard deviation of the normal distribution used
            for the initialization. Defaults to 0.02, which is the default value used in
            most transformer models
        **kwargs: Additional arguments. Refer to the documentation of
            :class:`torch.nn.Embedding` for details
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        init_scale: float = 0.02,
        **kwargs,
    ):
        self.init_scale = init_scale
        super().__init__(num_embeddings, embedding_dim, **kwargs)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        torch.nn.init.normal_(self.weight, mean=0, std=self.init_scale)
        self._fill_padding_idx_with_zero()

    # Vocabularies at or below this size use a one-hot matmul on Apple Silicon (MPS).
    _MPS_ONE_HOT_MAX_VOCAB = 64

    def forward(self, input):
        if (
            input.device.type == "mps"
            and self.num_embeddings <= self._MPS_ONE_HOT_MAX_VOCAB
            and self.padding_idx is None
            and self.max_norm is None
        ):
            # On MPS, the embedding backward pass scatter-adds into a handful of rows
            # and serializes on atomics (e.g. token-type embeddings: 4 rows, one lookup
            # per spike -> ~175 ms per step). A one-hot matmul computes the exact same
            # function, and its backward is a fast, contention-free matmul (~5 ms).
            one_hot = F.one_hot(input, self.num_embeddings).to(self.weight.dtype)
            return one_hot @ self.weight
        return super().forward(input)
