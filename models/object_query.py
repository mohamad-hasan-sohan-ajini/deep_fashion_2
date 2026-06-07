import torch
from torch import nn


class ObjectQuery(nn.Module):
    """Learnable object query vectors."""

    def __init__(self, embedding_dim: int, num_queries: int) -> None:
        super().__init__()

        self.num_queries = num_queries
        self.embedding_dim = embedding_dim
        self.query_vectors = nn.Parameter(torch.empty(num_queries, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query_vectors, mean=0.0, std=0.01)

    @property
    def vectors(self) -> torch.Tensor:
        return self.query_vectors

    def forward(self) -> torch.Tensor:
        """Return query vectors as ``[num_queries, dim]`` or ``[batch, num_queries, dim]``."""
        return self.query_vectors


if __name__ == "__main__":
    # Example usage
    embedding_dim = 256
    num_queries = 100

    object_query = ObjectQuery(embedding_dim=embedding_dim, num_queries=num_queries)
    query_vectors = object_query()
    print("Query vectors shape:", query_vectors.shape)
