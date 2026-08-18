import torch
import numpy as np

# ------------------------------------ VIRTUAL TOKEN PROJECTOR MODULE -----------------------------------------
class GenomicVirtualTokenProjector(torch.nn.Module):
    """
    Projects input feature vector (sequence/structure embedding) into a set of learnable 'virtual' token embeddings.
    Used to prepend modality info before language tokens for LLM Q&A.
    """

    def __init__(self, embedding_dim, llama_hidden_size, num_virtual_tokens, dropout=0.1):
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        self.llama_hidden_size = llama_hidden_size
        self.project = torch.nn.Linear(
            embedding_dim, num_virtual_tokens * llama_hidden_size
        )
        self.dropout = torch.nn.Dropout(dropout)
        # Optional "null" embedding for missing or invalid features
        self.null_emb = torch.nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, emb=None, batch_size=1):
        """
        Accepts None (for missing) or vector input (numpy or torch).
        Outputs [batch_size, num_virtual_tokens, llama_hidden_size].
        """
        if emb is None:
            emb = self.null_emb.unsqueeze(0).repeat(batch_size, 1)
        elif isinstance(emb, np.ndarray):
            emb = torch.tensor(emb, dtype=torch.float32, device=self.null_emb.device)
        else:
            emb = emb.to(dtype=torch.float32, device=self.null_emb.device)
        projected = self.project(emb)
        projected = self.dropout(projected)  
        return projected.view(-1, self.num_virtual_tokens, self.llama_hidden_size)
