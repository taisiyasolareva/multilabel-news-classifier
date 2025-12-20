"""Simple neural network classifier for news tag prediction."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SimpleClassifier(nn.Module):
    """
    Simple embedding-based classifier for multi-label news tag classification.
    
    Supports both title-only and title+snippet modes.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        output_dim: int,
        max_title_len: Optional[int] = None,
        max_snippet_len: Optional[int] = None,
        use_snippet: bool = False,
    ):
        """
        Initialize classifier.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            output_dim: Number of output classes (tags)
            max_title_len: Maximum title length (for snippet mode)
            max_snippet_len: Maximum snippet length (for snippet mode)
            use_snippet: Whether to use snippets in addition to titles
            
        Example:
            >>> model = SimpleClassifier(
            ...     vocab_size=10000,
            ...     embedding_dim=300,
            ...     output_dim=1000,
            ...     use_snippet=True
            ... )
        """
        super().__init__()
        self.use_snippet = use_snippet
        
        # Title embedding
        self.title_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        if use_snippet:
            # Snippet embedding (separate from title)
            self.snippet_embedding = nn.Embedding(vocab_size, embedding_dim)
            # Combined classifier
            self.linear1 = nn.Linear(embedding_dim * 2, 600)
            self.linear2 = nn.Linear(600, output_dim)
        else:
            # Title-only classifier
            self.fc = nn.Linear(embedding_dim, output_dim)
        
        logger.info(
            f"Initialized SimpleClassifier: vocab_size={vocab_size}, "
            f"embedding_dim={embedding_dim}, output_dim={output_dim}, "
            f"use_snippet={use_snippet}"
        )

    def forward(
        self,
        title: torch.Tensor,
        snippet: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            title: Title token indices [batch_size, seq_len]
            snippet: Optional snippet token indices [batch_size, seq_len]
            
        Returns:
            Logits [batch_size, output_dim]
        """
        # Embed and average title
        title_embedded = self.title_embedding(title)
        title_embedded = title_embedded.mean(dim=1)  # [batch_size, embedding_dim]
        
        if self.use_snippet and snippet is not None:
            # Embed and average snippet
            snippet_embedded = self.snippet_embedding(snippet)
            snippet_embedded = snippet_embedded.mean(dim=1)  # [batch_size, embedding_dim]
            
            # Concatenate and classify
            combined = torch.cat((title_embedded, snippet_embedded), dim=1)
            out = F.relu(self.linear1(combined))
            out = self.linear2(out)
        else:
            # Title-only classification
            out = self.fc(title_embedded)
        
        return out

