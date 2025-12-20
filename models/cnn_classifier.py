"""CNN-based classifier for news tag prediction."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class CNNClassifier(nn.Module):
    """
    CNN-based classifier using 1D convolutions for sequence processing.
    
    Uses convolutional layers to extract features from title and snippet,
    then combines them for classification.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        output_dim: int,
        max_title_len: int,
        max_snippet_len: int,
        conv_channels: list[int] = [128, 256],
        kernel_sizes: list[int] = [3, 3],
        pool_size: int = 2,
        hidden_dim: int = 600,
        dropout: float = 0.3,
    ):
        """
        Initialize CNN classifier.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            output_dim: Number of output classes (tags)
            max_title_len: Maximum title sequence length
            max_snippet_len: Maximum snippet sequence length
            conv_channels: List of output channels for each conv layer
            kernel_sizes: List of kernel sizes for each conv layer
            pool_size: Max pooling kernel size
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
            
        Example:
            >>> model = CNNClassifier(
            ...     vocab_size=10000,
            ...     embedding_dim=300,
            ...     output_dim=1000,
            ...     max_title_len=20,
            ...     max_snippet_len=50
            ... )
        """
        super().__init__()
        
        if len(conv_channels) != len(kernel_sizes):
            raise ValueError(
                f"conv_channels and kernel_sizes must have same length. "
                f"Got {len(conv_channels)} and {len(kernel_sizes)}"
            )
        
        self.embedding_dim = embedding_dim
        self.max_title_len = max_title_len
        self.max_snippet_len = max_snippet_len
        self.pool_size = pool_size
        
        # Embedding layers
        self.title_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.snippet_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Title CNN layers
        self.title_conv_layers = nn.ModuleList()
        in_channels = embedding_dim
        for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
            self.title_conv_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
            )
            in_channels = out_channels
        
        # Snippet CNN layers
        self.snippet_conv_layers = nn.ModuleList()
        in_channels = embedding_dim
        for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
            self.snippet_conv_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
            )
            in_channels = out_channels
        
        # Max pooling
        self.maxpool = nn.MaxPool1d(pool_size)
        
        # Calculate feature size after convolutions and pooling
        title_seq_len = self._calculate_output_length(max_title_len, len(conv_channels))
        snippet_seq_len = self._calculate_output_length(max_snippet_len, len(conv_channels))
        
        title_feat_size = conv_channels[-1] * title_seq_len
        snippet_feat_size = conv_channels[-1] * snippet_seq_len
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(title_feat_size + snippet_feat_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        
        logger.info(
            f"Initialized CNNClassifier: vocab_size={vocab_size}, "
            f"embedding_dim={embedding_dim}, output_dim={output_dim}, "
            f"title_feat_size={title_feat_size}, snippet_feat_size={snippet_feat_size}"
        )

    def _calculate_output_length(
        self,
        input_length: int,
        num_conv_layers: int
    ) -> int:
        """
        Calculate output sequence length after convolutions and pooling.
        
        Args:
            input_length: Input sequence length
            num_conv_layers: Number of convolution layers
            
        Returns:
            Output sequence length
        """
        length = input_length
        for _ in range(num_conv_layers):
            # MaxPool1d with kernel_size=2 reduces length by factor of 2
            length = length // self.pool_size
        return max(1, length)  # Ensure at least 1

    def _apply_cnn(
        self,
        embedded: torch.Tensor,
        conv_layers: nn.ModuleList
    ) -> torch.Tensor:
        """
        Apply CNN layers to embedded sequence.
        
        Args:
            embedded: Embedded sequence [batch_size, seq_len, embedding_dim]
            conv_layers: List of convolution layers
            
        Returns:
            Flattened feature vector [batch_size, feature_size]
        """
        # Permute to [batch_size, embedding_dim, seq_len] for Conv1d
        x = embedded.permute(0, 2, 1)
        
        # Apply convolutions and pooling
        for conv_layer in conv_layers:
            x = F.relu(conv_layer(x))
            x = self.maxpool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        return x

    def forward(
        self,
        title: torch.Tensor,
        snippet: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            title: Title token indices [batch_size, seq_len]
            snippet: Snippet token indices [batch_size, seq_len]
            
        Returns:
            Logits [batch_size, output_dim]
        """
        # Embed
        title_embedded = self.title_embedding(title)  # [batch_size, title_len, embedding_dim]
        snippet_embedded = self.snippet_embedding(snippet)  # [batch_size, snippet_len, embedding_dim]
        
        # Apply CNN
        title_feat = self._apply_cnn(title_embedded, self.title_conv_layers)
        snippet_feat = self._apply_cnn(snippet_embedded, self.snippet_conv_layers)
        
        # Concatenate features
        combined = torch.cat([title_feat, snippet_feat], dim=1)
        
        # Classify
        out = self.classifier(combined)
        
        return out

