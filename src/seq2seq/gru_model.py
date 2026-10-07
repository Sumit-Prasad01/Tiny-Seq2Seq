"""
4-Layer Gated Recurrent Unit (GRU) Seq2Seq Architecture.
Companion model to Sutskever et al. (2014) LSTM:
- 4-layer cuDNN GRU Encoder with packed sequences
- 4-layer cuDNN GRU Decoder directly initialized from Encoder final state
- Single recurrent state tensor h (no cell state c), saving 50% state memory
- Source reversal support
- Tied target embedding and linear projection layer
- Uniform weight initialization in [-0.08, 0.08]
"""

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class GRUEncoder(nn.Module):
    """
    4-layer cuDNN GRU Encoder.
    Processes packed sequences to extract hidden state at the last non-pad token.
    """

    def __init__(
        self,
        vocab_size: int = 16000,
        d_model: int = 896,
        n_layers: int = 4,
        dropout: float = 0.2,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        gru_dropout = dropout if n_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, src_ids: torch.Tensor, src_lens: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            src_ids: [batch_size, max_src_len] (reversed source tokens + EOS + PAD)
            src_lens: [batch_size] (true lengths including EOS)
        Returns:
            final_h: [n_layers, batch_size, d_model]
        """
        embedded = self.dropout(self.embedding(src_ids))

        # Pack sequence to avoid cuDNN processing padding tokens
        cpu_lens = src_lens.to(torch.device("cpu"), dtype=torch.int64)
        packed = pack_padded_sequence(
            embedded, cpu_lens, batch_first=True, enforce_sorted=False
        )

        _, final_h = self.gru(packed)
        return final_h


class GRUDecoder(nn.Module):
    """
    4-layer cuDNN GRU Decoder.
    Initial state is directly assigned from the Encoder's final hidden state h.
    """

    def __init__(
        self,
        vocab_size: int = 16000,
        d_model: int = 896,
        n_layers: int = 4,
        dropout: float = 0.2,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        gru_dropout = dropout if n_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt_in_ids: torch.Tensor,
        encoder_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Teacher-forced forward pass.
        Args:
            tgt_in_ids: [batch_size, max_tgt_len] (BOS + target tokens + PAD)
            encoder_state: h from encoder, shape [n_layers, batch_size, d_model]
        Returns:
            outputs: [batch_size, max_tgt_len, d_model]
            final_h: [n_layers, batch_size, d_model]
        """
        embedded = self.dropout(self.embedding(tgt_in_ids))
        outputs, final_h = self.gru(embedded, encoder_state)
        outputs = self.dropout(outputs)
        return outputs, final_h

    def step(
        self,
        token_id: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step decoding for autoregressive beam search.
        Args:
            token_id: [batch_size, 1] or [batch_size]
            state: h of shape [n_layers, batch_size, d_model]
        Returns:
            output: [batch_size, 1, d_model]
            new_state: [n_layers, batch_size, d_model]
        """
        if token_id.dim() == 1:
            token_id = token_id.unsqueeze(1)
        embedded = self.embedding(token_id)
        output, new_state = self.gru(embedded, state)
        return output, new_state


class GRUSeq2SeqModel(nn.Module):
    """
    Complete 4-Layer GRU Sequence-to-Sequence Model.
    Features:
    - 4-layer cuDNN GRU Encoder and Decoder
    - Tied target embedding and linear projection layer
    - Uniform initialization in [-0.08, 0.08]
    """

    def __init__(
        self,
        vocab_size: int = 16000,
        d_model: int = 896,
        n_layers: int = 4,
        dropout: float = 0.2,
        tie_weights: bool = True,
        init_uniform_bound: float = 0.08,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.dropout_rate = dropout
        self.tie_weights = tie_weights
        self.padding_idx = padding_idx

        self.encoder = GRUEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            padding_idx=padding_idx,
        )

        self.decoder = GRUDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            padding_idx=padding_idx,
        )

        self.fc_out = nn.Linear(d_model, vocab_size, bias=True)

        if tie_weights:
            # Tie output projection weights with target embedding weights
            self.fc_out.weight = self.decoder.embedding.weight

        self.init_weights(bound=init_uniform_bound)

    def init_weights(self, bound: float = 0.08) -> None:
        """Initializes all parameters uniformly in [-bound, bound]."""
        for name, param in self.named_parameters():
            if param.requires_grad:
                nn.init.uniform_(param.data, -bound, bound)

        # Ensure padding embeddings stay at zero
        with torch.no_grad():
            self.encoder.embedding.weight[self.padding_idx].fill_(0)
            self.decoder.embedding.weight[self.padding_idx].fill_(0)

    def forward(
        self,
        src_ids: torch.Tensor,
        src_lens: torch.Tensor,
        tgt_in_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Complete forward pass.
        Args:
            src_ids: [batch_size, max_src_len] (reversed source)
            src_lens: [batch_size]
            tgt_in_ids: [batch_size, max_tgt_len] (BOS + target)
        Returns:
            logits: [batch_size, max_tgt_len, vocab_size]
        """
        enc_h = self.encoder(src_ids, src_lens)
        dec_outputs, _ = self.decoder(tgt_in_ids, enc_h)
        logits = self.fc_out(dec_outputs)
        return logits

    def count_parameters(self) -> Dict[str, Any]:
        """Calculates total and unique trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        unique_params = {p.data_ptr(): p.numel() for p in self.parameters() if p.requires_grad}
        trainable_unique = sum(unique_params.values())

        return {
            "total_parameters": total_params,
            "trainable_unique_parameters": trainable_unique,
            "tied_weights": self.tie_weights,
            "model_type": "gru",
            "n_layers": self.n_layers,
            "d_model": self.d_model,
        }
