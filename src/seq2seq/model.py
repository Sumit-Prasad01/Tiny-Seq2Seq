"""
Seq2Seq Architecture for Tiny-Seq2Seq (Sutskever et al. 2014).
Scaled-down reproduction:
- 3-layer cuDNN LSTM Encoder with packed sequences
- 3-layer cuDNN LSTM Decoder initialized from Encoder final state
- Source reversal support
- Tied target embedding and linear projection layer (~29.0M params tied, ~37.2M untied)
- Uniform weight initialization in [-0.08, 0.08]
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Seq2SeqEncoder(nn.Module):
    """
    3-layer cuDNN LSTM Encoder.
    Processes packed sequences to ensure hidden/cell states are extracted
    at the last non-pad token of each sentence.
    """

    def __init__(
        self,
        vocab_size: int = 16000,
        d_model: int = 512,
        n_layers: int = 3,
        dropout: float = 0.2,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        # Dropout between LSTM layers (cuDNN requires n_layers > 1 for dropout)
        lstm_dropout = dropout if n_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, src_ids: torch.Tensor, src_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src_ids: [batch_size, max_src_len] (reversed source tokens + EOS + PAD)
            src_lens: [batch_size] (true lengths including EOS)
        Returns:
            final_h: [n_layers, batch_size, d_model]
            final_c: [n_layers, batch_size, d_model]
        """
        # [batch_size, max_src_len, d_model]
        embedded = self.dropout(self.embedding(src_ids))

        # Pack sequence to avoid cuDNN processing padding tokens
        # Ensure lengths are on CPU for pack_padded_sequence
        cpu_lens = src_lens.to(torch.device("cpu"), dtype=torch.int64)
        packed = pack_padded_sequence(
            embedded, cpu_lens, batch_first=True, enforce_sorted=False
        )

        _, (final_h, final_c) = self.lstm(packed)

        return final_h, final_c


class Seq2SeqDecoder(nn.Module):
    """
    3-layer cuDNN LSTM Decoder.
    Initial state is directly assigned from the Encoder's final state (h, c).
    """

    def __init__(
        self,
        vocab_size: int = 16000,
        d_model: int = 512,
        n_layers: int = 3,
        dropout: float = 0.2,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        lstm_dropout = dropout if n_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt_in_ids: torch.Tensor,
        encoder_states: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Teacher-forced forward pass.
        Args:
            tgt_in_ids: [batch_size, max_tgt_len] (BOS + target tokens + PAD)
            encoder_states: (h, c) from encoder, shape [n_layers, batch_size, d_model]
        Returns:
            outputs: [batch_size, max_tgt_len, d_model]
            (h, c): final decoder hidden states
        """
        embedded = self.dropout(self.embedding(tgt_in_ids))
        outputs, states = self.lstm(embedded, encoder_states)
        outputs = self.dropout(outputs)
        return outputs, states

    def step(
        self,
        token_id: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Single-step decoding for autoregressive beam search.
        Args:
            token_id: [batch_size, 1] or [batch_size]
            state: (h, c) of shape [n_layers, batch_size, d_model]
        Returns:
            output: [batch_size, 1, d_model]
            new_state: (h, c)
        """
        if token_id.dim() == 1:
            token_id = token_id.unsqueeze(1)
        embedded = self.embedding(token_id)
        output, new_state = self.lstm(embedded, state)
        return output, new_state


class Seq2SeqModel(nn.Module):
    """
    Complete Sequence-to-Sequence model replicating Sutskever et al. (2014).
    Features:
    - Separate Encoder and Decoder 3-layer LSTMs
    - Optional tied output projection to target embedding
    - Uniform initialization in [-0.08, 0.08]
    """

    def __init__(
        self,
        vocab_size: int = 16000,
        d_model: int = 512,
        n_layers: int = 3,
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

        self.encoder = Seq2SeqEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            padding_idx=padding_idx,
        )

        self.decoder = Seq2SeqDecoder(
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
        """Initializes all parameters uniformly in [-bound, bound] as in the paper."""
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
            src_ids: [batch_size, max_src_len] (reversed)
            src_lens: [batch_size]
            tgt_in_ids: [batch_size, max_tgt_len] (BOS + target)
        Returns:
            logits: [batch_size, max_tgt_len, vocab_size]
        """
        enc_h, enc_c = self.encoder(src_ids, src_lens)
        dec_outputs, _ = self.decoder(tgt_in_ids, (enc_h, enc_c))
        logits = self.fc_out(dec_outputs)
        return logits

    def count_parameters(self) -> Dict[str, int]:
        """Calculates total and unique trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        # Track unique parameter pointers to avoid double-counting tied weights
        unique_params = {p.data_ptr(): p.numel() for p in self.parameters() if p.requires_grad}
        trainable_unique = sum(unique_params.values())

        return {
            "total_parameters": total_params,
            "trainable_unique_parameters": trainable_unique,
            "tied_weights": self.tie_weights,
        }
