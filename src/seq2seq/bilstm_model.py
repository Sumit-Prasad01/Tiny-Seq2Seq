"""
4-Layer Bidirectional LSTM (BiLSTM) Seq2Seq Architecture for Tiny-Seq2Seq.
Features:
- 4-layer cuDNN Bidirectional LSTM Encoder with packed sequence processing
- Per-layer non-linear bridge projection (Linear + Tanh) mapping concatenated
  forward and backward hidden states [h_fwd; h_bwd] and cell states [c_fwd; c_bwd]
  from 2 * d_model to d_model
- 4-layer cuDNN Unidirectional LSTM Decoder initialized directly from bridged states
- Tied target embedding and linear projection layer
- Uniform weight initialization in [-0.08, 0.08] matching Sutskever et al. (2014)
"""

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class BiLSTMEncoder(nn.Module):
    """
    4-layer cuDNN Bidirectional LSTM Encoder.
    Processes packed sequences to extract bidirectional hidden and cell states at the last non-pad token,
    then projects them via a layer-wise bridge into the unidirectional decoder space.
    """

    def __init__(
        self,
        vocab_size: int = 16000,
        d_model: int = 704,
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
        lstm_dropout = dropout if n_layers > 1 else 0.0

        # Bidirectional cuDNN LSTM
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

        # Non-linear bridge projecting concatenated [fwd; bwd] states (2 * d_model) -> d_model
        self.bridge_h = nn.ModuleList(
            [nn.Linear(2 * d_model, d_model) for _ in range(n_layers)]
        )
        self.bridge_c = nn.ModuleList(
            [nn.Linear(2 * d_model, d_model) for _ in range(n_layers)]
        )

    def forward(
        self, src_ids: torch.Tensor, src_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src_ids: [batch_size, max_src_len] (source tokens + EOS + PAD)
            src_lens: [batch_size] (true lengths including EOS)
        Returns:
            final_h: [n_layers, batch_size, d_model]
            final_c: [n_layers, batch_size, d_model]
        """
        embedded = self.dropout(self.embedding(src_ids))

        # Pack sequence to avoid cuDNN processing padding tokens
        cpu_lens = src_lens.to(torch.device("cpu"), dtype=torch.int64)
        packed = pack_padded_sequence(
            embedded, cpu_lens, batch_first=True, enforce_sorted=False
        )

        _, (h_n, c_n) = self.bilstm(packed)
        # h_n and c_n have shape: [2 * n_layers, batch_size, d_model]
        # In PyTorch cuDNN LSTM:
        # Layer l forward direction is index 2*l, backward direction is index 2*l + 1

        batch_size = src_ids.shape[0]

        # Reshape to [n_layers, 2, batch_size, d_model] -> [n_layers, batch_size, 2 * d_model]
        h_reshaped = (
            h_n.view(self.n_layers, 2, batch_size, self.d_model)
            .permute(0, 2, 1, 3)
            .reshape(self.n_layers, batch_size, 2 * self.d_model)
        )
        c_reshaped = (
            c_n.view(self.n_layers, 2, batch_size, self.d_model)
            .permute(0, 2, 1, 3)
            .reshape(self.n_layers, batch_size, 2 * self.d_model)
        )

        # Project through per-layer bridge
        bridged_h = torch.stack(
            [torch.tanh(self.bridge_h[l](h_reshaped[l])) for l in range(self.n_layers)],
            dim=0,
        )
        bridged_c = torch.stack(
            [torch.tanh(self.bridge_c[l](c_reshaped[l])) for l in range(self.n_layers)],
            dim=0,
        )

        return bridged_h, bridged_c


class BiLSTMDecoder(nn.Module):
    """
    4-layer cuDNN Unidirectional LSTM Decoder.
    Initial recurrent states (h, c) are directly initialized from the BiLSTM Encoder bridged states.
    """

    def __init__(
        self,
        vocab_size: int = 16000,
        d_model: int = 704,
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
        lstm_dropout = dropout if n_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt_in_ids: torch.Tensor,
        encoder_state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Teacher-forced forward pass.
        Args:
            tgt_in_ids: [batch_size, max_tgt_len] (BOS + target tokens + PAD)
            encoder_state: (h, c) where each is [n_layers, batch_size, d_model]
        Returns:
            outputs: [batch_size, max_tgt_len, d_model]
            (final_h, final_c): each [n_layers, batch_size, d_model]
        """
        embedded = self.dropout(self.embedding(tgt_in_ids))
        outputs, (final_h, final_c) = self.lstm(embedded, encoder_state)
        outputs = self.dropout(outputs)
        return outputs, (final_h, final_c)

    def step(
        self,
        token_id: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Single-step decoding for autoregressive beam search.
        Args:
            token_id: [batch_size, 1] or [batch_size]
            state: (h, c) each of shape [n_layers, batch_size, d_model]
        Returns:
            output: [batch_size, 1, d_model]
            new_state: (new_h, new_c) each of shape [n_layers, batch_size, d_model]
        """
        if token_id.dim() == 1:
            token_id = token_id.unsqueeze(1)
        embedded = self.embedding(token_id)
        output, new_state = self.lstm(embedded, state)
        return output, new_state


class BiLSTMSeq2SeqModel(nn.Module):
    """
    Complete 4-Layer Bidirectional LSTM Sequence-to-Sequence Model.
    Features:
    - 4-layer cuDNN BiLSTM Encoder with layer-wise bridge projection
    - 4-layer cuDNN Unidirectional LSTM Decoder
    - Tied target embedding and linear projection layer
    - Uniform initialization in [-0.08, 0.08]
    """

    def __init__(
        self,
        vocab_size: int = 16000,
        d_model: int = 704,
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

        self.encoder = BiLSTMEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            padding_idx=padding_idx,
        )

        self.decoder = BiLSTMDecoder(
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
            src_ids: [batch_size, max_src_len] (source tokens)
            src_lens: [batch_size]
            tgt_in_ids: [batch_size, max_tgt_len] (BOS + target)
        Returns:
            logits: [batch_size, max_tgt_len, vocab_size]
        """
        enc_h, enc_c = self.encoder(src_ids, src_lens)
        dec_outputs, _ = self.decoder(tgt_in_ids, (enc_h, enc_c))
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
            "model_type": "bilstm",
            "n_layers": self.n_layers,
            "d_model": self.d_model,
        }
