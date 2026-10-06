"""
Vectorized GPU Beam Search Decoder for Tiny-Seq2Seq.
Implements:
- Batched GPU beam search for beam sizes K in {1, 2, 5, 12}
- Step 0 duplicate suppression (single live beam initialization)
- LSTM hidden and cell state permutation across surviving beams
- Length penalty normalization: ((5 + len) / 6) ^ alpha
- Early stopping when all hypotheses in a batch emit <eos>
- Full SentencePiece detokenization
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from src.seq2seq.data.tokenizer_manager import TokenizerManager
from src.seq2seq.model import Seq2SeqModel


class BeamSearchDecoder:
    """
    Vectorized CUDA Beam Search Decoder.
    Decodes entire batches concurrently with state reordering on GPU.
    """

    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3

    def __init__(
        self,
        model: Seq2SeqModel,
        tokenizer: Optional[TokenizerManager] = None,
        beam_size: int = 5,
        length_penalty_alpha: float = 0.6,
        max_decoding_len_ratio: float = 1.5,
        max_decoding_len_const: int = 10,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.alpha = length_penalty_alpha
        self.len_ratio = max_decoding_len_ratio
        self.len_const = max_decoding_len_const

    def compute_length_penalty(self, lengths: torch.Tensor) -> torch.Tensor:
        """
        Computes length penalty factor:
        lp(Y) = ((5 + |Y|) / 6) ^ alpha
        """
        if self.alpha == 0.0:
            return torch.ones_like(lengths, dtype=torch.float32)
        return ((5.0 + lengths.float()) / 6.0) ** self.alpha

    @torch.no_grad()
    def decode_batch(
        self,
        src_ids: torch.Tensor,
        src_lens: torch.Tensor,
        beam_size: Optional[int] = None,
        max_len: Optional[int] = None,
    ) -> List[List[int]]:
        """
        Decodes a batch using vectorized GPU beam search.
        Args:
            src_ids: [B, max_src_len] (reversed source tokens + EOS)
            src_lens: [B]
            beam_size: optional beam width override
            max_len: optional max output length override
        Returns:
            best_hypotheses: List of B lists containing generated token IDs (excluding BOS/EOS/PAD)
        """
        self.model.eval()
        device = src_ids.device
        B = src_ids.shape[0]
        K = beam_size or self.beam_size

        if max_len is None:
            max_len = int(src_lens.max().item() * self.len_ratio) + self.len_const
            max_len = min(max_len, 128)

        # 1. Encode source sequences
        enc_h, enc_c = self.model.encoder(src_ids, src_lens)
        # Shapes: [n_layers, B, d_model]

        # Fast path for Greedy Search (K = 1)
        if K == 1:
            return self._decode_greedy(enc_h, enc_c, B, max_len, device)

        # 2. Expand encoder states to [n_layers, B * K, d_model]
        h = enc_h.repeat_interleave(K, dim=1)
        c = enc_c.repeat_interleave(K, dim=1)

        # 3. Initialize beam log-probabilities
        # At step 0, only beam 0 is active (score 0), other beams set to -inf
        beam_scores = torch.full((B, K), -1e9, device=device)
        beam_scores[:, 0] = 0.0

        # Sequences: [B, K, 1] containing initial BOS
        sequences = torch.full((B, K, 1), self.BOS_ID, dtype=torch.long, device=device)
        finished = torch.zeros((B, K), dtype=torch.bool, device=device)

        batch_offsets = (torch.arange(B, device=device) * K).unsqueeze(1)  # [B, 1]

        for step in range(max_len):
            # Input tokens for current step: [B * K, 1]
            curr_tokens = sequences[:, :, -1].reshape(B * K, 1)

            # Step decoder LSTM
            dec_out, (h, c) = self.model.decoder.step(curr_tokens, (h, c))
            logits = self.model.fc_out(dec_out.squeeze(1))  # [B * K, V]
            vocab_size = logits.shape[-1]

            log_probs = torch.log_softmax(logits, dim=-1)  # [B * K, V]
            log_probs = log_probs.view(B, K, vocab_size)

            # Mask finished hypotheses: only allow PAD continuation with 0 log-prob delta
            if finished.any():
                log_probs = log_probs.masked_fill(
                    finished.unsqueeze(-1).expand_as(log_probs), -1e9
                )
                # Keep score frozen for finished beams
                log_probs[:, :, self.PAD_ID] = torch.where(
                    finished,
                    torch.zeros_like(log_probs[:, :, self.PAD_ID]),
                    log_probs[:, :, self.PAD_ID],
                )

            # Calculate candidate scores: [B, K, V]
            cand_scores = beam_scores.unsqueeze(-1) + log_probs
            cand_scores = cand_scores.view(B, K * vocab_size)

            # Select top-K candidates per sentence
            topk_scores, topk_indices = torch.topk(cand_scores, K, dim=-1)

            beam_idx = topk_indices // vocab_size   # [B, K] in [0, K-1]
            token_idx = topk_indices % vocab_size   # [B, K] in [0, V-1]

            beam_scores = topk_scores

            # Reorder LSTM hidden/cell states
            # Global indices: [B * K]
            global_beam_idx = (beam_idx + batch_offsets).view(-1)
            h = h[:, global_beam_idx, :]
            c = c[:, global_beam_idx, :]

            # Update sequences
            gather_indices = beam_idx.unsqueeze(-1).expand(B, K, sequences.shape[-1])
            surviving_seqs = torch.gather(sequences, 1, gather_indices)
            sequences = torch.cat([surviving_seqs, token_idx.unsqueeze(-1)], dim=-1)

            # Update finished status
            finished = torch.gather(finished, 1, beam_idx) | (token_idx == self.EOS_ID)

            if finished.all():
                break

        # 4. Length penalty ranking
        # Count non-BOS/non-PAD tokens in each hypothesis
        lengths = (sequences != self.PAD_ID) & (sequences != self.BOS_ID)
        hyp_lengths = lengths.sum(dim=-1)  # [B, K]
        lp = self.compute_length_penalty(hyp_lengths).to(device)

        penalized_scores = beam_scores / lp  # [B, K]
        best_beam_idx = torch.argmax(penalized_scores, dim=-1)  # [B]

        # Extract best hypothesis for each sentence in batch
        results = []
        for b in range(B):
            k = best_beam_idx[b].item()
            seq = sequences[b, k].tolist()
            # Filter special tokens (BOS, EOS, PAD)
            clean_tokens = [
                t for t in seq
                if t not in (self.BOS_ID, self.EOS_ID, self.PAD_ID)
            ]
            results.append(clean_tokens)

        return results

    def _decode_greedy(
        self,
        enc_h: torch.Tensor,
        enc_c: torch.Tensor,
        B: int,
        max_len: int,
        device: torch.device,
    ) -> List[List[int]]:
        """Fast path greedy decoding (K = 1)."""
        dec_state = (enc_h, enc_c)
        curr_tokens = torch.full((B, 1), self.BOS_ID, dtype=torch.long, device=device)

        generated = [[] for _ in range(B)]
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            out, dec_state = self.model.decoder.step(curr_tokens, dec_state)
            logits = self.model.fc_out(out.squeeze(1))
            next_tokens = torch.argmax(logits, dim=-1)  # [B]

            for b in range(B):
                if not finished[b]:
                    tok = next_tokens[b].item()
                    if tok == self.EOS_ID:
                        finished[b] = True
                    elif tok != self.PAD_ID:
                        generated[b].append(tok)

            if finished.all():
                break

            curr_tokens = next_tokens.unsqueeze(1)

        return generated

    def translate_batch(
        self,
        src_ids: torch.Tensor,
        src_lens: torch.Tensor,
        beam_size: Optional[int] = None,
    ) -> List[str]:
        """Translates batch directly to detokenized text strings."""
        if self.tokenizer is None:
            raise ValueError("TokenizerManager must be provided for text translation.")

        token_lists = self.decode_batch(src_ids, src_lens, beam_size=beam_size)
        return [self.tokenizer.decode(tokens) for tokens in token_lists]
