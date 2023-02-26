"""
(C) Copyright 2021 IBM Corp.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

from typing import Sequence, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torchtext

from fuse.dl.models.backbones.backbone_transformer import Transformer


class GRUEncoder(nn.Module):
    """
    Encoder is GRU with FC layers connected to last hidden unit
    """

    def __init__(
        self, emb_dim, h_dim, z_dim, biGRU, layers, p_dropout, single_output=False
    ):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=h_dim,
            num_layers=layers,
            dropout=p_dropout,
            bidirectional=biGRU,
            batch_first=True,
        )
        # Bidirectional GRU has 2*hidden_state
        self.biGRU_factor = 2 if biGRU else 1
        self.biGRU = biGRU
        self.single_output = single_output
        # Reparametrization
        self.q_mu = nn.Linear(self.biGRU_factor * h_dim, z_dim)
        if not single_output:
            self.q_logvar = nn.Linear(self.biGRU_factor * h_dim, z_dim)

    def forward(self, x):
        """
        Inputs is embeddings of: mbsize x seq_len x emb_dim
        """
        _, h = self.rnn(x, None)
        if self.biGRU:
            # Concatenates features from Forward and Backward
            # Uses the highest layer representation
            h = torch.cat((h[-2, :, :], h[-1, :, :]), 1)
        # Forward to latent
        mu = self.q_mu(h)
        if self.single_output:
            return mu

        logvar = self.q_logvar(h)
        return mu, logvar


class GRUDecoder(nn.Module):
    """
    Decoder is GRU with FC layers connected to last hidden unit
    """

    def __init__(
        self,
        emb_dim,
        output_dim,
        h_dim,
        num_tokens: int,
        p_out_dropout=0.3,
    ):
        super().__init__()
        self.rnn = nn.GRU(emb_dim, h_dim, batch_first=True)
        self.emb_dim = emb_dim
        self.h_dim = h_dim
        self.fc = nn.Sequential(nn.Dropout(p_out_dropout), nn.Linear(h_dim, output_dim))
        self._num_tokens = num_tokens

    def forward(self, z):
        """
        :param z: latent space. Shape [N, LATENT_SPACE_DIM]
        """
        init_h = torch.zeros(
            (1, z.shape[0], self.h_dim), device=z.device, dtype=z.dtype
        )

        x = z.unsqueeze(1).repeat((1, self._num_tokens, 1))
        # Compute outputs. # mbsize x seq_len x h_dim
        rnn_out, _ = self.rnn(x, init_h)

        y = self.fc(rnn_out)
        return y


class Embed(nn.Module):
    "Convert token ids to learned embedding"

    def __init__(
        self, n_vocab: int, emb_dim: int, key_in: str, key_out: str, **embedding_kwargs
    ):
        super().__init__()
        self._emb_dim = emb_dim
        self._word_emb = nn.Embedding(n_vocab, self._emb_dim, **embedding_kwargs)
        self._key_in = key_in
        self._key_out = key_out

    def forward(self, batch_dict: dict):
        tokens = batch_dict[self._key_in]
        tokens = tokens.to(device=next(iter(self._word_emb.parameters())).device)

        embds = self._word_emb(tokens)

        batch_dict[self._key_out] = embds

        return batch_dict


class WordDropout(nn.Module):
    """randomly set token ids to a specified value"""

    def __init__(
        self,
        p_word_dropout: float,
        key_in: str,
        key_out: str,
        mask_value: int = 0,
        p_word_dropout_eval: Optional[float] = None,
    ):
        super().__init__()
        self._p = p_word_dropout
        self._p_eval = (
            p_word_dropout if p_word_dropout_eval is None else p_word_dropout_eval
        )
        self._key_in = key_in
        self._key_out = key_out
        self._mask_value = mask_value

    def forward(self, batch_dict: dict):
        """
        Do word dropout: with prob `p_word_dropout`, set the word to '<unk>'.
        """
        x = batch_dict[self._key_in]

        data = x.clone().detach()

        # Sample masks: elems with val 1 will be set to <unk>
        p = self._p if self.training else self._p_eval

        mask = torch.from_numpy(
            np.random.choice(2, p=(1.0 - p, p), size=tuple(data.size())).astype("uint8")
        ).to(x.device)

        mask = mask.bool()
        # Set to <unk>
        data[mask] = self._mask_value

        batch_dict[self._key_out] = data
        return batch_dict


class RandomOverride(nn.Module):
    """randomly override token ids with value from values (also sampled randomly)"""

    def __init__(
        self,
        p_train: float,
        key_in: str,
        key_out: str,
        values: Sequence[int],
        p_eval: float = 0.0,
    ):
        super().__init__()
        self._p_train = p_train
        self._p_eval = p_eval
        self._key_in = key_in
        self._key_out = key_out
        self._values = values

    def forward(self, batch_dict: dict):
        """
        Do word dropout: with prob `p_word_dropout`, set the word to '<unk>'.
        """
        x = batch_dict[self._key_in]

        data = x.clone().detach()

        # Sample masks: elems with val 1 will be set to <unk>
        p = self._p_train if self.training else self._p_eval

        mask = torch.from_numpy(
            np.random.choice(2, p=(1.0 - p, p), size=tuple(data.size())).astype("uint8")
        ).to(x.device)

        mask = mask.bool()
        if mask.sum() > 0:
            data[mask] = torch.from_numpy(
                np.random.choice(self._values, int(mask.sum()))
            )

        batch_dict[self._key_out] = data
        return batch_dict


class RandomAdjacentSwap(nn.Module):
    """randomly swap token-id with adjacent token id"""

    def __init__(
        self,
        p_train: float,
        key_in: str,
        key_out: str,
        p_eval: float = 0.0,
    ):
        super().__init__()
        self._p_train = p_train
        self._p_eval = p_eval
        self._key_in = key_in
        self._key_out = key_out

    def forward(self, batch_dict: dict):
        """
        Do word dropout: with prob `p_word_dropout`, set the word to '<unk>'.
        """
        x = batch_dict[self._key_in]

        data = x.clone().detach()

        # Sample masks: elems with val 1 will be set to <unk>
        p = self._p_train if self.training else self._p_eval

        mask = torch.from_numpy(
            np.random.choice(2, p=(1.0 - p, p), size=tuple(data.size())).astype("uint8")
        ).to(x.device)

        mask = mask.bool()
        mask[:, -1] = False  # don't swap the last element
        # don't do two adjacent swaps
        swap_mask = torch.roll(mask, 1, 1)
        mask[np.logical_and(mask, swap_mask).bool()] = False
        if mask.sum() > 0:
            swap_mask = torch.roll(mask, 1, 1)
            # swap_values = torch.roll(data, 1, 1)
            data[mask] = x[swap_mask]
            data[swap_mask] = x[mask]

        batch_dict[self._key_out] = data
        return batch_dict


class RandomShift(nn.Module):
    """roll the sequence - the shift amount selected randomly"""

    def __init__(
        self,
        max_fraction_train: float,
        key_in: str,
        key_out: str,
        max_fraction_eval: float = 0.0,
    ):
        super().__init__()
        self._max_fraction_train = max_fraction_train
        self._max_fraction_eval = max_fraction_eval
        self._key_in = key_in
        self._key_out = key_out

    def forward(self, batch_dict: dict):
        """
        Do word dropout: with prob `p_word_dropout`, set the word to '<unk>'.
        """
        x = batch_dict[self._key_in]

        data = x.clone().detach()

        max_fraction_shift = (
            self._max_fraction_train if self.training else self._max_fraction_eval
        )
        if max_fraction_shift > 0.0:
            max_shift = int(x.shape[-1] * max_fraction_shift)
            shift = np.random.randint(0, max_shift, 1)
            data = torch.roll(data, int(shift), 1)

        batch_dict[self._key_out] = data
        return batch_dict


class RandomMix(nn.Module):
    """randomly mix a token embedding with another token embedding"""

    def __init__(
        self,
        p_train: float,
        key_in: str,
        key_out: str,
        values: Sequence[int],
        embed: nn.Module,
        weight: float = 0.2,
        p_eval: float = 0.0,
    ):
        super().__init__()
        self._p_train = p_train
        self._p_eval = p_eval
        self._key_in = key_in
        self._key_out = key_out
        self._values = values
        self._weight = weight
        self._embed = embed

    def forward(self, batch_dict: dict):
        """
        Do word dropout: with prob `p_word_dropout`, set the word to '<unk>'.
        """
        x = batch_dict[self._key_in]

        data = x.clone().detach()

        # Sample masks: elems with val 1 will be set to <unk>
        p = self._p_train if self.training else self._p_eval

        mask = torch.from_numpy(
            np.random.choice(2, p=(1.0 - p, p), size=tuple(data.shape[:2])).astype(
                "uint8"
            )
        ).to(x.device)

        mask = mask.bool()
        if mask.sum() > 0:
            tokens = torch.from_numpy(np.random.choice(self._values, int(mask.sum())))
            tokens = tokens.to(device=x.device)
            data[mask] = (
                data[mask]
                + self._weight
                * self._embed._word_emb(tokens.reshape(1, -1))[0].detach()
            )

        batch_dict[self._key_out] = data
        return batch_dict


class Sample(nn.Module):
    """Sample from a normal distribution"""

    def __init__(self, key_mu: str, key_logvar: str, key_out: int) -> None:
        super().__init__()
        self._key_mu = key_mu
        self._key_logvar = key_logvar
        self._key_out = key_out

    def forward(self, batch_dict: str):
        """
        Reparameterization trick: z = mu + std*eps; eps ~ N(0, I)
        """
        mu = batch_dict[self._key_mu]
        logvar = batch_dict[self._key_logvar]

        if self.training:
            eps = torch.randn(*mu.size()).to(logvar.device)
            batch_dict[self._key_out] = mu + torch.exp(logvar / 2) * eps
        else:
            batch_dict[self._key_out] = mu
        return batch_dict


class Tokenizer(torch.nn.Module):
    """simple tokenizer - every char is a token"""

    def __init__(
        self, seqs: Sequence[str], key_in: str, key_out: str, max_seq_len: int
    ):
        super().__init__()
        self._tokenizer = torchtext.data.Field(
            init_token="<start>",
            eos_token="<eos>",
            sequential=True,
            lower=True,
            tokenize=self._tokenizer_func,
            fix_length=max_seq_len,
            batch_first=True,
        )
        self._tokenizer.build_vocab(seqs)
        self._key_in = key_in
        self._key_out = key_out
        print(f"vocab={self._tokenizer.vocab.itos}")

    def forward(self, batch_dict: dict) -> dict:
        input = batch_dict[self._key_in]
        output = self._tokenizer.process(input)
        batch_dict[self._key_out] = output
        return batch_dict

    @staticmethod
    def _tokenizer_func(seq: str) -> List[str]:
        return [tok for tok in str.split(seq)]


class LogitsToSeq(torch.nn.Module):
    """Convert the logits back to tokens"""

    def __init__(self, key_in: str, key_out: str, itos: callable):
        super().__init__()
        self._key_in = key_in
        self._key_out = key_out
        self._itos = itos

    def forward(self, batch_dict: dict) -> dict:
        if self.training:
            return batch_dict
        logits = batch_dict[self._key_in]
        batch = []
        tokens = logits.argmax(dim=2)
        for sample_tokens in tokens:
            seq = ""
            for t in sample_tokens:
                ch = self._itos[t]
                if ch == "<eos>":
                    break
                if not ch.startswith("<"):
                    seq += ch
            batch.append(seq)

        batch_dict[self._key_out] = batch
        return batch_dict


class TransformerEncoder(Transformer):
    """Transformer based encoder"""

    def __init__(self, num_cls_tokens=2, **kwargs):
        super().__init__(num_cls_tokens=num_cls_tokens, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        if self.num_cls_tokens == 1:
            return out[:, 0], out[:, 1:]
        else:
            return [out[:, i] for i in range(self.num_cls_tokens)] + [
                out[:, self.num_cls_tokens :]
            ]


class TransformerDecoder(Transformer):
    """Transformer based decoder"""

    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        output_dim: int,
        out_dropout: float,
        **kwargs,
    ):
        super().__init__(num_tokens=num_tokens, token_dim=token_dim, **kwargs)
        self.fc = nn.Sequential(
            nn.Dropout(out_dropout), nn.Linear(token_dim, output_dim)
        )
        self.num_tokens = num_tokens

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        :param z: latent space. Shape [N, LATENT_SPACE_DIM]
        """
        x = z.unsqueeze(1).repeat((1, self.num_tokens, 1))
        # Compute outputs. # mbsize x seq_len x z_dim
        out = super().forward(x)

        y = self.fc(out[:, 1:])
        return y
