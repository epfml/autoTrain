import os
from copy import deepcopy
from typing import Dict, Iterable, List

import numpy as np
import spacy
import torch
import torch.nn as nn
import torchtext
from spacy.symbols import ORTH
from torch.utils.data import DataLoader


"""
This is another example of a task.
It is an implementation of language modelling.
The CifarTask is easier to understand and better documented.
"""


class Batch:
    def __init__(self, x, y, hidden):
        self.x = x
        self.y = y
        self.hidden = hidden


class LanguageModelingTask:
    def __init__(self):
        self.default_batch_size = 32
        self.target_test_loss = 2.0

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._seed = 34534
        self._epoch = 0

        torch.random.manual_seed(self._seed)
        self.text, self.train_loader, self.val_loader = define_dataset(
            self._device, "wikitext2", "data", batch_size=self.default_batch_size
        )

        global ITOS
        global STOI
        ITOS = self.text.vocab.itos
        STOI = self.text.vocab.stoi

        self._model = self._create_model()
        self._criterion = torch.nn.CrossEntropyLoss().to(self._device)

        self.state = [parameter.data for parameter in self._model.parameters()]
        self.buffers = [buffer for buffer in self._model.buffers()]
        self.parameter_names = [name for (name, _) in self._model.named_parameters()]
        self._hidden_container = {"hidden": None}

    def train_iterator(self, batch_size: int, shuffle: bool = False) -> Iterable[Batch]:
        """Shuffle is ignored .. text cannot be shuffled"""
        self._epoch += 1
        self._hidden_container["hidden"] = self._model.init_hidden(batch_size)
        _, train_loader, _ = define_dataset(
            self._device, "wikitext2", "data", batch_size=batch_size
        )
        return BatchLoader(
            train_loader, self._device, model=self._model, hidden_container=self._hidden_container
        )

    def batch_loss(self, batch: Batch) -> (float, Dict[str, float]):
        with torch.no_grad():
            prediction, hidden = self._model(batch.x, batch.hidden)
            self._hidden_container["hidden"] = hidden
            loss = self._criterion(
                prediction.view(-1, self._model.ntokens), batch.y.contiguous().view(-1)
            )
        return loss.item()

    def batch_loss_and_gradient(
        self, batch: Batch, rnn_clip=0.4
    ) -> (float, List[torch.Tensor], Dict[str, float]):
        self._zero_grad()
        prediction, hidden = self._model(batch.x, batch.hidden)
        self._hidden_container["hidden"] = hidden
        f = self._criterion(prediction.view(-1, self._model.ntokens), batch.y.contiguous().view(-1))
        f.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), rnn_clip)
        df = [parameter.grad for parameter in self._model.parameters()]
        return f.detach(), df

    def test(self, state_dict=None) -> float:
        self._hidden_container["hidden"] = self._model.init_hidden(self.default_batch_size)
        test_loader = BatchLoader(
            self.val_loader,
            self._device,
            model=self._model,
            hidden_container=self._hidden_container,
        )

        if state_dict:
            test_model = self._create_test_model(state_dict)
        else:
            test_model = self._model
            test_model.eval()

        losses = []

        for batch in test_loader:
            with torch.no_grad():
                prediction, hidden = self._model(batch.x, batch.hidden)
                self._hidden_container["hidden"] = hidden
                losses.append(self._criterion(prediction, batch.y).item())

        mean_f = np.mean(losses)
        if mean_f < self.target_test_loss:
            raise Done(mean_f)

        return mean_f

    def _create_model(self):
        torch.random.manual_seed(self._seed)
        model = define_model(self.text)
        model.to(self._device)
        model.train()
        return model

    def _create_test_model(self, state_dict):
        test_model = deepcopy(self._model)
        test_model.load_state_dict(state_dict)
        test_model.eval()
        return test_model

    def _zero_grad(self):
        self._model.zero_grad()


class BatchLoader:
    """
    Utility that transforms a dataloader that is an iterable over (x, y) tuples
    into an iterable over Batch() tuples, where its contents are already moved
    to the selected device.
    """

    def __init__(self, dataloader, device, model, hidden_container):
        self.dataloader = dataloader
        self.device = device
        self._model = model
        self._hidden_container = hidden_container

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for batch in self.dataloader:
            x = batch.text
            y = batch.target
            hidden = self._model.repackage_hidden(self._hidden_container["hidden"])
            yield Batch(x, y, hidden)


def define_dataset(
    device,
    dataset_name,
    dataset_path,
    batch_size,
    rnn_use_pretrained_emb=False,
    rnn_n_hidden=650,
    reshuffle_per_epoch=True,
    rnn_bptt_len=30,
):
    # create dataset.
    TEXT, train, valid, test = _get_dataset(dataset_name, dataset_path)

    # Build vocb.
    # we can use some precomputed word embeddings,
    # e.g., GloVe vectors with 100, 200, and 300.
    if rnn_use_pretrained_emb:
        try:
            vectors = "glove.6B.{}d".format(rnn_n_hidden)
            vectors_cache = os.path.join(dataset_path, ".vector_cache")
        except:
            vectors, vectors_cache = None, None
    else:
        vectors, vectors_cache = None, None
    TEXT.build_vocab(train, vectors=vectors, vectors_cache=vectors_cache)

    # Partition training data.
    train_loader, _ = torchtext.data.BPTTIterator.splits(
        (train, valid),
        batch_size=batch_size,
        bptt_len=rnn_bptt_len,
        device=device,
        shuffle=reshuffle_per_epoch,
    )
    _, val_loader = torchtext.data.BPTTIterator.splits(
        (train, valid),
        batch_size=batch_size,
        bptt_len=rnn_bptt_len,
        device=device,
        shuffle=reshuffle_per_epoch,
    )

    # get some stat.
    return TEXT, train_loader, val_loader


def define_model(TEXT, rnn_n_hidden=650, rnn_n_layers=3, rnn_tie_weights=True, drop_rate=0.4):
    # get embdding size and num_tokens.
    weight_matrix = TEXT.vocab.vectors

    if weight_matrix is not None:
        n_tokens, emb_size = weight_matrix.size(0), weight_matrix.size(1)
    else:
        n_tokens, emb_size = len(TEXT.vocab), rnn_n_hidden

    # create model.
    model = RNNModel(
        rnn_type="LSTM",
        ntoken=n_tokens,
        ninp=emb_size,
        nhid=rnn_n_hidden,
        nlayers=rnn_n_layers,
        tie_weights=rnn_tie_weights,
        dropout=drop_rate,
    )

    # init the model.
    if weight_matrix is not None:
        model.encoder.weight.data.copy_(weight_matrix)

    return model


def _get_text():
    spacy_en = spacy.load("en")
    spacy_en.tokenizer.add_special_case("<eos>", [{ORTH: "<eos>"}])
    spacy_en.tokenizer.add_special_case("<bos>", [{ORTH: "<bos>"}])
    spacy_en.tokenizer.add_special_case("<unk>", [{ORTH: "<unk>"}])

    def spacy_tok(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    TEXT = torchtext.data.Field(lower=True, tokenize=spacy_tok)
    return TEXT


def _get_dataset(name, datasets_path):
    TEXT = _get_text()

    # Load and split data.
    if "wikitext2" in name:
        train, valid, test = torchtext.datasets.WikiText2.splits(TEXT, root=datasets_path)
    elif "ptb" in name:
        train, valid, test = torchtext.datasets.PennTreebank.splits(TEXT, root=datasets_path)
    return TEXT, train, valid, test


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ["LSTM", "GRU"]:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {"RNN_TANH": "tanh", "RNN_RELU": "relu"}[rnn_type]
            except KeyError:
                raise ValueError(
                    """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']"""
                )
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError("When using the tied flag, nhid must be equal to emsize")
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.ntokens = ntoken

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == "LSTM":
            return (
                weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid),
            )
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

    def repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)


ITOS = None  # integer to string
STOI = None  # string to integer
