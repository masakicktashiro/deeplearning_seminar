import random
import torch
import torch.nn as nn

from beam_search import BeamSearch

class Seq2SeqModel(nn.Module):

    def __init__(self, encoder, decoder,
                decoding_style="greedy", special_tokens_dict=None,
                max_decoding_steps=128, beam_width=10):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        if decoding_style not in ["greedy", "beam_search"]:
            print(f"{decoding_style} is not allowed parameter")
            decoding_style = "greedy"
        self.decoding_style = decoding_style
        if special_tokens_dict is None:
            self.special_tokens_dict = {"<pad>":0, "<bos>":1, "<eos>":2, "<unk>":3}
        else:
            self.special_tokens_dict = special_tokens_dict
        self.max_decoding_steps = max_decoding_steps
        self._beam_search = BeamSearch(self.special_tokens_dict["<eos>"],
                                       max_decoding_steps,
                                       beam_width)

    def forward(self, src, oracle, teacher_forcing_ratio=0.5):
        state = self.encoder(src)
        if teacher_forcing_ratio == 1:
            outputs, state = self.decoder(state=state, oracle=oracle)
        else:
            len_oracle = oracle.shape[1]
            last_pred = oracle[:, 0].unsqueeze(dim=-1)
            outputs = []
            for i in range(len_oracle):
                input = oracle[:,i].unsqueeze(dim=-1) \
                    if random.random() < teacher_forcing_ratio else last_pred
                output, state = self.decoder(state=state, oracle=input)
                outputs.append(output)
                last_pred = output.argmax(dim=-1)
            outputs = torch.cat(outputs, dim=1)
        return outputs

    def decode(self, src):
        """
        src : tensor
            src.shape = (bs, length)
        preds : tensor
            preds.shape = (bs, max_length)
        """
        state = self.encoder(src)
        if self.decoding_style == "greedy":
            preds = self.greedy_search(state)
        else:
            preds = self.beam_search(state)
        return preds

    def greedy_search(self, state):
        bs = state["hidden"].size(1)
        device = [j for i, j in self.encoder.named_parameters()][0].device
        bos_token_id = self.special_tokens_dict["<bos>"]
        last_pred = torch.full((bs, 1), bos_token_id, dtype=torch.int64)  
        pred = []
        for i in range(self.max_decoding_steps):
            last_pred = last_pred.to(device)
            output, state = \
                self.decoder(state=state, oracle=last_pred)
            next_pred = output.argmax(dim=-1).detach().cpu().view(-1, 1)
            pred.append(next_pred)
            last_pred = next_pred
        pred = torch.cat(pred, dim=1)
        return pred

    def beam_search(self, state):
        bs = state["encoder_output"].size(0)
        device = [j for i, j in self.encoder.named_parameters()][0].device
        bos_token_id = self.special_tokens_dict["<bos>"]
        start_predictions = torch.full((bs,), bos_token_id, dtype=torch.int64, device=device)
        all_top_k_predictions, log_probabilities = self._beam_search.search(
            start_predictions, state, self.decoder.forward
        )
        ind = log_probabilities.argmax(dim=-1)
        return all_top_k_predictions[range(bs), ind, :]

class SimpleRNNEncoder(nn.Module):

    def __init__(self, vocab_size, embedding_size,
                hidden_size, num_layers=1, bidirectional=True,
                dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.RNN(embedding_size, hidden_size, batch_first=True,
                          bidirectional=bidirectional, num_layers=num_layers)
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        Parameters
        ==========
        src : tensor
            src.size = (bs, length)

        Return
        ==========
        output : tensor
            output.size = (bs, length, hidden_size * 2)
        state : tensor
            state.size = (bs, hidden_state * 2)
        """
        bs = src.size(0)
        embedding = self.dropout(self.embedding(src))
        output, state = self.rnn(embedding)
        return {"encoder_output" : output,
                "hidden" : state}

class SimpleGRUEncoder(nn.Module):

    def __init__(self, vocab_size, embedding_size,
                hidden_size, num_layers=1, bidirectional=True,
                dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True,
                          bidirectional=bidirectional, num_layers=num_layers)
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        Parameters
        ==========
        src : tensor
            src.size = (bs, length)

        Return
        ==========
        output : tensor
            output.size = (bs, length, hidden_size * 2)
        state : tensor
            state.size = (bs, hidden_state * 2)
        """
        bs = src.size(0)
        embedding = self.dropout(self.embedding(src))
        output, state = self.gru(embedding)
        if self.bidirectional:
            state = torch.cat(torch.chunk(state, 2, dim=0), dim=-1)
        return {"encoder_output" : output,
                "hidden" : state}

class SimpleGRUDecoder(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers=1, dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size,
                          batch_first=True, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

    def forward(self, oracle, state, dropout=0.5):
        """
        Parameters
        ==========
        hidden_state : tensor
            output.size = (num_layers, bs, hidden_size * num_direction)
        decoder_input : tensor
            decoder_input.size = (bs, 1)
        oracle : tensor
            oracle.size = (bs, length)
        encoder_output : tensor
            state.size = (bs, length, hidden_state * 2)

        Return
        ==========
        pred : tensor
            output.size = (bs, length, vocab_size)
        state : tensor
            state.size = (bs, hidden_state
            )
        """
        is_1d = False
        if len(oracle.shape) < 2:
            is_1d = True
            oracle = oracle.unsqueeze(dim=-1)
        oracle = self.embedding(oracle)
        hidden_state = state["hidden"]
        if hidden_state.size(0) != self.num_layers:
            if hidden_state.size(0) == 1:
                hidden_state = torch.cat([hidden_state] * self.num_layers, dim=0)
            elif self.num_layers == 1:
                hidden_state = torch.cat(
                                torch.chunk(hidden_state, hidden_state.size(0), dim=0), dim=-1)
            else:
                raise ValueError
        output, hidden_state = self.gru(oracle, hidden_state)
        pred = self.linear(output)
        state["hidden"] = hidden_state
        if is_1d:
            pred = pred.squeeze(dim=1)
        return self.log_softmax(pred), state

class SimpleLSTMEncoder(nn.Module):

    def __init__(self, vocab_size, embedding_size,
                hidden_size, num_layers=1, bidirectional=True,
                dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True,
                          bidirectional=bidirectional, num_layers=num_layers)
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        Parameters
        ==========
        src : tensor
            src.size = (bs, length)

        Return
        ==========
        output : tensor
            output.size = (bs, length, hidden_size * 2)
        state : tuple
            state[i].size = (num_layers, bs, hidden_state * num_dirction)
        """
        bs = src.size(0)
        embedding = self.dropout(self.embedding(src))
        output, hidden_state = self.lstm(embedding)
        if self.bidirectional:
            hidden_state = tuple([torch.cat(torch.chunk(s, 2, dim=0), dim=-1) for s in hidden_state])
        return {"encoder_output" : output,
                "hidden" : hidden_state[0],
                "cell" : hidden_state[1]}

class SimpleLSTMDecoder(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers=1,
                 dropout=0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size,
                          batch_first=True, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

    def forward(self, oracle, state):
        """
        Parameters
        ==========
        state : Dict[str, torch.Tensor]
        decoder_input : tensor
            decoder_input.size = (bs, 1)
        oracle : tensor
            oracle.size = (bs, length)

        Return
        ==========
        pred : tensor
            output.size = (bs, length, vocab_size)
        state : Dict[str, torch.Tensor]
        """
        is_1d = False
        if len(oracle.shape) < 2:
            is_1d = True
            oracle = oracle.unsqueeze(dim=-1)
        oracle = self.embedding(oracle)
        h_n = state["hidden"]
        c_n = state["cell"]
        if h_n.size(0) != self.num_layers:
            if h_n.size(0) == 1:
                h_n = torch.cat([h_n] * self.num_layers, dim=0)
                c_n = torch.cat([c_n] * self.num_layers, dim=0)
            elif self.num_layers == 1:
                h_n = torch.cat(torch.chunk(h_n, h_n.size(0), dim=0), dim=-1)
                c_n = torch.cat(torch.chunk(c_n, c_n.size(0), dim=0), dim=-1)
            else:
                raise ValueError
        hidden_state = (h_n, c_n)
        output, hidden_state = self.lstm(oracle, hidden_state)
        pred = self.linear(output)
        state["hidden"] = hidden_state[0]
        state["cell"] = hidden_state[1]
        if is_1d:
            pred = pred.squeeze(dim=1)
        return self.log_softmax(pred), state
