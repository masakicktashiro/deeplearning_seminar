import random
import torch
import torch.nn as nn

class AttnGRUDecoder(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size,
                encoder_hidden_size, num_layers=1, dropout=0.5,
                attn_hid_size=None):
        super().__init__()
        if attn_hid_size is None:
            attn_hid_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.attn_layer = AttentionLayer(hidden_size, encoder_hidden_size,
                            attn_hid_size)
        self.rnn = nn.GRU(embedding_size + encoder_hidden_size, hidden_size,
                          batch_first=True, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size + encoder_hidden_size, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, oracle, state, teacher_forcing_ratio=0.5):
        """
        Parameters
        ==========
        state : Dict[str, torch.Tensor]
        
        oracle : tensor
            oracle.size = (bs, length)
        encoder_output : tensor
            (bs, length, hid_size * num_direction)
        
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
        last_pred = oracle[:, :1]
        outputs = []
        hidden_state = state["hidden"]
        for i in range(oracle.size(1)):
            input = oracle[:, i].unsqueeze(dim=-1) \
                if random.random() < teacher_forcing_ratio else last_pred
            input = self.embedding(input)
            # attn.shape = (bs, length)
            attn = self.attn_layer(state)
            hidden_state = state["hidden"]
            context = torch.einsum("ijk,ij->ijk", state["encoder_output"], attn).sum(dim=1, keepdims=True)
            output, hidden_state = self.rnn(torch.cat([input, context], dim=-1), hidden_state)
            output = self.linear(torch.cat([output, context], dim=-1))
            output = self.log_softmax(output)
            state["hidden"] = hidden_state
            outputs.append(output)
            last_pred = output.argmax(dim=-1)
        outputs = torch.cat(outputs, dim=1)
        if is_1d:
            outputs = outputs.squeeze(dim=1)
        return outputs, state

class AttentionLayer(nn.Module):
    
    def __init__(self, decoder_hid_size, encoder_hid_size, hidden_size):
        super().__init__()
        self.decoder_hid_size = decoder_hid_size
        self.encoder_hid_size = encoder_hid_size
        self.linear = nn.Linear(decoder_hid_size + encoder_hid_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        """
        Param
        =====
        decoder_hidden : torch.Tensor
            (1, bs, hid_size * num_direction)
        encoder_hidden : torch.Tensor
            (bs, length, hid_size * num_direction)
        
        Output
        ======
        output : torch.Tensor
            (num_layers, bs, length)
        """
        decoder_hidden = state["hidden"].squeeze(dim=0).unsqueeze(dim=-2)
        encoder_hidden = state["encoder_output"]
        encoder_length = encoder_hidden.size(1)
        # (num_l, bs, length, hid_size * num_direction)
        decoder_hidden = decoder_hidden.expand(-1, encoder_length, -1)
        input = torch.cat([encoder_hidden, decoder_hidden], dim=-1)
        output = self.tanh(self.linear(input))
        output = self.linear2(output)
        output = self.softmax(output.squeeze(dim=-1))
        return output
