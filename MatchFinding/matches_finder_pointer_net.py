import torch
import torch.nn as nn
import math
import numpy as np
import dataset
from constants import constants


class Encoder(nn.Module):
    """Maps a graph represented as an input sequence
    to a hidden vector"""

    def __init__(self, input_dim, hidden_dim, use_cuda, bi_directional, num_layers):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim//2 if bi_directional else hidden_dim
        self.n_layers = num_layers * 2 if bi_directional else num_layers
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, bidirectional=bi_directional,
                            num_layers=num_layers).to(dtype=torch.double)
        if use_cuda:
            self.lstm = self.lstm.cuda()
        self.use_cuda = use_cuda
        self.h = torch.zeros(1).unsqueeze(0).unsqueeze(0).repeat(self.n_layers, dataset.batch_size, self.hidden_dim)
        self.c = torch.zeros(1).unsqueeze(0).unsqueeze(0).repeat(self.n_layers, dataset.batch_size, self.hidden_dim)
        self.h, self.c = nn.Parameter(self.h.double(), requires_grad=True), \
                         nn.Parameter(self.c.double(), requires_grad=True)
        if self.use_cuda:
            self.h, self.c = self.h.cuda(), self.c.cuda()

    def forward(self, x):
        output, hidden = self.lstm(x, (self.h, self.c))
        return output, hidden


class Attention(nn.Module):
    """A generic attention module for a decoder in seq2seq"""

    def __init__(self, dim, use_tanh=False, C=10, use_cuda=True):
        super(Attention, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(dim, dim).to(dtype=torch.double)
        self.project_ref = nn.Conv1d(dim, dim, 1, 1).to(dtype=torch.double)
        self.C = C  # tanh exploration
        self.tanh = nn.Tanh()

        v = torch.empty(dim, dtype=torch.double).uniform_(-(1. / math.sqrt(dim)), 1. / math.sqrt(dim))
        if use_cuda:
            v = v.cuda()
            self.project_query = self.project_query.cuda()
            self.project_ref = self.project_ref.cuda()

        self.v = nn.Parameter(v, requires_grad=True)

    def forward(self, query, ref):
        """
        Args:
            query: is the hidden state of the decoder at the current
                time step. batch, dim
            ref: the set of hidden states from the encoder.
                source_length, batch, hidden_dim
        """
        # ref is now [batch_size, hidden_dim, source_length]
        ref = ref.permute(1, 2, 0)
        q = self.project_query(query).unsqueeze(2)  # batch, dim, 1
        e = self.project_ref(ref)  # batch_size, hidden_dim, source_length
        # expand the query by source_length
        # batch, dim, source_length
        expanded_q = q.repeat(1, 1, e.size(2))
        # batch, 1, hidden_dim
        v_view = self.v.unsqueeze(0).expand(expanded_q.size(0), len(self.v)).unsqueeze(1)
        # [batch_size, 1, hidden_dim] * [batch_size, hidden_dim, source_length]
        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)

        logits = self.C * self.tanh(u) if self.use_tanh else u

        return e, logits


class Decoder(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 max_length,
                 tanh_exploration,
                 use_tanh,
                 n_glimpses=1,
                 use_cuda=True):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_glimpses = n_glimpses
        self.max_length = max_length
        self.use_cuda = use_cuda

        self.input_weights = nn.Linear(embedding_dim, 4 * hidden_dim).to(dtype=torch.double)
        self.hidden_weights = nn.Linear(hidden_dim, 4 * hidden_dim).to(dtype=torch.double)

        self.pointer = Attention(hidden_dim, use_tanh=use_tanh, C=tanh_exploration, use_cuda=self.use_cuda)
        self.glimpse = Attention(hidden_dim, use_tanh=False, use_cuda=self.use_cuda)
        self.sm = nn.Softmax(dim=-1)

    def forward(self, decoder_input, embedded_inputs, hidden, context, counts):
        """
        Args:
            decoder_input: The initial input to the decoder
                size is [batch_size, embedding_dim]. Trainable parameter.
            embedded_inputs: [source_length, batch_size, embedding_dim]
            hidden: the prev hidden state, size is [batch_size, hidden_dim].
                Initially this is set to (enc_h[-1], enc_c[-1])
            context: encoder outputs, [source_length, batch_size, hidden_dim]
            counts:
        """
        length = max(counts).item() + 1
        def recurrence(x, hidden):
            hx, cx = hidden  # batch_size, hidden_dim

            gates = self.input_weights(x) + self.hidden_weights(hx)
            in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)

            in_gate = torch.sigmoid(in_gate)
            forget_gate = torch.sigmoid(forget_gate)
            cell_gate = torch.tanh(cell_gate)
            out_gate = torch.sigmoid(out_gate)

            cy = (forget_gate * cx) + (in_gate * cell_gate)
            hy = out_gate * torch.tanh(cy)  # batch_size, hidden_dim

            g_l = hy
            for i in range(self.n_glimpses):
                ref, logits = self.glimpse(g_l, context)
                # [batch_size, h_dim, source_length] * [batch_size, source_length, 1] = [batch_size, h_dim, 1]
                g_l = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
            _, logits = self.pointer(g_l, context)

            probs = self.sm(logits)
            return hy, cy, probs

        outputs = []
        for _ in range(dataset.batch_size):
            outputs.append([])
        selections = []
        batch_size = context.size(1)
        finished_batches_mask = torch.ones(batch_size).int()  # batches indices that have already finished
        if self.use_cuda:
            finished_batches_mask = finished_batches_mask.cuda()
        finished_batches_matches_mask = torch.ones(batch_size).int()  # batches indices that their real matches have
        # already finished, used to prevent over training of zeros
        if self.use_cuda:
            finished_batches_matches_mask = finished_batches_matches_mask.cuda()

        zeros = torch.zeros_like(finished_batches_mask)
        step = 0
        MAX_STEPS = length * 5
        for _ in range(MAX_STEPS):
            hx, cx, probs = recurrence(decoder_input, hidden)
            hidden = (hx, cx)

            idxs = torch.distributions.Categorical(probs).sample()

            idxs = idxs * finished_batches_mask

            probs_mask1 = finished_batches_mask.reshape(dataset.batch_size, 1).repeat(1, constants['window_size'] + 1)
            probs_mask2 = \
                finished_batches_matches_mask.reshape(dataset.batch_size, 1).repeat(1, constants['window_size'] + 1)
            probs = probs * probs_mask1 * probs_mask2  # force train to use only first zero of matches

            for i in range(dataset.batch_size):
                outputs[i].append(probs[i, :])
            selections.append(idxs)

            finished_batches_mask = finished_batches_mask * idxs.bool().int()
            finished_batches_matches_mask[[True if step == counts[j].item() else False
                                           for j in range(dataset.batch_size)]] = 0

            if torch.equal(finished_batches_mask, zeros):
                current_length = len(outputs[0])
                for i in range(dataset.batch_size):
                    outputs[i] += [torch.zeros_like(outputs[i][0])] * (length - current_length)
                selections += [torch.zeros_like(selections[0])] * (length - len(selections))
                break

            decoder_input = embedded_inputs[idxs.data, range(batch_size), :]
            step += 1

        for i in range(dataset.batch_size):
            outputs[i] = torch.stack(outputs[i])
        outputs = torch.cat(outputs)

        return (outputs, selections), hidden


class PointerNetwork(nn.Module):
    """The pointer network, which is the core seq2seq model"""
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 max_decoding_len,
                 n_glimpses,
                 tanh_exploration,
                 use_tanh,
                 use_cuda,
                 encoder_bi_directional,
                 encoder_num_layers,
                 padding_value):
        super(PointerNetwork, self).__init__()

        self.encoder = Encoder(
            embedding_dim,
            hidden_dim,
            use_cuda,
            encoder_bi_directional,
            encoder_num_layers)

        self.decoder = Decoder(
            embedding_dim,
            hidden_dim,
            max_length=max_decoding_len,
            tanh_exploration=tanh_exploration,
            use_tanh=use_tanh,
            n_glimpses=n_glimpses,
            use_cuda=use_cuda,
        )

        # Trainable initial hidden states
        dec_in_0 = torch.empty(embedding_dim, dtype=torch.double).uniform_(-(1. / math.sqrt(embedding_dim)),
                                                                           1. / math.sqrt(embedding_dim))
        if use_cuda:
            dec_in_0 = dec_in_0.cuda()

        self.decoder_in_0 = nn.Parameter(dec_in_0, requires_grad=True)
        self.encoder_bi_directional = encoder_bi_directional
        self.encoder_num_layers = encoder_num_layers
        self.padding_value = padding_value

        self.use_cuda = use_cuda
        self.max_length = max_decoding_len

    def forward(self, inputs, max_length=None):
        inputs = inputs.transpose(0, 1)
        """ Propagate inputs through the network
        Args:
            inputs: [source_length, batch_size, embedding_dim]
            :param max_length:
        """
        if max_length is None:
            max_length = self.max_length
        if self.padding_value is not None:
            beginning_padding = torch.full(size=(1, inputs.size(1), inputs.size(2)),
                                           fill_value=self.padding_value, dtype=inputs.dtype)
            if self.use_cuda:
                beginning_padding = beginning_padding.cuda()
            inputs = torch.cat((beginning_padding, inputs), dim=0)

        # encoder forward pass
        enc_h, (enc_h_t, enc_c_t) = self.encoder(inputs)

        if self.encoder_bi_directional:
            dec_init_state = (torch.cat([enc_h_t[-2], enc_h_t[-1]], dim=-1),
                              torch.cat([enc_c_t[-2], enc_c_t[-1]], dim=-1))
        else:
            dec_init_state = (enc_h_t[-1], enc_c_t[-1])

        # repeat decoder_in_0 across batch
        decoder_input = self.decoder_in_0.unsqueeze(0).repeat(inputs.size(1), 1)

        (pointer_probs, input_idxs), dec_hidden_t = self.decoder(decoder_input,
                                                                 inputs,
                                                                 dec_init_state,
                                                                 enc_h,
                                                                 max_length)

        return pointer_probs, input_idxs

    def __str__(self):
        return "pointer net"
