import torch
import torch.nn as nn
import math
import numpy as np
import dataset
from constants import constants
import time

class Encoder(nn.Module):
    """Maps a graph represented as an input sequence
    to a hidden vector"""

    def __init__(self, input_dim, hidden_dim, use_cuda, bi_directional, num_layers):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=bi_directional,
                            num_layers=num_layers).to(dtype=torch.double)
        if use_cuda:
            self.lstm = self.lstm.cuda()
        self.use_cuda = use_cuda
        self.enc_init_state = self.init_hidden(hidden_dim)

    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        return output, hidden

    def init_hidden(self, hidden_dim):
        """Trainable initial hidden state"""
        enc_init_hx = torch.zeros(hidden_dim, dtype=torch.double)
        if self.use_cuda:
            enc_init_hx = enc_init_hx.cuda()

        enc_init_cx = torch.zeros(hidden_dim, dtype=torch.double)
        if self.use_cuda:
            enc_init_cx = enc_init_cx.cuda()

        enc_init_hx = nn.Parameter(enc_init_hx, requires_grad=True)
        enc_init_cx = nn.Parameter(enc_init_cx, requires_grad=True)
        return enc_init_hx, enc_init_cx


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

    def apply_chosen_elements_mask(self, logits, mask, prev_idxs):
        if mask is None:
            mask = torch.zeros(logits.size()).bool()
            if self.use_cuda:
                mask = mask.cuda()

        maskk = mask.clone()

        # to prevent them from being reselected. 
        # Or, allow re-selection and penalize in the objective function
        if prev_idxs is not None:
            # set most recently selected idx values to 1
            maskk[range(logits.size(0)), prev_idxs.data] = 1
            # maskk[[x for x in range(logits.size(0))], prev_idxs.data] = 1
            # since the first element can be reused never zero the first prob to avoid future nans
            maskk[range(logits.size(0)), torch.zeros_like(prev_idxs)] = 0
            logits[maskk] = -np.inf

        return logits, maskk

    def forward(self, decoder_input, embedded_inputs, hidden, context):
        """
        Args:
            decoder_input: The initial input to the decoder
                size is [batch_size, embedding_dim]. Trainable parameter.
            embedded_inputs: [source_length, batch_size, embedding_dim]
            hidden: the prev hidden state, size is [batch_size, hidden_dim].
                Initially this is set to (enc_h[-1], enc_c[-1])
            context: encoder outputs, [source_length, batch_size, hidden_dim]
        """
        def recurrence(x, hidden, chosen_elements_mask, prev_idxs):
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
                logits, chosen_elements_mask = self.apply_chosen_elements_mask(logits, chosen_elements_mask, prev_idxs)
                # [batch_size, h_dim, source_length] * [batch_size, source_length, 1] = [batch_size, h_dim, 1]
                g_l = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
            _, logits = self.pointer(g_l, context)

            logits, chosen_elements_mask = self.apply_chosen_elements_mask(logits, chosen_elements_mask, prev_idxs)

            probs = self.sm(logits)
            return hy, cy, probs, chosen_elements_mask

        outputs = []
        selections = []
        idxs = None
        chosen_elements_mask = None
        batch_size = context.size(1)
        finished_batches_mask = torch.ones(batch_size).int()  # batches indices that have already finished
        if self.use_cuda:
            finished_batches_mask = finished_batches_mask.cuda()

        for _ in range(self.max_length):
            hx, cx, probs, chosen_elements_mask = recurrence(decoder_input, hidden, chosen_elements_mask, idxs)
            hidden = (hx, cx)

            idxs = torch.distributions.Categorical(probs).sample()

            idxs = idxs * finished_batches_mask
            finished_batches_mask = finished_batches_mask * idxs.bool().int()

            decoder_input = embedded_inputs[idxs.data, range(batch_size), :]
            # use outs to point to next object
            outputs.append(probs)
            selections.append(idxs)

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
                 encoder_num_layers):
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

    def forward(self, inputs):
        """ Propagate inputs through the network
        Args: 
            inputs: [source_length, batch_size, embedding_dim]
        """
        (encoder_hx, encoder_cx) = self.encoder.enc_init_state
        if self.encoder_bi_directional or self.encoder_num_layers > 1:
            c = 2 if self.encoder_bi_directional else 1
            encoder_hx = encoder_hx.unsqueeze(0).repeat(c * self.encoder_num_layers, inputs.size(1), 1)
            encoder_cx = encoder_cx.unsqueeze(0).repeat(c * self.encoder_num_layers, inputs.size(1), 1)
        else:
            encoder_hx = encoder_hx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)
            encoder_cx = encoder_cx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)

        # encoder forward pass
        enc_h, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_hx, encoder_cx))

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
                                                                 enc_h)

        return pointer_probs, input_idxs


class CriticNetwork(nn.Module):
    """Useful as a baseline in REINFORCE updates"""

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 tanh_exploration,
                 use_tanh,
                 use_cuda,
                 n_process_block_iters=3):
        super(CriticNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_process_block_iters = n_process_block_iters

        self.encoder = Encoder(
            input_dim,
            hidden_dim,
            use_cuda,
            False,
            1)

        self.process_block = Attention(hidden_dim, use_tanh=use_tanh, C=tanh_exploration, use_cuda=use_cuda)
        self.sm = nn.Softmax(dim=-1)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).double()

        if use_cuda:
            self.sm = self.sm.cuda()
            self.decoder = self.decoder.cuda()

    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size, window_size, event_size]
        """
        inputs = inputs.transpose(0, 1)
        # inputs: [window_size, batch_size, event_size]
        (encoder_hx, encoder_cx) = self.encoder.enc_init_state
        encoder_hx = encoder_hx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)
        encoder_cx = encoder_cx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)

        # encoder forward pass
        enc_outputs, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_hx, encoder_cx))

        # grab the hidden state and process it via the process block
        process_block_state = enc_h_t[-1]
        for i in range(self.n_process_block_iters):
            ref, logits = self.process_block(process_block_state, enc_outputs)
            process_block_state = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)

        # produce the final scalar output
        out = self.decoder(process_block_state).squeeze(1)
        return out


class NeuralCombOptNet(nn.Module):
    """
    This module contains the PointerNetwork (actor) and
    CriticNetwork (critic). It requires
    an application-specific reward function
    """

    def __init__(self,
                 input_dim,
                 embedding_dim,
                 hidden_dim,
                 max_decoding_len,
                 n_glimpses,
                 tanh_exploration,
                 use_tanh,
                 is_train,
                 use_cuda,
                 encoder_bi_directional,
                 encoder_num_layers,
                 padding_value):

        super(NeuralCombOptNet, self).__init__()
        self.input_dim = input_dim
        self.is_train = is_train
        self.use_cuda = use_cuda

        self.actor_net = PointerNetwork(
            embedding_dim if embedding_dim is not None else input_dim,
            hidden_dim,
            max_decoding_len,
            n_glimpses,
            tanh_exploration,
            use_tanh,
            use_cuda,
            encoder_bi_directional,
            encoder_num_layers)

        # self.critic_net = CriticNetwork(
        #        embedding_dim,
        #        hidden_dim,
        #        n_process_block_iters,
        #        tanh_exploration,
        #        False,
        #        use_cuda)

        self.embedding = None
        if embedding_dim is not None:
            embedding_ = torch.empty(size=(input_dim, embedding_dim), dtype=torch.double) \
                .uniform_(-(1. / math.sqrt(embedding_dim)), 1. / math.sqrt(embedding_dim))
            if self.use_cuda:
                embedding_ = embedding_.cuda()
            self.embedding = nn.Parameter(embedding_, requires_grad=True)

        self.padding_value = padding_value

    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size, input_dim, source_length]
        """
        inputs = inputs.transpose(1, 2)
        batch_size = inputs.size(0)
        # input_dim = inputs.size(1)
        source_length = inputs.size(2)

        if self.padding_value is not None:
            beginning_padding = torch.full(size=(inputs.size(0), inputs.size(1), 1),
                                           fill_value=self.padding_value, dtype=inputs.dtype)
            if self.use_cuda:
                beginning_padding = beginning_padding.cuda()
            inputs = torch.cat((beginning_padding, inputs), dim=2)

        # inputs: [batch_size, input_dim, source_length + 1]

        # repeat embeddings across batch_size
        # result is [batch_size, input_dim, embedding_dim]
        if self.embedding is not None:
            embedding = self.embedding.repeat(batch_size, 1, 1)
            embedded_inputs = []
            # result is [batch_size, 1, input_dim, source_length]
            ips = inputs.unsqueeze(1)

            for i in range(source_length):
                # [batch_size, 1, input_dim] * [batch_size, input_dim, embedding_dim]
                # result is [batch_size, embedding_dim]
                embedded_inputs.append(torch.bmm(ips[:, :, :, i].double(), embedding).squeeze(1))

            # Result is [source_length, batch_size, embedding_dim]
            embedded_inputs = torch.cat(embedded_inputs).view(source_length, batch_size, embedding.size(2))

            # query the actor net for the input indices
            # making up the output, and the pointer attn
            embedded_inputs.to(dtype=torch.double)
        else:
            # Result is [source_length, batch_size, embedding_dim]
            embedded_inputs = inputs.transpose(0, 2).transpose(1, 2)
            embedded_inputs.to(dtype=torch.double)

        t1 = time.perf_counter()
        probs_, actions_idxs = self.actor_net(embedded_inputs)
        t2 = time.perf_counter()

        if self.is_train:
            # probs_ is a list of len source_length of [batch_size, source_length]
            probs = []
            for prob, action_id in zip(probs_, actions_idxs):
                probs.append(prob[[x for x in range(batch_size)], action_id.data])
        else:
            probs = None

        # get the critic value fn estimates for the baseline
        # [batch_size]
        # v = self.critic_net(embedded_inputs)

        # return v, probs, actions, actions_idxs

        log_probs = torch.zeros_like(probs[0], device=dataset.device)

        finished_batches_mask = torch.ones_like(probs[0], device=dataset.device).int()

        for prob, idxs in zip(probs, actions_idxs):
            # compute the sum of the log probs
            # for each tour in the batch
            log_prob = torch.log(prob)
            # zero the log_prob where previous(!) batch action is 0 (=finished)
            log_prob = log_prob * finished_batches_mask
            finished_batches_mask = finished_batches_mask * idxs.bool().int()
            log_probs += log_prob

        chosen_events = torch.zeros((batch_size, constants['window_size'] + 1))
        for i in range(constants['window_size'] + 1):
            idxs = actions_idxs[i]
            chosen_events[range(batch_size), idxs] = 1

        chosen_events = chosen_events[:, 1:]
        chosen_events = chosen_events.numpy()
        return chosen_events, log_probs, t2 - t1
