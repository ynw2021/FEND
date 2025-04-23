""" This code is based on the Trajectron++ repository.

    For usage, see the License of Trajectron++ under:
    https://github.com/StanfordASL/Trajectron-plus-plus
"""
import torch.nn as nn
import torch.nn.functional as F

from Trajectron_plus_plus.trajectron.model.components import *
from Trajectron_plus_plus.trajectron.model.model_utils import *
from Trajectron_plus_plus.trajectron.model.mgcvae import MultimodalGenerativeCVAE
import utilities

import torch
import torch.distributions as td
import numpy as np

import faiss
import torch.nn as nn

from random import sample
from Trajectron_plus_plus.trajectron.model.hyperlstm import HyperLSTM_re as HyperLSTM

class AdditiveAttention(nn.Module):
    # Implementing the attention module of Bahdanau et al. 2015 where
    # score(h_j, s_(i-1)) = v . tanh(W_1 h_j + W_2 s_(i-1))
    def __init__(self, encoder_hidden_state_dim, decoder_hidden_state_dim, internal_dim=None):
        super(AdditiveAttention, self).__init__()

        if internal_dim is None:
            internal_dim = int((encoder_hidden_state_dim + decoder_hidden_state_dim) / 2)

        self.w1 = nn.Linear(encoder_hidden_state_dim, internal_dim, bias=False)
        self.w2 = nn.Linear(decoder_hidden_state_dim, internal_dim, bias=False)
        self.v = nn.Linear(internal_dim, 1, bias=False)

    def score(self, encoder_state, decoder_state):
        # encoder_state is of shape (batch, enc_dim)
        # decoder_state is of shape (batch, dec_dim)
        # return value should be of shape (batch, 1)
        return self.v(torch.tanh(self.w1(encoder_state) + self.w2(decoder_state)))

    def forward(self, encoder_states, decoder_state):
        # encoder_states is of shape (batch, num_enc_states, enc_dim)
        # decoder_state is of shape (batch, dec_dim)
        score_vec = torch.cat([self.score(encoder_states[:, i], decoder_state) for i in range(encoder_states.shape[1])],
                              dim=1)
        # score_vec is of shape (batch, num_enc_states)

        attention_probs = torch.unsqueeze(F.softmax(score_vec, dim=1), dim=2)
        # attention_probs is of shape (batch, num_enc_states, 1)

        final_context_vec = torch.sum(attention_probs * encoder_states, dim=1)
        # final_context_vec is of shape (batch, enc_dim)

        return final_context_vec, attention_probs

class DiscreteLatent(object):
    def __init__(self, hyperparams, device):
        self.hyperparams = hyperparams
        self.z_dim = hyperparams['N'] * hyperparams['K']
        self.N = hyperparams['N']
        self.K = hyperparams['K']
        self.kl_min = hyperparams['kl_min']
        self.device = device
        self.temp = None  # filled in by MultimodalGenerativeCVAE.set_annealing_params
        self.z_logit_clip = None  # filled in by MultimodalGenerativeCVAE.set_annealing_params
        self.p_dist = None  # filled in by MultimodalGenerativeCVAE.encoder
        self.q_dist = None  # filled in by MultimodalGenerativeCVAE.encoder

    def dist_from_h(self, h, mode):
        logits_separated = torch.reshape(h, (-1, self.N, self.K))
        logits_separated_mean_zero = logits_separated - torch.mean(logits_separated, dim=-1, keepdim=True)
        if self.z_logit_clip is not None and mode == ModeKeys.TRAIN:
            c = self.z_logit_clip
            logits = torch.clamp(logits_separated_mean_zero, min=-c, max=c)
        else:
            logits = logits_separated_mean_zero

        return td.OneHotCategorical(logits=logits)

    def sample_q(self, num_samples, mode):
        bs = self.p_dist.probs.size()[0]
        num_components = self.N * self.K
        z_NK = torch.from_numpy(self.all_one_hot_combinations(self.N, self.K)).float().to(self.device).repeat(num_samples, bs)
        return torch.reshape(z_NK, (num_samples * num_components, -1, self.z_dim))

    def sample_p(self, num_samples, mode, most_likely_z=False, full_dist=True, all_z_sep=False):
        num_components = 1
        if full_dist:
            bs = self.p_dist.probs.size()[0]
            z_NK = torch.from_numpy(self.all_one_hot_combinations(self.N, self.K)).float().to(self.device).repeat(num_samples, bs)
            num_components = self.K ** self.N
            k = num_samples * num_components
        elif all_z_sep:
            bs = self.p_dist.probs.size()[0]
            z_NK = torch.from_numpy(self.all_one_hot_combinations(self.N, self.K)).float().to(self.device).repeat(1, bs)
            k = self.K ** self.N
            num_samples = k
        elif most_likely_z:
            # Sampling the most likely z from p(z|x).
            eye_mat = torch.eye(self.p_dist.event_shape[-1], device=self.device)
            argmax_idxs = torch.argmax(self.p_dist.probs, dim=2)
            z_NK = torch.unsqueeze(eye_mat[argmax_idxs], dim=0).expand(num_samples, -1, -1, -1)
            k = num_samples
        else:
            z_NK = self.p_dist.sample((num_samples,))
            k = num_samples

        if mode == ModeKeys.PREDICT:
            return torch.reshape(z_NK, (k, -1, self.N * self.K)), num_samples, num_components
        else:
            return torch.reshape(z_NK, (k, -1, self.N * self.K))

    def kl_q_p(self, log_writer=None, prefix=None, curr_iter=None):
        kl_separated = td.kl_divergence(self.q_dist, self.p_dist)
        if len(kl_separated.size()) < 2:
            kl_separated = torch.unsqueeze(kl_separated, dim=0)

        kl_minibatch = torch.mean(kl_separated, dim=0, keepdim=True)

        if log_writer is not None:
            log_writer.add_scalar(prefix + '/true_kl', torch.sum(kl_minibatch), curr_iter)

        if self.kl_min > 0:
            kl_lower_bounded = torch.clamp(kl_minibatch, min=self.kl_min)
            kl = torch.sum(kl_lower_bounded)
        else:
            kl = torch.sum(kl_minibatch)

        return kl

    def q_log_prob(self, z):
        k = z.size()[0]
        z_NK = torch.reshape(z, [k, -1, self.N, self.K])
        return torch.sum(self.q_dist.log_prob(z_NK), dim=2)

    def p_log_prob(self, z):
        k = z.size()[0]
        z_NK = torch.reshape(z, [k, -1, self.N, self.K])
        return torch.sum(self.p_dist.log_prob(z_NK), dim=2)

    def get_p_dist_probs(self):
        return self.p_dist.probs

    @staticmethod
    def all_one_hot_combinations(N, K):
        return np.eye(K).take(np.reshape(np.indices([K] * N), [N, -1]).T, axis=0).reshape(-1, N * K)  # [K**N, N*K]

    def summarize_for_tensorboard(self, log_writer, prefix, curr_iter):
        log_writer.add_histogram(prefix + "/latent/p_z_x", self.p_dist.probs, curr_iter)
        log_writer.add_histogram(prefix + "/latent/q_z_xy", self.q_dist.probs, curr_iter)
        log_writer.add_histogram(prefix + "/latent/p_z_x_logits", self.p_dist.logits, curr_iter)
        log_writer.add_histogram(prefix + "/latent/q_z_xy_logits", self.q_dist.logits, curr_iter)
        if self.z_dim <= 9:
            for i in range(self.N):
                for j in range(self.K):
                    log_writer.add_histogram(prefix + "/latent/q_z_xy_logit{0}{1}".format(i, j),
                                             self.q_dist.logits[:, i, j],
                                             curr_iter)





def contrastive_three_modes_loss(features, scores, temp=0.1, base_temperature=0.07,change=False):
    device = (torch.device('cuda') if features.is_cuda
              else torch.device('cpu'))
    batch_size = features.shape[0]
    scores = scores.contiguous().view(-1, 1)
    if change:
        mask_positives = (torch.abs(scores.sub(scores.T)) < 0.01).float().to(device)
        mask_negatives = (torch.abs(scores.sub(scores.T)) > 0.5).float().to(device)
    else:
        mask_positives = (torch.abs(scores.sub(scores.T)) < 0.1).float().to(device)
        mask_negatives = (torch.abs(scores.sub(scores.T)) > 2.0).float().to(device)
    mask_neutral = mask_positives + mask_negatives

    anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temp)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()   ###todo:this do normailze?

    logits_mask = torch.scatter(
        torch.ones_like(mask_positives), 1,
        torch.arange(batch_size).view(-1, 1).to(device), 0) * mask_neutral
    mask_positives = mask_positives * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-20)
    mean_log_prob_pos = (mask_positives * log_prob).sum(1) / (mask_positives.sum(1) + 1e-20)

    loss = - (temp / base_temperature) * mean_log_prob_pos
    loss = loss.view(1, batch_size).mean()
    return loss, mask_positives.sum(1).mean(), mask_negatives.sum(1).mean()


class MultimodalGenerativeCVAEEWTA(MultimodalGenerativeCVAE):
    def __init__(self,
                 env,
                 node_type,
                 model_registrar,
                 hyperparams,
                 device,
                 edge_types,
                 log_writer=None):
        super().__init__(
            env, node_type, model_registrar, hyperparams,
            device, edge_types, log_writer)
        dynamic_class = getattr(utilities, hyperparams['dynamic'][self.node_type]['name'])
        dyn_limits = hyperparams['dynamic'][self.node_type]['limits']
        self.dynamic = dynamic_class(self.env.scenes[0].dt, dyn_limits, device,
                                     self.model_registrar, self.x_size, self.node_type)

    def create_node_models(self):
        self.add_submodule(self.node_type + '/node_history_encoder',
                           model_if_absent=nn.LSTM(input_size=self.state_length,
                                                   hidden_size=self.hyperparams['enc_rnn_dim_history'],
                                                   batch_first=True))
        if self.hyperparams['edge_encoding']:
            if self.hyperparams['edge_influence_combine_method'] == 'bi-rnn':
                self.add_submodule(self.node_type + '/edge_influence_encoder',
                                   model_if_absent=nn.LSTM(input_size=self.hyperparams['enc_rnn_dim_edge'],
                                                           hidden_size=self.hyperparams['enc_rnn_dim_edge_influence'],
                                                           bidirectional=True,
                                                           batch_first=True))
                self.eie_output_dims = 4 * self.hyperparams['enc_rnn_dim_edge_influence']

            elif self.hyperparams['edge_influence_combine_method'] == 'attention':
                self.add_submodule(self.node_type + '/edge_influence_encoder',
                                   model_if_absent=AdditiveAttention(
                                       encoder_hidden_state_dim=self.hyperparams['enc_rnn_dim_edge_influence'],
                                       decoder_hidden_state_dim=self.hyperparams['enc_rnn_dim_history']))
                self.eie_output_dims = self.hyperparams['enc_rnn_dim_edge_influence']
        if self.hyperparams['use_map_encoding']:
            if self.node_type in self.hyperparams['map_encoder']:
                me_params = self.hyperparams['map_encoder'][self.node_type]
                self.add_submodule(self.node_type + '/map_encoder',
                                   model_if_absent=CNNMapEncoder(me_params['map_channels'],
                                                                 me_params['hidden_channels'],
                                                                 me_params['output_size'],
                                                                 me_params['masks'],
                                                                 me_params['strides'],
                                                                 me_params['patch_size']))
        self.latent = DiscreteLatent(self.hyperparams, self.device)

        x_size = self.hyperparams['enc_rnn_dim_history']
        if self.hyperparams['edge_encoding']:
            x_size += self.eie_output_dims
        if self.hyperparams['use_map_encoding'] and self.node_type in self.hyperparams['map_encoder']:
            x_size += self.hyperparams['map_encoder'][self.node_type]['output_size']

        z_size = self.hyperparams['N'] * self.hyperparams['K']

        if self.hyperparams['p_z_x_MLP_dims'] is not None:
            self.add_submodule(self.node_type + '/p_z_x',
                               model_if_absent=nn.Linear(x_size, self.hyperparams['p_z_x_MLP_dims']))
            hx_size = self.hyperparams['p_z_x_MLP_dims']
        else:
            hx_size = x_size

        self.add_submodule(self.node_type + '/hx_to_z',
                           model_if_absent=nn.Linear(hx_size, self.latent.z_dim))

        if self.hyperparams['q_z_xy_MLP_dims'] is not None:
            self.add_submodule(self.node_type + '/q_z_xy',
                               model_if_absent=nn.Linear(x_size + 4 * self.hyperparams['enc_rnn_dim_future'],
                                                         self.hyperparams['q_z_xy_MLP_dims']))
            hxy_size = self.hyperparams['q_z_xy_MLP_dims']
        else:
            hxy_size = x_size + 4 * self.hyperparams['enc_rnn_dim_future']

        self.add_submodule(self.node_type + '/hxy_to_z',
                           model_if_absent=nn.Linear(hxy_size, self.latent.z_dim))

        decoder_input_dims = self.pred_state_length * 20 + x_size
        self.add_submodule(self.node_type + '/decoder/state_action',
                           model_if_absent=nn.Sequential(
                               nn.Linear(self.state_length, self.pred_state_length)))
        self.add_submodule(self.node_type + '/decoder/rnn_cell',
                           model_if_absent=HyperLSTM(decoder_input_dims, self.hyperparams['dec_rnn_dim'],self.hyperparams['hyper_rnn_dim'],self.hyperparams['hyper_z_dim']))
        self.add_submodule(self.node_type + '/decoder/initial_h',
                           model_if_absent=nn.Linear(x_size, self.hyperparams['dec_rnn_dim']))

        if self.node_type =='VEHICLE' and self.hyperparams['use_map_encoding'] and self.node_type in self.hyperparams['map_encoder']:
              self.add_submodule(self.node_type + '/decoder/initial_hyper_h',
                           model_if_absent=nn.Linear(264, self.hyperparams['hyper_rnn_dim']))
        else:
            self.add_submodule(self.node_type + '/decoder/initial_hyper_h',
                               model_if_absent=nn.Linear(232, self.hyperparams['hyper_rnn_dim']))

        self.add_submodule(self.node_type + '/decoder/proj_to_GMM_mus',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     20 * self.pred_state_length))
        self.x_size = x_size
        self.z_size = z_size


        if self.node_type =='VEHICLE' and self.hyperparams['use_map_encoding'] and self.node_type in self.hyperparams['map_encoder']:
            # self.add_submodule(self.node_type + '/con_head',
            #                model_if_absent=nn.Linear(264, 232))   #todo:mlp
            self.add_submodule(self.node_type + '/con_head',
                           model_if_absent=nn.Sequential(nn.Linear(264,1024),
                                                         nn.ReLU(),
                                                         nn.Linear(1024,232)))   #todo:mlp
        else:
            self.add_submodule(self.node_type + '/con_head',
                               model_if_absent=nn.Linear(232, 232))  # todo:mlp

        self.add_submodule(self.node_type + '/cluster_con_head',
                               model_if_absent=nn.Linear(232, 232))  # todo:mlp

        # self.add_submodule(self.node_type + '/input_query_w',
        # model_if_absent =MLP(232,232,(512,512))
        # )
        # self.add_submodule(self.node_type + '/past_memory_w',
        #                    model_if_absent=MLP(232, 232, (512, 512))
        #                    )



    def obtain_encoded_tensors(self, mode, inputs, inputs_st, labels, labels_st,
                               first_history_indices, neighbors,
                               neighbors_edge_value, robot, map):
        initial_dynamics = dict()
        batch_size = inputs.shape[0]
        node_history = inputs
        node_pos = inputs[:, -1, 0:2]
        node_vel = inputs[:, -1, 2:4]
        node_history_st = inputs_st
        node_present_state_st = inputs_st[:, -1]
        n_s_t0 = node_present_state_st

        initial_dynamics['pos'] = node_pos
        initial_dynamics['vel'] = node_vel
        self.dynamic.set_initial_condition(initial_dynamics)

        node_history_encoded = self.encode_node_history(mode,
                                                        node_history_st,
                                                        first_history_indices)
        if self.hyperparams['edge_encoding']:
            node_edges_encoded = list()
            for edge_type in self.edge_types:
                encoded_edges_type = self.encode_edge(mode,
                                                      node_history,
                                                      node_history_st,
                                                      edge_type,
                                                      neighbors[edge_type],
                                                      neighbors_edge_value[edge_type],
                                                      first_history_indices)
                node_edges_encoded.append(encoded_edges_type)
            total_edge_influence = self.encode_total_edge_influence(mode,
                                                                    node_edges_encoded,
                                                                    node_history_encoded,
                                                                    batch_size)
        if self.hyperparams['use_map_encoding'] and self.node_type in self.hyperparams['map_encoder']:
            encoded_map = self.node_modules[self.node_type + '/map_encoder'](map * 2. - 1., (mode == ModeKeys.TRAIN))
            do = self.hyperparams['map_encoder'][self.node_type]['dropout']
            encoded_map = F.dropout(encoded_map, do, training=(mode == ModeKeys.TRAIN))

        x_concat_list = list()
        if self.hyperparams['edge_encoding']:
            x_concat_list.append(total_edge_influence)
        x_concat_list.append(node_history_encoded)
        if self.hyperparams['use_map_encoding'] and self.node_type in self.hyperparams['map_encoder']:
            x_concat_list.append(encoded_map)
        x = torch.cat(x_concat_list, dim=1)
        return x, n_s_t0

    def project_to_GMM_params(self, tensor):
        mus = self.node_modules[self.node_type + '/decoder/proj_to_GMM_mus'](tensor)
        return mus

    def p_y_xz(self, x, n_s_t0, prediction_horizon):
        ph = prediction_horizon  # 12
        cell = self.node_modules[self.node_type + '/decoder/rnn_cell']
        initial_h_model = self.node_modules[self.node_type + '/decoder/initial_h']
        initial_state = initial_h_model(x)
        mus = []
        a_0 = self.node_modules[self.node_type + '/decoder/state_action'](n_s_t0)
        state = initial_state
        c_state=initial_state
        input_ = torch.cat([x, a_0.repeat(1, 20)], dim=1)
        features = torch.cat([input_, state], dim=1)
        contrast_features = F.normalize(self.node_modules[self.node_type + '/con_head'](features),dim=-1)

        # contrast_features=torch.ones_like(features)
        hyper_initial_h_model = self.node_modules[self.node_type + '/decoder/initial_hyper_h']
        hyper_initial_state = hyper_initial_h_model(features)
        hyper_state = hyper_initial_state
        hyper_c_state = hyper_initial_state

        for j in range(ph):
            _,h_state,c_state,hyper_h_state,hyper_c_state = cell(input_,features,state,c_state,hyper_state,hyper_c_state)
            mu_t = self.project_to_GMM_params(h_state)
            mus.append(mu_t.reshape(-1, 20, 2))
            dec_inputs = [x, mu_t]
            input_ = torch.cat(dec_inputs, dim=1)
            state = h_state
            hyper_state=hyper_h_state
        mus = torch.stack(mus, dim=2)
        y = self.dynamic.integrate_samples(mus, x)
        return y, features

    def p_y_xz_futurefeature(self, x, n_s_t0, future_feats,prediction_horizon):
        ph = prediction_horizon  # 12
        cell = self.node_modules[self.node_type + '/decoder/rnn_cell']
        initial_h_model = self.node_modules[self.node_type + '/decoder/initial_h']
        initial_state = initial_h_model(x)
        hyper_initial_h_model =self.node_modules[self.node_type + '/decoder/initial_hyper_h']
        hyper_initial_state=hyper_initial_h_model(x)
        mus = []
        a_0 = self.node_modules[self.node_type + '/decoder/state_action'](n_s_t0)
        state = initial_state
        c_state=initial_state
        hyper_state=hyper_initial_state
        hyper_c_state = hyper_initial_state
        # input_ = torch.cat([x, a_0.repeat(1, 20),future_feats], dim=1)
        input_ = torch.cat([x, a_0.repeat(1, 20)], dim=1)
        features = torch.cat([input_, state], dim=1)

        for j in range(ph):
            _,h_state,c_state,hyper_h_state,hyper_c_state = cell(input_,future_feats,state,c_state,hyper_state,hyper_c_state)
            mu_t = self.project_to_GMM_params(h_state)
            mus.append(mu_t.reshape(-1, 20, 2))
            # dec_inputs = [x, mu_t,future_feats]
            dec_inputs=[x, mu_t]
            input_ = torch.cat(dec_inputs, dim=1)
            state = h_state
            hyper_state=hyper_h_state
        mus = torch.stack(mus, dim=2)
        y = self.dynamic.integrate_samples(mus, x)
        return y, features





    def decoder(self, x, n_s_t0, prediction_horizon):
        y, features = self.p_y_xz(x, n_s_t0, prediction_horizon)
        return y, features

    def decoder_futurefeature(self,x, n_s_t0,future_feats, prediction_horizon):
        y, features = self.p_y_xz_futurefeature(x, n_s_t0,future_feats, prediction_horizon)
        return y, features

    def ewta_loss(self, y, labels, mode='epe-all', top_n=1):
        # y has shape (bs, 20, 12 ,2)
        # labels has shape (bs, 12, 2)
        gts = torch.stack([labels for i in range(20)], dim=1)  # (bs, 20, 12, 2)
        diff = (y - gts) ** 2
        channels_sum = torch.sum(diff, dim=3)  # (bs, 20, 12)
        spatial_epes = torch.sqrt(channels_sum + 1e-20)  # (bs, 20, 12)

        sum_spatial_epe = torch.zeros(spatial_epes.shape[0])
        if mode=='epe-top-n' and top_n==1 :
            mode='epe'
        if mode == 'epe':
            spatial_epe, _ = torch.min(spatial_epes, dim=1)  # (bs, 12)
            sum_spatial_epe = torch.sum(spatial_epe, dim=1)
        elif mode == 'epe-top-n' and top_n > 1:
            spatial_epes_min, _ = torch.topk(-1 * spatial_epes, top_n, dim=1)
            spatial_epes_min = -1 * spatial_epes_min  # (bs, top_n, 12)
            sum_spatial_epe = torch.sum(spatial_epes_min, dim=(1, 2))
        elif mode == 'epe-all':
            sum_spatial_epe = torch.sum(spatial_epes, dim=(1, 2))

        return torch.mean(sum_spatial_epe)

    def train_loss(self,
                   inputs,
                   inputs_st,
                   first_history_indices,
                   labels,
                   labels_st,
                   neighbors,
                   neighbors_edge_value,
                   robot,
                   map,
                   prediction_horizon,
                   loss_type,
                   score,
                   contrastive=False,
                   factor_con=100,
                   temp=0.1,
                   change=False):
        mode = ModeKeys.TRAIN
        x, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                inputs=inputs,
                                                inputs_st=inputs_st,
                                                labels=labels,
                                                labels_st=labels_st,
                                                first_history_indices=first_history_indices,
                                                neighbors=neighbors,
                                                neighbors_edge_value=neighbors_edge_value,
                                                robot=robot,
                                                map=map)
        y, features = self.decoder(x, n_s_t0, prediction_horizon)
        features = F.normalize(self.node_modules[self.node_type + '/con_head'](features), dim=1)
        mode, top_n = loss_type, 1
        if 'top' in loss_type:
            mode = 'epe-top-n'
            top_n = int(loss_type.replace('epe-top-', ''))
        loss = self.ewta_loss(y, labels, mode=mode, top_n=top_n)
        if contrastive:
            con_loss, positive, negative = contrastive_three_modes_loss(features, score, temp=temp,change=change)
            final_loss = loss + factor_con * con_loss
            if self.log_writer is not None:
                self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'contrastive_loss'),
                                           con_loss, self.curr_iter)
                self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'positives'),
                                           positive, self.curr_iter)
                self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'negatives'),
                                           negative, self.curr_iter)
        else:
            final_loss = loss

        if self.log_writer is not None:
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'loss'),
                                       loss, self.curr_iter)
        return final_loss

    def predict(self,
                inputs,
                inputs_st,
                first_history_indices,
                neighbors,
                neighbors_edge_value,
                robot,
                map,
                prediction_horizon,
                return_original=False,
                return_feat=True):
        mode = ModeKeys.PREDICT
        x, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                inputs=inputs,
                                                inputs_st=inputs_st,
                                                labels=None,
                                                labels_st=None,
                                                first_history_indices=first_history_indices,
                                                neighbors=neighbors,
                                                neighbors_edge_value=neighbors_edge_value,
                                                robot=robot,
                                                map=map)
        y, features = self.decoder(x, n_s_t0, prediction_horizon)
        features_original=features[:]
        features = F.normalize(self.node_modules[self.node_type + '/con_head'](features), dim=1)
        cluster_features=F.normalize(self.node_modules[self.node_type + '/cluster_con_head'](features), dim=1)
        if return_original:
            return y, features,features_original
        else:
            return y , features,cluster_features

    def predict_social(self,
                inputs,
                inputs_st,
                first_history_indices,
                neighbors,
                neighbors_edge_value,
                robot,
                map,
                prediction_horizon,
                return_original=False,
                return_feat=True):
        mode = ModeKeys.PREDICT
        x, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                inputs=inputs,
                                                inputs_st=inputs_st,
                                                labels=None,
                                                labels_st=None,
                                                first_history_indices=first_history_indices,
                                                neighbors=neighbors,
                                                neighbors_edge_value=neighbors_edge_value,
                                                robot=robot,
                                                map=map)
        y, features = self.decoder(x, n_s_t0, prediction_horizon)
        features = F.normalize(self.node_modules[self.node_type + '/con_head'](features), dim=1)
        social_feature=x[:,:32]
        normalized_social_feature=F.normalize(social_feature[:,:32],dim=1)
        return social_feature,normalized_social_feature,features






    def pcl_contrast_loss(self,
                   inputs,
                   inputs_st,
                   first_history_indices,
                   labels,
                   labels_st,
                   neighbors,
                   neighbors_edge_value,
                   robot,
                   map,
                   prediction_horizon,
                   loss_type,
                   cluster_result,
                   index,
                   factor_con=100,
                   original=False,
                   temp=0.1):   #####todo:temp

        mode = ModeKeys.PREDICT
        x, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                inputs=inputs,
                                                inputs_st=inputs_st,
                                                labels=None,
                                                labels_st=None,
                                                first_history_indices=first_history_indices,
                                                neighbors=neighbors,
                                                neighbors_edge_value=neighbors_edge_value,
                                                robot=robot,
                                                map=map)
        y, features = self.decoder(x, n_s_t0, prediction_horizon)
        if not original:
             features = F.normalize(self.node_modules[self.node_type + '/con_head'](features), dim=1)
        else:
            features=F.normalize(features, dim=1)
        mode, top_n = loss_type, 1
        if 'top' in loss_type:
            mode = 'epe-top-n'
            top_n = int(loss_type.replace('epe-top-', ''))
        loss = self.ewta_loss(y, labels, mode=mode, top_n=top_n)

        criterion=nn.CrossEntropyLoss()

        if cluster_result is not None:
            proto_labels=[]
            proto_logits=[]
            for n,(tra2cluster,prototypes,density) in enumerate(zip(cluster_result['tra2cluster'],cluster_result['centroids'],cluster_result['density'])):
                pos_proto_id=tra2cluster[index]
                pos_prototypes=prototypes[pos_proto_id]

                all_proto_id=[i for i in range(tra2cluster.max())]
                neg_proto_id=set(all_proto_id)-set(pos_proto_id.tolist())
                neg_proto_id=sample(neg_proto_id,len(neg_proto_id))
                neg_prototypes=prototypes[neg_proto_id]

                proto_selected=torch.cat([pos_prototypes,neg_prototypes],dim=0)

                logits_proto=torch.mm(features,proto_selected.t())

                labels_proto=torch.linspace(0,features.size(0)-1,steps=features.size(0)).long().cuda()

                temp_proto=density[torch.cat([pos_proto_id,torch.LongTensor(neg_proto_id).cuda()],dim=0)]
                logits_proto /= temp_proto

                proto_labels.append(labels_proto)
                proto_logits.append(logits_proto)

            con_loss=0
            for n,(labels_proto,logits_proto) in enumerate(zip(proto_labels,proto_logits)):
                con_loss += criterion(logits_proto,labels_proto)

            con_loss  /= len(proto_labels)

        final_loss = loss + factor_con * con_loss
        if self.log_writer is not None:
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'contrastive_loss'),
                                       con_loss, self.curr_iter)

        if self.log_writer is not None:
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'ewta_loss'),
                                       loss, self.curr_iter)

        return final_loss

    def pcl_contrast_loss_reweight_by_loss_justlongtail(self,
                                           inputs,
                                           inputs_st,
                                           first_history_indices,
                                           labels,
                                           labels_st,
                                           neighbors,
                                           neighbors_edge_value,
                                           robot,
                                           map,
                                           prediction_horizon,
                                           loss_type,
                                           predict_loss,
                                           cluster_result_with_future,
                                           cluster_result,
                                           index,
                                           factor_con=100,
                                           original=False,
                                           temp=0.1):  #####todo:temp

        mode = ModeKeys.PREDICT
        x, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                inputs=inputs,
                                                inputs_st=inputs_st,
                                                labels=None,
                                                labels_st=None,
                                                first_history_indices=first_history_indices,
                                                neighbors=neighbors,
                                                neighbors_edge_value=neighbors_edge_value,
                                                robot=robot,
                                                map=map)
        y, features = self.decoder(x, n_s_t0, prediction_horizon)
        if not original:
            features = F.normalize(self.node_modules[self.node_type + '/con_head'](features), dim=1)
        else:
            features = F.normalize(features, dim=1)
        mode, top_n = loss_type, 1
        if 'top' in loss_type:
            mode = 'epe-top-n'
            top_n = int(loss_type.replace('epe-top-', ''))
        loss = self.ewta_loss(y, labels, mode=mode, top_n=top_n)

        thres = 0.0

        mask = predict_loss > thres  ####tensor
        idx_mask = index[mask]
        idx_res = index[mask.logical_not()]

        criterion = nn.CrossEntropyLoss(reduction='none')

        if cluster_result is not None:
            proto_labels = []
            proto_logits = []
            proto_labels_f = []
            proto_logits_f = []
            instance_loss = 0
            instance_loss_f = 0
            for n, (tra2cluster, prototypes, density,
                    tra2cluster_f, prototypes_f, density_f,) in enumerate(
                zip(cluster_result['tra2cluster'], cluster_result['centroids'], cluster_result['density'],
                    cluster_result_with_future['tra2cluster'], cluster_result_with_future['centroids'],
                    cluster_result_with_future['density'])):
                pos_proto_id = tra2cluster[index]
                pos_proto_id_f = tra2cluster_f[index]

                all_proto_id = [i for i in range(tra2cluster.max())]
                neg_proto_id = set(all_proto_id) - set(pos_proto_id.tolist())
                neg_proto_id = sample(neg_proto_id, len(neg_proto_id))

                instance_mask_positives = (
                        torch.abs(pos_proto_id.unsqueeze(-1).sub(pos_proto_id.unsqueeze(-1).T)) < 0.5).float().to(
                    features.device)

                instance_mask_positives_f = (
                        torch.abs(
                            pos_proto_id_f.unsqueeze(-1).sub(pos_proto_id_f.unsqueeze(-1).T)) < 0.5).float().to(
                    features.device)

                anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temp)
                logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
                logits = anchor_dot_contrast - logits_max.detach()

                batch_size = instance_mask_positives.shape[0]
                self_mask = torch.scatter(
                    torch.ones_like(instance_mask_positives), 1,
                    torch.arange(batch_size).view(-1, 1).to(instance_mask_positives.device), 0
                )

                f_mask = torch.ones_like(instance_mask_positives) * mask.unsqueeze(0) * mask.unsqueeze(-1)

                instance_mask_positives = instance_mask_positives * self_mask
                instance_mask_positives_f = instance_mask_positives_f * self_mask

                instance_mask_positives = instance_mask_positives * f_mask.logical_not()
                instance_mask_positives_f = instance_mask_positives_f * f_mask

                exp_logits = torch.exp(logits) * self_mask

                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-20)

                mean_log_prob_pos = (instance_mask_positives * log_prob).sum(1) / (
                        instance_mask_positives.sum(1) + 1e-20)
                mean_log_prob_pos_f = (instance_mask_positives_f * log_prob).sum(1) / (
                        instance_mask_positives_f.sum(1) + 1e-20)

                instance_loss_ = - mean_log_prob_pos
                instance_loss += instance_loss_.view(1, batch_size).mean()

                instance_loss_f_ = - mean_log_prob_pos_f
                instance_loss_f += instance_loss_f_.view(1, batch_size).mean()

                pos_proto_id = tra2cluster[index]
                proto_selected = prototypes[:, :232]

                logits_proto = torch.mm(features, proto_selected.t())

                labels_proto = pos_proto_id

                temp_proto = density
                logits_proto /= temp_proto

                pos_proto_id_f = tra2cluster_f[index]
                proto_selected_f = prototypes_f[:, :232]

                logits_proto_f = torch.mm(features, proto_selected_f.t())

                labels_proto_f = pos_proto_id_f

                temp_proto_f = density_f
                logits_proto_f /= temp_proto_f

                proto_labels.append(labels_proto)
                proto_logits.append(logits_proto)

                proto_labels_f.append(labels_proto_f)
                proto_logits_f.append(logits_proto_f)

            instance_loss /= len(proto_labels)
            instance_loss_f /= len(proto_labels)
            con_loss = 0
            con_loss_nof = 0
            con_loss_f = 0
            for n, (labels_proto, logits_proto, labels_proto_f, logits_proto_f) in enumerate(
                    zip(proto_labels, proto_logits,
                        proto_labels_f, proto_logits_f, )):
                con_loss_ = criterion(logits_proto, labels_proto) * mask.logical_not()
                con_loss_f_ = criterion(logits_proto_f, labels_proto_f) * mask
                con_loss = con_loss + torch.mean(con_loss_f_)
                con_loss_nof += torch.mean(con_loss_)
                con_loss_f += torch.mean(con_loss_f_)

            con_loss /= len(proto_labels)
            con_loss_nof /= len(proto_labels)
            con_loss_f /= len(proto_labels)

        final_loss = loss + factor_con * con_loss + factor_con * instance_loss_f
        if self.log_writer is not None:
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'contrastive_proto_loss'),
                                       con_loss, self.curr_iter)

        if self.log_writer is not None:
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'contrastive_proto_loss_nof'),
                                       con_loss_nof, self.curr_iter)

        if self.log_writer is not None:
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'contrastive_proto_loss_f'),
                                       con_loss_f, self.curr_iter)

        if self.log_writer is not None:
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'contrastive_ins_loss'),
                                       instance_loss, self.curr_iter)

        if self.log_writer is not None:
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'contrastive_ins_loss_f'),
                                       instance_loss_f, self.curr_iter)
        if self.log_writer is not None:
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'ewta_loss'),
                                       loss, self.curr_iter)

        return final_loss



    def predict_futurefeature(self,
                inputs,
                inputs_st,
                first_history_indices,
                neighbors,
                neighbors_edge_value,
                robot,
                map,
                prediction_horizon,
                return_original=False):
        mode = ModeKeys.PREDICT
        x, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                inputs=inputs,
                                                inputs_st=inputs_st,
                                                labels=None,
                                                labels_st=None,
                                                first_history_indices=first_history_indices,
                                                neighbors=neighbors,
                                                neighbors_edge_value=neighbors_edge_value,
                                                robot=robot,
                                                map=map)

        y, features = self.decoder(x, n_s_t0, prediction_horizon)
        features_original=features[:]
        features = F.normalize(self.node_modules[self.node_type + '/con_head'](features), dim=1)
        if return_original:
            return y, features,features_original
        else:
            return y , features









class LDAMLoss():
    def __init__(self):
        super(LDAMLoss, self).__init__()

    def forward(self,x,target,cls_num_list):
        max_m=0.5
        s=1
        weight=None
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        assert s > 0
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(s * output, target, weight=weight)







class model_encdec_re(nn.Module):
    def __init__(self, hyperparams, pretrained_model=None):
        super(model_encdec_re, self).__init__()

        self.name_model = 'AIO_autoencoder'
        self.dim_embedding_key = 232
        self.past_len = 8  ######todo:no shorter history
        self.future_len = hyperparams['prediction_horizon']


        # LAYERS for different modes
        self.norm_fut_encoder = st_encoder()
        self.norm_past_encoder = st_encoder()


        self.res_past_encoder = st_encoder()

        self.decoder = MLP(self.dim_embedding_key , self.future_len * 2, hidden_size=(1024, 512, 1024))
        self.decoder_x = MLP(self.dim_embedding_key , self.past_len * 2, hidden_size=(1024, 512, 1024))
        self.decoder_2 = MLP(self.dim_embedding_key * 2, self.future_len * 2, hidden_size=(1024, 512, 1024))
        self.decoder_2_x = MLP(self.dim_embedding_key * 2, self.past_len * 2, hidden_size=(1024, 512, 1024))

    def get_state_encoding(self, future):
        norm_fut_state = self.norm_fut_encoder(future)

        return norm_fut_state



    def decode_state_into_intention(self,norm_fut_state):
        # state concatenation and decoding
        input_fut = norm_fut_state
        batch_size=input_fut.shape[0]
        prediction_y1 = self.decoder(input_fut).contiguous().view(batch_size,-1, 2)
        reconstruction_x1 = self.decoder_x(input_fut).contiguous().view(-1, self.past_len, 2)

        # diff_past = past - reconstruction_x1  # B, T, 2   ######todo:cannot deal with diff
        #
        # diff_past_embed = self.res_past_encoder(diff_past)  # B, F
        #
        # state_conc_diff = torch.cat((diff_past_embed, norm_fut_state), 1)
        # prediction_y2 = self.decoder_2(state_conc_diff).contiguous().view(-1, 1, 2)
        # reconstruction_x2 = self.decoder_2_x(state_conc_diff).contiguous().view(-1, self.past_len, 2)
        #
        # prediction = prediction_y1 + prediction_y2
        # reconstruction = reconstruction_x1 + reconstruction_x2
        prediction = prediction_y1
        reconstruction = reconstruction_x1

        return prediction, reconstruction

    def forward(self, future):
        norm_fut_state = self.get_state_encoding(future)
        prediction, reconstruction = self.decode_state_into_intention(norm_fut_state)
        return prediction, reconstruction














class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers)-1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x



class st_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        channel_in = 2
        channel_out = 16
        dim_kernel = 3
        self.dim_embedding_key = 232
        self.spatial_conv = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
        self.temporal_encoder = nn.GRU(channel_out, self.dim_embedding_key, 1, batch_first=True)

        self.relu = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.spatial_conv.weight)
        nn.init.kaiming_normal_(self.temporal_encoder.weight_ih_l0)
        nn.init.kaiming_normal_(self.temporal_encoder.weight_hh_l0)
        nn.init.zeros_(self.spatial_conv.bias)
        nn.init.zeros_(self.temporal_encoder.bias_ih_l0)
        nn.init.zeros_(self.temporal_encoder.bias_hh_l0)

    def forward(self, X):
        '''
        X: b, T, 2

        return: b, F
        '''
        X_t = torch.transpose(X, 1, 2)
        X_after_spatial = self.relu(self.spatial_conv(X_t))
        X_embed = torch.transpose(X_after_spatial, 1, 2)

        output_x, state_x = self.temporal_encoder(X_embed)
        state_x = state_x.squeeze(0)

        return state_x

