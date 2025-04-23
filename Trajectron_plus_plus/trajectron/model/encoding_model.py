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
from Trajectron_plus_plus.trajectron.model.hyperlstm import HyperLSTM

import torch.nn.utils.rnn as rnn
from Trajectron_plus_plus.trajectron.model.mgcvaeEWTA import MultimodalGenerativeCVAEEWTA


def run_lstm_on_variable_length_seqs(lstm_module, original_seqs, lower_indices=None, upper_indices=None, total_length=None):
    bs, tf = original_seqs.shape[:2]
    if lower_indices is None:
        lower_indices = torch.zeros(bs, dtype=torch.int)
    if upper_indices is None:
        upper_indices = torch.ones(bs, dtype=torch.int) * (tf - 1)
    if total_length is None:
        total_length = max(upper_indices) + 1
    # This is done so that we can just pass in self.prediction_timesteps
    # (which we want to INCLUDE, so this will exclude the next timestep).
    inclusive_break_indices = upper_indices + 1

    pad_list = list()
    for i, seq_len in enumerate(inclusive_break_indices):
        pad_list.append(original_seqs[i, lower_indices[i]:seq_len])

    packed_seqs = rnn.pack_sequence(pad_list, enforce_sorted=False)
    packed_output, (h_n, c_n) = lstm_module(packed_seqs)
    output, _ = rnn.pad_packed_sequence(packed_output,
                                        batch_first=True,
                                        total_length=total_length)

    return output, (h_n, c_n)



class model_encdec_trajectron(nn.Module):
    def __init__(self, hyperparams,device,model_registrar,kwargs=None):
        super(model_encdec_trajectron, self).__init__()
        node_type='PEDESTRIAN'
        self.name_model = 'AIO_autoencoder'
        self.dim_embedding_key = 232
        self.past_len = 8  ######todo:no shorter history
        self.future_len = hyperparams['prediction_horizon']
        self.pred_state = self.hyperparams['pred_state'][node_type]
        self.state_length = int(np.sum([len(entity_dims) for entity_dims in self.state[node_type].values()]))


        # LAYERS for different modes
        self.norm_fut_encoder = nn.LSTM(input_size=self.pred_state_length,
                                                   hidden_size=self.hyperparams['enc_rnn_dim_future'],
                                                   bidirectional=True,
                                                   batch_first=True)

        self.norm_fut_initial_h=nn.Linear(self.state_length,
                                                   self.hyperparams['enc_rnn_dim_future'])
        self.norm_fut_initial_c=nn.Linear(self.state_length,
                                                     self.hyperparams['enc_rnn_dim_future'])


        self.norm_past_encoder=nn.LSTM(input_size=self.state_length,
                                                   hidden_size=self.hyperparams['enc_rnn_dim_history'],
                                                   batch_first=True)



        self.decoder_x=nn.GRUCell(decoder_input_dims, self.hyperparams['dec_rnn_dim'])
        self.decoder_x_initial_h=nn.Linear(x_size, self.hyperparams['dec_rnn_dim'])
        self.project_to_GMM_x=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     20 * self.pred_state_length)


        self.decoder_y=nn.GRUCell(decoder_input_dims, self.hyperparams['dec_rnn_dim'])
        self.decoder_y_initial_h = nn.Linear(x_size, self.hyperparams['dec_rnn_dim'])
        self.project_to_GMM_y = nn.Linear(self.hyperparams['dec_rnn_dim'],
                                        1 * self.pred_state_length)



        dynamic_class = getattr(utilities, hyperparams['dynamic']['PEDESTRIAN']['name'])
        dyn_limits = hyperparams['dynamic']['PEDESTRIAN']['limits']
        self.dynamic = dynamic_class(0.4, dyn_limits, device,model_registrar,64,'PEDESTRIAN')

    def project_to_GMM_params_x(self, tensor):
        mus = self.project_to_GMM_x(tensor)
        return mus
    def project_to_GMM_params_y(self, tensor):
        mus = self.project_to_GMM_y(tensor)
        return mus

    def p_y_xz(self, features):
        cell=self.decoder_rnn
        ph=self.future_len
        mus=[]
        x=features[:,:64]
        input_=features[:,:104]
        state=features[:,:128]
        features = torch.cat([input_, state], dim=1)
        for j in range(ph):
            h_state = cell(input_, state)
            mu_t = self.project_to_GMM_params(h_state)
            mus.append(mu_t.reshape(-1, 20, 2))
            dec_inputs = [x, mu_t]
            input_ = torch.cat(dec_inputs, dim=1)
            state = h_state
        mus = torch.stack(mus, dim=2)
        y = self.dynamic.integrate_samples(mus, x)
        return y, features

    def decoder(self,features):
        y, features = self.p_y_xz(features)
        return y


    def decode_state_into_intention(self, features):
        prediction=self.decoder(features)
        return prediction
    def train_loss(self,prediction,label):
        mode='epe-top-n'
        top_n=1
        loss = self.ewta_loss(prediction, label, mode=mode, top_n=top_n)
        return loss
    def ewta_loss(self,y, labels, mode='epe-all', top_n=1):
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

    def encode_node_future(self, mode, node_present, node_future) -> torch.Tensor:
        """
        Encodes the node future (during training) using a bi-directional LSTM

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param node_present: Current state of the node. [bs, state]
        :param node_future: Future states of the node. [bs, ph, state]
        :return: Encoded future.
        """
        initial_h_model = self.node_modules[self.node_type + '/node_future_encoder/initial_h']
        initial_c_model = self.node_modules[self.node_type + '/node_future_encoder/initial_c']

        # Here we're initializing the forward hidden states,
        # but zeroing the backward ones.
        initial_h = initial_h_model(node_present)
        initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=self.device)], dim=0)

        initial_c = initial_c_model(node_present)
        initial_c = torch.stack([initial_c, torch.zeros_like(initial_c, device=self.device)], dim=0)

        initial_state = (initial_h, initial_c)

        _, state = self.node_modules[self.node_type + '/node_future_encoder'](node_future, initial_state)
        state = unpack_RNN_state(state)
        state = F.dropout(state,
                          p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        return state

    def obtain_obs_tensor(self,past):
        initial_dynamics = dict()
        node_pos = past[:, -1, 0:2]
        node_vel = past[:, -1, 2:4]

        initial_dynamics['pos'] = node_pos
        initial_dynamics['vel'] = node_vel

        self.dynamic.set_initial_condition(initial_dynamics)

    def forward(self,future,future_st,past,past_st):
        self.obtain_obs_tensor(past)
        mode=ModeKeys.TRAIN
        n_s_t0=past_st[:,-1]
        features=self.encode_node_future(mode,n_s_t0,future_st)
        prediction=self.decode_state_into_intention(features)
        loss=self.train_loss(prediction,future)
        return prediction,loss




class mgcvaeEWTA_encdec(MultimodalGenerativeCVAEEWTA):
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

        self.state = self.hyperparams['state']
        self.pred_state = self.hyperparams['pred_state'][node_type]
        self.state_length = int(np.sum([len(entity_dims) for entity_dims in self.state[node_type].values()]))
        self.pred_state_length = int(np.sum([len(entity_dims) for entity_dims in self.pred_state.values()]))


        dynamic_class = getattr(utilities, hyperparams['dynamic'][self.node_type]['name'])
        dyn_limits = hyperparams['dynamic'][self.node_type]['limits']
        self.dynamic = dynamic_class(self.env.scenes[0].dt, dyn_limits, device,
                                     self.model_registrar, self.x_size, self.node_type)


     def create_node_models(self):
         #########################################original##############
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

         self.x_size = x_size
         self.z_size = z_size
         #######################################################################################


         decoder_input_dims = self.pred_state_length * 20 + x_size
         decoder_xy_input_dims = self.pred_state_length * 20 + x_size

         decoder_xy_concate_dims=self.hyperparams['enc_rnn_dim_future']*4 + x_size
         y_size=self.hyperparams['enc_rnn_dim_future']*4



         self.add_submodule(self.node_type + '/decoder_x/state_action',
                            model_if_absent=nn.Sequential(
                                nn.Linear(self.state_length, self.pred_state_length)))
         self.add_submodule(self.node_type + '/decoder_y/state_action',
                            model_if_absent=nn.Sequential(
                                nn.Linear(self.state_length, self.pred_state_length)))

         self.add_submodule(self.node_type + '/decoder_x/rnn_cell',
                            model_if_absent=nn.GRUCell(decoder_input_dims, self.hyperparams['dec_rnn_dim']))
         # self.add_submodule(self.node_type + '/decoder_y/rnn_cell',
         #                    model_if_absent=nn.GRUCell(self.hyperparams['enc_rnn_dim_future']*4+self.pred_state_length, self.hyperparams['dec_rnn_dim']))

         self.add_submodule(self.node_type + '/decoder_y/rnn_cell',
                            model_if_absent=nn.GRUCell(
                                x_size + self.pred_state_length,
                                self.hyperparams['dec_rnn_dim']))

         self.add_submodule(self.node_type + '/decoder_xy/rnn_cell',
                            model_if_absent=nn.GRUCell(
                                decoder_xy_input_dims,
                                self.hyperparams['dec_rnn_dim']))

         self.add_submodule(self.node_type + '/decoder_xy_single/rnn_cell',
                            model_if_absent=nn.GRUCell(
                                self.pred_state_length + x_size,
                                self.hyperparams['dec_rnn_dim']))

         self.add_submodule(self.node_type + '/decoder_x/initial_h',
                            model_if_absent=nn.Linear(x_size, self.hyperparams['dec_rnn_dim']))
         self.add_submodule(self.node_type + '/decoder_y/initial_h',
                            model_if_absent=nn.Linear(self.hyperparams['enc_rnn_dim_future']*4, self.hyperparams['dec_rnn_dim']))
         self.add_submodule(self.node_type + '/decoder_xy/initial_h',
                            model_if_absent=nn.Linear(decoder_xy_concate_dims, self.hyperparams['dec_rnn_dim']))


         self.add_submodule(self.node_type + '/decoder_x/proj_to_GMM_mus',
                            model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                    20 * self.pred_state_length))
         self.add_submodule(self.node_type + '/decoder_y/proj_to_GMM_mus',
                            model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                    1 * self.pred_state_length))
         self.add_submodule(self.node_type + '/decoder_xy/proj_to_GMM_mus',
                            model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                      20 * self.pred_state_length))
         self.add_submodule(self.node_type + '/decoder_xy_single/proj_to_GMM_mus',
                            model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                      1 * self.pred_state_length))



         self.add_submodule(self.node_type + '/node_future_encoder',
                            model_if_absent=nn.LSTM(input_size=self.pred_state_length,
                                                    hidden_size=self.hyperparams['enc_rnn_dim_future'],
                                                    bidirectional=True,
                                                    batch_first=True))
         # These are related to how you initialize states for the node future encoder.
         self.add_submodule(self.node_type + '/node_future_encoder/initial_h',
                            model_if_absent=nn.Linear(self.state_length,
                                                      self.hyperparams['enc_rnn_dim_future']))
         self.add_submodule(self.node_type + '/node_future_encoder/initial_c',
                            model_if_absent=nn.Linear(self.state_length,
                                                      self.hyperparams['enc_rnn_dim_future']))

         self.add_submodule(self.node_type + '/decoder_xy/input_projection',
                            model_if_absent=nn.Linear(decoder_xy_concate_dims,
                                                      x_size)
                            )
         self.add_submodule(self.node_type + '/decoder_y/input_projector',
                            model_if_absent=nn.Linear(y_size,
                                                      x_size)
                            )

         self.add_submodule(self.node_type + '/con_head_y',
                            model_if_absent=nn.Linear(232,
                                                      232)
                            )


     def encode_node_future(self, mode, node_present, node_future) -> torch.Tensor:
        """
        Encodes the node future (during training) using a bi-directional LSTM

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param node_present: Current state of the node. [bs, state]
        :param node_future: Future states of the node. [bs, ph, state]
        :return: Encoded future.
        """
        initial_h_model = self.node_modules[self.node_type + '/node_future_encoder/initial_h']
        initial_c_model = self.node_modules[self.node_type + '/node_future_encoder/initial_c']

        # Here we're initializing the forward hidden states,
        # but zeroing the backward ones.
        initial_h = initial_h_model(node_present)
        initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=self.device)], dim=0)

        initial_c = initial_c_model(node_present)
        initial_c = torch.stack([initial_c, torch.zeros_like(initial_c, device=self.device)], dim=0)

        initial_state = (initial_h, initial_c)

        _, state = self.node_modules[self.node_type + '/node_future_encoder'](node_future, initial_state)
        state = unpack_RNN_state(state)
        state = F.dropout(state,
                          p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        return state

     def decoder_x(self, x, n_s_t0, prediction_horizon):
         y, features = self.p_y_x(x, n_s_t0, prediction_horizon)
         return y, features

     def decoder_y(self,y_e,n_s_t0, prediction_horizon):
         y, features = self.p_y_y(y_e, n_s_t0, prediction_horizon)
         return y, features

     def decoder_xy(self, x, y_e,n_s_t0, prediction_horizon):
         y, features = self.p_y_xy(x,y_e, n_s_t0, prediction_horizon)
         return y, features

     def decoder_xy_single(self, x, y_e,n_s_t0, prediction_horizon):
         y, features = self.p_y_xy_single(x,y_e, n_s_t0, prediction_horizon)
         return y, features

     def decoder_xy_kd(self, x, y_e,n_s_t0, prediction_horizon):
         y, features,dec_features = self.p_y_xy_kd(x, y_e, n_s_t0, prediction_horizon)
         return y, features,dec_features

     def p_y_y(self,y_e, n_s_t0, prediction_horizon):
         ph = prediction_horizon
         cell = self.node_modules[self.node_type + '/decoder_y/rnn_cell']
         initial_h_model = self.node_modules[self.node_type + '/decoder_y/initial_h']
         initial_state=initial_h_model(y_e)
         mus = []
         state = initial_state
         input_y_e=self.node_modules[self.node_type + '/decoder_y/input_projector'](y_e)
         a_0 = self.node_modules[self.node_type + '/decoder_y/state_action'](n_s_t0)
         input_ =torch.cat([input_y_e,a_0],dim=-1)

         features_input=torch.cat([input_y_e,a_0.repeat(1,20)],dim=-1)
         features=torch.cat([features_input,state],dim=-1)


         for j in range(ph):
             h_state = cell(input_, state)
             mu_t = self.node_modules[self.node_type + '/decoder_y/proj_to_GMM_mus'](h_state)
             mus.append(mu_t.reshape(-1, 1, 2))
             dec_inputs = [input_y_e, mu_t]
             input_ = torch.cat(dec_inputs, dim=1)
             state = h_state
         mus = torch.stack(mus, dim=2)
         y = self.dynamic.integrate_samples(mus, x=None) ##########TODO:not unicycle
         return y, features


     def p_y_y_multimodal(self,y_e, n_s_t0, prediction_horizon):
         ph = prediction_horizon
         cell = self.node_modules[self.node_type + '/decoder_x/rnn_cell']
         initial_h_model = self.node_modules[self.node_type + '/decoder_y/initial_h']
         initial_state=initial_h_model(y_e)
         mus = []
         state = initial_state
         a_0 = self.node_modules[self.node_type + '/decoder_x/state_action'](n_s_t0)
         input_ =torch.cat([y_e,a_0],dim=-1)
         features=torch.cat([input_,state],dim=-1)

         for j in range(ph):
             h_state = cell(input_, state)
             mu_t = self.node_modules[self.node_type + '/decoder_y/proj_to_GMM_mus'](h_state)
             mus.append(mu_t.reshape(-1, 1, 2))
             dec_inputs = [y_e, mu_t]
             input_ = torch.cat(dec_inputs, dim=1)
             state = h_state
         mus = torch.stack(mus, dim=2)
         y = self.dynamic.integrate_samples(mus, x=None) ##########TODO:not unicycle
         return y, features


     def p_y_xy_single(self,x,y_e, n_s_t0, prediction_horizon):
         ph = prediction_horizon
         cell = self.node_modules[self.node_type + '/decoder_xy_single/rnn_cell']
         initial_h_model = self.node_modules[self.node_type + '/decoder_xy/initial_h']
         mus = []
         xy_e=torch.cat([x,y_e],dim=1)
         initial_state=initial_h_model(xy_e)
         a_0 = self.node_modules[self.node_type + '/decoder_x/state_action'](n_s_t0)
         state = initial_state
         input_xy=self.node_modules[self.node_type + '/decoder_xy/input_projection'](xy_e)
         input_ = torch.cat([input_xy, a_0.repeat(1, 1)], dim=1)
         features = torch.cat([input_, state], dim=1)
         for j in range(ph):
             h_state = cell(input_, state)
             mu_t = self.node_modules[self.node_type + '/decoder_xy_single/proj_to_GMM_mus'](h_state)
             mus.append(mu_t.reshape(-1, 1, 2))
             dec_inputs = [input_xy, mu_t]
             input_ = torch.cat(dec_inputs, dim=1)
             state = h_state
         mus = torch.stack(mus, dim=2)
         y = self.dynamic.integrate_samples(mus, x)
         return y, features

     def p_y_xy(self,x,y_e, n_s_t0, prediction_horizon):
         ph = prediction_horizon
         cell = self.node_modules[self.node_type + '/decoder_xy/rnn_cell']
         initial_h_model = self.node_modules[self.node_type + '/decoder_xy/initial_h']
         mus = []
         xy_e=torch.cat([x,y_e],dim=1)
         initial_state=initial_h_model(xy_e)
         a_0 = self.node_modules[self.node_type + '/decoder_x/state_action'](n_s_t0)
         state = initial_state
         input_xy=self.node_modules[self.node_type + '/decoder_xy/input_projection'](xy_e)
         input_ = torch.cat([input_xy, a_0.repeat(1, 20)], dim=1)
         features = torch.cat([input_, state], dim=1)
         for j in range(ph):
             h_state = cell(input_, state)
             mu_t = self.node_modules[self.node_type + '/decoder_xy/proj_to_GMM_mus'](h_state)
             mus.append(mu_t.reshape(-1, 20, 2))
             dec_inputs = [input_xy, mu_t]
             input_ = torch.cat(dec_inputs, dim=1)
             state = h_state
         mus = torch.stack(mus, dim=2)
         y = self.dynamic.integrate_samples(mus, x)
         return y, features


     def p_y_xy_kd(self,x,y_e, n_s_t0, prediction_horizon):
         ph = prediction_horizon
         cell = self.node_modules[self.node_type + '/decoder_xy/rnn_cell']
         initial_h_model = self.node_modules[self.node_type + '/decoder_xy/initial_h']
         mus = []
         xy_e=torch.cat([x,y_e],dim=1)
         initial_state=initial_h_model(xy_e)
         a_0 = self.node_modules[self.node_type + '/decoder_x/state_action'](n_s_t0)
         state = initial_state
         input_xy=self.node_modules[self.node_type + '/decoder_xy/input_projection'](xy_e)
         input_ = torch.cat([input_xy, a_0.repeat(1, 20)], dim=1)
         features = torch.cat([input_, state], dim=1)
         hidden_state_list=[]
         for j in range(ph):
             h_state = cell(input_, state)
             mu_t = self.node_modules[self.node_type + '/decoder_xy/proj_to_GMM_mus'](h_state)
             mus.append(mu_t.reshape(-1, 20, 2))
             dec_inputs = [input_xy, mu_t]
             input_ = torch.cat(dec_inputs, dim=1)
             state = h_state
             hidden_state_list.append(h_state)
         mus = torch.stack(mus, dim=2)
         hidden_state_t=torch.cat(hidden_state_list,dim=-1) #[batch_size,32*12]
         y = self.dynamic.integrate_samples(mus, x)
         return y, features,hidden_state_t

     def p_y_x(self,x, n_s_t0, prediction_horizon):
         ph = prediction_horizon  # 12
         cell = self.node_modules[self.node_type + '/decoder_x/rnn_cell']
         initial_h_model = self.node_modules[self.node_type + '/decoder_x/initial_h']
         initial_state = initial_h_model(x)
         mus = []
         a_0 = self.node_modules[self.node_type + '/decoder_x/state_action'](n_s_t0)
         state = initial_state
         input_ = torch.cat([x, a_0.repeat(1, 20)], dim=1)  # [64,40]
         features = torch.cat([input_, state], dim=1)  # [104,128]
         for j in range(ph):
             h_state = cell(input_, state)
             mu_t = self.node_modules[self.node_type + '/decoder_x/proj_to_GMM_mus'](h_state)
             mus.append(mu_t.reshape(-1, 20, 2))
             dec_inputs = [x, mu_t]
             input_ = torch.cat(dec_inputs, dim=1)
             state = h_state
         mus = torch.stack(mus, dim=2)
         y = self.dynamic.integrate_samples(mus, x)
         return y, features


     def train_loss_y_y(self,
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
                   ):

         mode = ModeKeys.TRAIN
         n_s_t0 = inputs_st[:, -1]

         initial_dynamics = dict()
         batch_size = inputs.shape[0]
         node_history = inputs
         node_pos = inputs[:, -1, 0:2]
         node_vel = inputs[:, -1, 2:4]

         initial_dynamics['pos'] = node_pos
         initial_dynamics['vel'] = node_vel
         self.dynamic.set_initial_condition(initial_dynamics)

         y_e = self.encode_node_future(mode,n_s_t0 ,labels_st)  #####[32]



         y, features = self.decoder_y(y_e, n_s_t0, prediction_horizon)
         loss_type='epe-top-1'
         mode, top_n = loss_type, 1
         if 'top' in loss_type:
             mode = 'epe-top-n'
             top_n = int(loss_type.replace('epe-top-', ''))
         loss = self.ewta_loss(y, labels, mode=mode, top_n=top_n)

         return loss

     def train_loss_y_y_pcl(self,
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
                   cluster_result,
                   index,
                   factor_con
                   ):

         mode = ModeKeys.TRAIN
         n_s_t0 = inputs_st[:, -1]

         initial_dynamics = dict()
         batch_size = inputs.shape[0]
         node_history = inputs
         node_pos = inputs[:, -1, 0:2]
         node_vel = inputs[:, -1, 2:4]

         initial_dynamics['pos'] = node_pos
         initial_dynamics['vel'] = node_vel
         self.dynamic.set_initial_condition(initial_dynamics)

         y_e = self.encode_node_future(mode,n_s_t0 ,labels_st)  #####[32]



         y, features = self.decoder_y(y_e, n_s_t0, prediction_horizon)

         loss_type='epe-top-1'
         mode, top_n = loss_type, 1
         if 'top' in loss_type:
             mode = 'epe-top-n'
             top_n = int(loss_type.replace('epe-top-', ''))
         loss = self.ewta_loss(y, labels, mode=mode, top_n=top_n)

         criterion = nn.CrossEntropyLoss()

         if cluster_result is not None:
             proto_labels = []
             proto_logits = []
             instance_loss = 0
             for n, (tra2cluster, prototypes, density) in enumerate(
                     zip(cluster_result['tra2cluster'], cluster_result['centroids'], cluster_result['density'])):
                 pos_proto_id = tra2cluster[index]



                 pos_proto_id = tra2cluster[index]
                 proto_selected = prototypes[:, :232]

                 logits_proto = torch.mm(features, proto_selected.t())

                 labels_proto = pos_proto_id

                 temp_proto = density
                 logits_proto /= temp_proto

                 proto_labels.append(labels_proto)
                 proto_logits.append(logits_proto)

         con_loss = 0
         for n, (labels_proto, logits_proto) in enumerate(zip(proto_labels, proto_logits)):
             con_loss += criterion(logits_proto, labels_proto)

         con_loss /= len(proto_labels)
         final_loss = loss + factor_con * con_loss + factor_con * instance_loss

         if self.log_writer is not None:
             self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'contrastive_proto_loss'),
                                        con_loss, self.curr_iter)


         if self.log_writer is not None:
             self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'ewta_loss'),
                                        loss, self.curr_iter)

         return final_loss

     def train_loss_y_x(self,
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
                        ):

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
         y, features = self.decoder_x(x, n_s_t0, prediction_horizon)
         loss_type='epe-top-1'
         mode, top_n = loss_type, 1
         if 'top' in loss_type:
             mode = 'epe-top-n'
             top_n = int(loss_type.replace('epe-top-', ''))
         loss = self.ewta_loss(y, labels, mode=mode, top_n=top_n)

         return loss


     def train_loss_y_xy(self,
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
                        ):
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
         y_e = self.encode_node_future(mode,n_s_t0 ,labels_st)  #####[128]
         y, features = self.decoder_xy(x, y_e, n_s_t0, prediction_horizon)
         loss_type='epe-top-1'
         mode, top_n = loss_type, 1
         if 'top' in loss_type:
             mode = 'epe-top-n'
             top_n = int(loss_type.replace('epe-top-', ''))
         loss = self.ewta_loss(y, labels, mode=mode, top_n=top_n)

         return loss

     def train_loss_y_xy_single(self,
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
                        ):
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
         y_e = self.encode_node_future(mode,n_s_t0 ,labels_st)  #####[128]
         y, features = self.decoder_xy_single(x, y_e, n_s_t0, prediction_horizon)
         loss_type='epe-top-1'
         mode, top_n = loss_type, 1
         if 'top' in loss_type:
             mode = 'epe-top-n'
             top_n = int(loss_type.replace('epe-top-', ''))
         loss = self.ewta_loss(y, labels, mode=mode, top_n=top_n)

         return loss



     def predict_y_y(self,
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
                ):
        mode = ModeKeys.PREDICT
        n_s_t0 = inputs_st[:, -1]

        initial_dynamics = dict()
        batch_size = inputs.shape[0]
        node_history = inputs
        node_pos = inputs[:, -1, 0:2]
        node_vel = inputs[:, -1, 2:4]

        initial_dynamics['pos'] = node_pos
        initial_dynamics['vel'] = node_vel
        self.dynamic.set_initial_condition(initial_dynamics)

        y_e = self.encode_node_future(mode, n_s_t0, labels_st)
        y, features = self.decoder_y(y_e, n_s_t0, prediction_horizon)
        features=F.normalize(features,dim=-1)
        # features = F.normalize(self.node_modules[self.node_type+'/con_head_y'](features),dim=-1)
        return y, features

     def predict_y_x(self,
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
                ):




        mode = ModeKeys.PREDICT
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

        y, features = self.decoder_x(x, n_s_t0, prediction_horizon)
        return y, features

     def predict_y_xy_single(self,
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
                ):
        mode = ModeKeys.PREDICT
        x_e, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                        inputs=inputs,
                                        inputs_st=inputs_st,
                                        labels=labels,
                                        labels_st=labels_st,
                                        first_history_indices=first_history_indices,
                                        neighbors=neighbors,
                                        neighbors_edge_value=neighbors_edge_value,
                                        robot=robot,
                                        map=map)
        y_e = self.encode_node_future(mode, n_s_t0, labels_st)
        y, features = self.decoder_xy_single(x_e,y_e, n_s_t0, prediction_horizon)
        features = F.normalize(features, dim=-1)
        return y, features

     def predict_y_xy(self,
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
                ):
        mode = ModeKeys.PREDICT
        x_e, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                        inputs=inputs,
                                        inputs_st=inputs_st,
                                        labels=labels,
                                        labels_st=labels_st,
                                        first_history_indices=first_history_indices,
                                        neighbors=neighbors,
                                        neighbors_edge_value=neighbors_edge_value,
                                        robot=robot,
                                        map=map)
        y_e = self.encode_node_future(mode, n_s_t0, labels_st)
        y, features = self.decoder_xy(x_e,y_e, n_s_t0, prediction_horizon)
        features = F.normalize(features, dim=-1)
        return y, features

     def predict_y_xy_kd(self,
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
                ):
        mode = ModeKeys.PREDICT
        x_e, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                        inputs=inputs,
                                        inputs_st=inputs_st,
                                        labels=labels,
                                        labels_st=labels_st,
                                        first_history_indices=first_history_indices,
                                        neighbors=neighbors,
                                        neighbors_edge_value=neighbors_edge_value,
                                        robot=robot,
                                        map=map)
        y_e = self.encode_node_future(mode, n_s_t0, labels_st)
        y, features,dec_features = self.decoder_xy_kd(x_e,y_e, n_s_t0, prediction_horizon)
        features = F.normalize(features, dim=-1)
        return y, features,dec_features







class model_encdec_future(nn.Module):
    def __init__(self, hyperparams):
        super(model_encdec_future, self).__init__()

        self.name_model = 'AIO_autoencoder'
        self.dim_embedding_key = 232
        self.past_len = 8  ######todo:no shorter history
        self.future_len = hyperparams['prediction_horizon']


        # LAYERS for different modes
        self.norm_fut_encoder = st_encoder_future()

        self.res_past_encoder = st_encoder_future()

        self.decoder = MLP(self.dim_embedding_key , self.future_len * 2, hidden_size=(1024, 512, 1024))
        self.decoder_x = MLP(self.dim_embedding_key , self.past_len * 2, hidden_size=(1024, 512, 1024))
        self.decoder_2 = MLP(self.dim_embedding_key * 2, self.future_len * 2, hidden_size=(1024, 512, 1024))
        self.decoder_2_x = MLP(self.dim_embedding_key * 2, self.past_len * 2, hidden_size=(1024, 512, 1024))

    def get_state_encoding(self, future):
        norm_fut_state = self.norm_fut_encoder(future)

        norm_fut_state=F.normalize(norm_fut_state,dim=-1)

        return norm_fut_state


    def decode_state_into_intention(self,norm_fut_state):
        # state concatenation and decoding
        input_fut = norm_fut_state
        batch_size=input_fut.shape[0]
        prediction_y1 = self.decoder(input_fut).contiguous().view(batch_size,-1, 2)
        reconstruction_x1 = self.decoder_x(input_fut).contiguous().view(-1, self.past_len, 2)

        prediction = prediction_y1
        reconstruction = reconstruction_x1

        return prediction, reconstruction

    def forward(self, future):
        norm_fut_state = self.get_state_encoding(future)
        prediction, reconstruction = self.decode_state_into_intention(norm_fut_state)
        return prediction, reconstruction


# class model_encdec_re_x_pred_y(nn.Moudle):  #####TODO


class model_encdec_re(nn.Module):
    def __init__(self, hyperparams, pretrained_model=None):
        super(model_encdec_re, self).__init__()

        self.name_model = 'AIO_autoencoder'
        self.dim_embedding_key = 232
        self.past_len = hyperparams['maximum_history_length'] + 1
        self.future_len = hyperparams['prediction_horizon']


        # LAYERS for different modes
        self.norm_fut_encoder = st_encoder()
        self.norm_past_encoder = st_encoder()


        self.res_past_encoder = st_encoder()

        self.decoder = MLP(self.dim_embedding_key*2 , self.future_len * 2, hidden_size=(1024, 512, 1024))
        self.decoder_x = MLP(self.dim_embedding_key*2 , self.past_len * 2, hidden_size=(1024, 512, 1024))
        self.decoder_2 = MLP(self.dim_embedding_key * 2, self.future_len * 2, hidden_size=(1024, 512, 1024))
        self.decoder_2_x = MLP(self.dim_embedding_key * 2, self.past_len * 2, hidden_size=(1024, 512, 1024))

    def get_state_encoding(self, future,past,first_history_index):
        norm_fut_state = self.norm_fut_encoder(future)
        norm_past_state=self.norm_past_encoder(past,past=True,first_history_index=first_history_index)

        norm_fut_state=F.normalize(norm_fut_state,dim=-1)
        norm_past_state = F.normalize(norm_past_state, dim=-1)
        return norm_fut_state,norm_past_state



    def decode_state_into_intention(self,norm_fut_state,norm_past_state):
        # state concatenation and decoding
        input_fut = torch.cat((norm_past_state, norm_fut_state), -1)
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

    def forward(self, future,past,first_history_index):
        norm_fut_state,norm_past_state = self.get_state_encoding(future,past,first_history_index)
        prediction, reconstruction = self.decode_state_into_intention(norm_fut_state,norm_past_state)
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

class st_encoder_future(nn.Module):
    def __init__(self):
        super().__init__()
        channel_in = 2
        channel_out = 16
        dim_kernel = 3
        self.dim_embedding_key = 232
        self.spatial_conv = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
        self.temporal_encoder = nn.LSTM(channel_out, self.dim_embedding_key, 1,batch_first=True)

        self.relu = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.spatial_conv.weight)
        nn.init.kaiming_normal_(self.temporal_encoder.weight_ih_l0)
        nn.init.kaiming_normal_(self.temporal_encoder.weight_hh_l0)
        nn.init.zeros_(self.spatial_conv.bias)
        nn.init.zeros_(self.temporal_encoder.bias_ih_l0)
        nn.init.zeros_(self.temporal_encoder.bias_hh_l0)

    def forward(self, X,past=False,first_history_index=None):
        '''
        X: b, T, 2

        return: b, F
        '''
        X_t = torch.transpose(X, 1, 2)
        X_after_spatial = self.relu(self.spatial_conv(X_t))
        X_embed = torch.transpose(X_after_spatial, 1, 2)


        if past:
            assert first_history_index is not None
            output_x, (hx,cx)=run_lstm_on_variable_length_seqs(self.temporal_encoder,original_seqs=X_embed,
                                                               lower_indices=first_history_index)
        else:
            output_x, (hx,cx) = self.temporal_encoder(X_embed)
        state_x = hx[-1]

        return state_x


class st_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        channel_in = 2
        channel_out = 16
        dim_kernel = 3
        self.dim_embedding_key = 232
        # self.spatial_conv = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
        self.temporal_encoder = nn.LSTM(2, self.dim_embedding_key, 3, bidirectional=True,batch_first=True)

        self.relu = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.kaiming_normal_(self.spatial_conv.weight)
        nn.init.kaiming_normal_(self.temporal_encoder.weight_ih_l0)
        nn.init.kaiming_normal_(self.temporal_encoder.weight_hh_l0)
        # nn.init.zeros_(self.spatial_conv.bias)
        nn.init.zeros_(self.temporal_encoder.bias_ih_l0)
        nn.init.zeros_(self.temporal_encoder.bias_hh_l0)

    def forward(self, X,past=False,first_history_index=None):
        '''
        X: b, T, 2

        return: b, F
        '''
        # X_t = torch.transpose(X, 1, 2)
        # X_after_spatial = self.relu(self.spatial_conv(X_t))
        # X_embed = torch.transpose(X_after_spatial, 1, 2)

        X_embed=X

        if past:
            assert first_history_index is not None
            output_x, (hx,cx)=run_lstm_on_variable_length_seqs(self.temporal_encoder,original_seqs=X_embed,
                                                               lower_indices=first_history_index)
        else:
            output_x, (hx,cx) = self.temporal_encoder(X_embed)
        state_x = hx[-1]

        return state_x


