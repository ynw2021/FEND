""" This code is based on the Trajectron++ repository.

    For usage, see the License of Trajectron++ under:
    https://github.com/StanfordASL/Trajectron-plus-plus
"""
import torch
import numpy as np
from Trajectron_plus_plus.trajectron.model.trajectron import Trajectron
from Trajectron_plus_plus.trajectron.model.dataset import get_timesteps_data, restore
import sys
sys.path.append("/data2/wyn/Contrastive-Future-Trajectory-Prediction-main/Trajectron_plus_plus/trajectron/model")
from mgcvaeEWTA import MultimodalGenerativeCVAEEWTA
# from mgcvaeEWTA_hyper_re_trajectron import MultimodalGenerativeCVAEEWTA




class TrajectronEWTA(Trajectron):
    def __init__(self, model_registrar,
                 hyperparams, log_writer,
                 device):
        super().__init__(
            model_registrar, hyperparams, log_writer, device)

    def set_environment(self, env):
        self.env = env
        self.node_models_dict.clear()
        edge_types = env.get_edge_types()
        for node_type in env.NodeType:
            if node_type in self.pred_state.keys():
                self.node_models_dict[node_type] = MultimodalGenerativeCVAEEWTA(
                    env, node_type, self.model_registrar, self.hyperparams,
                    self.device, edge_types, log_writer=self.log_writer)

    def train_loss(self, batch, node_type, loss_type,
                   lambda_kalman=1.0, lambda_sim=1.0, temp=0.1,
                   contrastive=False):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map, score) = batch
        neighbors_data_st_0_dict = restore(neighbors_data_st)
        neighbors_edge_value_0_dict = restore(neighbors_edge_value)
        x = x_t.to(self.device)
        score = score.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)
        model = self.node_models_dict[node_type]
        loss = model.train_loss(inputs=x,
                                inputs_st=x_st_t,
                                first_history_indices=first_history_index,
                                labels=y,
                                labels_st=y_st_t,
                                neighbors=neighbors_data_st_0_dict,
                                neighbors_edge_value=neighbors_edge_value_0_dict,
                                robot=robot_traj_st_t,
                                map=map,
                                prediction_horizon=self.ph,
                                loss_type=loss_type,
                                score=score,
                                contrastive=contrastive,
                                factor_con=lambda_kalman,
                                temp=temp)
        return loss

    def predict(self,
                scene,
                timesteps,
                ph,
                min_future_timesteps=0,
                min_history_timesteps=1,
                selected_node_type=None,
                return_original=False,
                return_feats=True):
        predictions_dict = {}
        features = None
        features_original = None
        for node_type in self.env.NodeType:
            # assert node_type=='PEDESTRIAN'
            if node_type not in self.pred_state:
                continue

            if selected_node_type is not None and node_type != selected_node_type:
                continue

            model = self.node_models_dict[node_type]
            batch = get_timesteps_data(env=self.env, scene=scene, t=timesteps, node_type=node_type, state=self.state,
                                       pred_state=self.pred_state, edge_types=model.edge_types,
                                       min_ht=min_history_timesteps, max_ht=self.max_ht, min_ft=min_future_timesteps,
                                       max_ft=min_future_timesteps, hyperparams=self.hyperparams)
            if batch is None:
                continue
            (first_history_index,
             x_t, y_t, x_st_t, y_st_t,
             neighbors_data_st,
             neighbors_edge_value,
             robot_traj_st_t,
             map), nodes, timesteps_o = batch
            x = x_t.to(self.device)
            x_st_t = x_st_t.to(self.device)
            if robot_traj_st_t is not None:
                robot_traj_st_t = robot_traj_st_t.to(self.device)
            if type(map) == torch.Tensor:
                map = map.to(self.device)
            if return_original:
                predictions, features, features_original = model.predict(
                    inputs=x,
                    inputs_st=x_st_t,
                    first_history_indices=first_history_index,
                    neighbors=neighbors_data_st,
                    neighbors_edge_value=neighbors_edge_value,
                    robot=robot_traj_st_t,
                    map=map,
                    prediction_horizon=ph,
                    return_original=return_original,
                    return_feat=return_feats)
            else:
                predictions, features,cluster_features = model.predict(
                    inputs=x,
                    inputs_st=x_st_t,
                    first_history_indices=first_history_index,
                    neighbors=neighbors_data_st,
                    neighbors_edge_value=neighbors_edge_value,
                    robot=robot_traj_st_t,
                    map=map,
                    prediction_horizon=ph,
                    return_original=return_original,
                    return_feat=return_feats)

            predictions = predictions.permute(1, 0, 2, 3)
            predictions_np = predictions.cpu().detach().numpy()
            for i, ts in enumerate(timesteps_o):
                if ts not in predictions_dict.keys():
                    predictions_dict[ts] = dict()
                predictions_dict[ts][nodes[i]] = np.transpose(predictions_np[:, [i]], (1, 0, 2, 3))

        if return_original:
            return predictions_dict, features,features_original

        return predictions_dict, features

    def predict_futurefeature(self,
                scene,
                timesteps,
                ph,
                cluster_result,
                num_samples=1,
                min_future_timesteps=0,
                min_history_timesteps=1,
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False,
                return_original=False,
                return_feats=True):

        predictions_dict = {}
        for node_type in self.env.NodeType:
            if node_type not in self.pred_state:
                continue

            model = self.node_models_dict[node_type]

            # Get Input data for node type and given timesteps
            batch = get_timesteps_data(env=self.env, scene=scene, t=timesteps, node_type=node_type, state=self.state,
                                       pred_state=self.pred_state, edge_types=model.edge_types,
                                       min_ht=min_history_timesteps, max_ht=self.max_ht, min_ft=min_future_timesteps,
                                       max_ft=min_future_timesteps, hyperparams=self.hyperparams)
            # There are no nodes of type present for timestep
            if batch is None:
                continue
            (first_history_index,
             x_t, y_t, x_st_t, y_st_t,
             neighbors_data_st,
             neighbors_edge_value,
             robot_traj_st_t,
             map), nodes, timesteps_o = batch

            x = x_t.to(self.device)
            x_st_t = x_st_t.to(self.device)
            if robot_traj_st_t is not None:
                robot_traj_st_t = robot_traj_st_t.to(self.device)
            if type(map) == torch.Tensor:
                map = map.to(self.device)

            # Run forward pass
            if return_original:
                predictions, features, features_original = model.predict_futurefeature(
                    inputs=x,
                    inputs_st=x_st_t,
                    first_history_indices=first_history_index,
                    neighbors=neighbors_data_st,
                    neighbors_edge_value=neighbors_edge_value,
                    robot=robot_traj_st_t,
                    map=map,
                    prediction_horizon=ph,
                    cluster_result=cluster_result,
                    return_original=return_original)
            else:
                predictions, features,cluster_features = model.predict_futurefeature(
                    inputs=x,
                    inputs_st=x_st_t,
                    first_history_indices=first_history_index,
                    neighbors=neighbors_data_st,
                    neighbors_edge_value=neighbors_edge_value,
                    robot=robot_traj_st_t,
                    map=map,
                    prediction_horizon=ph,
                    cluster_result = cluster_result,
                    return_original=return_original)

            predictions_np = predictions.cpu().detach().numpy()

            # Assign predictions to node
            for i, ts in enumerate(timesteps_o):
                if ts not in predictions_dict.keys():
                    predictions_dict[ts] = dict()
                predictions_dict[ts][nodes[i]] =predictions_np[i]

        if return_original:
            return predictions_dict, features,features_original

        return predictions_dict, features,cluster_features


    def encoding(self,batch,node_type,device=None,return_feat=True):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch
        if device==None:
            device=self.device
        x = x_t.to(device)
        x_st_t= x_st_t.to(device)
        y = y_t.to(device)
        neighbors_data_st_0_dict = restore(neighbors_data_st)
        neighbors_edge_value_0_dict = restore(neighbors_edge_value)
        model=self.node_models_dict[node_type]
        predictions, features,original_features = model.predict(
            inputs=x,
            inputs_st=x_st_t,
            first_history_indices=first_history_index,
            neighbors=neighbors_data_st_0_dict,
            neighbors_edge_value=neighbors_edge_value_0_dict,
            robot=robot_traj_st_t,
            map=map,
            prediction_horizon=self.ph,return_original=True,
            return_feat=return_feat)
        # predictions = predictions.permute(1, 0, 2, 3)
        return predictions,features,original_features

    def encoding_split(self,batch,node_type,device=None):

        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch
        if device==None:
            device=self.device
        x = x_t.to(device)
        x_st_t= x_st_t.to(device)
        y = y_t.to(device)
        neighbors_data_st_0_dict = restore(neighbors_data_st)
        neighbors_edge_value_0_dict = restore(neighbors_edge_value)
        model=self.node_models_dict[node_type]
        predictions,features_history,features_social,features_a_0,features_state= model.predict_split(
            inputs=x,
            inputs_st=x_st_t,
            first_history_indices=first_history_index,
            neighbors=neighbors_data_st_0_dict,
            neighbors_edge_value=neighbors_edge_value_0_dict,
            robot=robot_traj_st_t,
            map=map,
            prediction_horizon=self.ph,return_original=True)
        return predictions,features_history,features_social,features_a_0,features_state


    def vis_encoding(self,batch,node_type,device=None):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch
        if device==None:
            device=self.device
        first_history_index=torch.tensor(first_history_index).unsqueeze(0).to(device)
        x = x_t.to(device).unsqueeze(0)
        x_st_t= x_st_t.to(device).unsqueeze(0)
        y = y_t.to(device)
        for key in neighbors_data_st.keys():
            aa=list()
            aa.append(neighbors_data_st[key])
            neighbors_data_st[key] = aa
        for key in neighbors_edge_value.keys():
            aa=list()
            aa.append(neighbors_edge_value[key])
            neighbors_edge_value[key] =aa
        model=self.node_models_dict[node_type]
        predictions, features,original_features = model.predict(
            inputs=x,
            inputs_st=x_st_t,
            first_history_indices=first_history_index,
            neighbors=neighbors_data_st,
            neighbors_edge_value=neighbors_edge_value,
            robot=robot_traj_st_t,
            map=map,
            prediction_horizon=self.ph,return_original=True)
        # predictions = predictions.permute(1, 0, 2, 3)

        return predictions,features,original_features