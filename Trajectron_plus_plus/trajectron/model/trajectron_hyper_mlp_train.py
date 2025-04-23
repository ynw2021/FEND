import torch
import numpy as np
from .mgcvaeEWTA_hyper_re import MultimodalGenerativeCVAEEWTA
from model.dataset import get_timesteps_data, restore
import faiss
import torch.nn as nn


class Trajectron(object):
    def __init__(self, model_registrar,
                 hyperparams, log_writer,
                 device):
        super(Trajectron, self).__init__()
        self.hyperparams = hyperparams
        self.log_writer = log_writer
        self.device = device
        self.curr_iter = 0

        self.model_registrar = model_registrar
        self.node_models_dict = dict()
        self.nodes = set()

        self.env = None

        self.min_ht = self.hyperparams['minimum_history_length']
        self.max_ht = self.hyperparams['maximum_history_length']
        self.ph = self.hyperparams['prediction_horizon']
        self.state = self.hyperparams['state']
        self.state_length = dict()
        for state_type in self.state.keys():
            self.state_length[state_type] = int(
                np.sum([len(entity_dims) for entity_dims in self.state[state_type].values()])
            )
        self.pred_state = self.hyperparams['pred_state']

    def set_environment(self, env):
        self.env = env

        self.node_models_dict.clear()
        edge_types = env.get_edge_types()

        for node_type in env.NodeType:
            # Only add a Model for NodeTypes we want to predict
            if node_type in self.pred_state.keys():
                self.node_models_dict[node_type] = MultimodalGenerativeCVAEEWTA(env,
                                                                            node_type,
                                                                            self.model_registrar,
                                                                            self.hyperparams,
                                                                            self.device,
                                                                            edge_types,
                                                                            log_writer=self.log_writer)

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter
        for node_str, model in self.node_models_dict.items():
            model.set_curr_iter(curr_iter)

    def set_annealing_params(self):
        for node_str, model in self.node_models_dict.items():
            model.set_annealing_params()

    def step_annealers(self, node_type=None):
        if node_type is None:
            for node_type in self.node_models_dict:
                self.node_models_dict[node_type].step_annealers()
        else:
            self.node_models_dict[node_type].step_annealers()

    def classfier_loss(self,batch, node_type,index,cluster_result,classifier_mode):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)

            # Run forward pass
        model = self.node_models_dict[node_type]
        loss = model.classifier_loss(inputs=x,
                                inputs_st=x_st_t,
                                first_history_indices=first_history_index,
                                labels=y,
                                labels_st=y_st_t,
                                neighbors=restore(neighbors_data_st),
                                neighbors_edge_value=restore(neighbors_edge_value),
                                robot=robot_traj_st_t,
                                map=map,
                                prediction_horizon=self.ph,
                                index=index,
                                cluster_result=cluster_result,
                                classifier_mode=classifier_mode)
        return loss




    def train_loss(self, batch, node_type,loss_type,score,contrastive,change=False,factor_con=50):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)

        # Run forward pass
        model = self.node_models_dict[node_type]
        loss = model.train_loss(inputs=x,
                                inputs_st=x_st_t,
                                first_history_indices=first_history_index,
                                labels=y,
                                labels_st=y_st_t,
                                neighbors=restore(neighbors_data_st),
                                neighbors_edge_value=restore(neighbors_edge_value),
                                robot=robot_traj_st_t,
                                map=map,
                                prediction_horizon=self.ph,
                                loss_type=loss_type,
                                score=score,
                                contrastive=contrastive,
                                factor_con=factor_con,
                                change=change)
        return loss


    def pcl_contrast_loss(self,batch,node_type,loss_type,cluster_result,index,factor_con=50):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)

        model = self.node_models_dict[node_type]
        loss = model.pcl_contrast_loss(inputs=x,
                                inputs_st=x_st_t,
                                first_history_indices=first_history_index,
                                labels=y,
                                labels_st=y_st_t,
                                neighbors=restore(neighbors_data_st),
                                neighbors_edge_value=restore(neighbors_edge_value),
                                robot=robot_traj_st_t,
                                map=map,
                                prediction_horizon=self.ph,
                                loss_type=loss_type,
                                cluster_result=cluster_result,
                                index=index,
                                factor_con=factor_con)
        return loss


    def pcl_contrast_loss_instance_reweight_by_loss_justlongtail(self,batch,node_type,loss_type,predict_loss,cluster_result_with_future,cluster_result,index,factor_con=50,original=False):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)

        model = self.node_models_dict[node_type]
        loss = model.pcl_contrast_loss_reweight_by_loss_justlongtail(inputs=x,
                                inputs_st=x_st_t,
                                first_history_indices=first_history_index,
                                labels=y,
                                labels_st=y_st_t,
                                neighbors=restore(neighbors_data_st),
                                neighbors_edge_value=restore(neighbors_edge_value),
                                robot=robot_traj_st_t,
                                map=map,
                                prediction_horizon=self.ph,
                                loss_type=loss_type,
                                predict_loss=predict_loss,
                                cluster_result_with_future=cluster_result_with_future,
                                cluster_result=cluster_result,
                                index=index,
                                factor_con=factor_con,
                                original=original)
        return loss








    def eval_loss(self, batch, node_type):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)

        # Run forward pass
        model = self.node_models_dict[node_type]
        nll = model.eval_loss(inputs=x,
                              inputs_st=x_st_t,
                              first_history_indices=first_history_index,
                              labels=y,
                              labels_st=y_st_t,
                              neighbors=restore(neighbors_data_st),
                              neighbors_edge_value=restore(neighbors_edge_value),
                              robot=robot_traj_st_t,
                              map=map,
                              prediction_horizon=self.ph)

        return nll.cpu().detach().numpy()

    def eval_loss_for_hard_sample(self, batch, node_type,score):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)

        # Run forward pass
        model = self.node_models_dict[node_type]
        nll = model.eval_loss(inputs=x,
                              inputs_st=x_st_t,
                              first_history_indices=first_history_index,
                              labels=y,
                              labels_st=y_st_t,
                              neighbors=restore(neighbors_data_st),
                              neighbors_edge_value=restore(neighbors_edge_value),
                              robot=robot_traj_st_t,
                              map=map,
                              prediction_horizon=self.ph)

        return nll.cpu().detach().numpy()

    def eval_loss_with_center(self, batch, node_type,cluster_center):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)

        # Run forward pass
        model = self.node_models_dict[node_type]
        nll = model.eval_loss(inputs=x,
                              inputs_st=x_st_t,
                              first_history_indices=first_history_index,
                              labels=y,
                              labels_st=y_st_t,
                              neighbors=restore(neighbors_data_st),
                              neighbors_edge_value=restore(neighbors_edge_value),
                              robot=robot_traj_st_t,
                              map=map,
                              prediction_horizon=self.ph,
                              cluster_center=cluster_center)

        return nll.cpu().detach().numpy()

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
        if type(map) == torch.Tensor:
            map = map.to(self.device)
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

    def encoding_social(self,batch,node_type,device=None,return_feat=True):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch
        if device == None:
            device = self.device
        x = x_t.to(device)
        x_st_t = x_st_t.to(device)
        y = y_t.to(device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)
        neighbors_data_st_0_dict = restore(neighbors_data_st)
        neighbors_edge_value_0_dict = restore(neighbors_edge_value)
        model = self.node_models_dict[node_type]
        social_features,normalized_social_feature,features = model.predict_social(
            inputs=x,
            inputs_st=x_st_t,
            first_history_indices=first_history_index,
            neighbors=neighbors_data_st_0_dict,
            neighbors_edge_value=neighbors_edge_value_0_dict,
            robot=robot_traj_st_t,
            map=map,
            prediction_horizon=self.ph, return_original=True,
            return_feat=return_feat)
        # predictions = predictions.permute(1, 0, 2, 3)

        return social_features,normalized_social_feature,features


    def prediction_by_feature(self,batch,node_type,features,device=None):
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
        predictions, features, features_original = model.predict_by_feature(
            inputs=x,
            inputs_st=x_st_t,
            first_history_indices=first_history_index,
            neighbors=neighbors_data_st,
            neighbors_edge_value=neighbors_edge_value,
            robot=robot_traj_st_t,
            map=map,
            prediction_horizon=self.ph,
            features=features,
            return_original=True,
            return_feat=True)

        return predictions, features, features_original






    def encoding_two_head(self,batch,node_type,device=None,return_feat=True):
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
        predictions, features_social,features_history = model.predict(
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

        return predictions,features_social,features_history

    def encoding_with_center(self,batch,node_type,device=None,return_feat=True):
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
        features = model.encode(
            inputs=x,
            inputs_st=x_st_t,
            first_history_indices=first_history_index,
            neighbors=neighbors_data_st_0_dict,
            neighbors_edge_value=neighbors_edge_value_0_dict,
            robot=robot_traj_st_t,
            map=map,
            prediction_horizon=self.ph, return_original=True,
            return_feat=return_feat
        )
        # predictions = predictions.permute(1, 0, 2, 3)

        return features

    def encoding_future(self,batch,node_type,device=None):
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
        predictions, features,cluster_features = model.predict(
            inputs=x,
            inputs_st=x_st_t,
            first_history_indices=first_history_index,
            neighbors=neighbors_data_st_0_dict,
            neighbors_edge_value=neighbors_edge_value_0_dict,
            robot=robot_traj_st_t,
            map=map,
            prediction_horizon=self.ph,
            return_feat=False)
        # predictions = predictions.permute(1, 0, 2, 3)
        return predictions

    def predict_by_score(self,
                         scene,
                         timesteps,
                         ph,
                         mask,
                         num_samples=1,
                         min_future_timesteps=0,
                         min_history_timesteps=1,
                         z_mode=False,
                         gmm_mode=False,
                         full_dist=True,
                         all_z_sep=False,
                         return_original=False):

        predictions_dict_easy = {}
        predictions_dict_hard = {}
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
                predictions, features, features_original = model.predict(
                    inputs=x,
                    inputs_st=x_st_t,
                    first_history_indices=first_history_index,
                    neighbors=neighbors_data_st,
                    neighbors_edge_value=neighbors_edge_value,
                    robot=robot_traj_st_t,
                    map=map,
                    prediction_horizon=ph,
                    return_original=return_original)
            else:
                predictions, features, cluster_features = model.predict(
                    inputs=x,
                    inputs_st=x_st_t,
                    first_history_indices=first_history_index,
                    neighbors=neighbors_data_st,
                    neighbors_edge_value=neighbors_edge_value,
                    robot=robot_traj_st_t,
                    map=map,
                    prediction_horizon=ph,
                    return_original=return_original)

            predictions_np = predictions.cpu().detach().numpy()
            predictions_np_hard=predictions_np[mask]
            predictions_np_easy = predictions_np[1-mask]



            # Assign predictions to node
            for i, ts in enumerate(timesteps_o):
                if ts not in predictions_dict_easy.keys():
                    predictions_dict_easy[ts] = dict()
                predictions_dict_easy[ts][nodes[i]] = predictions_np_easy[i]
            for i, ts in enumerate(timesteps_o):
                if ts not in predictions_dict_hard.keys():
                    predictions_dict_hard[ts] = dict()
                predictions_dict_hard[ts][nodes[i]] = predictions_np_hard[i]


        return predictions_dict_easy,predictions_dict_hard


    def predict(self,
                scene,
                timesteps,
                ph,
                num_samples=1,
                min_future_timesteps=0,
                min_history_timesteps=1,
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False,
                return_original=False):

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
                predictions, features, features_original = model.predict(
                    inputs=x,
                    inputs_st=x_st_t,
                    first_history_indices=first_history_index,
                    neighbors=neighbors_data_st,
                    neighbors_edge_value=neighbors_edge_value,
                    robot=robot_traj_st_t,
                    map=map,
                    prediction_horizon=ph,
                    return_original=return_original)
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





