from torch.utils import data
import numpy as np
from .preprocessing import get_node_timestep_data

import torch

class EnvironmentDataset(object):
    def __init__(self, env, state, pred_state, node_freq_mult, scene_freq_mult, hyperparams,scores=None, **kwargs):
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.scores=None
        if scores is not None:
           self.scores=scores
        self.max_ht = self.hyperparams['maximum_history_length']
        self.max_ft = kwargs['min_future_timesteps']
        self.node_type_datasets = list()
        self._augment = False
        for node_type in env.NodeType:
            if node_type not in hyperparams['pred_state']:
                continue
            self.node_type_datasets.append(NodeTypeDataset(env, node_type, state, pred_state, node_freq_mult,
                                                           scene_freq_mult, hyperparams,scores, **kwargs))

    @property
    def augment(self):
        return self._augment

    @augment.setter
    def augment(self, value):
        self._augment = value
        for node_type_dataset in self.node_type_datasets:
            node_type_dataset.augment = value

    def __iter__(self):
        return iter(self.node_type_datasets)


class NodeTypeDataset(data.Dataset):
    def __init__(self, env, node_type, state, pred_state, node_freq_mult,
                 scene_freq_mult, hyperparams, scores=None,augment=False, **kwargs):
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.scores = None
        if scores is not None:
            self.scores=scores
        self.max_ht = self.hyperparams['maximum_history_length']
        self.max_ft = kwargs['min_future_timesteps']

        self.augment = augment

        self.node_type = node_type
        self.index = self.index_env(node_freq_mult, scene_freq_mult, **kwargs)
        self.len = len(self.index)
        self.edge_types = [edge_type for edge_type in env.get_edge_types() if edge_type[0] is node_type]

    def index_env(self, node_freq_mult, scene_freq_mult, **kwargs):
        index = list()
        for scene in self.env.scenes:
            present_node_dict = scene.present_nodes(np.arange(0, scene.timesteps), type=self.node_type, **kwargs)
            for t, nodes in present_node_dict.items():
                for node in nodes:
                    index += [(scene, t, node)] *\
                             (scene.frequency_multiplier if scene_freq_mult else 1) *\
                             (node.frequency_multiplier if node_freq_mult else 1)

        return index

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        (scene, t, node) = self.index[i]

        if self.augment:
            original_scene=scene
            original_node=original_scene.get_node_by_id(node.id)
            scene = scene.augment()
            node = scene.get_node_by_id(node.id)

        if self.scores is not None:
          score=self.scores[i]
          if  self.augment:
              return (get_node_timestep_data(self.env, scene, t, node, self.state, self.pred_state,
                                             self.edge_types, self.max_ht, self.max_ft, self.hyperparams),
                      # score,
                      i,
                      get_node_timestep_data(self.env, original_scene, t, original_node, self.state, self.pred_state,
                                             self.edge_types, self.max_ht, self.max_ft, self.hyperparams)
                      )
          return (get_node_timestep_data(self.env, scene, t, node, self.state, self.pred_state,
                                             self.edge_types, self.max_ht, self.max_ft, self.hyperparams),
                  i)
                      # score,


        if self.augment:
            # return (get_node_timestep_data(self.env, scene, t, node, self.state, self.pred_state,
            #                                self.edge_types, self.max_ht, self.max_ft, self.hyperparams),
            #         # score,
            #         i,
            #         get_node_timestep_data(self.env, original_scene, t, original_node, self.state, self.pred_state,
            #                                self.edge_types, self.max_ht, self.max_ft, self.hyperparams)
            #         )
            return (get_node_timestep_data(self.env, scene, t, node, self.state, self.pred_state,
                                       self.edge_types, self.max_ht, self.max_ft, self.hyperparams),
                # score,
                i,)

        return (get_node_timestep_data(self.env, scene, t, node, self.state, self.pred_state,
                                       self.edge_types, self.max_ht, self.max_ft, self.hyperparams),
               i)


class ClassNodeTypeDataset(data.Dataset):
    def __init__(self, env, node_type, state, pred_state, node_freq_mult,
                 scene_freq_mult, hyperparams, scores=None,augment=False, **kwargs):
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.scores = None
        if scores is not None:
            self.scores=scores
            self.scores_int=scores.astype(int)

        self.max_ht = self.hyperparams['maximum_history_length']
        self.max_ft = kwargs['min_future_timesteps']

        self.augment = augment

        self.node_type = node_type
        self.index = self.index_env(node_freq_mult, scene_freq_mult, **kwargs)
        self.len = len(self.index)
        self.edge_types = [edge_type for edge_type in env.get_edge_types() if edge_type[0] is node_type]

    def index_env(self, node_freq_mult, scene_freq_mult, **kwargs):
        index = list()
        for scene in self.env.scenes:
            present_node_dict = scene.present_nodes(np.arange(0, scene.timesteps), type=self.node_type, **kwargs)
            for t, nodes in present_node_dict.items():
                for node in nodes:
                    index += [(scene, t, node)] *\
                             (scene.frequency_multiplier if scene_freq_mult else 1) *\
                             (node.frequency_multiplier if node_freq_mult else 1)

        return index

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        (scene, t, node) = self.index[i]

        if self.augment:
            original_scene=scene
            original_node=original_scene.get_node_by_id(node.id)
            scene = scene.augment()
            node = scene.get_node_by_id(node.id)

        if self.scores is not None:
          score=self.scores[i]
          if  self.augment:
              return (get_node_timestep_data(self.env, scene, t, node, self.state, self.pred_state,
                                             self.edge_types, self.max_ht, self.max_ft, self.hyperparams),
                      # score,
                      i,
                      get_node_timestep_data(self.env, original_scene, t, original_node, self.state, self.pred_state,
                                             self.edge_types, self.max_ht, self.max_ft, self.hyperparams)
                      )
          return (get_node_timestep_data(self.env, scene, t, node, self.state, self.pred_state,
                                             self.edge_types, self.max_ht, self.max_ft, self.hyperparams),
                      # score,
                      i,)

        if self.augment:
            return (get_node_timestep_data(self.env, scene, t, node, self.state, self.pred_state,
                                           self.edge_types, self.max_ht, self.max_ft, self.hyperparams),
                    # score,
                    i,
                    get_node_timestep_data(self.env, original_scene, t, original_node, self.state, self.pred_state,
                                           self.edge_types, self.max_ht, self.max_ft, self.hyperparams)
                    )
        return (get_node_timestep_data(self.env, scene, t, node, self.state, self.pred_state,
                                       self.edge_types, self.max_ht, self.max_ft, self.hyperparams),
                i)
                # get_node_timestep_data(self.env, original_scene, t, original_node, self.state, self.pred_state,
                #                        self.edge_types, self.max_ht, self.max_ft, self.hyperparams))


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, scores,cls_num,indices=None, num_samples=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(scores))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count=cls_num

        beta = 0.9999
        effective_num = 1.0 - np.power(beta, label_to_count)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)

        # weight for each sample
        weights = [per_cls_weights[np.clip(scores[idx].astype(int),0,12)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples