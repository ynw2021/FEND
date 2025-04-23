import torch
from torch import nn, optim, utils
import numpy as np
import os
import time
import dill
import json
import random
import pathlib
import warnings
import sys

sys.path.append('Trajectron_plus_plus')
sys.path.append('Trajectron_plus_plus/trajectron')
sys.path.append('../../Trajectron_plus_plus')
sys.path.append('../../Trajectron_plus_plus/trajectron')
sys.path.append('./')
sys.path.append("./Trajectron_plus_plus")
sys.path.append("./Trajectron_plus_plus/trajectron")
sys.path.append("./Trajectron_plus_plus/experiments/nuScenes")

from tqdm import tqdm
import visualization
import evaluation
import matplotlib.pyplot as plt
from argument_parser import args
from model.trajectron_hyper_mlp_train import Trajectron
from model.model_registrar import ModelRegistrar
from model.model_utils import cyclical_lr
from model.dataset import EnvironmentDataset, collate
from torch.utils.tensorboard import SummaryWriter
# torch.autograd.set_detect_anomaly(True)

import faiss

# from model.mgcvaeEWTA import model_encdec
from model.encoding_model import model_encdec_re
from normalize_utils import normalize_y
from contrast_rec_training import pretrain_full_aug,pretrain_full_aug_warmup_for_nuscenes


if not torch.cuda.is_available() or args.device == 'cpu':
    args.device = torch.device('cpu')
else:
    if torch.cuda.device_count() == 1:
        # If you have CUDA_VISIBLE_DEVICES set, which you should,
        # then this will prevent leftover flag arguments from
        # messing with the device allocation.
        args.device = 'cuda:0'

    args.device_idx = args.device
    args.device = torch.device(args.device)


if args.eval_device is None:
    args.eval_device = torch.device('cpu')

# This is needed for memory pinning using a DataLoader (otherwise memory is pinned to cuda:0 by default)
torch.cuda.set_device(args.device)

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


args.change_epoch=50
# args.num_clusters=[100,250,500]
# args.num_clusters=[50,100,500]
args.num_clusters_p=[20,50,100]
args.num_clusters_v=[5,12,25]
# args.num_clusters_v=[20,50,100]
# args.num_clusters_v=[200,500,1000]
# args.num_clusters_v=[400,1000,2000]
args.temperature=0.1

args.output_dim=232


def run_kmeans(x, num_clusters,hyperparams,device):
    print('performing Kmeans cluster')
    results = {
        'tra2cluster': [], 'centroids': [], 'density': []
    }
    for seed, num_cluster in enumerate(num_clusters):
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = device
        index = faiss.GpuIndexFlatL2(res, d, cfg)

        clus.train(x, index)

        D, I = index.search(x, 1)
        tra2cluster = [int(n[0]) for n in I]

        centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)
        Dcluster = [[] for c in range(k)]
        for tra, i in enumerate(tra2cluster):
            Dcluster[i].append(D[tra][0])

        density = np.zeros(k)
        for i, dist in enumerate(Dcluster):
            if len(dist) > 1:
                d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)  ######todo
                density[i] = d

        d_max = density.max()
        for i, dist in enumerate(Dcluster):
            if len(dist) <= 1:
                density[i] = d_max

        density = density.clip(np.percentile(density, 10), np.percentile(density, 90))  #####todo
        density = hyperparams['temperature'] * density / density.mean()

        centroids = torch.tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)  ################todo

        tra2cluster = torch.LongTensor(tra2cluster).cuda()
        density = torch.Tensor(density).cuda()

        results['centroids'].append(centroids)
        results['density'].append(density)
        results['tra2cluster'].append(tra2cluster)

    return results


def load_model(model_dir, env, ts=100):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)
    trajectron = Trajectron(model_registrar, hyperparams, None, 'cpu')
    trajectron.set_environment(env)
    trajectron.set_annealing_params()
    return trajectron, hyperparams


def momentumupdate(model_registrar, model_registrar_moment, m=0.9, freeze=True):
    namekeys = list(model_registrar_moment.model_dict.keys())
    namekeys1 = list(model_registrar.model_dict.keys())
    assert len(namekeys) == len(namekeys1)
    for idx, model_name in enumerate(namekeys):
        for param_q, param_k in zip(model_registrar.model_dict[model_name].parameters(),
                                    model_registrar_moment.model_dict[model_name].parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)
            if m == 0 and freeze:
                param_k.requires_grad = False


def main():
    # Load hyperparameters from json
    args.conf='models/nuScenes_ewta/config.json'
    if not os.path.exists(args.conf):
        print('Config json not found!')
    with open(args.conf, 'r', encoding='utf-8') as conf_json:
        hyperparams = json.load(conf_json)

    PEDESTRIAN_kalman_score_path='kalman/nuScenes_PEDESTRIAN_train.pkl'
    with open(PEDESTRIAN_kalman_score_path, 'rb') as f:
        PEDESTRIAN_scores = dill.load(f, encoding='latin1')
    VEHICLE_kalman_score_path='kalman/nuScenes_VEHICLE_train.pkl'
    with open(VEHICLE_kalman_score_path, 'rb') as f:
        VEHICLE_scores = dill.load(f, encoding='latin1')

    scores=dict()
    scores['PEDESTRIAN']=PEDESTRIAN_scores
    scores['VEHICLE'] =VEHICLE_scores



    #### hyperparams for faiss kmeans
    hyperparams['PEDESTRIAN/num_clusters']=args.num_clusters_p
    hyperparams['VEHICLE/num_clusters'] = args.num_clusters_v
    hyperparams['temperature']=args.temperature

    hyperparams['node_freq_mult_train'] = args.node_freq_mult_train


    hyperparams['hyper_rnn_dim']=64
    hyperparams['hyper_z_dim'] = 32

    hyperparams['prefix'] = args.prefix

    hyperparams['pred_state'] ={"VEHICLE": {"position": ["x", "y"]}}  ######todo: just VEHICLE


    print('-----------------------')
    print('| TRAINING PARAMETERS |')
    print('-----------------------')
    print('| batch_size: %d' % args.batch_size)
    print('| device: %s' % args.device)
    print('| eval_device: %s' % args.eval_device)
    print('| Offline Scene Graph Calculation: %s' % args.offline_scene_graph)
    print('| EE state_combine_method: %s' % args.edge_state_combine_method)
    print('| EIE scheme: %s' % args.edge_influence_combine_method)
    print('| dynamic_edges: %s' % args.dynamic_edges)
    print('| robot node: %s' % args.incl_robot_node)
    print('| edge_addition_filter: %s' % args.edge_addition_filter)
    print('| edge_removal_filter: %s' % args.edge_removal_filter)
    print('| MHL: %s' % hyperparams['minimum_history_length'])
    print('| PH: %s' % hyperparams['prediction_horizon'])
    print('-----------------------')

    log_writer = None
    model_dir = None
    if not args.debug:
        # Create the log and model directiory if they're not present.
        model_dir = os.path.join(args.log_dir,
                                 args.log_tag)
                                 # 'models_' + time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime()) + args.log_tag)
        pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
        # Save config to model directory
        with open(os.path.join(model_dir, 'config.json'), 'w') as conf_json:
            json.dump(hyperparams, conf_json)

        log_writer = SummaryWriter(log_dir=model_dir)

    # Load training and evaluation environments and scenes
    args.train_data_dict='nuScenes_train_full.pkl'
    args.eval_data_dict='nuScenes_val_full.pkl'

    train_scenes = []
    train_data_path = os.path.join(args.data_dir, args.train_data_dict)
    with open(train_data_path, 'rb') as f:
        train_env = dill.load(f, encoding='latin1')

    for attention_radius_override in args.override_attention_radius:
        node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
        train_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    if train_env.robot_type is None and hyperparams['incl_robot_node']:
        train_env.robot_type = train_env.NodeType[0]  # TODO: Make more general, allow the user to specify?
        for scene in train_env.scenes:
            scene.add_robot_from_nodes(train_env.robot_type)

    train_scenes = train_env.scenes
    train_scenes_sample_probs = train_env.scenes_freq_mult_prop if args.scene_freq_mult_train else None

    train_dataset = EnvironmentDataset(train_env,
                                       hyperparams['state'],
                                       hyperparams['pred_state'],
                                       scene_freq_mult=hyperparams['scene_freq_mult_train'],
                                       node_freq_mult=hyperparams['node_freq_mult_train'],
                                       hyperparams=hyperparams,
                                       scores=None,
                                       min_history_timesteps=hyperparams['minimum_history_length'],
                                       min_future_timesteps=hyperparams['prediction_horizon'],
                                       return_robot=not args.incl_robot_node)
    train_data_loader = dict()
    for node_type_data_set in train_dataset:
        if len(node_type_data_set) == 0:
            continue

        node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                     collate_fn=collate,
                                                     pin_memory=False if args.device is 'cpu' else True,
                                                     batch_size=args.batch_size,
                                                     shuffle=True,
                                                     num_workers=args.preprocess_workers)
        train_data_loader[node_type_data_set.node_type] = node_type_dataloader

    print(f"Loaded training data from {train_data_path}")


    feature_data_loader = dict()
    for node_type_data_set in train_dataset:
        if len(node_type_data_set) == 0:
            continue

        node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                     collate_fn=collate,
                                                     pin_memory=False if args.device is 'cpu' else True,
                                                     batch_size=args.batch_size,
                                                     shuffle=False,
                                                     num_workers=args.preprocess_workers)
        feature_data_loader[node_type_data_set.node_type] = node_type_dataloader

    print(f"Loaded training data from {train_data_path}")



    eval_scenes = []
    eval_scenes_sample_probs = None
    if args.eval_every is not None:
        eval_data_path = os.path.join(args.data_dir, args.eval_data_dict)
        with open(eval_data_path, 'rb') as f:
            eval_env = dill.load(f, encoding='latin1')

        for attention_radius_override in args.override_attention_radius:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            eval_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

        if eval_env.robot_type is None and hyperparams['incl_robot_node']:
            eval_env.robot_type = eval_env.NodeType[0]  # TODO: Make more general, allow the user to specify?
            for scene in eval_env.scenes:
                scene.add_robot_from_nodes(eval_env.robot_type)

        eval_scenes = eval_env.scenes
        eval_scenes_sample_probs = eval_env.scenes_freq_mult_prop if args.scene_freq_mult_eval else None

        eval_dataset = EnvironmentDataset(eval_env,
                                          hyperparams['state'],
                                          hyperparams['pred_state'],
                                          scene_freq_mult=hyperparams['scene_freq_mult_eval'],
                                          node_freq_mult=hyperparams['node_freq_mult_eval'],
                                          hyperparams=hyperparams,
                                          scores=None,
                                          min_history_timesteps=hyperparams['minimum_history_length'],
                                          min_future_timesteps=hyperparams['prediction_horizon'],
                                          return_robot=not args.incl_robot_node)
        eval_data_loader = dict()
        for node_type_data_set in eval_dataset:
            if len(node_type_data_set) == 0:
                continue

            node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                         collate_fn=collate,
                                                         pin_memory=False if args.eval_device is 'cpu' else True,
                                                         batch_size=args.eval_batch_size,
                                                         shuffle=False,
                                                         num_workers=args.preprocess_workers)
            eval_data_loader[node_type_data_set.node_type] = node_type_dataloader

        print(f"Loaded evaluation data from {eval_data_path}")




####################################################################################################
    # Offline Calculate Scene Graph
    if hyperparams['offline_scene_graph'] == 'yes':
        print(f"Offline calculating scene graphs")
        for i, scene in enumerate(train_scenes):
            scene.calculate_scene_graph(train_env.attention_radius,
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])
            print(f"Created Scene Graph for Training Scene {i}")

        for i, scene in enumerate(eval_scenes):
            scene.calculate_scene_graph(eval_env.attention_radius,
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])
            print(f"Created Scene Graph for Evaluation Scene {i}")

    model_registrar = ModelRegistrar(model_dir, args.device)
    model_registrar_moment = ModelRegistrar(model_dir, args.device)
   #############################

    trajectron = Trajectron(model_registrar,
                            hyperparams,
                            log_writer,
                            args.device)

    trajectron.set_environment(train_env)
    trajectron.set_annealing_params()
    print('Created Training Model.')

    eval_trajectron = None
    if args.eval_every is not None or args.vis_every is not None:
        eval_trajectron = Trajectron(model_registrar,
                                     hyperparams,
                                     log_writer,
                                     args.eval_device)
        eval_trajectron.set_environment(eval_env)
        eval_trajectron.set_annealing_params()
    print('Created Evaluation Model.')



#######################momentum network ##############################
    moment_trajectron=None
    moment_trajectron = Trajectron(model_registrar_moment,
                                   hyperparams,
                                   log_writer,
                                   args.eval_device)
    moment_trajectron.set_environment(train_env)
    moment_trajectron.set_annealing_params()

    model_registrar.to(args.device)
    model_registrar_moment.to(args.device)
    momentumupdate(model_registrar, model_registrar_moment, m=0)


    optimizer = dict()
    lr_scheduler = dict()
    for node_type in train_env.NodeType:
        if node_type not in hyperparams['pred_state']:
            continue
        optimizer[node_type] = optim.Adam([{'params': model_registrar.get_all_but_name_match('map_encoder').parameters()},
                                           {'params': model_registrar.get_name_match('map_encoder').parameters(), 'lr':0.0008}], lr=hyperparams['learning_rate'])
        # Set Learning Rate
        if hyperparams['learning_rate_style'] == 'const':
            lr_scheduler[node_type] = optim.lr_scheduler.ExponentialLR(optimizer[node_type], gamma=1.0)
        elif hyperparams['learning_rate_style'] == 'exp':
            lr_scheduler[node_type] = optim.lr_scheduler.ExponentialLR(optimizer[node_type],
                                                                       gamma=hyperparams['learning_decay_rate'])




################################################score_trajectron
    if args.reload:
        load_dir = 'new_model/' + args.reload_prefix
        model_registrar_temp = ModelRegistrar(load_dir, args.device)
        model_registrar_temp.load_models(args.reload_epoch)
        model_registrar.to(args.device)
        model_registrar_temp.to(args.device)
        with torch.no_grad():
             momentumupdate(model_registrar_temp,model_registrar,m=0,freeze=False)
             namekeys = list(model_registrar_temp.model_dict.keys())
             for idx, model_name in enumerate(namekeys):
                 for param_k in (model_registrar_temp.model_dict[model_name].parameters()):
                     param_k.requires_grad = False


        model_registrar_moment_temp = ModelRegistrar(load_dir, args.device)
        model_registrar_moment_temp.load_models(args.reload_epoch,momentum=True)
        with torch.no_grad():
             momentumupdate(model_registrar_moment_temp,model_registrar_moment,m=0,freeze=False)



        # for node_type in train_env.NodeType:
        for node_type in ['VEHICLE']:
            opt_statedict = \
                torch.load('new_model/' + args.reload_prefix + '/last_optimizer',
                           map_location=args.device)[node_type]
            optimizer[node_type].load_state_dict(opt_statedict.state_dict())


        # for node_type in train_env.NodeType:
        for node_type in ['VEHICLE']:
            schedular_statedict = \
                torch.load('new_model/' + args.reload_prefix + '/last_schedular',
                           map_location=args.device)[node_type]
            lr_scheduler[node_type].load_state_dict(schedular_statedict.state_dict())





    #################################
    #           TRAINING            #
    #################################
    curr_iter_node_type = {node_type: 0 for node_type in train_data_loader.keys()}
    changing_scores=dict()
    for score_node_type in ['VEHICLE','PEDESTRIAN']:
        changing_scores[score_node_type] = torch.tensor(scores[score_node_type]).float().to(args.device).unsqueeze(-1)

    max_hl = hyperparams['maximum_history_length']
    ph = hyperparams['prediction_horizon']
    change=False
    warmup_epoch=15
    pretrain_flag=0
    update_flag=0
    cluster_flag=0
    if args.reload:
        start_epoch=args.reload_epoch + 1
    else:
        start_epoch=1
    for epoch in range(start_epoch, args.train_epochs + 1):
        ewta_k = max(int(20 / 2 ** (int(epoch / 5))), 1)
        # ewta_k=1
        if  epoch>warmup_epoch and pretrain_flag==0:
            train_dataset.augment = args.augment  #####todo: augment!
            #######################################################pretrain#################################
            # for pretrain_node_type in ['VEHICLE','PEDESTRIAN']:
            # for pretrain_node_type in ['VEHICLE']:
                # train_dataset_split.augment=args.augment
                # pretrain_full_aug_warmup_for_nuscenes(hyperparams,log_writer,args,pretrain_node_type,train_data_loader_split,feature_data_loader_split,eval_data_loader_split)
            ################################################################################################
            train_dataset.augment =False
            pretrain_flag=1
            pretrain_encdec_dict=dict()
            for pretrain_node_type in ['VEHICLE']:
               pretrain_encdec = model_encdec_re(hyperparams)
               # model_to_load=torch.load('full_normalization_reconstruction_aug_momentum_'+ args.prefix+ 'warmup500model_ae_',map_location=args.device)
               model_to_load = torch.load(
                    'Trajectron_plus_plus/trajectron/full_normalization_reconstruction_aug_momentum_' + args.prefix +pretrain_node_type+str(hyperparams[pretrain_node_type+'/num_clusters'])+ 'warmupmodel_ae_best_',
                    map_location=args.device)

               # model_to_load=torch.load(model_dir+'model_ae_')
               # model_to_load=torch.load('../experiments/pedestrians/models/old/future_momentum_EM_hyper_zara1'+'model_ae_')
               pretrain_encdec.load_state_dict(model_to_load.state_dict())
               pretrain_encdec.to(args.device)
               pretrain_encdec_dict[pretrain_node_type]=pretrain_encdec


        if epoch>warmup_epoch:
            with torch.no_grad():
                if epoch==warmup_epoch + 1:
                    momentumupdate(model_registrar, model_registrar_moment,m=0)
                if cluster_flag==0:
                    cluster_result_f = dict()
                    future_feature_list = []
                    past_feature_list = []
                    # cluster_result = {'tra2cluster': [[] for i in range(len(hyperparams['num_clusters']))],
                    #                   'centroids': [[] for i in range(len(hyperparams['num_clusters']))],
                    #                   'density': [[] for i in range(len(hyperparams['num_clusters']))]}

                    for node_type, data_loader in feature_data_loader.items():
                        pbar = tqdm(data_loader, ncols=80)
                        for (batch, index) in pbar:
                            (first_history_index,
                             x_t, y_t, x_st_t, y_st_t,
                             neighbors_data_st,
                             neighbors_edge_value,
                             robot_traj_st_t,
                             map) = batch

                            xst = x_st_t.to(args.device)[:, :, :2]
                            yst = y_st_t.to(args.device)
                            last_dis_tan = yst[:, 0, 1] / (yst[:, 0, 0] + 1e-10)
                            last_angle = -torch.arctan(last_dis_tan)
                            last_angle = torch.where(yst[:, 0, 0] > 0, last_angle, -(torch.tensor(np.pi).to(last_angle.device) - last_angle))
                            yst = normalize_y(yst, last_angle)
                            xst = normalize_y(xst, last_angle)

                            fut_features, past_features = pretrain_encdec_dict[node_type].get_state_encoding(yst, xst,
                                                                                             first_history_index)
                            del batch
                            del map

                            fut_features = fut_features.cpu().numpy()
                            past_features = past_features.cpu().numpy()
                            future_feature_list.append(fut_features)
                            past_feature_list.append(past_features)

                        future_feature_array = np.concatenate(future_feature_list, axis=0)
                        past_feature_array = np.concatenate(past_feature_list, axis=0)
                        cluster_array = np.concatenate([past_feature_array, future_feature_array], axis=-1)
                        cluster_result_f[node_type] = run_kmeans(cluster_array, hyperparams[node_type + '/num_clusters'], hyperparams,
                                                    args.device_idx)
                    cluster_flag=1

                #################mometnum update##############
                model_registrar_moment.to(args.device)
                model_registrar.to(args.device)
                momentumupdate(model_registrar, model_registrar_moment)


                for node_type, data_loader in feature_data_loader.items():
                    if epoch == start_epoch and epoch > warmup_epoch:
                        feature_list = []
                        pbar = tqdm(data_loader, ncols=80)
                        for  (batch,index) in pbar:
                            _,features,_=moment_trajectron.encoding(batch,node_type,device=args.device)
                            feature_list.append(features)
                            del batch

                        feature_tensor = torch.cat(feature_list, dim=0)


                    for idx, num_cluster in enumerate(hyperparams[node_type +'/num_clusters']):
                         cluster_center_list=[]
                         density_list = []
                         dist_list = []
                         for clus_i in range(num_cluster):
                             center = torch.mean(feature_tensor[cluster_result_f[node_type]['tra2cluster'][idx] == clus_i], dim=0)
                             center = nn.functional.normalize(center, p=2, dim=-1)
                             dist = torch.norm(feature_tensor[cluster_result_f[node_type]['tra2cluster'][idx] == clus_i] - center,
                                               p=2,
                                               dim=-1)
                             density = torch.mean(dist) / np.log(len(dist) + 10)
                             density_list.append(density.unsqueeze(0))
                             cluster_center_list.append(center.unsqueeze(0))
                         cluster_center = torch.cat(cluster_center_list, dim=0)
                         density = torch.cat(density_list, dim=0)
                         cluster_result_f[node_type]['centroids'][idx] = cluster_center.to(args.device)
                         density = density.cpu().numpy()
                         d_max = np.max(density)
                         for i, dist in enumerate(dist_list):
                             if dist <= 1:
                                 density[i] = d_max

                         density = density.clip(np.percentile(density, 10), np.percentile(density, 90))  #####todo
                         density = hyperparams['temperature'] * density / density.mean()
                         density = torch.tensor(density).to(center.device)
                         cluster_result_f[node_type]['density'][idx] = density.to(args.device)



        loss_type='epe-top-'+str(ewta_k)
        model_registrar.to(args.device)
        train_dataset.augment = args.augment   #####todo: augment!


        for node_type, data_loader in train_data_loader.items():
            if epoch > warmup_epoch - 1:
                feature_tensor = torch.zeros((len(train_dataset.node_type_datasets[0]),args.output_dim)).to(args.device)  ###TODO:just one node type!for more node type must fix bug
            curr_iter = curr_iter_node_type[node_type]
            pbar = tqdm(data_loader, ncols=80)
            for (batch,index) in pbar:
                index=index.long().to(args.device)
                if epoch > warmup_epoch - 1:
                    _,moment_features,_=moment_trajectron.encoding(batch,node_type,device=args.device)
                    feature_tensor[index]=moment_features
                score=changing_scores[node_type][index,:].squeeze(-1)
                trajectron.set_curr_iter(curr_iter)
                trajectron.step_annealers(node_type)
                optimizer[node_type].zero_grad()
                factorcon_list = [1, 0.2, 0.2]
                # factorcon_list = [50,50,50]
                if epoch>warmup_epoch:
                    changing_factorcon_id = int((epoch - warmup_epoch) / 5)
                    changing_factorcon = factorcon_list[changing_factorcon_id]
                    train_loss = trajectron.pcl_contrast_loss_instance_reweight_by_loss_justlongtail(batch, node_type, loss_type=loss_type,
                                                                       predict_loss=score,cluster_result_with_future=cluster_result_f[node_type],
                                                                       cluster_result=cluster_result_f[node_type], index=index,
                                                                       factor_con=changing_factorcon,
                                                                       original=False)
                else:
                    train_loss = trajectron.train_loss(batch, node_type, loss_type=loss_type, score=score,
                                                       contrastive= False, change=False, factor_con=1)
                    del batch

                pbar.set_description(f"Epoch {epoch}, {node_type} L: {train_loss.item():.2f}")
                train_loss.backward()
                # Clipping gradients.
                if hyperparams['grad_clip'] is not None:
                    nn.utils.clip_grad_value_(model_registrar.parameters(), hyperparams['grad_clip'])
                optimizer[node_type].step()

                # Stepping forward the learning rate scheduler and annealers.
                lr_scheduler[node_type].step()

                if not args.debug:
                    log_writer.add_scalar(f"{node_type}/train/learning_rate",
                                          lr_scheduler[node_type].get_lr()[0],
                                          curr_iter)
                    log_writer.add_scalar(f"{node_type}/train/loss", train_loss, curr_iter)

                curr_iter += 1
            curr_iter_node_type[node_type] = curr_iter
        train_dataset.augment = False
        if args.eval_every is not None or args.vis_every is not None:
            eval_trajectron.set_curr_iter(epoch)

        #################################
        #        VISUALIZATION          #
        #################################
        if args.vis_every is not None and not args.debug and epoch % args.vis_every == 0 and epoch > 0:
            max_hl = hyperparams['maximum_history_length']
            ph = hyperparams['prediction_horizon']
            with torch.no_grad():
                # Predict random timestep to plot for train data set
                if args.scene_freq_mult_viz:
                    scene = np.random.choice(train_scenes, p=train_scenes_sample_probs)
                else:
                    scene = np.random.choice(train_scenes)
                timestep = scene.sample_timesteps(1, min_future_timesteps=ph)
                predictions,features,_ = trajectron.predict(scene,
                                                 timestep,
                                                 ph,
                                                 min_future_timesteps=ph,
                                                 z_mode=True,
                                                 gmm_mode=True,
                                                 all_z_sep=False,
                                                 full_dist=False)

                # Plot predicted timestep for random scene
                fig, ax = plt.subplots(figsize=(10, 10))
                visualization.visualize_prediction(ax,
                                                   predictions,
                                                   scene.dt,
                                                   max_hl=max_hl,
                                                   ph=ph,
                                                   map=scene.map['VISUALIZATION'] if scene.map is not None else None)
                ax.set_title(f"{scene.name}-t: {timestep}")
                log_writer.add_figure('train/prediction', fig, epoch)

                model_registrar.to(args.eval_device)
                # Predict random timestep to plot for eval data set
                if args.scene_freq_mult_viz:
                    scene = np.random.choice(eval_scenes, p=eval_scenes_sample_probs)
                else:
                    scene = np.random.choice(eval_scenes)
                timestep = scene.sample_timesteps(1, min_future_timesteps=ph)
                predictions,features,_ = eval_trajectron.predict(scene,
                                                      timestep,
                                                      ph,
                                                      num_samples=20,
                                                      min_future_timesteps=ph,
                                                      z_mode=False,
                                                      full_dist=False)

                # Plot predicted timestep for random scene
                fig, ax = plt.subplots(figsize=(10, 10))
                visualization.visualize_prediction(ax,
                                                   predictions,
                                                   scene.dt,
                                                   max_hl=max_hl,
                                                   ph=ph,
                                                   map=scene.map['VISUALIZATION'] if scene.map is not None else None)
                ax.set_title(f"{scene.name}-t: {timestep}")
                log_writer.add_figure('eval/prediction', fig, epoch)

                # Predict random timestep to plot for eval data set
                predictions,features,_ = eval_trajectron.predict(scene,
                                                      timestep,
                                                      ph,
                                                      min_future_timesteps=ph,
                                                      z_mode=True,
                                                      gmm_mode=True,
                                                      all_z_sep=True,
                                                      full_dist=False)

                # Plot predicted timestep for random scene
                fig, ax = plt.subplots(figsize=(10, 10))
                visualization.visualize_prediction(ax,
                                                   predictions,
                                                   scene.dt,
                                                   max_hl=max_hl,
                                                   ph=ph,
                                                   map=scene.map['VISUALIZATION'] if scene.map is not None else None)
                ax.set_title(f"{scene.name}-t: {timestep}")
                log_writer.add_figure('eval/prediction_all_z', fig, epoch)

        #################################
        #           EVALUATION          #
        #################################
        if args.eval_every is not None and not args.debug and epoch % args.eval_every == 0 and epoch > 0:
            max_hl = hyperparams['maximum_history_length']
            ph = hyperparams['prediction_horizon']
            model_registrar.to(args.eval_device)
            with torch.no_grad():
                # Calculate evaluation loss
                for node_type, data_loader in eval_data_loader.items():
                    eval_loss = []
                    print(f"Starting Evaluation @ epoch {epoch} for node type: {node_type}")
                    pbar = tqdm(data_loader, ncols=80)
                    for (batch,index) in pbar:
                        eval_loss_node_type = eval_trajectron.eval_loss(batch, node_type)
                        pbar.set_description(f"Epoch {epoch}, {node_type} L: {eval_loss_node_type.item():.2f}")
                        eval_loss.append({node_type: {'nll': [eval_loss_node_type]}})
                        del batch

                    evaluation.log_batch_errors(eval_loss,
                                                log_writer,
                                                f"{node_type}/eval_loss",
                                                epoch)

                # Predict batch timesteps for evaluation dataset evaluation
                eval_batch_errors = []
                for scene in tqdm(eval_scenes, desc='Sample Evaluation', ncols=80):
                    timesteps = scene.sample_timesteps(args.eval_batch_size)

                    predictions,features,_ = eval_trajectron.predict(scene,
                                                          timesteps,
                                                          ph,
                                                          num_samples=50,
                                                          min_future_timesteps=ph,
                                                          full_dist=False)


                    eval_batch_errors.append(evaluation.compute_batch_statistics(predictions,
                                                                                 scene.dt,
                                                                                 max_hl=max_hl,
                                                                                 ph=ph,
                                                                                 node_type_enum=eval_env.NodeType,
                                                                                 map=scene.map))

                evaluation.log_batch_errors(eval_batch_errors,
                                            log_writer,
                                            'eval',
                                            epoch,
                                            bar_plot=['kde'],
                                            box_plot=['ade', 'fde'])

                # Predict maximum likelihood batch timesteps for evaluation dataset evaluation
                eval_batch_errors_ml = []
                for scene in tqdm(eval_scenes, desc='MM Evaluation', ncols=80):
                    timesteps = scene.sample_timesteps(scene.timesteps)

                    predictions,features,_ = eval_trajectron.predict(scene,
                                                          timesteps,
                                                          ph,
                                                          num_samples=1,
                                                          min_future_timesteps=ph,
                                                          z_mode=True,
                                                          gmm_mode=True,
                                                          full_dist=False)

                    eval_batch_errors_ml.append(evaluation.compute_batch_statistics(predictions,
                                                                                    scene.dt,
                                                                                    max_hl=max_hl,
                                                                                    ph=ph,
                                                                                    map=scene.map,
                                                                                    node_type_enum=eval_env.NodeType,
                                                                                    kde=False))

                evaluation.log_batch_errors(eval_batch_errors_ml,
                                            log_writer,
                                            'eval/ml',
                                            epoch)

        if args.save_every is not None and args.debug is False and epoch % args.save_every == 0:
            model_registrar.save_models(epoch)
            if epoch>0:
                 model_registrar_moment.save_models(epoch,momentum=True)
            torch.save(optimizer, model_dir + '/last_optimizer')
            torch.save(lr_scheduler, model_dir + '/last_schedular')
            if epoch in [5,10,15,20,25]:
                torch.save(optimizer, model_dir + '/last_optimizer_'+str(epoch))
                torch.save(lr_scheduler, model_dir + '/last_schedular_'+str(epoch))









if __name__ == '__main__':
    main()
