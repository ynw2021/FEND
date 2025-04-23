""" This code is based on the Trajectron++ repository.

    For usage, see the License of Trajectron++ under:
    https://github.com/StanfordASL/Trajectron-plus-plus
"""
import os
import sys
import dill
import json
import argparse
import torch
import numpy as np
import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

sys.path.append('./')
sys.path.append("./Trajectron_plus_plus")
sys.path.append("./Trajectron_plus_plus/trajectron")
sys.path.append("./Trajectron_plus_plus/experiments/pedestrians")

# from trajectronEWTA import TrajectronEWTA
from trajectronEWTA_hyper_mlp import TrajectronEWTA
from Trajectron_plus_plus.trajectron.model.model_registrar import ModelRegistrar
from Trajectron_plus_plus.trajectron.evaluation import evaluation


from torch import nn, optim, utils
from Trajectron_plus_plus.trajectron.model.dataset import EnvironmentDataset, collate

PARAMS = {
    'eth-ucy': (7, 12),
    'nuScenes': (1, 8)
}



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model full path", type=str, 
                        default='./models/nuScenes_model/trajectron_map_int_fend_ewta_withoutfrequency')
    parser.add_argument("--checkpoint", help="model checkpoint to evaluate", type=int,
                        default=25)
    parser.add_argument("--data", help="full path to data file", type=str,
                          # default='data/eth_test.pkl')
                        default='./Trajectron_plus_plus/experiments/processed/nuScenes_test_full.pkl')
    parser.add_argument("--node_type", type=str,default='VEHICLE')
    # parser.add_argument("--kalman", type=str,default='kalman/nuScenes_VEHICLE_test_ewta_6.pkl')
    parser.add_argument("--baseline_error", type=str, 
                        default='./kalman/nuScenes_VEHICLE_test_ewta.pkl')
                        # default='./kalman/nuScenes_VEHICLE_test_ewta_baseline_withfrequency.pkl')
    parser.add_argument("--prefix", type=str, default='nuscenes_v')

    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_dir, env, ts=100):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    # model_registrar.load_models(ts)
    if 'ewta' in model_dir and 'nuScenes' not in model_dir:
        model_registrar.load_models(ts)
    elif 'justlongtail' in model_dir:
        model_registrar.load_models(ts)
    elif 'new_model' in model_dir:
        model_registrar.load_models(ts)    
    else:
        model_registrar.model_dict.clear()
        checkpoint_path = os.path.join(model_dir, 'model_registrar-%d.pt' % ts)
        checkpoint = torch.load(checkpoint_path, map_location=model_registrar.device)
        model_registrar.model_dict = checkpoint['model_dict']
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)
    hyperparams['hyper_rnn_dim']=64
    hyperparams['hyper_z_dim'] = 32
    trajectron = TrajectronEWTA(model_registrar, hyperparams, None, 'cpu')
    trajectron.set_environment(env)
    trajectron.set_annealing_params()
    return trajectron, hyperparams











if __name__ == "__main__":
    set_seed(0)
    args = parse_arguments()

    args.error_est = True

    args.vis=False


    args.test_future=False


    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')
    eval_stg, hyperparams = load_model(args.model, env, ts=args.checkpoint)
    if 'override_attention_radius' in hyperparams:
        for attention_radius_override in hyperparams['override_attention_radius']:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            env.attention_radius[(node_type1, node_type2)] = float(attention_radius)


    scenes = env.scenes
    for scene in tqdm.tqdm(scenes):
        scene.calculate_scene_graph(env.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])
    ph = hyperparams['prediction_horizon']
    max_hl = hyperparams['maximum_history_length']
    prediction_parameters = PARAMS['nuScenes'] if 'nuScenes' in args.data else PARAMS['eth-ucy']

    # hyperparams['minimum_history_length']=1
    if args.test_future:
        eval_scenes = scenes

        eval_dataset = EnvironmentDataset(env,
                                          hyperparams['state'],
                                          hyperparams['pred_state'],
                                          scene_freq_mult=hyperparams['scene_freq_mult_eval'],
                                          node_freq_mult=hyperparams['node_freq_mult_eval'],
                                          hyperparams=hyperparams,
                                          scores=None,
                                          min_history_timesteps=hyperparams['minimum_history_length'],
                                          min_future_timesteps=hyperparams['prediction_horizon'],
                                          )

        eval_data_loader = dict()
        for node_type_data_set in eval_dataset:
            if len(node_type_data_set) == 0:
                continue

            node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                         collate_fn=collate,
                                                         pin_memory=False,
                                                         batch_size=256,
                                                         shuffle=False,
                                                         num_workers=5)
            eval_data_loader[node_type_data_set.node_type] = node_type_dataloader



    if args.baseline_error:
       with open(args.baseline_error, 'rb') as f:
           baseline_errors = dill.load(f, encoding='latin1')
    with torch.no_grad():
        print('processing %s' % args.node_type)
        eval_ade_batch_errors = np.array([])
        eval_fde_batch_errors = np.array([])
        baseline_index=0
        scene_index=0
        features_list=[]
        for scene in tqdm.tqdm(scenes):
            timesteps = np.arange(scene.timesteps)
            predictions, features,features_original = eval_stg.predict(
            # predictions = eval_stg.predict_future(
                scene, timesteps, ph, min_history_timesteps=prediction_parameters[0],
                min_future_timesteps=prediction_parameters[1],
                # z_mode=False,
                # gmm_mode=False,
                # full_dist=False)
                selected_node_type=args.node_type,
            return_original=True,
            return_feats=True)


            batch_error_dict = evaluation.compute_batch_statistics(
                predictions, scene.dt, max_hl=max_hl, ph=ph,
                node_type_enum=env.NodeType, map=None,
                best_of=True,
                prune_ph_to_future=True)
            eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))
            eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))
            batch_length=len(batch_error_dict[args.node_type]['fde'])
            baseline_batch_errors=baseline_errors[baseline_index:baseline_index+batch_length]
            baseline_index=baseline_index+batch_length

            largest_batch_error=np.argsort(baseline_batch_errors)
            challenging=largest_batch_error[-int(3/100*batch_length):]


            scene_index += 1

        total_number_testing_samples = eval_fde_batch_errors.shape[0]
        print('All         (ADE/FDE): %.2f, %.2f   --- %d' % (
            eval_ade_batch_errors.mean(),
            eval_fde_batch_errors.mean(),
            total_number_testing_samples))




        if args.error_est:
            assert baseline_errors.shape[0] == eval_fde_batch_errors.shape[0]
            largest_errors_indexes = np.argsort(eval_fde_batch_errors)
            mask = np.ones(eval_ade_batch_errors.shape, dtype=bool)
            for top_index in range(1, 4):
                challenging = largest_errors_indexes[-int(
                    total_number_testing_samples * top_index / 100):]
                fde_errors_challenging = np.copy(eval_fde_batch_errors)
                ade_errors_challenging = np.copy(eval_ade_batch_errors)
                mask[challenging] = False
                fde_errors_challenging[mask] = 0
                ade_errors_challenging[mask] = 0
                print('Test result on Challenging Top %d selected by this net prediction error (ADE/FDE): %.2f, %.2f   --- %d' %
                      (top_index,
                       np.sum(ade_errors_challenging) / len(challenging),
                       np.sum(fde_errors_challenging) / len(challenging),
                       len(challenging)))



            largest_errors_indexes = np.argsort(baseline_errors)
            mask = np.ones(eval_ade_batch_errors.shape, dtype=bool)
            for top_index in range(1, 6):
                challenging = largest_errors_indexes[-int(
                    total_number_testing_samples * top_index / 100):]
                fde_errors_challenging = np.copy(eval_fde_batch_errors)
                ade_errors_challenging = np.copy(eval_ade_batch_errors)
                mask[challenging] = False
                fde_errors_challenging[mask] = 0
                ade_errors_challenging[mask] = 0
                print('Test result on Challenging Top %d selected by kalman error (ADE/FDE): %.2f, %.2f   --- %d' %
                      (top_index,
                       np.sum(ade_errors_challenging) / len(challenging),
                       np.sum(fde_errors_challenging) / len(challenging),
                       len(challenging)))

                np.save(args.prefix + str(top_index) + '_ade',np.sum(ade_errors_challenging) / len(challenging), )
                np.save(args.prefix + str(top_index) + '_fde',np.sum(fde_errors_challenging) / len(challenging), )

            mask = np.ones(eval_ade_batch_errors.shape, dtype=bool)
            normal = largest_errors_indexes[:int(
                total_number_testing_samples * 95 / 100)]
            fde_errors_normal = np.copy(eval_fde_batch_errors)
            ade_errors_normal = np.copy(eval_ade_batch_errors)
            mask[normal] = False
            fde_errors_normal[mask] = 0
            ade_errors_normal[mask] = 0
            print('Test result on normal selected by baseline prediction error (ADE/FDE): %.5f, %.5f   --- %d' %
                  (
                   np.sum(ade_errors_normal) / len(normal),
                   np.sum(fde_errors_normal) / len(normal),
                   len(normal)))

            np.save(args.prefix + 'normal' + '_ade',np.sum(ade_errors_normal) / len(normal))
            np.save(args.prefix + 'normal' + '_fde',np.sum(fde_errors_normal) / len(normal))

            mask = np.ones(eval_ade_batch_errors.shape, dtype=bool)
            normal1 = largest_errors_indexes[:int(
                total_number_testing_samples * 30 / 100)]
            fde_errors_normal = np.copy(eval_fde_batch_errors)
            ade_errors_normal = np.copy(eval_ade_batch_errors)
            mask[normal1] = False
            fde_errors_normal[mask] = 0
            ade_errors_normal[mask] = 0
            print('Test result on normal 30 selected by baseline prediction error (ADE/FDE): %.5f, %.5f   --- %d' %
                  (
                   np.sum(ade_errors_normal) / len(normal1),
                   np.sum(fde_errors_normal) / len(normal1),
                   len(normal1)))

            mask = np.ones(eval_ade_batch_errors.shape, dtype=bool)
            normal2 = largest_errors_indexes[-int(
                total_number_testing_samples * 70 / 100):]
            fde_errors_normal = np.copy(eval_fde_batch_errors)
            ade_errors_normal = np.copy(eval_ade_batch_errors)
            mask[normal2] = False
            fde_errors_normal[mask] = 0
            ade_errors_normal[mask] = 0
            print('Test result on hard 70 selected by baseline prediction error (ADE/FDE): %.5f, %.5f   --- %d' %
                  (
                   np.sum(ade_errors_normal) / len(normal2),
                   np.sum(fde_errors_normal) / len(normal2),
                   len(normal2)))






        ade_mean=np.mean(eval_ade_batch_errors)
        fde_mean=np.mean(eval_fde_batch_errors)
        print('MEAN  (ADE/FDE): %.5f, %.5f  ' %
              (ade_mean,fde_mean)
              )

        np.save(args.prefix + 'mean' + '_ade', ade_mean)
        np.save(args.prefix + 'mean' + '_fde', fde_mean)

        ade_var=np.var(eval_ade_batch_errors)
        fde_var=np.var(eval_fde_batch_errors)
        print('VAR  (ADE/FDE): %.2f, %.2f  ' %
              (ade_var,fde_var)
              )

        ade_VaR95=np.sort(eval_ade_batch_errors)[int(len(eval_ade_batch_errors)*0.95)]
        fde_VaR95 =np.sort(eval_fde_batch_errors)[int(len(eval_fde_batch_errors)*0.95)]
        print('VaR95  (ADE/FDE): %.2f, %.2f  ' %
              (ade_VaR95,fde_VaR95)
              )


        ade_VaR98=np.sort(eval_ade_batch_errors)[int(len(eval_ade_batch_errors)*0.98)]
        fde_VaR98 =np.sort(eval_fde_batch_errors)[int(len(eval_fde_batch_errors)*0.98)]
        print('VaR98  (ADE/FDE): %.2f, %.2f  ' %
              (ade_VaR98,fde_VaR98)
              )

        ade_VaR99=np.sort(eval_ade_batch_errors)[int(len(eval_ade_batch_errors)*0.99)]
        fde_VaR99 =np.sort(eval_fde_batch_errors)[int(len(eval_fde_batch_errors)*0.99)]
        print('VaR99  (ADE/FDE): %.2f, %.2f  ' %
              (ade_VaR99,fde_VaR99)
              )


        ade_skew=np.sum(((eval_ade_batch_errors-ade_mean)/np.sqrt(ade_var))**3)/len(eval_ade_batch_errors)
        fde_skew = np.sum(((eval_fde_batch_errors - fde_mean) / np.sqrt(fde_var)) ** 3) / len(eval_fde_batch_errors)
        print('SKEW  (ADE/FDE): %.2f, %.2f  ' %
              (ade_skew,fde_skew )
              )


        ade_kurtosis=np.sum(((eval_ade_batch_errors-ade_mean)/np.sqrt(ade_var))**4)/len(eval_ade_batch_errors)
        fde_kurtosis = np.sum(((eval_fde_batch_errors - fde_mean) / np.sqrt(fde_var)) ** 4) / len(eval_fde_batch_errors)
        print('KURTOSIS  (ADE/FDE): %.2f, %.2f  ' %
              (ade_kurtosis,fde_kurtosis )
              )

        ade_max_error=np.max(eval_ade_batch_errors)
        fde_max_error=np.max(eval_fde_batch_errors)
        print('MAX  (ADE/FDE): %.2f, %.2f  ' %
              (ade_max_error,fde_max_error)
              )



