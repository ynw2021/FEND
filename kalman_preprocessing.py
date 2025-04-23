import sys
import os
import dill
import json
import argparse
import torch
import numpy as np
import pandas as pd

sys.path.append("Trajectron_plus_plus/trajectron")
sys.path.append("Trajectron_plus_plus/")
sys.path.append("Trajectron_plus_plus/trajectron/model")
from tqdm import tqdm
from utilities import estimate_kalman_filter
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
import evaluation
import utils
from scipy.interpolate import RectBivariateSpline
from utils import prediction_output_to_trajectories
import math



seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model full path", type=str,
                    default='models/nuScenes_ewta')
parser.add_argument("--checkpoint", help="model checkpoint to evaluate", type=int,
                    default=25)
parser.add_argument("--data", help="full path to data file", type=str,
                    default='Trajectron_plus_plus/experiments/processed/nuScenes_train_full.pkl')
parser.add_argument("--node_type", help="node type to evaluate", type=str,
                    default='PEDESTRIAN')
parser.add_argument("--prediction_horizon", nargs='+', help="prediction horizon", type=int,
                    default=[6])
parser.add_argument("--save_output", type=str,default='kalman/nuScenes_PEDESTRIAN_train')
args = parser.parse_args()


def load_model(model_dir, env, ts=100):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    if 'ewta' in model_dir and 'nuScenes' not in model_dir:
        model_registrar.load_models(ts)
    else:
        model_registrar.model_dict.clear()
        checkpoint_path = os.path.join(model_dir, 'model_registrar-%d.pt' % ts)
        checkpoint = torch.load(checkpoint_path, map_location=model_registrar.device)
        model_registrar.model_dict = checkpoint['model_dict']
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    trajectron = Trajectron(model_registrar, hyperparams, None, 'cpu')

    trajectron.set_environment(env)
    trajectron.set_annealing_params()
    return trajectron, hyperparams


def calculate_epe(pred, gt):
    diff_x = (gt[0] - pred[0]) * (gt[0] - pred[0])
    diff_y = (gt[1] - pred[1]) * (gt[1] - pred[1])
    epe = math.sqrt(diff_x+diff_y)
    return epe

if __name__ == "__main__":
    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')

    eval_stg, hyperparams = load_model(args.model, env, ts=args.checkpoint)

    if 'override_attention_radius' in hyperparams:
        for attention_radius_override in hyperparams['override_attention_radius']:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    scenes = env.scenes

    print("-- Preparing Node Graph")
    for scene in tqdm(scenes):
        scene.calculate_scene_graph(env.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])

    for ph in args.prediction_horizon:
        print(f"Prediction Horizon: {ph}")
        max_hl = hyperparams['maximum_history_length']

        with torch.no_grad():
            epes = []
            for scene in tqdm(scenes):
                timesteps = np.arange(scene.timesteps)

                predictions,features,_ = eval_stg.predict(scene,
                                               timesteps,
                                               ph,
                                               num_samples=1,
                                               min_history_timesteps=1,
                                               min_future_timesteps=6,
                                               z_mode=False,
                                               gmm_mode=False,
                                               full_dist=False)

                (prediction_dict,
                 histories_dict,
                 futures_dict) = prediction_output_to_trajectories(predictions,
                                                                   scene.dt,
                                                                   max_hl,
                                                                   ph,
                                                                   prune_ph_to_future=True)

                for t in prediction_dict.keys():
                    for node in prediction_dict[t].keys():
                        if node.type == args.node_type:
                            z_future = estimate_kalman_filter(histories_dict[t][node], ph)
                            epe = calculate_epe(z_future, futures_dict[t][node][-1, :])
                            if 'test' in args.data:
                                epes.append(epe)
                            else:
                                for i in range(node.frequency_multiplier):
                                    epes.append(epe)

            kalman_errors = np.array(epes)
            print('Kalman (FDE): %.2f' % (np.mean(kalman_errors)))
            print(kalman_errors.shape)
            with open(args.save_output + '.pkl', 'wb') as f_writer:
                dill.dump(kalman_errors, f_writer)