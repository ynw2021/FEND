import numpy as np
import torch
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_dilation
from scipy.stats import gaussian_kde
from utils import prediction_output_to_trajectories,prediction_output_to_trajectories_nll
import visualization
from matplotlib import pyplot as plt


def compute_ade(predicted_trajs, gt_traj):
    error = np.linalg.norm(predicted_trajs - gt_traj, axis=-1)
    ade = np.mean(error, axis=-1)
    return ade.flatten()


def compute_fde(predicted_trajs, gt_traj):
    if predicted_trajs.ndim==4:
        predicted_trajs=predicted_trajs[0]
    guess_num=predicted_trajs.shape[0]
    shape1,shape2=gt_traj.shape[0],gt_traj.shape[1]
    gt_traj=np.expand_dims(gt_traj,0).repeat(guess_num,axis=0)
    final_error = np.linalg.norm(predicted_trajs[:, -1, :] - gt_traj[:,-1,:], axis=-1)
    return final_error.flatten()


def compute_kde_nll(predicted_trajs, gt_traj):
    kde_ll = 0.
    log_pdf_lower_bound = -20
    num_timesteps = gt_traj.shape[0]
    num_batches = predicted_trajs.shape[0]

    for batch_num in range(num_batches):
        for timestep in range(num_timesteps):
            try:
                kde = gaussian_kde(predicted_trajs[batch_num, :, timestep].T)
                pdf = np.clip(kde.logpdf(gt_traj[timestep].T), a_min=log_pdf_lower_bound, a_max=None)[0]
                kde_ll += pdf / (num_timesteps * num_batches)
            except np.linalg.LinAlgError:
                kde_ll = np.nan

    return -kde_ll

def compute_nll(predicted_trajs, gt_traj):
    mu, log_sigma, corr, log_pi=predicted_trajs
    log_sigma = np.clip(log_sigma, a_min=1e-5,a_max=1e5)
    log_pi = np.clip(log_pi, a_min=1e-5,a_max=1e5)
    # log_pi = log_pi - np.logsumexp(log_pi, dim=-1, keepdim=True)
    log_pi = log_pi - np.log(np.sum(np.exp(log_pi), axis=-1, keepdims=True))
    sigma = np.sqrt(np.exp(log_sigma)) + 1e-5
    corr = np.tanh(corr)
    # diff = (mu - gts) ** 2
    diff = bivariate_log_prob(mu, sigma, corr, log_pi, log_sigma, gt_traj)  # (bs,20)
    return -np.log(np.sum(np.exp(diff),axis=-1))


def bivariate_log_prob(mus, sigmas, corrs, log_pi, log_sigmas, value):
    dx = value - mus  ##[..,N,2]
    exp_nominator = (np.sum((dx / sigmas) ** 2, axis=-1)
                     - 2 * corrs * np.prod(dx, axis=-1) / np.prod(sigmas, axis=-1))

    one_minus_rho2 = 1 - corrs ** 2
    one_minus_rho2 = np.clip(one_minus_rho2, a_min=1e-5, a_max=1)

    component_log_p = -(2 * np.log(2 * np.pi)
                        + np.log(one_minus_rho2)
                        + 2 * np.sum(log_sigmas, axis=-1)
                        + exp_nominator / one_minus_rho2
                        ) / 2

    return np.sum(component_log_p, axis=-1) + log_pi

def compute_obs_violations(predicted_trajs, map):
    obs_map = map.data

    interp_obs_map = RectBivariateSpline(range(obs_map.shape[1]),
                                         range(obs_map.shape[0]),
                                         binary_dilation(obs_map.T, iterations=4),
                                         kx=1, ky=1)

    old_shape = predicted_trajs.shape
    pred_trajs_map = map.to_map_points(predicted_trajs.reshape((-1, 2)))

    traj_obs_values = interp_obs_map(pred_trajs_map[:, 0], pred_trajs_map[:, 1], grid=False)
    traj_obs_values = traj_obs_values.reshape((old_shape[0], old_shape[1]))
    num_viol_trajs = np.sum(traj_obs_values.max(axis=1) > 0, dtype=float)

    return num_viol_trajs


def compute_batch_statistics(prediction_output_dict,
                             dt,
                             max_hl,
                             ph,
                             node_type_enum,
                             kde=False,
                             obs=False,
                             map=None,
                             prune_ph_to_future=False,
                             best_of=True,
                             return_head=False):

    (prediction_dict,
     _,
     futures_dict) = prediction_output_to_trajectories(prediction_output_dict,
                                                       dt,
                                                       max_hl,
                                                       ph,
                                                       prune_ph_to_future=prune_ph_to_future)

    batch_error_dict = dict()
    for node_type in node_type_enum:
        batch_error_dict[node_type] =  {'ade': list(), 'fde': list(), 'kde': list(), 'obs_viols': list()}

    minade_idx_dict=dict()
    minfde_idx_dict=dict()
    for node_type in node_type_enum:
        minade_idx_dict[node_type]=list()
        minfde_idx_dict[node_type] = list()

    full_ade_dict=dict()
    full_fde_dict=dict()
    for node_type in node_type_enum:
        full_ade_dict[node_type]=list()
        full_fde_dict[node_type] = list()

    for t in prediction_dict.keys():
        for node in prediction_dict[t].keys():
            ade_errors = compute_ade(prediction_dict[t][node], futures_dict[t][node])
            fde_errors = compute_fde(prediction_dict[t][node], futures_dict[t][node])
            if kde:
                kde_ll = compute_kde_nll(prediction_dict[t][node], futures_dict[t][node])
            else:
                kde_ll = 0
            if obs:
                obs_viols = compute_obs_violations(prediction_dict[t][node], map)
            else:
                obs_viols = 0
            if best_of:
                if return_head:
                    minade_idx = np.argmin(ade_errors)
                    minfde_idx = np.argmin(fde_errors)
                    full_ade_errors=ade_errors
                    full_fde_errors=fde_errors
                ade_errors = np.min(ade_errors, keepdims=True)
                fde_errors = np.min(fde_errors, keepdims=True)

                kde_ll = np.min(kde_ll)
            batch_error_dict[node.type]['ade'].extend(list(ade_errors))
            batch_error_dict[node.type]['fde'].extend(list(np.array(fde_errors)))
            batch_error_dict[node.type]['kde'].extend([kde_ll])
            batch_error_dict[node.type]['obs_viols'].extend([obs_viols])
            if return_head:
              minade_idx_dict[node.type].extend([minade_idx])
              minfde_idx_dict[node.type].extend([minfde_idx])
              full_ade_dict[node.type].extend([full_ade_errors])
              full_fde_dict[node.type].extend([full_fde_errors])


    if return_head:
         return batch_error_dict, minade_idx_dict,minfde_idx_dict,full_ade_dict,full_fde_dict

    return batch_error_dict


def compute_batch_statistics_nll(prediction_output_dict,
                             dt,
                             max_hl,
                             ph,
                             node_type_enum,
                             kde=False,
                             obs=False,
                             map=None,
                             prune_ph_to_future=False,
                             best_of=True,
                             return_head=False):

    (prediction_dict,
     _,
     futures_dict) = prediction_output_to_trajectories_nll(prediction_output_dict,
                                                       dt,
                                                       max_hl,
                                                       ph,
                                                       prune_ph_to_future=prune_ph_to_future)

    batch_error_dict = dict()
    for node_type in node_type_enum:
        batch_error_dict[node_type] =  {'ade': list(), 'fde': list(), 'kde': list(), 'obs_viols': list(),'nll':list()}

    minade_idx_dict=dict()
    minfde_idx_dict=dict()
    for node_type in node_type_enum:
        minade_idx_dict[node_type]=list()
        minfde_idx_dict[node_type] = list()

    full_ade_dict=dict()
    full_fde_dict=dict()
    for node_type in node_type_enum:
        full_ade_dict[node_type]=list()
        full_fde_dict[node_type] = list()

    for t in prediction_dict.keys():
        for node in prediction_dict[t].keys():
            ade_errors = compute_ade(prediction_dict[t][node][0], futures_dict[t][node])
            fde_errors = compute_fde(prediction_dict[t][node][0], futures_dict[t][node])
            nll = compute_nll(prediction_dict[t][node], futures_dict[t][node])
            if best_of:
                ade_errors = np.min(ade_errors, keepdims=True)
                fde_errors = np.min(fde_errors, keepdims=True)

            batch_error_dict[node.type]['nll'].extend([nll])
            batch_error_dict[node.type]['ade'].extend(list(ade_errors))
            batch_error_dict[node.type]['fde'].extend(list(np.array(fde_errors)))

    return batch_error_dict


def log_batch_errors(batch_errors_list, log_writer, namespace, curr_iter, bar_plot=[], box_plot=[]):
    for node_type in batch_errors_list[0].keys():
        for metric in batch_errors_list[0][node_type].keys():
            metric_batch_error = []
            for batch_errors in batch_errors_list:
                metric_batch_error.extend(batch_errors[node_type][metric])

            if len(metric_batch_error) > 0:
                # log_writer.add_histogram(f"{node_type.name}/{namespace}/{metric}", metric_batch_error, curr_iter)
                log_writer.add_scalar(f"{node_type.name}/{namespace}/{metric}_mean", np.mean(metric_batch_error), curr_iter)
                log_writer.add_scalar(f"{node_type.name}/{namespace}/{metric}_median", np.median(metric_batch_error), curr_iter)

                if metric in bar_plot:
                    pd = {'dataset': [namespace] * len(metric_batch_error),
                                  metric: metric_batch_error}
                    kde_barplot_fig, ax = plt.subplots(figsize=(5, 5))
                    visualization.visualization_utils.plot_barplots(ax, pd, 'dataset', metric)
                    log_writer.add_figure(f"{node_type.name}/{namespace}/{metric}_bar_plot", kde_barplot_fig, curr_iter)

                if metric in box_plot:
                    mse_fde_pd = {'dataset': [namespace] * len(metric_batch_error),
                                  metric: metric_batch_error}
                    fig, ax = plt.subplots(figsize=(5, 5))
                    visualization.visualization_utils.plot_boxplots(ax, mse_fde_pd, 'dataset', metric)
                    log_writer.add_figure(f"{node_type.name}/{namespace}/{metric}_box_plot", fig, curr_iter)


def print_batch_errors(batch_errors_list, namespace, curr_iter):
    for node_type in batch_errors_list[0].keys():
        for metric in batch_errors_list[0][node_type].keys():
            metric_batch_error = []
            for batch_errors in batch_errors_list:
                metric_batch_error.extend(batch_errors[node_type][metric])

            if len(metric_batch_error) > 0:
                print(f"{curr_iter}: {node_type.name}/{namespace}/{metric}_mean", np.mean(metric_batch_error))
                print(f"{curr_iter}: {node_type.name}/{namespace}/{metric}_median", np.median(metric_batch_error))
