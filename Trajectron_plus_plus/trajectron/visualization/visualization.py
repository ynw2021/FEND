
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns




def prediction_output_to_trajectories(prediction_output_dict,
                                      dt,
                                      max_h,
                                      ph,
                                      map=None,
                                      prune_ph_to_future=False):

    prediction_timesteps = prediction_output_dict.keys()

    output_dict = dict()
    histories_dict = dict()
    futures_dict = dict()

    for t in prediction_timesteps:
        histories_dict[t] = dict()
        output_dict[t] = dict()
        futures_dict[t] = dict()
        prediction_nodes = prediction_output_dict[t].keys()
        for node in prediction_nodes:
            predictions_output = prediction_output_dict[t][node]
            position_state = {'position': ['x', 'y']}

            history = node.get(np.array([t - max_h, t]), position_state)  # History includes current pos
            history = history[~np.isnan(history.sum(axis=1))]

            future = node.get(np.array([t + 1, t + ph]), position_state)
            future = future[~np.isnan(future.sum(axis=1))]

            if prune_ph_to_future:
                predictions_output = predictions_output[:, :, :future.shape[0]]
                if predictions_output.shape[2] == 0:
                    continue

            trajectory = predictions_output

            if map is None:
                histories_dict[t][node] = history
                output_dict[t][node] = trajectory
                futures_dict[t][node] = future
            else:
                histories_dict[t][node] = map.to_map_points(history)
                output_dict[t][node] = map.to_map_points(trajectory)
                futures_dict[t][node] = map.to_map_points(future)

    return output_dict, histories_dict, futures_dict




def plot_trajectories(ax,
                      prediction_dict,
                      histories_dict,
                      futures_dict,
                      color_key='b',
                      line_alpha=0.7,
                      line_width=0.2,
                      edge_width=2,
                      circle_edge_width=0.5,
                      node_circle_size=0.3,
                      batch_num=0,
                      kde=False,
                      bold=False):

    cmap = ['k', 'b', 'y', 'g', 'r']

    for node in histories_dict:
        history = histories_dict[node]
        future = futures_dict[node]
        predictions = prediction_dict[node]

        if np.isnan(history[-1]).any():
            continue

        ax.plot(history[:, 0], history[:, 1], 'k--')

        for sample_num in range(prediction_dict[node].shape[1]):

            if kde and predictions.shape[1] >= 50:
                line_alpha = 0.2
                for t in range(predictions.shape[2]):
                    sns.kdeplot(predictions[batch_num, :, t, 0], predictions[batch_num, :, t, 1],
                                ax=ax, shade=True, shade_lowest=False,
                                color=np.random.choice(cmap), alpha=0.8)

            if bold:
                ax.plot(predictions[batch_num, sample_num, :, 0], predictions[batch_num, sample_num, :, 1],
                        # color=cmap[node.type.value],
                        color=color_key,
                        linewidth=line_width * 10, alpha=line_alpha)
            else:
                ax.plot(predictions[batch_num, sample_num, :, 0], predictions[batch_num, sample_num, :, 1],
                        # color=cmap[node.type.value],
                        color=color_key,
                        linewidth=line_width * 5, alpha=line_alpha)

            ax.plot(future[:, 0],
                    future[:, 1],
                    'c--',
                    path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()])

            # Current Node Position
            circle = plt.Circle((history[-1, 0],
                                 history[-1, 1]),
                                node_circle_size,
                                facecolor='r',
                                edgecolor='k',
                                lw=circle_edge_width,
                                zorder=3)
            ax.add_artist(circle)

    ax.axis('equal')


def plot_trajectories_no_prediction(ax,
                      histories_dict,
                      futures_dict,
                      line_alpha=0.7,
                      line_width=0.2,
                      edge_width=2,
                      circle_edge_width=0.5,
                      node_circle_size=0.3,
                      batch_num=0,
                      kde=False):

    cmap = ['k', 'b', 'y', 'g', 'r']

    for node in histories_dict:
        history = histories_dict[node]
        future = futures_dict[node]

        if np.isnan(history[-1]).any():
            continue

        ax.plot(history[:, 0], history[:, 1], 'k--')

        ax.plot(future[:, 0],
                future[:, 1],
                'w--',
                path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()])

        # Current Node Position
        circle = plt.Circle((history[-1, 0],
                             history[-1, 1]),
                            node_circle_size,
                            facecolor='g',
                            edgecolor='k',
                            lw=circle_edge_width,
                            zorder=3)
        ax.add_artist(circle)




    ax.axis('equal')


def visualize_prediction(ax,
                         prediction_output_dict,
                         dt,
                         max_hl,
                         ph,
                         robot_node=None,
                         map=None,
                         **kwargs):

    prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(prediction_output_dict,
                                                                                      dt,
                                                                                      max_hl,
                                                                                      ph,
                                                                                      map=map)

    assert(len(prediction_dict.keys()) <= 1)
    if len(prediction_dict.keys()) == 0:
        return
    ts_key = list(prediction_dict.keys())[0]

    prediction_dict = prediction_dict[ts_key]
    histories_dict = histories_dict[ts_key]
    futures_dict = futures_dict[ts_key]

    if map is not None:
        ax.imshow(map.as_image(), origin='lower', alpha=0.5)
    plot_trajectories(ax, prediction_dict, histories_dict, futures_dict, *kwargs)


def visualize_best_prediction(ax,
                         prediction_output_dict,
                         prediction_contrast_dict,
                         # prediction_F_dict,
                         # prediction_H_dict,
                         # prediction_PCL_dict,
                         prediction_ewta_dict,
                         other_predictions,
                         dt,
                         max_hl,
                         ph,
                         robot_node=None,
                         map=None,
                         **kwargs):

    prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(prediction_output_dict,
                                                                                      dt,
                                                                                      max_hl,
                                                                                      ph,
                                                                                      map=map)
    contrast_dict, _,_ = prediction_output_to_trajectories(prediction_contrast_dict,
                                                                                      dt,
                                                                                      max_hl,
                                                                                      ph,
                                                                                      map=map)

    # F_dict, _,_ = prediction_output_to_trajectories(prediction_F_dict,
    #                                                                                   dt,
    #                                                                                   max_hl,
    #                                                                                   ph,
    #                                                                                   map=map)
    #
    #
    # H_dict, _,_ = prediction_output_to_trajectories(prediction_H_dict,
    #                                                                                   dt,
    #                                                                                   max_hl,
    #                                                                                   ph,
    #                                                                                   map=map)
    #
    # PCL_dict, _,_ = prediction_output_to_trajectories(prediction_PCL_dict,
    #                                                                                   dt,
    #                                                                                   max_hl,
    #                                                                                   ph,
    #                                                                                   map=map)

    ewta_dict, _,_ = prediction_output_to_trajectories(prediction_ewta_dict,
                                                                                      dt,
                                                                                      max_hl,
                                                                                      ph,
                                                                                      map=map)
    _, other_historys,other_futures = prediction_output_to_trajectories(other_predictions,
                                                                                      dt,
                                                                                      max_hl,
                                                                                      ph,
                                                                                      map=map)



    assert(len(prediction_dict.keys()) <= 1)
    if len(prediction_dict.keys()) == 0:
        return
    ts_key = list(prediction_dict.keys())[0]

    prediction_dict = prediction_dict[ts_key]
    histories_dict = histories_dict[ts_key]
    futures_dict = futures_dict[ts_key]
    contrast_dict = contrast_dict[ts_key]
    ewta_dict = ewta_dict[ts_key]
    # F_dict = F_dict[ts_key]
    # H_dict = H_dict[ts_key]
    # PCL_dict = PCL_dict[ts_key]
    other_historys=other_historys[ts_key]
    other_futures = other_futures[ts_key]



    if map is not None:
        ax.imshow(map.as_image(), origin='lower', alpha=0.5)

    plot_trajectories_no_prediction(ax, other_historys, other_futures, *kwargs)
    plot_trajectories(ax, prediction_dict, histories_dict, futures_dict, *kwargs)
    plot_trajectories(ax, contrast_dict, histories_dict, futures_dict,color_key='m', *kwargs)
    # plot_trajectories(ax, F_dict, histories_dict, futures_dict, color_key='m',*kwargs)
    # plot_trajectories(ax, H_dict, histories_dict, futures_dict, color_key='c', *kwargs)
    # plot_trajectories(ax, PCL_dict, histories_dict, futures_dict, color_key='k', *kwargs)
    plot_trajectories(ax, ewta_dict, histories_dict, futures_dict,color_key='r', *kwargs)




def visualize_distribution(ax,
                           prediction_distribution_dict,
                           map=None,
                           pi_threshold=0.05,
                           **kwargs):
    if map is not None:
        ax.imshow(map.as_image(), origin='lower', alpha=0.5)

    for node, pred_dist in prediction_distribution_dict.items():
        if pred_dist.mus.shape[:2] != (1, 1):
            return

        means = pred_dist.mus.squeeze().cpu().numpy()
        covs = pred_dist.get_covariance_matrix().squeeze().cpu().numpy()
        pis = pred_dist.pis_cat_dist.probs.squeeze().cpu().numpy()

        for timestep in range(means.shape[0]):
            for z_val in range(means.shape[1]):
                mean = means[timestep, z_val]
                covar = covs[timestep, z_val]
                pi = pis[timestep, z_val]

                if pi < pi_threshold:
                    continue

                v, w = linalg.eigh(covar)
                v = 2. * np.sqrt(2.) * np.sqrt(v)
                u = w[0] / linalg.norm(w[0])

                # Plot an ellipse to show the Gaussian component
                angle = np.arctan(u[1] / u[0])
                angle = 180. * angle / np.pi  # convert to degrees
                ell = patches.Ellipse(mean, v[0], v[1], 180. + angle, color='blue' if node.type.name == 'VEHICLE' else 'orange')
                ell.set_edgecolor(None)
                ell.set_clip_box(ax.bbox)
                ell.set_alpha(pi/10)
                ax.add_artist(ell)
