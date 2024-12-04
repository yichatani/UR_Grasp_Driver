"""
    visualize affordances distribution
"""

import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.font_manager
matplotlib.rc('font',family='Times New Roman')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer, QuantileTransformer



if __name__ == '__main__':

    # load ground truth affordance of scenes, offline
    graspness = np.load('../policy/graspness_statistic.npy')
    suctioness = np.load('../policy/suctioness_statistic.npy')

    # load learned calibration model
    grasp_model = pickle.load(open('../policy/grasp_model.pkl', 'rb'))
    suction_model = pickle.load(open('../policy/suction_model.pkl', 'rb'))

    # transform new data based on learned model
    graspness_T = grasp_model.transform(graspness.reshape(-1, 1))
    graspness_T = suction_model.inverse_transform(graspness_T).reshape(-1)
    # # graspness percent 0 - 0.1 --- 0.9 - 1.0
    # grasp_percent_list, graspness_distribution = [], np.zeros([25600, 10])
    # for i in range(10):
    #     mask = (graspness_T > 0 + 0.1 * i) & (graspness_T <= 0.1 + 0.1 * i)
    #     count = np.count_nonzero(mask)
    #     grasp_percent = count / graspness_T.size
    #     grasp_percent_list.append(grasp_percent) # len: 10
    # graspness_distribution[i, :] = np.array(grasp_percent_list)
    # graspness_distribution = np.sum(graspness_distribution, axis=0).reshape(-1) # (10,)
    # graspness_distribution = list(graspness_distribution / np.sum(graspness_distribution))
    # print('graspness_distribution: ', graspness_distribution)
    # exit()

    # padding as the ValueError: All arrays must be of the same length
    max_len = max(len(graspness), len(suctioness))
    graspness = np.pad(graspness, (0, max_len - len(graspness)), 'constant')
    suctioness = np.pad(suctioness, (0, max_len - len(suctioness)), 'constant')

    max_len = max(len(graspness_T), len(suctioness))
    graspness_T = np.pad(graspness_T, (0, max_len - len(graspness_T)), 'constant')
    suctioness = np.pad(suctioness, (0, max_len - len(suctioness)), 'constant')


    # prepare data for plot
    df_stat_origin = pd.DataFrame({
            "graspness": graspness,
            "suctioness": suctioness
        })
    df_stat_trans = pd.DataFrame({
            "graspness": graspness_T,
            "suctioness": suctioness
        })

    # plot
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    fig_1 = sns.histplot(data=df_stat_origin, bins=9, binrange=[0.1, 1], stat='percent', \
                         common_norm=False, legend=False, ax=axs[0])
    fig_2 = sns.histplot(data=df_stat_trans, bins=9, binrange=[0.1, 1], stat='percent', \
                         common_norm=False, legend=False, ax=axs[1])

    for ax in axs:
        ax.set_xlabel('Value', fontsize=18)
        ax.set_ylabel('Percent', fontsize=18)
        ax.tick_params(axis='both', labelsize=16)
        ax.legend(['grasp', 'suction'], fontsize=16)
    axs[0].set_title('Original Distribution', fontsize=18)
    axs[1].set_title('Transformed Distribution', fontsize=18)
    # fig.suptitle('Density Histogram', fontsize=18)
    plt.show()








# import matplotlib as mpl
# from matplotlib import pyplot as plt
# from matplotlib import cm

# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import minmax_scale
# from sklearn.preprocessing import MaxAbsScaler
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import RobustScaler
# from sklearn.preprocessing import Normalizer
# from sklearn.preprocessing import QuantileTransformer
# from sklearn.preprocessing import PowerTransformer

# Graspness, Suctioness = load_data(cfgs.split)
# X = np.concatenate((Graspness.reshape(-1, 1), Suctioness.reshape(-1, 1)), axis=1)

# # y = minmax_scale(y_full)
# # cmap = getattr(cm, "plasma_r", cm.hot_r)
# distributions = [
#     ("Unscaled data", X),
#     ("Data after standard scaling", StandardScaler().fit_transform(X)),
#     ("Data after min-max scaling", MinMaxScaler().fit_transform(X)),
#     ("Data after max-abs scaling", MaxAbsScaler().fit_transform(X)),
#     (
#         "Data after robust scaling",
#         RobustScaler(quantile_range=(25, 75)).fit_transform(X),
#     ),
#     (
#         "",
#         PowerTransformer(method="yeo-johnson").fit_transform(X),
#     ),
#     # (
#     #     "Data after power transformation (Box-Cox)",
#     #     PowerTransformer(method="box-cox").fit_transform(X),
#     # ),
#     (
#         "Data after quantile transformation (uniform pdf)",
#         QuantileTransformer(output_distribution="uniform").fit_transform(X),
#     ),
#     (
#         "Data after quantile transformation (gaussian pdf)",
#         QuantileTransformer(output_distribution="normal").fit_transform(X),
#     ),
#     ("Data after sample-wise L2 normalizing", Normalizer().fit_transform(X)),
# ]


# def create_axes(title, figsize=(16, 5)):
#     fig = plt.figure(figsize=figsize)
#     fig.suptitle(title)

#     # define the axis for the first plot
#     left, width = 0.1, 0.22
#     bottom, height = 0.1, 0.7
#     bottom_h = height + 0.15
#     left_h = left + width + 0.02

#     rect_scatter = [left, bottom, width, height]
#     rect_histx = [left, bottom_h, width, 0.1]
#     rect_histy = [left_h, bottom, 0.05, height]

#     ax_scatter = plt.axes(rect_scatter)
#     ax_histx = plt.axes(rect_histx)
#     ax_histy = plt.axes(rect_histy)

#     # define the axis for the zoomed-in plot
#     left = width + left + 0.2
#     left_h = left + width + 0.02

#     rect_scatter = [left, bottom, width, height]
#     rect_histx = [left, bottom_h, width, 0.1]
#     rect_histy = [left_h, bottom, 0.05, height]

#     ax_scatter_zoom = plt.axes(rect_scatter)
#     ax_histx_zoom = plt.axes(rect_histx)
#     ax_histy_zoom = plt.axes(rect_histy)

#     # define the axis for the colorbar
#     left, width = width + left + 0.13, 0.01

#     return (
#         (ax_scatter, ax_histy, ax_histx),
#         (ax_scatter_zoom, ax_histy_zoom, ax_histx_zoom),
#     )


# def plot_distribution(axes, X, hist_nbins=10, title="", x0_label="", x1_label=""):
#     ax, hist_X1, hist_X0 = axes

#     ax.set_title(title)
#     ax.set_xlabel(x0_label)
#     ax.set_ylabel(x1_label)

#     # The scatter plot
#     ax.scatter(X[:, 0], X[:, 1], alpha=0.5, marker="o", s=5, lw=0, c='r')
#     # y=x base line
#     ax.plot([0,1], [0,1], alpha=0.5, c='b')

#     # axis limit
#     # ax.set_xlim([np.min(X[:, 0]), np.max(X[:, 0])])
#     # ax.set_ylim([np.min(X[:, 1]), np.max(X[:, 1])])
#     ax.set_xlim([0, 1])
#     ax.set_ylim([0, 1])

#     # Removing the top and the right spine for aesthetics
#     # make nice axis layout
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.get_xaxis().tick_bottom()
#     ax.get_yaxis().tick_left()
#     ax.spines["left"].set_position(("outward", 10))
#     ax.spines["bottom"].set_position(("outward", 10))

#     # Histogram for axis X1 (suctioness)
#     hist_X1.set_ylim(ax.get_ylim())
#     hist_X1.hist(
#         X[:, 1], bins=hist_nbins, orientation="horizontal", color="grey", ec="grey"
#     )
#     hist_X1.axis("off")

#     # Histogram for axis X0 (graspness)
#     hist_X0.set_xlim(ax.get_xlim())
#     hist_X0.hist(
#         X[:, 0], bins=hist_nbins, orientation="vertical", color="grey", ec="grey"
#     )
#     hist_X0.axis("off")


# def make_plot(item_idx):
#     title, X_T = distributions[item_idx]
#     X_T[:, 0] = (X_T[:, 0] - np.min(X_T[:, 0])) / (np.max(X_T[:, 0]) - np.min(X_T[:, 0]))
#     X_T[:, 1] = (X_T[:, 1] - np.min(X_T[:, 1])) / (np.max(X_T[:, 1]) - np.min(X_T[:, 1]))


#     ax_zoom_out, ax_zoom_in = create_axes(title)
#     axarr = (ax_zoom_out, ax_zoom_in)
#     plot_distribution(
#         axarr[0],
#         X,
#         hist_nbins=20,
#         x0_label='graspness',
#         x1_label='suctioness',
#         title="Origin",
#     )

#     plot_distribution(
#         axarr[1],
#         X_T,
#         hist_nbins=20,
#         x0_label='graspness',
#         x1_label='suctioness',
#         title="Transformed",
#     )

#     plt.show()