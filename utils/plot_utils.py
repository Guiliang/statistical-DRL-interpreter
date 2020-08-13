import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_decision_boundary(input_data, target_data, tree_model, plot_step = 0.02, cm = plt.cm.get_cmap('RdYlBu_r')):
    x_min, x_max = input_data[:, 0].min()-0.01, input_data[:, 0].max()+0.05
    y_min, y_max = input_data[:, 1].min()-0.01, input_data[:, 1].max()+0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = tree_model.predict(np.c_[xx.ravel(), yy.ravel()])  # TODO: implement predict for mcts
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=cm, alpha=0.5)

    # unique_predictions = np.sort(np.unique(target_data))
    # n_classes = len(unique_predictions)
    # for i in range(n_classes):
    #     idx = np.where(target_data == unique_predictions[i])

    sc = plt.scatter(input_data[:, 0], input_data[:, 1], c=target_data, cmap=cm, marker='+')
    plt.colorbar(sc)
    plt.xlabel("Latent dimension 3")
    plt.ylabel("Latent dimension 5")
    plt.savefig('./decision_boundary.png')

def plot_values_by_node(x_values_method_all, y_values_method_all, plotting_target, game_name, methods):
    method_name_dict= {'cart-fvae': 'CART',
                       'vr-lmt-fvae': 'VR-LMT',
                       'gn-lmt-fave': 'GM-LMT',
                       'mcts': 'MCTR'
                       }
    method_markers_dict = {
        'cart-fvae': "o",
        'vr-lmt-fvae': "v",
        'gn-lmt-fave': "*",
        'mcts': "X"
    }

    plt.figure(figsize=(8,6))
    plt.xticks(size=15)
    plt.yticks(size=15)

    if plotting_target == 'Variance Reduction':
        if game_name == 'SpaceInvaders-v0':
            y_lim = [0, 0.012]
            plt.ylim(y_lim)
        elif game_name == 'flappybird':
            y_lim = [0, 0.08]
            plt.ylim(y_lim)
        location = 'upper left'
    elif plotting_target == 'MAE':
        if game_name == 'SpaceInvaders-v0':
            y_lim = [0.121, 0.1275]
            plt.ylim(y_lim)
        location = 'upper right'
    elif plotting_target == 'RMSE':
        location = 'upper right'
        if game_name == 'SpaceInvaders-v0':
            y_lim = [0.20, 0.23]
            plt.ylim(y_lim)

    for method_index in range(len(x_values_method_all)):
        plt.scatter(x_values_method_all[method_index], y_values_method_all[method_index],
                    label=method_name_dict[methods[method_index]],
                    marker=method_markers_dict[methods[method_index]],
                    s=120)
        plt.plot(x_values_method_all[method_index], y_values_method_all[method_index])
    plt.xlabel("The Number of Leaves", fontsize=18)
    plt.ylabel(plotting_target, fontsize=18)
    plt.legend(fontsize=15, loc=location)
    plt.grid(linestyle='dotted')
    plt.savefig('../results/plot_results/{0}_{1}_by_node.png'.format(plotting_target, game_name))


