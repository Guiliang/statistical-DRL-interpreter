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


