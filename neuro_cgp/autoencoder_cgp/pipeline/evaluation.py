import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, roc_curve, auc, confusion_matrix
import tensorflow as tf
import matplotlib
import autoencoder_cgp.pipeline.preprocessing

# font_size = 12
# matplotlib.rcParams.update({
# #     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,#
# #     'pgf.rcfonts': False,
#     'axes.labelsize': font_size, # fontsize for x and y labels (was 10)
#     'axes.titlesize': font_size,
#     'font.size': font_size, # was 10
#     'legend.fontsize': font_size, # was 10
#     'xtick.labelsize': font_size,
#     'ytick.labelsize': font_size,
#     # 'figure.figsize' : (10,5)
# })

import os
import matplotlib.ticker as mticker


def top_k_mean(recon_loss_segment):
    k = 3
    return recon_loss_segment.sort_values(ascending=False).values[0:k].mean()


DEFAULT_AGG_FUNC = top_k_mean


def mean_recon_loss(X_true, X_pred):
    X_true_flat = X_true.reshape(X_true.shape[0], -1)
    X_pred_flat = X_pred.reshape(X_true.shape[0], -1)

    return mean_squared_error(X_true_flat, X_pred_flat)


def _map_prediction_to_recon_loss(X_true, X_pred, df_label_segment):
    '''
    Calculates the recon_loss per segment based on the prediction and information in the given dataframe: df_label_segment

    :param df_label_segment: Expects a dataframe containing three columns: segment, machine_id, label.
    '''

    X_true_flat = X_true.reshape(X_true.shape[0], -1)
    X_pred_flat = X_pred.reshape(X_true.shape[0], -1)

    reconstruction_loss = np.mean(np.power(X_true_flat - X_pred_flat, 2), axis=1)

    assert np.mean(reconstruction_loss) - mean_squared_error(X_true_flat,
                                                             X_pred_flat) < 0.001, "Reconstruction loss calculation is wrong"

    df_eval = df_label_segment.copy()
    df_eval["recon_loss"] = reconstruction_loss

    return df_eval


def _calculate_roc_curve(X_val, X_val_pred, df_val, agg_func=DEFAULT_AGG_FUNC):
    df_eval, y_val = _get_df_eval(X_val, X_val_pred, df_val, agg_func)
    agg_recon_loss = _aggregate_recon_loss(df_eval, agg_func)
    false_pos_rate, true_pos_rate, thresholds = roc_curve(y_val, agg_recon_loss)
    return false_pos_rate, true_pos_rate, thresholds


def _aggregate_recon_loss(df_eval, agg_func=DEFAULT_AGG_FUNC):
    aggregation = df_eval.groupby("segment")["recon_loss"].agg([agg_func])
    assert len(aggregation.columns) == 1
    return aggregation.iloc[:, 0]


def _get_df_eval(X_val, X_val_pred, df_val, agg_func=DEFAULT_AGG_FUNC):
    y_val = df_val.groupby("segment")["anomaly"].agg([agg_func])
    assert len(y_val.columns) == 1
    df_eval = _map_prediction_to_recon_loss(X_val, X_val_pred, df_val)
    return df_eval, y_val.iloc[:, 0]


def plot_roc_auc(X_val, X_val_pred, df_val, path=None, agg_func=DEFAULT_AGG_FUNC, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    false_pos_rate, true_pos_rate, thresholds = _calculate_roc_curve(X_val, X_val_pred, df_val, agg_func)
    _plot_roc_auc_custom_rates(ax, false_pos_rate, true_pos_rate, path)
    plt.close()


def _plot_roc_auc_custom_rates(ax, false_pos_rate, true_pos_rate, path=None):
    roc_auc = auc(false_pos_rate, true_pos_rate, )
    ax.plot(false_pos_rate, true_pos_rate, linewidth=2, label='AUC = %0.3f' % roc_auc)
    ax.plot([0, 1], [0, 1], linewidth=2)
    ax.set_xlim([-0.01, 1])
    ax.set_ylim([0, 1.01])
    ax.legend(loc='lower right')
    ax.set_title('Receiver operating characteristic curve (ROC)')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    if path is not None:
        plt.savefig(path)
    if ax is None:
        plt.show()


def roc_auc(X_val, X_val_pred, df_val, agg_func=DEFAULT_AGG_FUNC):
    false_pos_rate, true_pos_rate, thresholds = _calculate_roc_curve(X_val, X_val_pred, df_val, agg_func)
    return auc(false_pos_rate, true_pos_rate, )


def plot_generations_vs_fitness(history, path=None, fitness_metric="Fitness"):
    plt.scatter(history["generation"], history["fitness"])
    plt.xlabel("Generation")
    plt.ylabel(fitness_metric)
    num_generations = len(set(history["generation"]))
    step_size = int(num_generations / 10)
    step_size = 1 if step_size == 0 else step_size
    plt.xticks(np.arange(0, num_generations, step_size))
    if path is not None:
        plt.savefig(path)
    plt.show()
    plt.close()


def plot_generations_vs_best_fitness(history, path=None, fitness_metric="Fitness"):
    generations = np.arange(len(history["best_fitness"]))
    plt.plot(generations, history["best_fitness"])
    plt.xlabel("Generation")
    plt.ylabel(fitness_metric)
    num_generations = len(generations)
    step_size = int(num_generations / 10)
    step_size = 1 if step_size == 0 else step_size
    plt.xticks(np.arange(0, num_generations, step_size))
    if path is not None:
        plt.savefig(path)
    plt.show()
    plt.close()


def optimal_treshold_youden(false_pos_rate, true_pos_rate, thresholds):
    """
    @param false_pos_rate:
    @param true_pos_rate:
    @param thresholds:
    @return: optimal treshold and optimal_roc_point = (fpr, tpr)
    """
    J = true_pos_rate - false_pos_rate
    optimal_index = np.argmax(J)
    return thresholds[optimal_index], (false_pos_rate[optimal_index], true_pos_rate[optimal_index])


def save_best_architecture_graph(model, path):
    tf.keras.utils.plot_model(model, to_file=path, show_shapes=True, show_layer_names=False)


def plot_trainings_curve(history, path=None):
    """

    @param history: keras model fit history
    @param path:
    @return:
    """
    plt.plot(history.history["loss"][1:], label="train")
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    if path is not None:
        plt.savefig(path)
    plt.show()
    plt.close()


def rename_binary_labels(y_val, y_val_pred):
    y_val_pred = y_val_pred.map({True: "Anomalie", False: "Normal"})
    y_val = y_val.map({True: "Anomalie", False: "Normal"})
    return y_val, y_val_pred


def plot_simple_confusion_matrix(y_val, y_val_pred, ):
    y_val, y_val_pred = rename_binary_labels(y_val, y_val_pred)

    conf_matrix = confusion_matrix(y_val, y_val_pred, y_val.unique())
    unique_labels = y_val.unique()

    result = pd.DataFrame(conf_matrix,
                          index=['true:{:}'.format(x) for x in unique_labels],
                          columns=['pred:{:}'.format(x) for x in unique_labels])
    print(result)
    return result


def plot_confusion_matrix(y_val, y_val_pred,
                          normalize=True,
                          show_absolute_values_additionally=True,
                          title=None,
                          cmap=plt.cm.Blues,
                          ax=None):
    """
    This function prints and plots the confusion matrix.
    Adopted from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """

    if show_absolute_values_additionally:
        assert normalize is True, "If you only want to see the absolute values, set normalize=False" \
                                  " and show_absolute_values_additionally=False."
    # Compute confusion matrix
    y_val, y_val_pred = rename_binary_labels(y_val, y_val_pred)
    assert set(y_val.unique()) == set(y_val_pred.unique())
    classes = y_val.unique()

    cm = confusion_matrix(y_val, y_val_pred, y_val.unique())

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    if ax is None:
        fig, ax = plt.subplots()
    # plt.grid(b=None)
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Wahre Klasse',
           xlabel='Vorhergesagte Klasse')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="center",  # rotation=45,
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if show_absolute_values_additionally:
                ax.text(j, i, f"{format(cm_norm[i, j].copy(), '.2f')} ({format(cm[i, j].copy(), 'd')})",
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
            else:
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    if ax is None:
        fig.tight_layout()
    return ax

def plot_multiple_images_side_by_side(data, indices, title="", show_recon_loss=False):
    n = len(indices)

    fig, ax_array = plt.subplots(1, n, figsize=(n * 4, 4), sharey=True)

    if title is not None:
        fig.suptitle(title, fontsize=16)
    for i in range(n):
        im = ax_array[i].imshow(data.loc[indices[i], "signal_spectrum"], origin="lower")
        if show_recon_loss:
            recon_loss = data.loc[indices[i], "recon_loss"]
            recon_loss = np.round(recon_loss, 4)
            ax_array[i].set_title(f"Fehler: {recon_loss}", fontsize=18)
        ax_array[i].set_xticks([])
        ax_array[i].set_yticks([])


def plot_confusion_with_optimal_treshold(df_test_dict, models_dict, mid, INPUT_X, INPUT_Y):
    df_val = df_test_dict[mid]
    model = models_dict[mid]

    X_val = autoencoder_cgp.pipeline.preprocessing.df_to_neural_network_format(df_val, INPUT_X, INPUT_Y)
    X_val_pred = model.predict(X_val)

    df_eval = _map_prediction_to_recon_loss(X_val, X_val_pred, df_val)

    df_eval, y_val = _get_df_eval(X_val, X_val_pred, df_val)
    agg_recon_loss = _aggregate_recon_loss(df_eval)

    false_pos_rate, true_pos_rate, thresholds = _calculate_roc_curve(X_val, X_val_pred, df_val)

    treshold, roc_point = optimal_treshold_youden(false_pos_rate, true_pos_rate, thresholds)

    y_val_pred = agg_recon_loss >=  treshold
    plot_confusion_matrix(y_val, y_val_pred, title=None, normalize=True)
    plt.tight_layout()
    plt.show()