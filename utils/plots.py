from io import StringIO

import numpy as np
import pydotplus
import torch
from IPython.core.display import Image
from matplotlib import pyplot as plt
from sklearn.tree import export_graphviz


def pred_contours(x, y, model, device):
    """Given input data, compute the contours of the prediciont to draw the contour plots.

        Parameters
        ----------
        x: Input data as meshgrid

        y: Labels as meshgrid

        model: Trained deep model


        Returns
        -------
        y_pred : Model predictions as predictions contours
    """

    data = np.c_[x.ravel(), y.ravel()]
    y_pred = []

    for d in data:
        y_hat = model(torch.tensor(d, dtype=torch.float, device=device))
        # y_hat = model(torch.tensor(d, dtype=torch.float, device='cuda:0'))
        y_pred.append(y_hat.detach().cpu().numpy())

    y_pred = np.array(y_pred)
    if y_pred.shape[1] == 1:
        # apply sigmoid
        y_pred = torch.sigmoid(torch.tensor(y_pred, dtype=torch.float, device=device)).detach().cpu().numpy()
        y_pred = np.where(y_pred > 0.5, 1, 0)
    else:
        # apply softmax
        y_pred = np.argmax(y_pred, 1)

    return y_pred, data


def model_contour_plot(space, model, plot_title, fig_file_name, X=None, y=None,
                       device='None'):
    """
    Draw contour plot for deep model.

    Parameters
    -------

    space: Feature space

    model: Target deep model

    plot_title: Plot title

    fig_file_name: Data name for saving the figure

    X: Input features, default None

    y: Labels, default None
    """

    xx, yy = np.linspace(space[0][0], space[0][1], 100), np.linspace(space[1][0], space[1][1], 100)
    xx, yy = np.meshgrid(xx, yy)
    Z, _ = pred_contours(xx, yy, model, device)
    Z = Z.reshape(xx.shape)

    fig = plt.figure()
    CS = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
    #plt.colorbar()
    # plt.contour(xx, yy, Z, CS.levels, colors='k', linewidths=1.5)
    if X is not None:
        plt.scatter(*X.T, c=colormap(y), edgecolors='k')
    plt.xlim([space[0][0], space[0][1]])
    plt.ylim([space[1][0], space[1][1]])
    plt.title(plot_title)
    #fig.tight_layout()
    plt.savefig(fig_file_name)
    plt.close(fig)


def plot_tree(clf, space, path_contour, path_tree, epoch,
              class_names, fid, acc, APL, feature_names=None, contour_plot=True):
    """
    Plot tree and corresponding contour plot.
    """

    dot_data = StringIO()
    export_graphviz(
        decision_tree=clf,
        out_file=dot_data,
        filled=True,
        rounded=True,
        special_characters=True,
        feature_names=feature_names,
        class_names=class_names)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(f'{path_tree}.png')
    Image(graph.create_png())

    if contour_plot:
        xx, yy = np.meshgrid(np.linspace(space[0][0], space[0][1], 100),
                             np.linspace(space[0][0], space[0][1], 100))
        # plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        fig_contour = plt.figure()
        plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
        # plt.scatter(*X_test.T, c=colormap(y_test), edgecolors='k')
        plt.title(f'Epoch {epoch},'
                  f' Train Accuracy {acc},'
                  f' Train Fidelity {fid},'
                  f' APL {APL}')
        # plt.tight_layout()
        plt.savefig(f'{path_contour}_contourplot.png')
        plt.close(fig_contour)

def plot_data(X_train, y_train, x_decision_fun, y_decision_fun,
              space, writer, path):
    """
    Plot feature space with decision function and samples, and the error zone.
    """

    fig = plt.figure()
    plt.scatter(*X_train.T, c=colormap(y_train), edgecolors='k')
    plt.xlim([space[0][0], space[0][1]])
    plt.ylim([space[1][0], space[1][1]])
    plt.title('Training data')
    plt.plot(x_decision_fun, y_decision_fun, 'k-', linewidth=2.5)
    plt.plot(x_decision_fun, y_decision_fun - 0.2, linewidth=2, color='#808080')
    plt.plot(x_decision_fun, y_decision_fun + 0.2, linewidth=2, color='#808080')
    plt.savefig(f'{path}/samples_training_plot.png')
    writer.add_figure('Training samples', figure=fig)
    plt.close(fig)
    # data_summary = f'Training data shape: {X_train.shape}  \nValidation data shape: {X_val.shape}, \nTest data shape: {X_test.shape}'
    # writer.add_text('Training data Summary', data_summary)

def plot_data_and_decision_function(trainer, inputs, targets,
                                    grid_xlim, grid_ylim,
                                    tree_reg=1.0, save_path=None):

    preds_proba = trainer.predict(inputs)[0, :]
    preds = np.rint(preds_proba)
    xx, yy = np.meshgrid(np.arange(grid_xlim[0], grid_xlim[1], 0.01),
                         np.arange(grid_ylim[0], grid_ylim[1], 0.01))
    Z = np.rint(trainer.predict(np.c_[xx.ravel(), yy.ravel()].T))
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(5, 5))
    plt.plot(
        [
            inputs[0, i] for i in range(inputs.shape[1])
            if targets[0, i] == 1 and preds[i] == 1
        ],
        [
            inputs[1, i] for i in range(inputs.shape[1])
            if targets[0, i] == 1 and preds[i] == 1
        ],
        'o', color='red', label='true positives'
    )
    plt.plot(
        [
            inputs[0,i] for i in range(inputs.shape[1])
            if targets[0, i] == 0 and preds[i] == 0
        ],
        [
            inputs[1,i] for i in range(inputs.shape[1])
            if targets[0, i] == 0 and preds[i] == 0
        ],
        'o', color='orange', label='true negatives'
    )
    plt.plot(
        [
            inputs[0, i] for i in range(inputs.shape[1])
            if targets[0, i] == 0 and preds[i] == 1
        ],
        [
            inputs[1, i] for i in range(inputs.shape[1])
            if targets[0, i] == 0 and preds[i] == 1
        ],
        'o', color='blue', label='false positives'
    )
    plt.plot(
        [
            inputs[0, i] for i in range(inputs.shape[1])
            if targets[0, i] == 1 and preds[i] == 0
        ],
        [
            inputs[1, i] for i in range(inputs.shape[1])
            if targets[0, i] == 1 and preds[i] == 0
        ],
        'o', color='green', label='false negatives'
    )
    plt.ylim(grid_ylim)
    plt.xlim(grid_xlim)
    plt.contour(xx, yy, Z)
    plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    plt.show() if save_path is None else plt.savefig(save_path)

def plot(X, y, fun, error, space):
    """
    Plot feature space with decision function and samples, and the error zone.
    """

    x_lower = lambda x: fun(x) - error
    x_upper = lambda x: fun(x) + error

    x_decision_fun = np.linspace(space[0][0], space[0][1], 100)
    y_decision_fun = fun(x_decision_fun)

    fig = plt.figure()
    plt.scatter(*X.T, c=colormap(y), edgecolors='k', alpha=0.4)
    plt.xlim([space[0][0], space[0][1]])
    plt.ylim([space[1][0], space[1][1]])
    plt.title('Samples')
    plt.plot(x_decision_fun, y_decision_fun, 'k-', linewidth=2)
    plt.plot(x_decision_fun, x_lower(x_decision_fun), linewidth=2, color='#454545')
    plt.plot(x_decision_fun, x_upper(x_decision_fun), linewidth=2, color='#454545')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.show()
    plt.close(fig)

    fig = plt.figure()
    plt.xlim([space[0][0], space[0][1]])
    plt.ylim([space[1][0], space[1][1]])
    plt.title('Noise Zone')
    plt.plot(x_decision_fun, y_decision_fun, 'k-', linewidth=2)
    plt.plot(x_decision_fun, x_lower(x_decision_fun), linewidth=2, color='#454545')
    plt.plot(x_decision_fun, x_upper(x_decision_fun), linewidth=2, color='#454545')
    plt.fill_between(x_decision_fun, x_lower(x_decision_fun), x_upper(x_decision_fun))
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.show()
    plt.close(fig)


def colormap(Y):
    """Convert labels Y into a color-coding list. If y = 0, the color is 'r' (red), otherwise 'b' (blue)

        Parameters
        ----------
        Y: Labels

        Returns
        -------
        colormap: color-coding for Y
    """
    colormap = ['b' if y == 1 else 'r' for y in Y]

    return colormap
