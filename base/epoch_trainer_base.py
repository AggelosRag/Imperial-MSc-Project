import os
from io import StringIO

import numpy as np
import pydotplus
from matplotlib import pyplot as plt
from IPython.core.display import Image
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz


class EpochTrainerBase:
    def __init__(self):
        pass

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _valid_epoch(self, epoch):
        raise NotImplementedError

    def _calculate_APL(self, min_samples_leaf1, C_pred, outputs):

        min_samples_leaf = min_samples_leaf1
        tree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
        # C_pred = self.model.mn_model.concept_predictor(
        #     torch.tensor(input, dtype=torch.float32)).detach()
        # # C_pred = C_batch
        # outputs = self.model.mn_model.label_predictor(C_pred).detach().numpy()
        C_pred = C_pred.detach().numpy()
        outputs = outputs.detach().numpy()

        if outputs.shape[1] == 1:
            preds = np.where(outputs > 0.5, 1, 0).reshape(-1)
        elif outputs.shape[1] >= 3:
            preds = np.argmax(outputs, axis=1)
        else:
            raise ValueError('Invalid number of output classes')

        tree.fit(C_pred, preds)
        # tree = reduced_error_prune(tree, C_val, y_val)
        # tree_train = reduced_error_prune(tree_train, C_pred, y_train)
        y_pred = tree.predict(C_pred)
        fid = accuracy_score(preds, y_pred)
        #print(f'Fidelity {mode}: {fid}')

        # path_length = 0
        # for c in C_pred:
        #     path_length += tree_train.decision_path([c]).toarray().sum()
        # print(f'Train: Average path length: {path_length / len(C_pred)}')

        # Get the total number of nodes
        total_nodes = tree.tree_.node_count
        #print(f'Total number of nodes {mode}: {total_nodes}')

        # preds = np.eye(3)[preds]
        # p_l = get_path_length(torch.tensor(C_pred, dtype=torch.float32),
        #                       torch.tensor(preds, dtype=torch.long),
        #                       min_samples_leaf=min_samples_leaf)
        # print(f'Train: Average path length2: {p_l}')

        APL = tree.tree_.node_count
        # APL = path_length / len(C_pred)

        return APL, fid, list(tree.feature_importances_), tree

    def _visualize_tree(self, tree, config, epoch, APL, train_acc, val_acc):

        # export tree
        plt.figure(figsize=(20, 20))
        # plot_tree(model, filled=True)
        dot_data = StringIO()
        export_graphviz(
            decision_tree=tree,
            out_file=dot_data,
            filled=True,
            rounded=True,
            special_characters=True,
            feature_names=config['dataset']['feature_names'],
            class_names=config['dataset']['class_names'],
        )

        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

        fig_path = str(self.config.log_dir) + '/trees'
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        # graph.write_png(fig_path + f'/tree_{epoch}_nodes_'
        #                 f'{APL}_train_acc_{train_acc:.4f}_'
        #                 f'test_acc_{val_acc:.4f}.png')
        graph.write_png(fig_path + f'/tree_{epoch}_nodes_'
                        f'{APL}.png')
        Image(graph.create_png())
