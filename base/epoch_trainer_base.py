import os

import numpy as np
import graphviz
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from base import TrainerBase
from experimentation.tree_comparison_utils import prune
from networks.custom_decision_treeB import CustomDecisionTree, tree_to_dict, \
    export_tree
from utils.tree_utils import get_light_colors, replace_splits, \
    modify_dot_with_colors


class EpochTrainerBase(TrainerBase):
    def __init__(self, arch, config, expert=None):

        super(EpochTrainerBase, self).__init__(arch, config, expert)
        self.or_class_names = config['dataset']['class_names']
        self.class_mapping = [i for i in range(len(self.or_class_names))]
        self.reduced_class_names = self.or_class_names
        self.config = config
        self.expert = expert
        # self.or_colors = list(mcolors.CSS4_COLORS.values())[:len(self.or_class_names)]
        self.or_colors = get_light_colors(len(self.or_class_names))

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _valid_epoch(self, epoch):
        raise NotImplementedError

    def _calculate_APL(self, min_samples_leaf, C_pred, outputs):

        tree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
        C_pred = C_pred.detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()

        if outputs.shape[1] == 1:
            preds = np.where(outputs > 0.5, 1, 0).reshape(-1)
        elif outputs.shape[1] >= 3:
            preds = np.argmax(outputs, axis=1)
        else:
            raise ValueError('Invalid number of output classes')

        self.reduced_class_names = [self.or_class_names[i] for i in self.class_mapping if i in preds]
        self.reduced_colors = [self.or_colors[i] for i in self.class_mapping if i in preds]
        self.reduced_colors_dict = {i: self.reduced_colors[i] for i in range(len(self.reduced_class_names))}

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

        # prune tree
        APL = tree.tree_.node_count
        prune(tree.tree_)
        # APL = path_length / len(C_pred)

        return APL, fid, list(tree.feature_importances_), tree

    def _visualize_tree(self, tree, config, epoch, APL, train_acc, val_acc,
                        mode, iteration=None):

        # export tree
        # plot_tree(model, filled=True)
        #dot_data = StringIO()
        dot_data = export_graphviz(
            decision_tree=tree,
            out_file=None,
            filled=False,
            rounded=True,
            special_characters=True,
            feature_names=config['dataset']['concept_names'],
            class_names=self.reduced_class_names,
        )

        # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

        fig_path = str(self.config.log_dir) + '/trees'
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        # graph.write_png(fig_path + f'/tree_{epoch}_nodes_'
        #                 f'{APL}_train_acc_{train_acc:.4f}_'
        #                 f'test_acc_{val_acc:.4f}.png')
        if iteration is not None:
            name = f'mode_{mode}_expert_{iteration}_nodes_{APL}'
        else:
            name = f'mode_{mode}_epoch_{epoch}_nodes_{APL}'

        # graph.write_png(fig_path + name)
        # Image(graph.create_png())

        # Modify the dot data to include the specific colors
        dot_data_with_colors = replace_splits(dot_data)
        dot_data_with_colors = modify_dot_with_colors(
            dot_data_with_colors, self.reduced_colors_dict, tree.tree_
        )
        # Render the graph
        graph = graphviz.Source(dot_data_with_colors, directory=fig_path)
        graph.render(name, format="pdf", cleanup=True)

    def _build_tree_with_fixed_roots(self, min_samples_leaf, C_pred, outputs,
                                     tree, mode, epoch, iteration=None):
        fixed_splits = tree_to_dict(tree)

        C_pred = C_pred.detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()

        if outputs.shape[1] == 1:
            preds = np.where(outputs > 0.5, 1, 0).reshape(-1)
        elif outputs.shape[1] >= 3:
            preds = np.argmax(outputs, axis=1)
        else:
            raise ValueError('Invalid number of output classes')

        self.reduced_class_names = [self.or_class_names[i] for i in
                                    self.class_mapping if i in preds]
        self.reduced_colors = [self.or_colors[i] for i in self.class_mapping if i in preds]
        self.reduced_colors_dict = {i: self.reduced_colors[i] for i in range(len(self.reduced_class_names))}

        custom_tree = CustomDecisionTree(fixed_splits,
                                         max_depth=100,
                                         min_samples_leaf=min_samples_leaf)
        custom_tree.fit(C_pred, preds)
        y_pred = custom_tree.predict(C_pred)
        fid = accuracy_score(preds, y_pred)
        APL = custom_tree.node_count
        #prune(custom_tree.tree_)

        #return APL, fid, list(custom_tree.feature_importances_), custom_tree

        if iteration is not None:
            name = f'mode_{mode}_expert_{iteration}_nodes_{APL}_fid_{fid:.2f}'
        else:
            name = f'mode_{mode}_epoch_{epoch}_nodes_{APL}_fid_{fid:.2f}'

        fig_path = str(self.config.log_dir) + '/trees'
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        dot_data = export_tree(custom_tree, self.config['dataset']['concept_names'],
                               self.or_colors, self.or_class_names)
        graph = graphviz.Source(dot_data, directory=fig_path)
        graph.render(name, format="pdf", cleanup=True)

    def _calculate_APL_gt(self, min_samples_leaf, C_pred, outputs):

        tree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
        C_pred = C_pred.detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()

        self.reduced_class_names = [self.or_class_names[i] for i in self.class_mapping if i in outputs]
        self.reduced_colors = [self.or_colors[i] for i in self.class_mapping if i in outputs]
        self.reduced_colors_dict = {i: self.reduced_colors[i] for i in range(len(self.reduced_class_names))}

        tree.fit(C_pred, outputs)
        # tree = reduced_error_prune(tree, C_val, y_val)
        # tree_train = reduced_error_prune(tree_train, C_pred, y_train)
        y_pred = tree.predict(C_pred)
        fid = accuracy_score(outputs, y_pred)
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

        # prune tree
        APL = tree.tree_.node_count
        #prune(tree.tree_)
        # APL = path_length / len(C_pred)

        return APL, fid, list(tree.feature_importances_), tree