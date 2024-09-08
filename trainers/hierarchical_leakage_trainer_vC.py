import os
import copy
import pickle as pkl

import numpy as np
import torch.utils.data
from matplotlib import pyplot as plt
import graphviz
from sklearn.metrics import accuracy_score

from epoch_trainers.xc_epoch_trainer import XC_Epoch_Trainer
from networks.custom_dt_gini_with_entropy_metrics import \
    build_combined_tree, CustomDecisionTree
from utils.tree_utils import get_light_colors, get_leaf_samples_and_features, \
    get_decision_path_features_and_thresholds


class HierarchicalLeakageTrainervC:

    def __init__(self, arch, config, device, data_loader, valid_data_loader,
                 reg=None, expert=None):

        self.arch = arch
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.expert = expert

        if self.expert is not None:
            self.path = f'expert_{self.expert}'
        else:
            self.path = ''
        self.save_dir = os.path.join(self.config.save_dir, self.path)
        self.log_dir = os.path.join(self.config.log_dir, self.path)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.xc_epochs = config['trainer']['xc_epochs']
        self.cy_epochs = config['trainer']['cy_epochs']
        self.num_concepts = config['dataset']['num_concepts']
        self.concept_names = config['dataset']['concept_names']
        self.reg = reg

        # define the x->c model
        self.xc_epoch_trainer = XC_Epoch_Trainer(
            self.arch, self.config,
            self.device, self.data_loader,
            self.valid_data_loader)

    def train(self):

        save_dicts = self.config["trainer"]["save_train_tensors"]
        logger = self.config.get_logger('train')
        if self.expert == 1 and "pretrained_concept_predictor" not in self.config["model"]:
            # train the x->c model
            print("\nTraining x->c")
            logger.info("Training x->c")
            self.xc_epoch_trainer._training_loop(self.xc_epochs)
            self.plot_xc()

        # create a new dataloader for the c->y model
        all_C, all_y, all_C_val, all_y_val = self.extract_cy_data()
        all_C = all_C.detach().cpu().numpy()
        all_y = all_y.detach().cpu().numpy()
        all_C_val = all_C_val.detach().cpu().numpy()
        all_y_val = all_y_val.detach().cpu().numpy()

        # train the c->y model
        print("\nTraining hard c->y DT label predictor")
        logger.info("\nTraining hard c->y DT label predictor")
        self.arch.label_predictor.fit(all_C, all_y)

        y_pred = self.arch.label_predictor.predict(all_C)
        print(f'Training Accuracy: {accuracy_score(all_y, y_pred)}')
        y_pred = self.arch.label_predictor.predict(all_C_val)
        print(f'Validation Accuracy: {accuracy_score(all_y_val, y_pred)}')

        original_tree = copy.deepcopy(self.arch.label_predictor)
        self.hard_tree = original_tree
        self._visualize_DT_label_predictor(self.arch.label_predictor, X=all_C, path='/original_tree')

        # Extract samples per leaf and features used in the decision paths
        leaf_samples_indices, leaf_features_per_path = get_leaf_samples_and_features(self.arch.label_predictor, all_C)
        thresholds_per_path = get_decision_path_features_and_thresholds(self.arch.label_predictor, all_C)
        leaf_samples_indices_val, leaf_features_per_path_val = get_leaf_samples_and_features(self.arch.label_predictor, all_C_val)

        # evaluate x->c
        train_C_pred, train_y = self.xc_epoch_trainer._predict(data_loader=self.data_loader, use_data_loader=True)
        val_C_pred, val_y = self.xc_epoch_trainer._predict(data_loader=self.valid_data_loader, use_data_loader=True)
        train_C_pred = train_C_pred.detach().cpu().numpy()
        train_y = train_y.detach().cpu().numpy()
        val_C_pred = val_C_pred.detach().cpu().numpy()
        val_y = val_y.detach().cpu().numpy()

        # print("\nTraining soft c->y DT label predictor")
        # logger.info("\nTraining soft c->y DT label predictor")
        # tree = CustomDecisionTree(min_samples_leaf=self.config["regularisation"]["min_samples_leaf"],
        #                           n_classes=self.config["dataset"]["num_classes"])
        # tree.fit(train_C_pred, train_y)
        #
        # y_pred = tree.predict(train_C_pred)
        # print(f'Training Accuracy: {accuracy_score(train_y, y_pred)}')
        # y_pred = tree.predict(val_C_pred)
        # print(f'Validation Accuracy: {accuracy_score(val_y, y_pred)}')
        #
        # self._visualize_DT_label_predictor(tree, X=train_C_pred, path='/soft_tree')

        # Execute the algorithm using the concept probs from the sequential CBM
        trees_seq = self._fit_subtrees(path_name='sequential_CBM', save_dicts=save_dicts, logger=logger,
                           leaf_samples_indices=leaf_samples_indices,
                           leaf_features_per_path=leaf_features_per_path,
                           leaf_samples_indices_val=leaf_samples_indices_val,
                           leaf_features_per_path_val=leaf_features_per_path_val,
                           all_C=all_C, all_y=all_y, all_C_val=all_C_val, all_y_val=all_y_val,
                           train_C_pred=train_C_pred, train_y=train_y, val_C_pred=val_C_pred, val_y=val_y,
                           original_tree=original_tree, thresholds_per_path=thresholds_per_path)

        # Assign the joint concept predictor to the model
        self.xc_epoch_trainer.model.concept_predictor = self.arch.concept_predictor_joint

        # Execute the algorithm using the concept probs from the joint CBM
        trees_joint = self._fit_subtrees(path_name='joint_CBM', save_dicts=save_dicts, logger=logger,
                           leaf_samples_indices=leaf_samples_indices,
                           leaf_features_per_path=leaf_features_per_path,
                           leaf_samples_indices_val=leaf_samples_indices_val,
                           leaf_features_per_path_val=leaf_features_per_path_val,
                           all_C=all_C, all_y=all_y, all_C_val=all_C_val, all_y_val=all_y_val,
                           train_C_pred=train_C_pred, train_y=train_y, val_C_pred=val_C_pred, val_y=val_y,
                           original_tree=original_tree, thresholds_per_path=thresholds_per_path)

        # Merge the two leave dictionaries: for each leaf, compare the two trees:
        # first check the number of nodes, if one of the trees have more than one, keep that tree
        # if both have one node, or both have more than one node, use the tree from the joint CBM

        self.leaf_trees = {}
        self.leaf_cbm_used = {}
        for leaf in leaf_samples_indices.keys():
            if trees_seq[leaf].node_count > 1 and trees_joint[leaf].node_count == 1:
                self.leaf_trees[leaf] = trees_seq[leaf]
                self.leaf_cbm_used[leaf] = 'seq'
            elif trees_seq[leaf].node_count == 1 and trees_joint[leaf].node_count > 1:
                self.leaf_trees[leaf] = trees_joint[leaf]
                self.leaf_cbm_used[leaf] = 'joint'
            else:
                self.leaf_trees[leaf] = trees_joint[leaf]
                self.leaf_cbm_used[leaf] = 'joint'

        # merge into a final tree
        self.combined_tree = build_combined_tree(original_tree, self.leaf_trees, all_C, all_y)

        y_pred = self.combined_tree.predict(train_C_pred)
        print(f'\n Final Tree: Training Accuracy of the combined tree: {accuracy_score(train_y, y_pred)}')
        y_pred = self.combined_tree.predict(val_C_pred)
        print(f' Final Tree: Validation Accuracy of the combined tree: {accuracy_score(val_y, y_pred)}')

        self._visualize_DT_label_predictor(self.combined_tree, X=all_C,
                                           path='/final_tree/combined_tree',
                                           hard_tree=True)



    def test(self, test_data_loader, hard_cbm=False):

        logger = self.config.get_logger('train')

        if self.arch.concept_predictor_joint is not None:
            self.xc_epoch_trainer.model.concept_predictor = self.arch.concept_predictor

        save_dicts = self.config["trainer"]["save_test_tensors"]
        # evaluate x->c
        tensor_C_binary_pred, tensor_y = self.xc_epoch_trainer._test(test_data_loader, hard_cbm=True)
        leaf_samples_indices, leaf_features_per_path = get_leaf_samples_and_features(self.arch.label_predictor, tensor_C_binary_pred)

        tensor_C_binary_pred = tensor_C_binary_pred.detach().cpu().numpy()
        tensor_y = tensor_y.detach().cpu().numpy()
        y_pred = self.arch.label_predictor.predict(tensor_C_binary_pred)
        print(f'\nTest Accuracy using the Hard CBM: {accuracy_score(tensor_y, y_pred)}')
        logger.info(f'\nTest Accuracy using the Hard CBM: {accuracy_score(tensor_y, y_pred)}')

        if save_dicts:
            new_leaves_per_leaf_samples_indices = {}
            new_leaves_per_leaf_features_per_path = {}
            C_leaf_pred_dict = {}
            C_leaf_dict = {}
            y_leaf_dict = {}
            y_original_pred_dict = {}
            y_pred_dict = {}
            X_leaf_dict = {}

        accuracy_per_original_path_dict = {}
        accuracy_per_new_path_dict = {}
        y_pred_all = []
        y_all = []
        for leaf, sample_indices in leaf_samples_indices.items():
            C_leaf, y_leaf = tensor_C_binary_pred[sample_indices], tensor_y[sample_indices]
            X_leaf = self.extract_x_subset_data(test_data_loader, sample_indices)
            y_original_pred = self.arch.label_predictor.predict(C_leaf)
            cbm_used = self.leaf_cbm_used[leaf]
            if cbm_used == 'seq':
                self.xc_epoch_trainer.model.concept_predictor = self.arch.concept_predictor
            else:
                self.xc_epoch_trainer.model.concept_predictor = self.arch.concept_predictor_joint
            C_leaf_pred = self.xc_epoch_trainer._predict(X=X_leaf, use_data_loader=False)
            leaf_features_not_used = list(set(range(self.num_concepts)) - set(leaf_features_per_path[leaf]))
            hard_concepts = C_leaf[:, leaf_features_not_used]
            C_leaf_pred[:, leaf_features_not_used] = hard_concepts
            tree = self.leaf_trees[leaf]
            y_pred = tree.predict(C_leaf_pred)
            y_pred_all.extend(y_pred)
            y_all.extend(y_leaf.tolist())
            print(f"\nTesting for leaf: {leaf}")
            print(f'Test Accuracy of the original path: {accuracy_score(y_leaf, y_original_pred)}')
            print(f'Test Accuracy of the new path: {accuracy_score(y_leaf, y_pred)}')
            logger.info(f'Test Accuracy of the original path: {accuracy_score(y_leaf, y_original_pred)}')
            logger.info(f'Test Accuracy of the new path: {accuracy_score(y_leaf, y_pred)}')
            accuracy_per_original_path_dict[leaf] = accuracy_score(y_leaf, y_original_pred)
            accuracy_per_new_path_dict[leaf] = accuracy_score(y_leaf, y_pred)

            if save_dicts:
                new_leaves_per_leaf_samples_indices[leaf], new_leaves_per_leaf_features_per_path[leaf] =\
                    get_leaf_samples_and_features(tree, C_leaf_pred)
                C_leaf_pred_dict[leaf] = C_leaf_pred
                C_leaf_dict[leaf] = C_leaf
                y_leaf_dict[leaf] = y_leaf
                y_original_pred_dict[leaf] = y_original_pred
                y_pred_dict[leaf] = y_pred
                X_leaf_dict[leaf] = X_leaf

        if save_dicts:
            # save C_leaf_pred and y_leaf per leaf
            output_path = os.path.join(self.save_dir, "test_dicts")
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            with open(os.path.join(output_path, 'leaf_samples_indices.pkl'), 'wb') as f:
                pkl.dump(leaf_samples_indices, f)
            with open(os.path.join(output_path, 'leaf_features_per_path.pkl'), 'wb') as f:
                pkl.dump(leaf_features_per_path, f)
            with open(os.path.join(output_path, 'new_leaves_per_leaf_samples_indices.pkl'), 'wb') as f:
                pkl.dump(new_leaves_per_leaf_samples_indices, f)
            with open(os.path.join(output_path, 'new_leaves_per_leaf_features_per_path.pkl'), 'wb') as f:
                pkl.dump(new_leaves_per_leaf_features_per_path, f)
            with open(os.path.join(output_path, 'C_leaf_pred_dict.pkl'), 'wb') as f:
                pkl.dump(C_leaf_pred_dict, f)
            with open(os.path.join(output_path, 'C_leaf_dict.pkl'), 'wb') as f:
                pkl.dump(C_leaf_dict, f)
            with open(os.path.join(output_path, 'y_leaf_dict.pkl'), 'wb') as f:
                pkl.dump(y_leaf_dict, f)
            with open(os.path.join(output_path, 'y_original_pred_dict.pkl'), 'wb') as f:
                pkl.dump(y_original_pred_dict, f)
            with open(os.path.join(output_path, 'y_pred_dict.pkl'), 'wb') as f:
                pkl.dump(y_pred_dict, f)
            with open(os.path.join(output_path, 'X_leaf_dict.pkl'), 'wb') as f:
                pkl.dump(X_leaf_dict, f)

        with open(os.path.join(self.save_dir, 'leaf_cbm_used.pkl'), 'wb') as f:
                pkl.dump(self.leaf_cbm_used, f)
        with open(os.path.join(self.save_dir, 'accuracy_per_original_path_dict.pkl'), 'wb') as f:
            pkl.dump(accuracy_per_original_path_dict, f)
        with open(os.path.join(self.save_dir, 'accuracy_per_new_path_dict.pkl'), 'wb') as f:
            pkl.dump(accuracy_per_new_path_dict, f)
        with open(os.path.join(self.save_dir, 'leaf_samples_indices.pkl'), 'wb') as f:
            pkl.dump(leaf_samples_indices, f)

        print(f'\nTest Accuracy of the combined tree: {accuracy_score(y_all, y_pred_all)}')
        logger.info(f'\nTest Accuracy of the combined tree: {accuracy_score(y_all, y_pred_all)}')
        return y_all, y_pred_all

    def plot_xc(self):
        results_trainer = self.xc_epoch_trainer.metrics_tracker.result()
        train_bce_losses = results_trainer['train_loss_per_concept']
        val_bce_losses = results_trainer['val_loss_per_concept']
        train_total_bce_losses = results_trainer['train_concept_loss']
        val_total_bce_losses = results_trainer['val_concept_loss']

        # Plotting the results
        epochs = range(1, self.xc_epochs + 1)
        epochs_less = range(11, self.xc_epochs + 1)
        plt.figure(figsize=(18, 30))

        plt.subplot(5, 2, 1)
        for i in range(self.config["dataset"]['num_concepts']):
            plt.plot(epochs_less,
                     [train_bce_losses[j][i] for j in range(10, self.xc_epochs)],
                     label=f'Train BCE Loss {self.concept_names[i]}')
        plt.title('Training BCE Loss per Concept')
        plt.xlabel('Epochs')
        plt.ylabel('BCE Loss')
        plt.legend(loc='upper left')

        plt.subplot(5, 2, 2)
        for i in range(self.config["dataset"]['num_concepts']):
            plt.plot(epochs_less,
                     [val_bce_losses[j][i] for j in range(10, self.xc_epochs)],
                     label=f'Val BCE Loss {self.concept_names[i]}')
        plt.title('Validation BCE Loss per Concept')
        plt.xlabel('Epochs')
        plt.ylabel('BCE Loss')
        plt.legend(loc='upper left')

        plt.subplot(5, 2, 3)
        plt.plot(epochs_less, train_total_bce_losses[10:], 'b',
                 label='Total Train BCE loss')
        plt.plot(epochs_less, val_total_bce_losses[10:], 'r',
                 label='Total Val BCE loss')
        plt.title('Total BCE Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Total BCE Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig(str(self.log_dir) + '/xc_plots.png')
        #plt.show()

    def plot_cy(self):
        results_trainer = self.cy_epoch_trainer.metrics_tracker.result()
        train_target_losses = results_trainer['train_target_loss']
        val_target_losses = results_trainer['val_target_loss']
        train_accuracies = results_trainer['train_accuracy']
        val_accuracies = results_trainer['val_accuracy']
        APLs_train = results_trainer['train_APL']
        APLs_test = results_trainer['val_APL']
        if self.reg == 'Tree':
            APL_predictions_train = results_trainer['train_APL_predictions']
            APL_predictions_test = results_trainer['val_APL_predictions']
        fidelities_train = results_trainer['train_fidelity']
        fidelities_test = results_trainer['val_fidelity']
        FI_train = results_trainer['train_feature_importance']
        FI_test = results_trainer['val_feature_importance']

        # Plotting the results
        epochs = range(1, self.cy_epochs + 1)
        plt.figure(figsize=(18, 30))

        plt.subplot(3, 2, 1)
        plt.plot(epochs, train_target_losses, 'b', label='Training Target loss')
        plt.plot(epochs, val_target_losses, 'r', label='Validation Target loss')
        plt.title('Training and Validation Target Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Target Loss')
        plt.legend()

        plt.subplot(3, 2, 2)
        plt.plot(epochs, train_accuracies, 'b', label='Training accuracy')
        plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(3, 2, 3)
        plt.plot(epochs, APLs_train, 'b', label='Train')
        plt.plot(epochs, APLs_test, 'r', label='Test')
        if self.reg == 'Tree':
            plt.plot(epochs, APL_predictions_train, 'g',
                     label='Train Predictions')
            plt.plot(epochs, APL_predictions_test, 'y',
                     label='Test Predictions')
        plt.title('APL')
        plt.xlabel('Epochs')
        plt.ylabel('APL')
        plt.legend()

        plt.subplot(3, 2, 4)
        plt.plot(epochs, fidelities_train, 'b', label='Train')
        plt.plot(epochs, fidelities_test, 'r', label='Test')
        plt.title('Fidelity')
        plt.xlabel('Epochs')
        plt.ylabel('Fidelity')
        plt.legend()

        plt.subplot(3, 2, 5)
        for i in range(self.config["dataset"]['num_concepts']):
            plt.plot(epochs, [fi[i] for fi in FI_train],
                     label=f'{self.concept_names[i]}')
        plt.title('Feature Importances (Train)')
        plt.xlabel('Epochs')
        plt.ylabel('Concept Importances')
        plt.legend(loc='upper left')

        plt.subplot(3, 2, 6)
        for i in range(self.config["dataset"]['num_concepts']):
            plt.plot(epochs, [fi[i] for fi in FI_test],
                     label=f'{self.concept_names[i]}')
        plt.title('Feature Importances (Test)')
        plt.xlabel('Epochs')
        plt.ylabel('Concept Importances')
        plt.legend(loc='upper left')

        plt.tight_layout()
        if self.expert is not None:
            plt.savefig(str(self.log_dir) + '/cy_plots_expert_' + str(self.expert) + '.png')
        else:
            plt.savefig(str(self.log_dir) + '/cy_plots.png')
        #plt.show()

    def extract_cy_data(self):

        if isinstance(self.data_loader.dataset, torch.utils.data.TensorDataset):
            all_C = self.data_loader.dataset[:][1]
            all_y = self.data_loader.dataset[:][2]
            all_C_val = self.valid_data_loader.dataset[:][1]
            all_y_val = self.valid_data_loader.dataset[:][2]
        else:
            all_C, all_y = self.data_loader.dataset.get_all_data_in_tensors(
                batch_size=self.config["data_loader"]["args"]["batch_size"], shuffle=True,
                return_loader=False
            )
            all_C_val, all_y_val = self.valid_data_loader.dataset.get_all_data_in_tensors(
                batch_size=self.config["data_loader"]["args"]["batch_size"], shuffle=False,
                return_loader=False
            )

        return all_C, all_y, all_C_val, all_y_val

    def extract_x_subset_data(self, dataloader, subset_indices):

        if isinstance(dataloader.dataset, torch.utils.data.TensorDataset):
            X = dataloader.dataset[:][0][subset_indices]
        else:
            X = []
            for i, sample in enumerate(dataloader.dataset.data):
                if i in subset_indices:
                    X.append(dataloader.dataset.__getitem__(i)[0])
            X = torch.stack(X)
        return X

    def _fit_subtrees(self, path_name=None, save_dicts=False, logger=None,
                      leaf_samples_indices=None, leaf_features_per_path=None,
                      leaf_samples_indices_val=None, leaf_features_per_path_val=None,
                      all_C=None, all_y=None, all_C_val=None, all_y_val=None,
                      train_C_pred=None, train_y=None, val_C_pred=None, val_y=None,
                      original_tree=None, thresholds_per_path=None):

        leaf_trees = {}

        if save_dicts:
            new_leaves_per_leaf_samples_indices = {}
            new_leaves_per_leaf_features_per_path = {}
            C_leaf_pred_dict = {}
            C_leaf_dict = {}
            y_leaf_dict = {}
            y_original_pred_dict = {}
            y_pred_dict = {}
            X_leaf_dict = {}

        for leaf, sample_indices in leaf_samples_indices.items():
            C_leaf, y_leaf = all_C[sample_indices], all_y[sample_indices]
            X_leaf = self.extract_x_subset_data(self.data_loader,sample_indices)
            y_original_pred = original_tree.predict(C_leaf)
            C_leaf_pred = self.xc_epoch_trainer._predict(X=X_leaf, use_data_loader=False)
            leaf_features_not_used = list(set(range(self.num_concepts)) - set(leaf_features_per_path[leaf]))
            hard_concepts = C_leaf[:, leaf_features_not_used]
            C_leaf_pred[:, leaf_features_not_used] = hard_concepts
            for feature_idx, feature_presence in thresholds_per_path[leaf].items():
                if feature_presence == 1:
                    C_leaf_pred[C_leaf_pred[:, feature_idx] < 0.5, feature_idx] = 0.5
                else:
                    C_leaf_pred[C_leaf_pred[:, feature_idx] >= 0.5, feature_idx] = 0.5

            print(f"\n {path_name}: Training soft c->y DT label predictor for leaf: {leaf}")
            logger.info(f"\n {path_name}: Training soft c->y DT label predictor for leaf: {leaf}")
            tree = CustomDecisionTree(min_samples_leaf=self.config["regularisation"]["min_samples_leaf"],
                                      n_classes=self.config["dataset"]["num_classes"]
            )
            tree.fit(C_leaf_pred, y_leaf)
            y_pred = tree.predict(C_leaf_pred)
            print(f' {path_name}: Training Accuracy of the original path: {accuracy_score(y_leaf, y_original_pred)}')
            print(f' {path_name}: Training Accuracy of the new path: {accuracy_score(y_leaf, y_pred)}')
            logger.info(f' {path_name}: Training Accuracy of the original path: {accuracy_score(y_leaf, y_original_pred)}')
            logger.info(f' {path_name}: Training Accuracy of the new path: {accuracy_score(y_leaf, y_pred)}')

            # prune(self.arch.label_predictor.tree_)
            leaf_trees[leaf] = tree
            self._visualize_DT_label_predictor(tree, path=f'/{path_name}/leaf_{leaf}')

            if save_dicts:
                new_leaves_per_leaf_samples_indices[leaf], \
                new_leaves_per_leaf_features_per_path[leaf] = \
                    get_leaf_samples_and_features(tree, C_leaf_pred)
                C_leaf_pred_dict[leaf] = C_leaf_pred
                C_leaf_dict[leaf] = C_leaf
                y_leaf_dict[leaf] = y_leaf
                y_original_pred_dict[leaf] = y_original_pred
                y_pred_dict[leaf] = y_pred
                X_leaf_dict[leaf] = X_leaf

        if save_dicts:
            # save C_leaf_pred and y_leaf per leaf
            output_path = os.path.join(self.save_dir, f"{path_name}/train_dicts")
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            with open(os.path.join(output_path, 'leaf_samples_indices.pkl'),'wb') as f:
                pkl.dump(leaf_samples_indices, f)
            with open(os.path.join(output_path, 'leaf_features_per_path.pkl'),'wb') as f:
                pkl.dump(leaf_features_per_path, f)
            with open(os.path.join(output_path, 'new_leaves_per_leaf_samples_indices.pkl'),'wb') as f:
                pkl.dump(new_leaves_per_leaf_samples_indices, f)
            with open(os.path.join(output_path, 'new_leaves_per_leaf_features_per_path.pkl'),'wb') as f:
                pkl.dump(new_leaves_per_leaf_features_per_path, f)
            with open(os.path.join(output_path, 'C_leaf_pred_dict.pkl'), 'wb') as f:
                pkl.dump(C_leaf_pred_dict, f)
            with open(os.path.join(output_path, 'C_leaf_dict.pkl'), 'wb') as f:
                pkl.dump(C_leaf_dict, f)
            with open(os.path.join(output_path, 'y_leaf_dict.pkl'), 'wb') as f:
                pkl.dump(y_leaf_dict, f)
            with open(os.path.join(output_path, 'y_original_pred_dict.pkl'),'wb') as f:
                pkl.dump(y_original_pred_dict, f)
            with open(os.path.join(output_path, 'y_pred_dict.pkl'), 'wb') as f:
                pkl.dump(y_pred_dict, f)
            with open(os.path.join(output_path, 'X_leaf_dict.pkl'), 'wb') as f:
                pkl.dump(X_leaf_dict, f)

        for leaf, sample_indices in leaf_samples_indices_val.items():
            C_leaf_val, y_leaf_val = all_C_val[sample_indices], all_y_val[sample_indices]
            X_leaf_val = self.extract_x_subset_data(self.valid_data_loader, sample_indices)
            y_original_pred = original_tree.predict(C_leaf_val)
            C_leaf_val_pred = self.xc_epoch_trainer._predict(X=X_leaf_val,
                                                             use_data_loader=False)
            leaf_features_not_used_val = list(
                set(range(self.num_concepts)) - set(
                    leaf_features_per_path_val[leaf]))
            hard_concepts = C_leaf_val[:, leaf_features_not_used_val]
            C_leaf_val_pred[:, leaf_features_not_used_val] = hard_concepts
            tree = leaf_trees[leaf]
            y_pred = tree.predict(C_leaf_val_pred)
            print(f"\n {path_name}: Validating for leaf: {leaf}")
            logger.info(f"\n {path_name}: Validating for leaf: {leaf}")
            print(f' {path_name}: Validation Accuracy of the original path: {accuracy_score(y_leaf_val, y_original_pred)}')
            print(f' {path_name}: Validation Accuracy of the new path: {accuracy_score(y_leaf_val, y_pred)}')
            logger.info(f' {path_name}: Validation Accuracy of the original path: {accuracy_score(y_leaf_val, y_original_pred)}')
            logger.info(f' {path_name}: Validation Accuracy of the new path: {accuracy_score(y_leaf_val, y_pred)}')

        combined_tree = build_combined_tree(original_tree, leaf_trees,
                                                 all_C, all_y)

        y_pred = combined_tree.predict(train_C_pred)
        print(f'\n {path_name}: Training Accuracy of the combined tree: {accuracy_score(train_y, y_pred)}')
        y_pred = combined_tree.predict(val_C_pred)
        print(f' {path_name}: Validation Accuracy of the combined tree: {accuracy_score(val_y, y_pred)}')

        self._visualize_DT_label_predictor(combined_tree, X=all_C,
                                           path=f'/{path_name}/combined_tree',
                                           hard_tree=True)

        return leaf_trees

    def _visualize_DT_label_predictor(self, tree, X=None, path=None, hard_tree=False):

        colors = get_light_colors(len(self.config['dataset']['class_names']))
        colors_dict = {i: colors[i] for i in range(len(self.config['dataset']['class_names']))}

        APL = tree.node_count

        # dot_data = export_graphviz(
        #     decision_tree=tree,
        #     out_file=None,
        #     filled=False,
        #     rounded=True,
        #     special_characters=True,
        #     feature_names=self.config['dataset']['concept_names'],
        #     class_names=self.config['dataset']['class_names'],
        # )

        fig_path = str(self.log_dir) + '/trees'
        if path is not None:
            fig_path = fig_path + path
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        # # Modify the dot data to include the specific colors
        # dot_data_with_colors = replace_splits(dot_data)
        # dot_data_with_colors = modify_dot_with_colors(
        #     dot_data_with_colors, colors_dict, tree.tree_, node_color=node_color
        # )

        dot_data = tree.export_tree(feature_names = self.config['dataset']['concept_names'],
                                    class_names = self.config['dataset']['class_names'],
                                    class_colors = colors)

        # Render the graph
        graph = graphviz.Source(dot_data, directory=fig_path)
        name = f'dt_label_predictor_nodes_{APL}'
        graph.render(name, format="pdf", cleanup=True)

        if X is not None:
            if hard_tree:
                tree_used_for_paths = self.hard_tree
            else:
                tree_used_for_paths = tree
            leaf_indices = tree_used_for_paths.apply(X)

            for leaf in np.unique(leaf_indices):
                sample_indices = np.where(leaf_indices == leaf)[0]
                decision_paths = tree_used_for_paths.decision_path(X[sample_indices])
                tree.export_decision_paths_with_subtree(decision_paths,
                                           feature_names = self.config['dataset']['concept_names'],
                                           class_colors=colors,
                                           class_names = self.config['dataset']['class_names'],
                                           output_dir = fig_path + '/decision_paths',
                                           leaf_id = leaf)
                or_leaf_node_ids = self.hard_tree.get_leaf_node_ids()
                decision_paths = tree.decision_path(X[sample_indices])
                tree.export_decision_paths_with_subtree_simplified(decision_paths,
                                           original_node_ids=or_leaf_node_ids,
                                           feature_names = self.config['dataset']['concept_names'],
                                           feature_groups = self.config['dataset']['concept_groups'],
                                           class_colors=colors,
                                           class_names = self.config['dataset']['class_names'],
                                           output_dir = fig_path + '/decision_paths_simplified',
                                           leaf_id = leaf)


