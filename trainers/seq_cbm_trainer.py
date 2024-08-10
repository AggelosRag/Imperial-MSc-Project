import torch.utils.data
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import graphviz
import numpy as np
import os

from epoch_trainers.xc_epoch_trainer import XC_Epoch_Trainer
from epoch_trainers.cy_epoch_trainer import CY_Epoch_Trainer
from epoch_trainers.cy_epoch_tree_trainer import CY_Epoch_Tree_Trainer
from utils.tree_utils import get_light_colors


class SequentialCBMTrainer:

    def __init__(self, arch, config, device, data_loader, valid_data_loader,
                 reg=None, expert=None):

        self.arch = arch
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.expert = expert

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

        # check if the label predictor is a tree or not
        if self.arch.model.label_predictor.__class__.__name__ in ['DecisionTreeClassifier', 'CustomDecisionTree']:
            self.tree_label_predictor = True

    def train(self):

        logger = self.config.get_logger('train')
        if self.expert == 1 or "pretrained_concept_predictor" not in self.config["model"]:
            # train the x->c model
            print("\nTraining x->c")
            logger.info("Training x->c")
            self.xc_epoch_trainer._training_loop(self.xc_epochs)
            self.plot_xc()

        # evaluate x->c
        train_C_pred, train_y = self.xc_epoch_trainer._test(self.data_loader, hard_cbm=False)
        val_C_pred, val_y = self.xc_epoch_trainer._test(self.valid_data_loader, hard_cbm=False)

        # train the c->y model
        print("\nTraining c->y")
        logger.info("Training c->y")
        if self.tree_label_predictor:
            train_C_pred = train_C_pred.detach().cpu().numpy()
            train_y = train_y.detach().cpu().numpy()
            val_C_pred = val_C_pred.detach().cpu().numpy()
            val_y = val_y.detach().cpu().numpy()

            # train the c->y model
            print("\nTraining hard c->y DT label predictor")
            logger.info("\nTraining hard c->y DT label predictor")
            self.arch.label_predictor.fit(train_C_pred, train_y)

            y_pred = self.arch.label_predictor.predict(train_C_pred)
            print(f'Training Accuracy: {accuracy_score(train_y, y_pred)}')
            y_pred = self.arch.label_predictor.predict(val_C_pred)
            print(f'Validation Accuracy: {accuracy_score(val_y, y_pred)}')
            self._visualize_DT_label_predictor(self.arch.label_predictor,
                                               X=train_C_pred, path='')
        else:
            # create a new dataloader for the c->y model
            train_data_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(train_C_pred, train_y),
                batch_size=self.config["data_loader"]["args"]["batch_size"], shuffle=True
            )
            val_data_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(val_C_pred, val_y),
                batch_size=self.config["data_loader"]["args"]["batch_size"], shuffle=False
            )

            # define the c->y model
            if self.reg == 'Tree':
                self.cy_epoch_trainer = CY_Epoch_Tree_Trainer(
                    self.arch, self.config,
                    self.device, train_data_loader,
                    val_data_loader)
            else:
                self.cy_epoch_trainer = CY_Epoch_Trainer(
                    self.arch, self.config,
                    self.device, train_data_loader,
                    val_data_loader, expert=self.expert)

            self.cy_epoch_trainer._training_loop(self.cy_epochs)
            self.plot_cy()

    def test(self, test_data_loader, hard_cbm=False):
        # evaluate x->c
        tensor_C_pred, tensor_y = self.xc_epoch_trainer._test(test_data_loader, hard_cbm)

        # evaluate c->y
        if self.tree_label_predictor:
            tensor_C_pred = tensor_C_pred.detach().cpu().numpy()
            tensor_y = tensor_y.detach().cpu().numpy()
            y_pred = self.arch.label_predictor.predict(tensor_C_pred)
            print(f'\nTest Accuracy using the Soft CBM: {accuracy_score(tensor_y, y_pred)}')
        else:
            # create a new dataloader for the c->y model
            test_data_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(tensor_C_pred, tensor_y),
                batch_size=self.config["data_loader"]["args"]["batch_size"], shuffle=False
            )
            self.cy_epoch_trainer._test(test_data_loader)

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
        plt.savefig(str(self.config.log_dir) + '/xc_plots.png')
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
            plt.savefig(str(self.config.log_dir) + '/cy_plots_expert_' + str(self.expert) + '.png')
        else:
            plt.savefig(str(self.config.log_dir) + '/cy_plots.png')
        #plt.show()

    def create_cy_dataloaders(self):

        if isinstance(self.data_loader.dataset, torch.utils.data.TensorDataset):
            all_C = self.data_loader.dataset[:][1]
            all_y = self.data_loader.dataset[:][2]
            all_C_val = self.valid_data_loader.dataset[:][1]
            all_y_val = self.valid_data_loader.dataset[:][2]
            train_dataset = torch.utils.data.TensorDataset(all_C, all_y)
            val_dataset = torch.utils.data.TensorDataset(all_C_val, all_y_val)
            train_data_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.config["data_loader"]["args"]["batch_size"], shuffle=True
            )
            val_data_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=self.config["data_loader"]["args"]["batch_size"], shuffle=False
            )
        else:
            train_data_loader = self.data_loader.dataset.get_all_data_in_tensors(
                batch_size=self.config["data_loader"]["args"]["batch_size"], shuffle=True
            )
            val_data_loader = self.valid_data_loader.dataset.get_all_data_in_tensors(
                batch_size=self.config["data_loader"]["args"]["batch_size"], shuffle=False
            )

        return train_data_loader, val_data_loader

    def _visualize_DT_label_predictor(self, tree, X=None, path=None, hard_tree=None):

        colors = get_light_colors(len(self.config['dataset']['class_names']))
        colors_dict = {i: colors[i] for i in range(len(self.config['dataset']['class_names']))}

        APL = tree.node_count

        fig_path = str(self.config.log_dir) + '/trees'
        if path is not None:
            fig_path = fig_path + path
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        dot_data = tree.export_tree(feature_names = self.config['dataset']['concept_names'],
                                    class_names = self.config['dataset']['class_names'],
                                    class_colors = colors)

        # Render the graph
        graph = graphviz.Source(dot_data, directory=fig_path)
        name = f'dt_label_predictor_nodes_{APL}'
        graph.render(name, format="pdf", cleanup=True)

        if X is not None:
            if hard_tree is not None:
                tree_used_for_paths = hard_tree
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