import torch.utils.data
from matplotlib import pyplot as plt

from epoch_trainers.xc_epoch_trainer import XC_Epoch_Trainer
from epoch_trainers.cy_epoch_trainer import CY_Epoch_Trainer
from epoch_trainers.cy_epoch_tree_trainer import CY_Epoch_Tree_Trainer


class IndependentCBMTrainer:

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

        # define the x->c model
        self.xc_epoch_trainer = XC_Epoch_Trainer(
            self.arch, self.config,
            self.device, self.data_loader,
            self.valid_data_loader)

        # create a new dataloader for the c->y model
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
        # define the c->y model
        self.reg = reg
        if reg == 'Tree':
            self.cy_epoch_trainer = CY_Epoch_Tree_Trainer(
                self.arch, self.config,
                self.device, train_data_loader,
                val_data_loader)
        else:
            self.cy_epoch_trainer = CY_Epoch_Trainer(
                self.arch, self.config,
                self.device, train_data_loader,
                val_data_loader, expert=self.expert)

    def train(self):

        logger = self.config.get_logger('train')
        if self.expert == 1 or "pretrained_concept_predictor" not in self.config["model"]:
            # train the x->c model
            print("\nTraining x->c")
            logger.info("Training x->c")
            self.xc_epoch_trainer._training_loop(self.xc_epochs)
            self.plot_xc()

        # train the c->y model
        print("\nTraining c->y")
        logger.info("Training c->y")
        self.cy_epoch_trainer._training_loop(self.cy_epochs)
        self.plot_cy()

    def test(self):

        logger = self.config.get_logger('test')
        # get x->c predictions
        predictions = self.xc_epoch_trainer._test_epoch()
        # get c->y predictions
        self.cy_epoch_trainer._test_epoch(predictions)

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


    def test(self):
        pass