from matplotlib import pyplot as plt

from epoch_trainers.xcy_epoch_trainer import XCY_Epoch_Trainer
from epoch_trainers.xcy_epoch_tree_trainer import XCY_Tree_Epoch_Trainer


class JointCBMTrainer:

    def __init__(self, arch, config, device, data_loader, valid_data_loader,
                 reg=None, iteration=None):

        self.arch = arch
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.iteration = iteration

        self.epochs = config['trainer']['epochs']
        self.num_concepts = config['dataset']['num_concepts']
        self.concept_names = config['dataset']['concept_names']

        self.reg = reg
        if reg == 'Tree':
            self.epoch_trainer = XCY_Tree_Epoch_Trainer(
                self.arch, self.config,
                self.device, self.data_loader,
                'joint',
                self.valid_data_loader,
                self.iteration
            )
        else:
            self.epoch_trainer = XCY_Epoch_Trainer(self.arch, self.config,
                                                   self.device,
                                                   self.data_loader,
                                                   self.valid_data_loader,
                                                   self.iteration)

    def train(self):
        self.epoch_trainer._training_loop(self.epochs)
        self.plot()

    def test(self, test_data_loader):
        self.epoch_trainer._test(test_data_loader)

    def plot(self):

        results_trainer = self.epoch_trainer.metrics_tracker.result()

        train_bce_losses = results_trainer['train_loss_per_concept']
        val_bce_losses = results_trainer['val_loss_per_concept']
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
        train_losses = results_trainer['train_loss']
        val_losses = results_trainer['val_loss']
        train_total_bce_losses = results_trainer['train_concept_loss']
        val_total_bce_losses = results_trainer['val_concept_loss']

        # Plotting the results
        epochs = range(1, self.epochs + 1)
        epochs_less = range(11, self.epochs + 1)
        plt.figure(figsize=(18, 30))

        plt.subplot(5, 2, 1)
        for i in range(self.config["dataset"]['num_concepts']):
            plt.plot(epochs_less,
                     [train_bce_losses[j][i] for j in range(10, self.epochs)],
                     label=f'Train BCE Loss {self.concept_names[i]}')
        plt.title('Training BCE Loss per Concept')
        plt.xlabel('Epochs')
        plt.ylabel('BCE Loss')
        plt.legend(loc='upper left')

        plt.subplot(5, 2, 2)
        for i in range(self.config["dataset"]['num_concepts']):
            plt.plot(epochs_less,
                     [val_bce_losses[j][i] for j in range(10, self.epochs)],
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

        plt.subplot(5, 2, 4)
        plt.plot(epochs, train_target_losses, 'b', label='Training Target loss')
        plt.plot(epochs, val_target_losses, 'r', label='Validation Target loss')
        plt.title('Training and Validation Target Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Target Loss')
        plt.legend()

        plt.subplot(5, 2, 5)
        plt.plot(epochs_less, train_losses[10:], 'b',
                 label='Training Total loss')
        plt.plot(epochs_less, val_losses[10:], 'r',
                 label='Validation Total loss')
        plt.title('Training and Validation Total Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Total Loss')
        plt.legend()

        plt.subplot(5, 2, 6)
        plt.plot(epochs, train_accuracies, 'b', label='Training accuracy')
        plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(5, 2, 7)
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

        plt.subplot(5, 2, 8)
        plt.plot(epochs, fidelities_train, 'b', label='Train')
        plt.plot(epochs, fidelities_test, 'r', label='Test')
        plt.title('Fidelity')
        plt.xlabel('Epochs')
        plt.ylabel('Fidelity')
        plt.legend()

        plt.subplot(5, 2, 9)
        for i in range(self.config["dataset"]['num_concepts']):
            plt.plot(epochs, [fi[i] for fi in FI_train],
                     label=f'{self.concept_names[i]}')
        plt.title('Feature Importances (Train)')
        plt.xlabel('Epochs')
        plt.ylabel('Feature Importances')
        plt.legend(loc='upper left')

        plt.subplot(5, 2, 10)
        for i in range(self.config["dataset"]['num_concepts']):
            plt.plot(epochs, [fi[i] for fi in FI_test],
                     label=f'{self.concept_names[i]}')
        plt.title('Feature Importances (Test)')
        plt.xlabel('Epochs')
        plt.ylabel('Feature Importances')
        # put legend to the top left of the plot
        plt.legend(loc='upper left')

        plt.tight_layout()
        if self.iteration is not None:
            name = f'/joint_cbm_plots_expert_{self.iteration}.png'
        else:
            name = '/plots.png'
        plt.savefig(str(self.config.log_dir) + name)
        #plt.show()