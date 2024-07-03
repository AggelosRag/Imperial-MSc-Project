from matplotlib import pyplot as plt

from base.trainer_base import TrainerBase
from epoch_trainers.xy_epoch_trainer import XY_Epoch_Trainer
from epoch_trainers.xy_epoch_tree_trainer import XY_Tree_Epoch_Trainer
from logger import TensorboardWriter


class NNTrainer(TrainerBase):

    def __init__(self, arch, metric_ftns, config, device, data_loader, valid_data_loader, reg=None):

        super(NNTrainer, self).__init__(arch.model, config, arch.optimizer)

        self.arch = arch
        self.metric_ftns = metric_ftns
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader

        self.epochs = config['trainer']['epochs']
        self.num_features = config['dataset']['num_features']
        self.feature_names = config['dataset']['feature_names']

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir,
                                        self.logger,
                                        config['trainer']['tensorboard'])

        self.reg = reg
        if reg == 'Tree':
            self.epoch_trainer = XY_Tree_Epoch_Trainer(
                self.arch, self.epochs, self.writer, self.metric_ftns, self.config,
                self.device, self.data_loader, self.valid_data_loader
            )
        else:
            self.epoch_trainer = XY_Epoch_Trainer(
                self.arch, self.epochs, self.writer, self.metric_ftns, self.config,
                self.device, self.data_loader, self.valid_data_loader
            )


    def train(self):
        epoch_trainer = self.epoch_trainer
        epochs = self.epochs
        self._training_loop(epochs, epoch_trainer)

        self.plot()


    def plot(self):

        results_trainer = self.epoch_trainer.train_metrics.all_results()
        results_val = self.epoch_trainer.valid_metrics.all_results()

        train_target_losses = results_trainer['target_loss'].to_numpy()
        val_target_losses = results_val['target_loss'].to_numpy()
        train_accuracies = results_trainer['accuracy'].to_numpy()
        val_accuracies = results_val['accuracy'].to_numpy()
        APLs_train = results_trainer['APL'].to_numpy()
        APLs_test = results_val['APL'].to_numpy()
        if self.reg == 'Tree':
            APL_predictions_train = results_trainer['APL_predictions'].to_numpy()
            APL_predictions_test = results_val['APL_predictions'].to_numpy()
        fidelities_train = results_trainer['fidelity'].to_numpy()
        fidelities_test = results_val['fidelity'].to_numpy()
        FI_train = results_trainer['Feauture Importances'].to_numpy()
        FI_test = results_val['Feauture Importances'].to_numpy()

        # Plotting the results
        epochs = range(1, self.epochs + 1)
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
        for i in range(self.num_features):
            plt.plot(epochs, [fi[i] for fi in FI_train],
                     label=f'{self.feature_names[i]}')
        plt.title('Feature Importances (Train)')
        plt.xlabel('Epochs')
        plt.ylabel('Feature Importances')
        plt.legend(loc='upper left')

        plt.subplot(3, 2, 6)
        for i in range(self.num_features):
            plt.plot(epochs, [fi[i] for fi in FI_test],
                     label=f'{self.feature_names[i]}')
        plt.title('Feature Importances (Test)')
        plt.xlabel('Epochs')
        plt.ylabel('Feature Importances')
        # put legend to the top left of the plot
        plt.legend(loc='upper left')

        plt.tight_layout()
        plt.savefig(str(self.config.log_dir) + '/plots.png')
        plt.show()

        # save as pickle file the dataframes
        results_trainer.to_pickle(
            str(self.config.log_dir) + '/results_trainer.pkl')
        results_val.to_pickle(str(self.config.log_dir) + '/results_val.pkl')

        print('Plots saved in the logs directory')