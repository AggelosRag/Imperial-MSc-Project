# from matplotlib import pyplot as plt
# from base.trainer_base import TrainerBase
# from logger import TensorboardWriter
#
#
# class SequentialCBMTrainer(TrainerBase):
#
#     def __init__(self, model, criterion, metric_ftns, optimizer, config,
#                  reg=None):
#
#         super(SequentialCBMTrainer, self).__init__(model, criterion, config, optimizer)
#
#         self.criterion = criterion
#         self.metric_ftns = metric_ftns
#         self.config = config
#
#         self.num_concepts = config['dataset']['num_concepts']
#         self.epochs_xc = config['trainer']['epochs_xc']
#         self.epochs_cy = config['trainer']['epochs_cy']
#
#         self.xc_epoch_trainer = XY_Epoch_Trainer(model, criterion,
#                                                  metric_ftns,
#                                                  optimizer, config)
#
#         # setup visualization writer instance
#         self.writer = TensorboardWriter(config.log_dir,
#                                         self.logger,
#                                         config['trainer']['tensorboard'])
#
#         self.reg = reg
#         # TODO: Add L1 and L2 regularization
#         if reg == 'Tree':
#             self.cy_epoch_trainer = XY_Tree_Epoch_Trainer(model, criterion,
#                                                           metric_ftns,
#                                                           optimizer, config)
#         else:
#             self.cy_epoch_trainer = XY_Epoch_Trainer(model, criterion,
#                                                      metric_ftns,
#                                                      optimizer, config)
#
#     def train(self):
#
#         # Training the XC model
#         epoch_trainer = self.xc_epoch_trainer
#         epochs = self.epochs_xc
#         self._training_loop(epochs, epoch_trainer)
#
#         # Freezing the XC model
#         for param in self.model.concept_predictor.parameters():
#             param.requires_grad = False
#
#         # Training the CY model
#         epoch_trainer = self.cy_epoch_trainer
#         epochs = self.epochs_cy
#         self._training_loop(epochs, epoch_trainer)
#
#         self.plot()
#
#
#     def plot(self):
#
#         # Plotting the results
#         epochs_cp = range(1, self.epochs_xc + 1)
#         epochs_lp = range(1, self.epochs_cy + 1)
#         epochs_less = range(11, self.epochs_xc + 1)
#         plt.figure(figsize=(18, 30))
#
#         plt.subplot(5, 2, 1)
#         for i in range(self.num_concepts):
#             plt.plot(epochs_less,
#                      [train_bce_losses[j][i] for j in range(10, self.epochs_xc)],
#                      label=f'Train BCE Loss {feature_names[i]}')
#         plt.title('Training BCE Loss per Concept')
#         plt.xlabel('Epochs')
#         plt.ylabel('BCE Loss')
#         plt.legend(loc='upper left')
#
#         plt.subplot(5, 2, 2)
#         for i in range(self.num_concepts):
#             plt.plot(epochs_less,
#                      [val_bce_losses[j][i] for j in range(10, self.epochs_xc)],
#                      label=f'Val BCE Loss {feature_names[i]}')
#         plt.title('Validation BCE Loss per Concept')
#         plt.xlabel('Epochs')
#         plt.ylabel('BCE Loss')
#         plt.legend(loc='upper left')
#
#         plt.subplot(5, 2, 3)
#         plt.plot(epochs_less, train_total_bce_losses[10:], 'b',
#                  label='Total Train BCE loss')
#         plt.plot(epochs_less, val_total_bce_losses[10:], 'r',
#                  label='Total Val BCE loss')
#         plt.title('Total BCE Loss')
#         plt.xlabel('Epochs')
#         plt.ylabel('Total BCE Loss')
#         plt.legend()
#
#         plt.subplot(5, 2, 4)
#         plt.plot(epochs_lp, train_target_losses, 'b',
#                  label='Training Target loss')
#         plt.plot(epochs_lp, val_target_losses, 'r',
#                  label='Validation Target loss')
#         plt.title('Training and Validation Target Loss')
#         plt.xlabel('Epochs')
#         plt.ylabel('Target Loss')
#         plt.legend()
#
#         plt.subplot(5, 2, 5)
#         plt.plot(epochs_lp, train_losses, 'b', label='Training Total loss')
#         plt.plot(epochs_lp, val_losses, 'r', label='Validation Total loss')
#         plt.title('Training and Validation Total Loss')
#         plt.xlabel('Epochs')
#         plt.ylabel('Total Loss')
#         plt.legend()
#
#         plt.subplot(5, 2, 6)
#         plt.plot(epochs_lp, train_accuracies, 'b', label='Training accuracy')
#         plt.plot(epochs_lp, val_accuracies, 'r', label='Validation accuracy')
#         plt.title('Training and Validation Accuracy')
#         plt.xlabel('Epochs')
#         plt.ylabel('Accuracy')
#         plt.legend()
#
#         plt.subplot(5, 2, 7)
#         plt.plot(epochs_lp, APLs_train, 'b', label='Train')
#         plt.plot(epochs_lp, APLs_test, 'r', label='Test')
#         if self.reg == 'Tree':
#             plt.plot(epochs_lp, APL_predictions_train, 'g',
#                      label='Train Predictions')
#             plt.plot(epochs_lp, APL_predictions_test, 'y',
#                      label='Test Predictions')
#         plt.title('APL')
#         plt.xlabel('Epochs')
#         plt.ylabel('APL')
#         plt.legend()
#
#         plt.subplot(5, 2, 8)
#         plt.plot(epochs_lp, fidelities_train, 'b', label='Train')
#         plt.plot(epochs_lp, fidelities_test, 'r', label='Test')
#         plt.title('Fidelity')
#         plt.xlabel('Epochs')
#         plt.ylabel('Fidelity')
#         plt.legend()
#
#         plt.subplot(5, 2, 9)
#         for i in range(self.num_concepts):
#             plt.plot(epochs_lp, [fi[i] for fi in FI_train],
#                      label=f'{feature_names[i]}')
#         plt.title('Feature Importances (Train)')
#         plt.xlabel('Epochs')
#         plt.ylabel('Feature Importances')
#         plt.legend(loc='upper left')
#
#         plt.subplot(5, 2, 10)
#         for i in range(self.num_concepts):
#             plt.plot(epochs_lp, [fi[i] for fi in FI_test],
#                      label=f'{feature_names[i]}')
#         plt.title('Feature Importances (Test)')
#         plt.xlabel('Epochs')
#         plt.ylabel('Feature Importances')
#         # put legend to the top left of the plot
#         plt.legend(loc='upper left')
#
#         plt.tight_layout()
#         plt.show()
#
#         # print the BCE losses per concept
#         print('Train BCE losses per concept')
#         for i in range(self.num_concepts):
#             print(f'{feature_names[i]}: {train_bce_losses[-1][i]}')
#         print('Val BCE losses per concept')
#         for i in range(num_concepts):
#             print(f'{feature_names[i]}: {val_bce_losses[-1][i]}')