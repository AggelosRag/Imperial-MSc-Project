import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from base.epoch_trainer_base import EpochTrainerBase
from utils import SimplerMetricTracker

class XCY_Epoch_Trainer(EpochTrainerBase):
    """
    Trainer Epoch class using Tree Regularization
    """

    def __init__(self, arch, epochs, writer, metric_ftns, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None):

        super(XCY_Epoch_Trainer, self).__init__()

        # Extract the configuration parameters
        self.writer = writer
        self.epochs = epochs
        self.config = config
        self.device = device
        self.train_loader = data_loader
        self.val_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.model = arch.model
        self.alpha = config['model']['alpha']
        self.criterion_concept = arch.criterion_concept
        self.criterion_label = arch.criterion_label
        self.optimizer = arch.optimizer
        self.num_concepts = config['dataset']['num_concepts']
        self.min_samples_leaf = config['regularisation']['min_samples_leaf']
        self.metric_ftns = metric_ftns

        self.do_validation = self.val_loader is not None
        self.log_step = int(np.sqrt(data_loader.batch_size))

        # Define the metric trackers
        self.train_metrics = SimplerMetricTracker(self.epochs, metric_ftns,
                                                  writer=self.writer,
                                                  mode='Train')
        self.valid_metrics = SimplerMetricTracker(self.epochs, metric_ftns,
                                                  writer=self.writer,
                                                  mode='Valid')

    def _train_epoch(self, epoch):
        # self.writer.set_step(epoch)
        self.model.concept_predictor.train()
        self.model.label_predictor.train()

        for (X_batch, C_batch, y_batch), (X_rest, C_rest, y_rest) in self.train_loader:
            X_batch = X_batch.to(self.device)
            C_batch = C_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward pass
            C_pred = self.model.concept_predictor(X_batch)
            # C_pred = C_batch
            y_pred = self.model.label_predictor(C_pred)

            # Calculate Concept losses
            loss_concept = self.criterion_concept(C_pred, C_batch)
            bce_loss_per_concept = torch.mean(loss_concept, dim=0)
            loss_concept_total = bce_loss_per_concept.sum()

            if y_pred.shape[1] == 1:
                y_hat_pred = torch.where(y_pred > 0.5, 1, 0).cpu()
            elif y_pred.shape[1] >= 3:
                y_hat_pred = torch.argmax(y_pred, 1).cpu()
                y_batch = y_batch.long()
            else:
                raise ValueError('Invalid number of output classes')

            loss_label = self.criterion_label(y_pred, y_batch)

            # Track target training loss and accuracy
            self.train_metrics.append_batch_result('concept_loss', loss_concept_total.detach().numpy())
            self.train_metrics.append_batch_result('target_loss', loss_label.item())

            total_train = y_batch.size(0)
            correct_train = (y_hat_pred == y_batch).sum().item()
            self.train_metrics.append_batch_result('correct', correct_train)
            self.train_metrics.append_batch_result('total', total_train)

            # if we operate in SGD mode, then X_batch + X_rest = X
            # We still need the complete dataset to compute the APL
            # In full-batch GD, X_batch = X and X_rest = None
            if X_rest is not None:
                X_rest = X_rest.to(self.device)

                self.model.freeze_model()
                self.model.eval()
                C_pred_rest = self.model.concept_predictor(X_rest)
                y_pred_rest = self.model.label_predictor(C_pred_rest)
                self.model.unfreeze_model()
                self.model.train()

                C_pred = torch.vstack([C_pred, C_pred_rest])
                y_pred = torch.vstack([y_pred, y_pred_rest])

        # Update the epoch metrics
        self.train_metrics.update_epoch(epoch)
        self.train_metrics.append_epoch_result(epoch,'bce_loss_per_concept',
                                               list(bce_loss_per_concept.detach().numpy()))
        # Calculate the APL
        self.model.freeze_model()
        self.model.eval()
        APL, self.train_metrics, tree = self._calculate_APL(self.model, self.min_samples_leaf, C_pred, y_pred,
                                                      self.train_metrics, epoch, mode='Training')
        self.model.unfreeze_model()
        self.model.train()

        loss = self.alpha * loss_concept_total + loss_label
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_metrics.append_epoch_result(epoch, 'total_loss', loss.item())

        log = self.train_metrics.result(epoch)

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        #visualize tree
        if epoch % self.config['regularisation']['snapshot_epochs'] == 0:
            self._visualize_tree(tree, self.config, epoch,
                                 self.valid_metrics,
                                 self.train_metrics)
        return log



    def _valid_epoch(self, epoch):

        self.model.concept_predictor.eval()
        self.model.label_predictor.eval()

        with torch.no_grad():
            for (X_batch, C_batch, y_batch) in self.val_loader:
                X_batch = X_batch.to(self.device)
                C_batch = C_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Only full-batch training is currently supported
                assert X_batch.size(0) == len(self.val_loader.dataset)

                C_pred = self.model.concept_predictor(X_batch)
                # C_pred = C_batch
                y_pred = self.model.label_predictor(C_pred)

                if y_pred.shape[1] == 1:
                    y_hat_pred = torch.where(y_pred > 0.5, 1, 0).cpu()
                elif y_pred.shape[1] >= 3:
                    y_hat_pred = torch.argmax(y_pred, 1).cpu()
                    y_batch = y_batch.long()
                else:
                    raise ValueError('Invalid number of output classes')

                loss_concept = self.criterion_concept(C_pred, C_batch)
                bce_loss_per_concept = torch.mean(loss_concept, dim=0)
                loss_concept_total = bce_loss_per_concept.sum()

                loss_label = self.criterion_label(y_pred, y_batch)

                # Track training loss and accuracy
                self.valid_metrics.append_batch_result('concept_loss',
                                                       loss_concept_total.detach().numpy())
                self.valid_metrics.append_batch_result('target_loss',
                                                       loss_label.item())

                total_val = y_batch.size(0)
                correct_val = (y_hat_pred == y_batch).sum().item()
                self.valid_metrics.append_batch_result('correct', correct_val)
                self.valid_metrics.append_batch_result('total', total_val)

        # Update the epoch metrics
        self.valid_metrics.update_epoch(epoch)
        self.valid_metrics.append_epoch_result(epoch,'bce_loss_per_concept',
                                               list(bce_loss_per_concept.detach().numpy()))

        APL_test, self.valid_metrics, tree = self._calculate_APL(self.model, self.min_samples_leaf, C_pred,
                                                                  y_pred, self.valid_metrics, epoch,
                                                                  mode='Validation')

        loss = self.alpha * loss_concept_total + loss_label

        self.valid_metrics.append_epoch_result(epoch, 'total_loss', loss.item())

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result(epoch)