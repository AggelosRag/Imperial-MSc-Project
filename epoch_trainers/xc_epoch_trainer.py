import numpy as np
import torch
from tqdm import tqdm
import sys

from loggers.joint_cbm_logger import JointCBMLogger
from loggers.xc_logger import XCLogger

from base.epoch_trainer_base import EpochTrainerBase


class XC_Epoch_Trainer(EpochTrainerBase):
    """
    Trainer Epoch class using Tree Regularization
    """

    def __init__(self, arch, config, device, data_loader,
                 valid_data_loader=None, lr_scheduler=None):

        super(XC_Epoch_Trainer, self).__init__(arch, config)

        # Extract the configuration parameters
        self.config = config
        self.device = device
        self.train_loader = data_loader
        self.val_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.arch = arch
        self.model = arch.model.to(self.device)
        self.criterion_per_concept = arch.criterion_per_concept
        self.criterion = arch.criterion_concept
        self.min_samples_leaf = config['regularisation']['min_samples_leaf']

        self.do_validation = self.val_loader is not None

        # Initialize the metrics tracker
        self.metrics_tracker = XCLogger(config, iteration=1,
                                       tb_path=str(self.config.log_dir),
                                       output_path=str(self.config.save_dir),
                                       train_loader=self.train_loader,
                                       val_loader=self.val_loader,
                                       device=self.device)
        self.metrics_tracker.begin_run()
        print("Device: ", self.device)

        self.optimizer = arch.xc_optimizer
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)


    def _train_epoch(self, epoch):

        print(f"Training Epoch {epoch}")
        self.metrics_tracker.begin_epoch()
        self.model.concept_predictor.train()

        with tqdm(total=len(self.train_loader), file=sys.stdout) as t:
            for batch_idx, (X_batch, C_batch, y_batch) in enumerate(self.train_loader):
                batch_size = X_batch.size(0)
                X_batch = X_batch.to(self.device)
                C_batch = C_batch.to(self.device)

                # Forward pass
                C_pred = self.model.concept_predictor(X_batch)

                # Calculate Concept losses
                loss_concept_total = self.criterion(C_pred, C_batch)
                bce_loss_per_concept = self.criterion_per_concept(C_pred, C_batch)
                bce_loss_per_concept = torch.mean(bce_loss_per_concept, dim=0)
                self.metrics_tracker.update_batch(update_dict_or_key='concept_loss',
                                                  value=loss_concept_total.detach().cpu().item(),
                                                  batch_size=batch_size,
                                                  mode='train')
                self.metrics_tracker.update_batch(update_dict_or_key='loss_per_concept',
                                                  value=list(bce_loss_per_concept.detach().cpu().numpy()),
                                                  batch_size=batch_size,
                                                  mode='train')
                # Track target training loss and accuracy
                self.metrics_tracker.track_total_train_correct_per_epoch_per_concept(
                    preds=C_pred, labels=C_batch
                )

                # Track target training loss and accuracy
                # self.metrics_tracker.track_total_train_correct_per_epoch(
                #     preds=outputs["prediction_out"], labels=y_batch
                # )

                self.optimizer.zero_grad()
                loss_concept_total.backward()
                self.optimizer.step()
                self.metrics_tracker.update_batch(update_dict_or_key='loss',
                                                  value=loss_concept_total.detach().cpu().item(),
                                                  batch_size=batch_size,
                                                  mode='train')

                t.set_postfix(
                    batch_id='{0}'.format(batch_idx + 1))
                t.update()

        if self.do_validation:
            self._valid_epoch(epoch)

        # Update the epoch metrics
        self.metrics_tracker.end_epoch()
        log = self.metrics_tracker.result_epoch()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log


    def _valid_epoch(self, epoch):

        print(f"Validation Epoch {epoch}")
        self.model.concept_predictor.eval()

        with torch.no_grad():
            with tqdm(total=len(self.val_loader), file=sys.stdout) as t:
                for batch_idx, (X_batch, C_batch, y_batch) in enumerate(self.val_loader):
                    batch_size = X_batch.size(0)
                    X_batch = X_batch.to(self.device)
                    C_batch = C_batch.to(self.device)

                    # Forward pass
                    C_pred = self.model.concept_predictor(X_batch)

                    # Calculate Concept losses
                    loss_concept_total = self.criterion(C_pred, C_batch)
                    bce_loss_per_concept = self.criterion_per_concept(C_pred, C_batch)
                    bce_loss_per_concept = torch.mean(bce_loss_per_concept, dim=0)
                    self.metrics_tracker.update_batch(
                        update_dict_or_key='concept_loss',
                        value=loss_concept_total.detach().cpu().item(),
                        batch_size=batch_size,
                        mode='val')
                    self.metrics_tracker.update_batch(
                        update_dict_or_key='loss_per_concept',
                        value=list(bce_loss_per_concept.detach().cpu().numpy()),
                        batch_size=batch_size,
                        mode='val')
                    # Track target training loss and accuracy
                    self.metrics_tracker.track_total_val_correct_per_epoch_per_concept(
                        preds=C_pred, labels=C_batch
                    )

                    # Track target training loss and accuracy
                    # self.metrics_tracker.track_total_val_correct_per_epoch(
                    #     preds=outputs["prediction_out"], labels=y_batch
                    # )

                    self.metrics_tracker.update_batch(update_dict_or_key='loss',
                                                      value=loss_concept_total.detach().cpu().item(),
                                                      batch_size=batch_size,
                                                      mode='val')

                    t.set_postfix(
                        batch_id='{0}'.format(batch_idx + 1))
                    t.update()