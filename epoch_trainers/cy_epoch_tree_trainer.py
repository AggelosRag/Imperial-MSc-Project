import numpy as np
import torch
from loggers.cy_logger import CYLogger

from base.epoch_trainer_base import EpochTrainerBase


class CY_Epoch_Tree_Trainer(EpochTrainerBase):
    """
    Trainer Epoch class using Tree Regularization
    """

    def __init__(self, arch, config, device,
                 data_loader, valid_data_loader=None,
                 lr_scheduler=None):

        super(CY_Epoch_Tree_Trainer, self).__init__(arch, config)

        # Extract the configuration parameters
        self.config = config
        self.device = device
        self.train_loader = data_loader
        self.val_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.arch = arch
        self.model = arch.model
        self.criterion = arch.criterion_target
        self.optimizer = arch.optimizer_target
        self.min_samples_leaf = config['regularisation']['min_samples_leaf']

        self.do_validation = self.val_loader is not None
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.metrics_tracker = CYLogger(config, iteration=1,
                                        tb_path=str(self.config.log_dir),
                                        output_path=str(self.config.save_dir),
                                        train_loader=self.train_loader,
                                        val_loader=self.val_loader,
                                        selectivenet=False,
                                        device=self.device)
        self.metrics_tracker.begin_run()

        # check if selective net is used
        if "selectivenet" in config.config.keys():
            self.selective_net = True
        else:
            self.selective_net = False

    def _train_epoch(self, epoch):

        print(f"Training epoch {epoch}")
        self.metrics_tracker.begin_epoch()
        self.model.label_predictor.train()
        if self.selective_net:
            self.arch.selector.train()
            self.arch.aux_model.train()

        for (X_batch, C_batch, y_batch), (
        X_rest, C_rest, y_rest) in self.train_loader:
            batch_size = X_batch.size(0)
            C_batch = C_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward pass
            y_pred = self.model.label_predictor(C_batch)
            outputs = {"prediction_out": y_pred}

            if self.selective_net:
                out_selector = self.arch.selector(C_batch)
                out_aux = self.arch.aux_model(C_batch)
                outputs = {"prediction_out": y_pred,
                           "selection_out": out_selector, "out_aux": out_aux}

            # Calculate Label losses
            loss_label = self.criterion(outputs, y_batch)
            self.metrics_tracker.update_batch(
                update_dict_or_key=loss_label,
                batch_size=batch_size,
                mode='train')

            # Track target training loss and accuracy
            self.metrics_tracker.track_total_train_correct_per_epoch(
                preds=outputs["prediction_out"], labels=y_batch
            )

            # if we operate in SGD mode, then X_batch + X_rest = X
            # We still need the complete dataset to compute the APL
            # In full-batch GD, X_batch = X and X_rest = None
            # if X_rest is not None:
            #     X_rest = X_rest.to(self.device)
            #
            #     with torch.no_grad():
            #         C_pred_rest = self.model.concept_predictor(X_rest)
            #         y_pred_rest = self.model.label_predictor(C_pred_rest)
            #
            #     C_pred = torch.vstack([C_pred, C_pred_rest])
            #     y_pred = torch.vstack([y_pred, y_pred_rest])
            #
            # # Calculate the APL
            # APL, fid, fi, tree = self._calculate_APL(
            #     self.min_samples_leaf, C_pred, y_pred
            # )
            # self.metrics_tracker.update_batch(update_dict_or_key='APL',
            #                                   value=APL,
            #                                   batch_size=batch_size,
            #                                   mode='train')
            # self.metrics_tracker.update_batch(update_dict_or_key='fidelity',
            #                                   value=fid,
            #                                   batch_size=batch_size,
            #                                   mode='train')
            # self.metrics_tracker.update_batch(update_dict_or_key='feature_importance',
            #                                   value=fi,
            #                                   batch_size=batch_size,
            #                                   mode='train')

            loss = loss_label["target_loss"]
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.metrics_tracker.update_batch(update_dict_or_key='loss',
                                              value=loss.detach().item(),
                                              batch_size=batch_size,
                                              mode='train')

        if self.do_validation:
            self._valid_epoch(epoch)

        # Update the epoch metrics
        self.metrics_tracker.end_epoch(selectivenet=self.selective_net)
        log = self.metrics_tracker.result_epoch()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # visualize last tree
        # if epoch % self.config['regularisation']['snapshot_epochs'] == 0:
        #     self._visualize_tree(tree, self.config, epoch, APL,
        #                          'None', 'None')
        return log

    def _valid_epoch(self, epoch):

        print(f"Validating epoch {epoch}")
        self.model.label_predictor.eval()
        if self.selective_net:
            self.arch.selector.eval()
            self.arch.aux_model.eval()

        with torch.no_grad():
            for (X_batch, C_batch, y_batch), (
            X_rest, C_rest, y_rest) in self.val_loader:
                batch_size = X_batch.size(0)
                C_batch = C_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward pass
                y_pred = self.model.label_predictor(C_batch)
                outputs = {"prediction_out": y_pred}

                if self.selective_net:
                    out_selector = self.arch.selector(C_batch)
                    out_aux = self.arch.aux_model(C_batch)
                    outputs = {"prediction_out": y_pred,
                               "selection_out": out_selector,
                               "out_aux": out_aux}

                # save outputs for selectivenet
                if self.selective_net:
                    self.metrics_tracker.update_batch(
                        update_dict_or_key='out_put_sel_proba',
                        value=out_selector.detach().cpu(),
                        batch_size=batch_size,
                        mode='val')
                    self.metrics_tracker.update_batch(
                        update_dict_or_key='out_put_class',
                        value=y_pred.detach().cpu(),
                        batch_size=batch_size,
                        mode='val')
                    self.metrics_tracker.update_batch(
                        update_dict_or_key='out_put_target',
                        value=y_batch.detach().cpu(),
                        batch_size=batch_size,
                        mode='val')

                # Calculate Label losses
                loss_label = self.criterion(outputs, y_batch)
                self.metrics_tracker.update_batch(
                    update_dict_or_key=loss_label,
                    batch_size=batch_size,
                    mode='val')

                # Track target training loss and accuracy
                self.metrics_tracker.track_total_val_correct_per_epoch(
                    preds=outputs["prediction_out"], labels=y_batch
                )

                # if we operate in SGD mode, then X_batch + X_rest = X
                # We still need the complete dataset to compute the APL
                # In full-batch GD, X_batch = X and X_rest = None
                # if X_rest is not None:
                #     X_rest = X_rest.to(self.device)
                #
                #     with torch.no_grad():
                #         C_pred_rest = self.model.concept_predictor(X_rest)
                #         y_pred_rest = self.model.label_predictor(C_pred_rest)
                #
                #     C_pred = torch.vstack([C_pred, C_pred_rest])
                #     y_pred = torch.vstack([y_pred, y_pred_rest])
                #
                # # Calculate the APL
                # APL, fid, fi, tree = self._calculate_APL(
                #     self.min_samples_leaf, C_pred, y_pred
                # )
                # self.metrics_tracker.update_batch(update_dict_or_key='APL',
                #                                   value=APL,
                #                                   batch_size=batch_size,
                #                                   mode='val')
                # self.metrics_tracker.update_batch(update_dict_or_key='fidelity',
                #                                   value=fid,
                #                                   batch_size=batch_size,
                #                                   mode='val')
                # self.metrics_tracker.update_batch(
                #     update_dict_or_key='feature_importance',
                #     value=fi,
                #     batch_size=batch_size,
                #     mode='val')

                loss = loss_label["target_loss"]
                self.metrics_tracker.update_batch(update_dict_or_key='loss',
                                                  value=loss.detach().item(),
                                                  batch_size=batch_size,
                                                  mode='val')

        # Update the epoch metrics
        if self.selective_net:
            # evaluate g for correctly selected samples (pi >= 0.5)
            # should be higher
            self.metrics_tracker.evaluate_correctly(
                selection_threshold=self.config['selectivenet'][
                    'selection_threshold'])

            # evaluate g for correctly rejected samples (pi < 0.5)
            # should be lower
            self.metrics_tracker.evaluate_incorrectly(
                selection_threshold=self.config['selectivenet'][
                    'selection_threshold'])
            self.metrics_tracker.evaluate_coverage_stats(
                selection_threshold=self.config['selectivenet'][
                    'selection_threshold'])
