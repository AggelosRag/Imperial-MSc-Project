import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from base.epoch_trainer_base import EpochTrainerBase
from loggers.joint_cbm_logger import JointCBMLogger
from loggers.cy_logger import CYLogger

class XCY_Tree_Epoch_Trainer(EpochTrainerBase):
    """
    Trainer Epoch class using Tree Regularization
    """

    def __init__(self, arch, config, device, data_loader, cbm_mode,
                 valid_data_loader=None, iteration=None, lr_scheduler=None):

        super(XCY_Tree_Epoch_Trainer, self).__init__(arch, config, iteration)

        # Extract the configuration parameters
        self.tree_reg_mode = config['regularisation']['tree_reg_mode']
        self.config = config
        self.device = device
        self.train_loader = data_loader
        self.val_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.arch = arch
        self.model = arch.model
        self.epochs_warm_up = config['regularisation']['warm_up_epochs']
        self.alpha = config['model']['alpha']
        self.criterion_concept = arch.criterion_concept
        self.criterion_label = arch.criterion_label
        self.criterion_sr = arch.criterion_sr
        self.optimizer = arch.optimizer
        self.optimizer_mn = arch.optimizer_mn
        self.optimizer_sr = arch.optimizer_sr
        self.num_concepts = config['dataset']['num_concepts']
        self.reg_strength = config['regularisation']['reg_strength']
        self.mse_loss_strength = config['regularisation']['mse_loss_strength']
        self.min_samples_leaf = config['regularisation']['min_samples_leaf']
        self.iteration = iteration

        self.do_validation = self.val_loader is not None
        self.log_step = int(np.sqrt(data_loader.batch_size))

        # if we use tree regularisation in sequential mode, we need to store
        # all intermediate APL values and the corresponding predictions
        if self.tree_reg_mode == 'Sequential':
            self.sr_training_freq = config['regularisation']['sequential']['sr_training_freq']
            self.APLs_truth = []
            self.all_preds = []

        self.cbm_mode = cbm_mode
        # Initialize the metrics tracker
        if cbm_mode == 'joint':
            self.metrics_tracker = JointCBMLogger(config, iteration=1,
                                                  tb_path=str(self.config.log_dir),
                                                  output_path=str(self.config.save_dir),
                                                  train_loader=self.train_loader,
                                                  val_loader=self.val_loader,
                                                  device=self.device)
        elif cbm_mode == 'sequential':
            self.metrics_tracker = CYLogger(config, iteration=1,
                                           tb_path=str(self.config.log_dir),
                                           output_path=str(self.config.save_dir),
                                           train_loader=self.train_loader,
                                           val_loader=self.val_loader,
                                           device=self.device)
        else:
            raise ValueError(f"Unknown CBM mode: {cbm_mode}")
        self.metrics_tracker.begin_run()

        # check if selective net is used
        if "selectivenet" in config.config.keys():
            self.selective_net = True
        else:
            self.selective_net = False

    def _train_epoch(self, epoch):

        print(f"Training Epoch {epoch}:")
        self.metrics_tracker.begin_epoch()

        self.model.mn_model.concept_predictor.train()
        self.model.mn_model.label_predictor.train()
        self.model.sr_model.train()
        if self.selective_net:
            self.arch.selector.train()
            self.arch.aux_model.train()

        for (X_batch, C_batch, y_batch), (X_rest, C_rest, y_rest) in self.train_loader:
            batch_size = X_batch.size(0)
            X_batch = X_batch.to(self.device)
            C_batch = C_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward pass
            C_pred = self.model.mn_model.concept_predictor(X_batch)
            y_pred = self.model.mn_model.label_predictor(C_pred)
            outputs = {"prediction_out": y_pred}

            if self.selective_net:
                out_selector = self.arch.selector(C_pred)
                out_aux = self.arch.aux_model(C_pred)
                outputs = {"prediction_out": y_pred, "selection_out": out_selector, "out_aux": out_aux}

            # Calculate Concept losses
            loss_concept = self.criterion_concept(C_pred, C_batch)
            bce_loss_per_concept = torch.mean(loss_concept, dim=0)
            loss_concept_total = bce_loss_per_concept.sum()
            self.metrics_tracker.update_batch(update_dict_or_key='concept_loss',
                                              value=loss_concept_total.detach().item(),
                                              batch_size=batch_size,
                                              mode='train')
            self.metrics_tracker.update_batch(update_dict_or_key='loss_per_concept',
                                              value=list(bce_loss_per_concept.detach().numpy()),
                                              batch_size=batch_size,
                                              mode='train')

            # if we do warm-up, detach the gradient for the surrogate training
            if epoch <= self.epochs_warm_up:
                y_hat_sr = y_pred.flatten().detach()
            else:
                y_hat_sr = y_pred.flatten()

            # Calculate Label losses
            loss_label = self.criterion_label(outputs, y_batch)
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
            if X_rest is not None:
                X_rest = X_rest.to(self.device)

                with torch.no_grad():
                    C_pred_rest = self.model.mn_model.concept_predictor(X_rest)
                    y_pred_rest = self.model.mn_model.label_predictor(C_pred_rest)
                    y_hat_sr_rest = y_pred_rest.flatten()

                C_pred = torch.vstack([C_pred, C_pred_rest])
                y_pred = torch.vstack([y_pred, y_pred_rest])
                y_hat_sr = torch.cat([y_hat_sr, y_hat_sr_rest])

            # Calculate the APL
            APL, fid, fi, tree = self._calculate_APL(self.min_samples_leaf,
                                                     C_pred, y_pred)
            self.metrics_tracker.update_batch(update_dict_or_key='APL',
                                              value=APL,
                                              batch_size=batch_size,
                                              mode='train')
            self.metrics_tracker.update_batch(update_dict_or_key='fidelity',
                                              value=fid,
                                              batch_size=batch_size,
                                              mode='train')
            self.metrics_tracker.update_batch(update_dict_or_key='feature_importance',
                                              value=fi,
                                              batch_size=batch_size,
                                              mode='train')

            # Calculate the APL prediction and store results if in sequential tree mode
            # only for usefull APL predictions
            omega = self.model.sr_model(y_hat_sr)
            self.metrics_tracker.update_batch(update_dict_or_key='APL_predictions',
                                              value=omega.item(),
                                              batch_size=batch_size,
                                              mode='train')
            if (APL > 1 or epoch == 0) and self.tree_reg_mode == 'Sequential':
                self.APLs_truth.append(APL)
                self.all_preds.append(y_hat_sr)

            # Calculate the surrogate loss
            sr_loss = self.criterion_sr(input=omega, target=torch.tensor(APL, dtype=torch.float))

            # Optimise either the two losses separately in warm-up mode or the total loss
            if epoch <= self.epochs_warm_up:
                loss = self.alpha * loss_concept_total + loss_label["target_loss"]
                self.optimizer_mn.zero_grad()
                loss.backward()
                self.optimizer_mn.step()

                self.optimizer_sr.zero_grad()
                sr_loss.backward()
                self.optimizer_sr.step()
            else:
                if self.tree_reg_mode == 'Sequential':
                    loss = self.alpha * loss_concept_total + loss_label["target_loss"] + self.reg_strength * omega
                else:
                    loss = self.alpha * loss_concept_total + loss_label["target_loss"] + self.reg_strength * omega + self.mse_loss_strength * sr_loss

                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.metrics_tracker.update_batch(update_dict_or_key='loss',
                                              value=loss.detach().item(),
                                              batch_size=batch_size,
                                              mode='train')

        # in sequential mode, train the surrogate model
        if self.tree_reg_mode == 'Sequential':
            if epoch % self.sr_training_freq == 0:
                self._train_surrogate_sequential_mode(epoch)

        if self.do_validation:
            self._valid_epoch(epoch)

        # Update the epoch metrics
        self.metrics_tracker.end_epoch(selectivenet=self.selective_net)
        log = self.metrics_tracker.result_epoch()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        #visualize last tree
        if epoch % self.config['regularisation']['snapshot_epochs'] == 0:
            self._visualize_tree(tree, self.config, epoch, APL, 'None', 'None',
                                 mode='train', iteration=self.iteration)
        return log



    def _valid_epoch(self, epoch):

        print(f"Validation Epoch {epoch}:")
        self.model.mn_model.concept_predictor.eval()
        self.model.mn_model.label_predictor.eval()
        self.model.sr_model.eval()
        if self.selective_net:
            self.arch.selector.eval()
            self.arch.aux_model.eval()

        with torch.no_grad():
            for (X_batch, C_batch, y_batch), (X_rest, C_rest, y_rest) in self.val_loader:
                batch_size = X_batch.size(0)
                X_batch = X_batch.to(self.device)
                C_batch = C_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                C_pred = self.model.mn_model.concept_predictor(X_batch)
                y_pred = self.model.mn_model.label_predictor(C_pred)
                outputs = {"prediction_out": y_pred}

                if self.selective_net:
                    out_selector = self.arch.selector(C_pred)
                    out_aux = self.arch.aux_model(C_pred)
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

                y_hat_sr = y_pred.flatten()
                omega = self.model.sr_model(y_hat_sr)

                # Calculate Concept losses
                loss_concept = self.criterion_concept(C_pred, C_batch)
                bce_loss_per_concept = torch.mean(loss_concept, dim=0)
                loss_concept_total = bce_loss_per_concept.sum()
                self.metrics_tracker.update_batch(
                    update_dict_or_key='concept_loss',
                    value=loss_concept_total.detach().item(),
                    batch_size=batch_size,
                    mode='val')
                self.metrics_tracker.update_batch(
                    update_dict_or_key='loss_per_concept',
                    value=list(bce_loss_per_concept.detach().numpy()),
                    batch_size=batch_size,
                    mode='val')

                # Calculate Label losses
                loss_label = self.criterion_label(outputs, y_batch)
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
                if X_rest is not None:
                    X_rest = X_rest.to(self.device)

                    with torch.no_grad():
                        C_pred_rest = self.model.concept_predictor(X_rest)
                        y_pred_rest = self.model.label_predictor(C_pred_rest)

                    C_pred = torch.vstack([C_pred, C_pred_rest])
                    y_pred = torch.vstack([y_pred, y_pred_rest])

                # Calculate the APL
                APL, fid, fi, tree = self._calculate_APL(self.min_samples_leaf,
                                                         C_pred, y_pred)
                self.metrics_tracker.update_batch(update_dict_or_key='APL',
                                                  value=APL,
                                                  batch_size=batch_size,
                                                  mode='val')
                self.metrics_tracker.update_batch(update_dict_or_key='fidelity',
                                                  value=fid,
                                                  batch_size=batch_size,
                                                  mode='val')
                self.metrics_tracker.update_batch(
                    update_dict_or_key='feature_importance',
                    value=fi,
                    batch_size=batch_size,
                    mode='val')
                self.metrics_tracker.update_batch(
                    update_dict_or_key='APL_predictions',
                    value=omega.item(),
                    batch_size=batch_size,
                    mode='val')

                sr_loss = self.criterion_sr(input=omega, target=torch.tensor(APL, dtype=torch.float))

                if epoch <= self.epochs_warm_up:
                    loss = self.alpha * loss_concept_total + loss_label["target_loss"]
                else:
                    if self.tree_reg_mode == 'Sequential':
                        loss = self.alpha * loss_concept_total + loss_label["target_loss"] + self.reg_strength * omega
                    else:
                        loss = self.alpha * loss_concept_total + loss_label["target_loss"] + self.reg_strength * omega + self.mse_loss_strength * sr_loss

                self.metrics_tracker.update_batch(update_dict_or_key='loss',
                                                  value=loss.detach().item(),
                                                  batch_size=batch_size,
                                                  mode='val')

        # Update the epoch metrics
        if self.selective_net:
            # evaluate g for correctly selected samples (pi >= 0.5)
            # should be higher
            self.metrics_tracker.evaluate_correctly(selection_threshold=self.config['selectivenet']['selection_threshold'])

            # evaluate g for correctly rejected samples (pi < 0.5)
            # should be lower
            self.metrics_tracker.evaluate_incorrectly(selection_threshold=self.config['selectivenet']['selection_threshold'])
            self.metrics_tracker.evaluate_coverage_stats(selection_threshold=self.config['selectivenet']['selection_threshold'])


    def _train_surrogate_sequential_mode(self, epoch):
        """
        Train the surrogate model in sequential mode
        """
        self.model.unfreeze_surrogate()
        self.model.sr_model.train()
        self.model.freeze_model()
        self.model.reset_surrogate_weights()
        if epoch > 0:
            surrogate_training_loss = self.train_surrogate_model(self.all_preds,
                                                            self.APLs_truth,
                                                            self.criterion_sr,
                                                            # optimizer,
                                                            self.optimizer_sr,
                                                            self.model)
            print(f'Surrogate Training Loss: {surrogate_training_loss[-1]:.4f}')

        self.model.unfreeze_model()
        self.model.freeze_surrogate()


    def train_surrogate_model(self, X, y, criterion, optimizer, model):

        # X_train = torch.vstack(X)
        X_train = torch.vstack(X).detach()
        y_train = torch.tensor([y], dtype=torch.float).T.to(self.device)

        model.surrogate_network.to(self.device)

        num_epochs = self.config['regularisation']['sequential']['sr_epochs']
        batch_size = self.config['regularisation']['sequential']['sr_batch_size']

        data_train = TensorDataset(X_train, y_train)
        data_train_loader = DataLoader(dataset=data_train,
                                       batch_size=batch_size, shuffle=True)

        training_loss = []

        model.surrogate_network.train()

        for epoch in range(num_epochs):
            batch_loss = []

            for (x, y) in data_train_loader:
                y_hat = model.surrogate_network(x)
                loss = criterion(input=y_hat, target=y)
                optimizer.zero_grad()
                # loss.backward()
                loss.backward()
                optimizer.step()

                batch_loss.append(
                    loss.item() / (torch.var(y_train).item() + 0.01))

            training_loss.append(np.array(batch_loss).mean())

            if epoch == 0 or (epoch + 1) % 10 == 0:
                # if epoch:
                print(
                    f'Surrogate Model: Epoch [{epoch + 1}/{num_epochs},'
                    f' Loss: {np.array(batch_loss).mean():.4f}]')

        del X
        del y

        return training_loss
