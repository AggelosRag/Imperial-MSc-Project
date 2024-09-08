import os
import pickle
import sys

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from loggers import XYLogger
from utils.plots import model_contour_plot
from utils.util import get_correct
from base.epoch_trainer_base import EpochTrainerBase


class XY_Epoch_Tree_Trainer(EpochTrainerBase):
    """
    Trainer Epoch class using Tree Regularization
    """

    def __init__(self, arch, config, device,
                 data_loader, valid_data_loader=None,
                 lr_scheduler=None, expert=None):

        super(XY_Epoch_Tree_Trainer, self).__init__(arch, config, expert)

        # Extract the configuration parameters
        self.tree_reg_mode = config['regularisation']['tree_reg_mode']
        self.epochs_warm_up = config['regularisation']['warm_up_epochs']
        self.optimizer_mn = arch.optimizer_mn
        self.optimizer_sr = arch.optimizer_sr
        self.criterion_sr = arch.criterion_sr
        self.reg_strength = config['regularisation']['reg_strength']
        self.mse_loss_strength = config['regularisation']['mse_loss_strength']

        self.config = config
        self.device = device
        self.train_loader = data_loader
        self.val_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.arch = arch
        self.model = arch.model.to(self.device)
        self.criterion = arch.criterion_label
        self.min_samples_leaf = config['regularisation']['min_samples_leaf']
        self.epochs = config['trainer']['epochs']

        # if we use tree regularisation in sequential mode, we need to store
        # all intermediate APL values and the corresponding predictions
        if self.tree_reg_mode == 'Sequential':
            self.sr_training_freq = config['regularisation']['sequential']['sr_training_freq']
            self.APLs_truth = []
            self.all_preds = []

        self.do_validation = self.val_loader is not None
        self.metrics_tracker = XYLogger(config, expert=expert,
                                      tb_path=str(self.config.log_dir),
                                      output_path=str(self.config.save_dir),
                                      train_loader=self.train_loader,
                                      val_loader=self.val_loader,
                                      selectivenet=False,
                                      device=self.device)
        self.metrics_tracker.begin_run()
        print("Device: ", self.device)

        # if we load a pre-trained model, we need to load the
        # optimizer state to device
        self.optimizer = arch.optimizer
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

    def _train_epoch(self, epoch):

        print(f"Training epoch {epoch}")
        self.metrics_tracker.begin_epoch()
        self.model.train()

        tensor_X = torch.FloatTensor().to(self.device)
        tensor_y_pred = torch.FloatTensor().to(self.device)
        tensor_y = torch.FloatTensor().to(self.device)
        tensor_y_pred_sr = torch.FloatTensor().to(self.device)

        with tqdm(total=len(self.train_loader), file=sys.stdout) as t:
            for batch_idx, (X_batch, y_batch) in enumerate(self.train_loader):
                batch_size = X_batch.size(0)
                X_batch = X_batch.to(self.device)
                tensor_X = torch.cat((tensor_X, X_batch), dim=0)
                y_batch = y_batch.to(self.device)
                tensor_y = torch.cat((tensor_y, y_batch), dim=0)

                # Forward pass
                y_pred = self.model.mn_model(X_batch)
                tensor_y_pred = torch.cat((tensor_y_pred, y_pred), dim=0)
                outputs = {"prediction_out": y_pred}

                # if we do warm-up, detach the gradient for the surrogate training
                if epoch <= self.epochs_warm_up:
                    y_hat_sr = y_pred.flatten().detach()
                else:
                    y_hat_sr = y_pred.flatten()
                tensor_y_pred_sr = torch.cat((tensor_y_pred_sr, y_hat_sr), dim=0)

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
                self.metrics_tracker.track_total_train_correct_per_epoch_per_class(
                    preds=outputs["prediction_out"], labels=y_batch
                )

                # In the final batch, calculate the APL
                if (batch_idx == len(self.train_loader) - 1):
                    APL, fid, fi, tree = self._calculate_APL(self.min_samples_leaf,
                                                             tensor_X, tensor_y_pred)
                    self.metrics_tracker.update_batch(update_dict_or_key='APL',
                                                      value=APL,
                                                      batch_size=len(self.train_loader.dataset),
                                                      mode='train')
                    self.metrics_tracker.update_batch(update_dict_or_key='fidelity',
                                                      value=fid,
                                                      batch_size=len(self.train_loader.dataset),
                                                      mode='train')
                    self.metrics_tracker.update_batch(
                        update_dict_or_key='feature_importance',
                        value=fi,
                        batch_size=len(self.train_loader.dataset),
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
                sr_loss = self.criterion_sr(input=omega, target=torch.tensor([APL], dtype=torch.float))

                # Optimise either the two losses separately in warm-up mode or the total loss
                if epoch <= self.epochs_warm_up:
                    loss = loss_label["target_loss"]
                    self.optimizer_mn.zero_grad()
                    loss.backward()
                    self.optimizer_mn.step()

                    self.optimizer_sr.zero_grad()
                    sr_loss.backward()
                    self.optimizer_sr.step()
                else:
                    if self.tree_reg_mode == 'Sequential':
                        loss = loss_label["target_loss"] + self.reg_strength * omega
                    else:
                        loss = loss_label["target_loss"] + self.reg_strength * omega + self.mse_loss_strength * sr_loss

                    # Backward pass and optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                self.metrics_tracker.update_batch(update_dict_or_key='loss',
                                                  value=loss.detach().item(),
                                                  batch_size=batch_size,
                                                  mode='train')
                t.set_postfix(
                    batch_id='{0}'.format(batch_idx + 1))
                t.update()

        # in sequential mode, train the surrogate model
        if self.tree_reg_mode == 'Sequential':
            if epoch % self.sr_training_freq == 0:
                self._train_surrogate_sequential_mode(epoch)

        if self.do_validation:
            self._valid_epoch(epoch)

        # Update the epoch metrics
        self.metrics_tracker.end_epoch()
        log = self.metrics_tracker.result_epoch()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # visualize last tree
        if epoch != 0 and (epoch % self.config['regularisation']['snapshot_epochs'] == 0
                           or epoch == self.epochs_warm_up):
            self._visualize_tree(tree=tree,
                                 config=self.config,
                                 epoch=epoch,
                                 APL=APL,
                                 train_acc='None',
                                 val_acc='None',
                                 mode='train',
                                 expert=None)

            if self.config["dataset"]["contour_plot"]:

                fig_path = str(self.config.log_dir) + '/trees/predicted_network_contour'
                if not os.path.exists(fig_path):
                    os.makedirs(fig_path)

                plot_title = f'Network Contourplot'
                fig_file_name = f'{fig_path}/fig_train_prediction-snapshot-epoch-{epoch}.png'
                tensor_X, tensor_y = tensor_X.detach().cpu().numpy(), tensor_y.detach().cpu().numpy()
                model_contour_plot(self.model.mn_model, plot_title, fig_file_name,
                                   X=tensor_X, y=tensor_y, device=self.device)

        return log

    def _valid_epoch(self, epoch):

        print(f"Validation epoch {epoch}")
        self.model.eval()

        tensor_X = torch.FloatTensor().to(self.device)
        tensor_y_pred = torch.FloatTensor().to(self.device)

        with torch.no_grad():
            with tqdm(total=len(self.val_loader), file=sys.stdout) as t:
                for batch_idx, (X_batch, y_batch) in enumerate(self.val_loader):
                    batch_size = X_batch.size(0)
                    X_batch = X_batch.to(self.device)
                    tensor_X = torch.cat((tensor_X, X_batch), dim=0)
                    y_batch = y_batch.to(self.device)

                    # Forward pass
                    y_pred = self.model.mn_model(X_batch)
                    tensor_y_pred = torch.cat((tensor_y_pred, y_pred), dim=0)
                    outputs = {"prediction_out": y_pred}

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
                    self.metrics_tracker.track_total_val_correct_per_epoch_per_class(
                        preds=outputs["prediction_out"], labels=y_batch
                    )

                    # In the final batch, calculate the APL
                    if (batch_idx == len(self.val_loader) - 1):
                        APL, fid, fi, tree = self._calculate_APL(
                            self.min_samples_leaf,
                            tensor_X, tensor_y_pred)
                        self.metrics_tracker.update_batch(
                            update_dict_or_key='APL',
                            value=APL,
                            batch_size=len(self.val_loader.dataset),
                            mode='val')
                        self.metrics_tracker.update_batch(
                            update_dict_or_key='fidelity',
                            value=fid,
                            batch_size=len(self.val_loader.dataset),
                            mode='val')
                        self.metrics_tracker.update_batch(
                            update_dict_or_key='feature_importance',
                            value=fi,
                            batch_size=len(self.val_loader.dataset),
                            mode='val')

                    loss = loss_label["target_loss"]
                    self.metrics_tracker.update_batch(update_dict_or_key='loss',
                                                      value=loss.detach().item(),
                                                      batch_size=batch_size,
                                                      mode='val')
                    t.set_postfix(
                        batch_id='{0}'.format(batch_idx + 1))
                    t.update()


    def _test(self, test_data_loader):

        self.model.eval()

        tensor_y_pred = torch.FloatTensor().to(self.device)
        tensor_y = torch.FloatTensor().to(self.device)
        test_metrics = {"loss": 0, "target_loss": 0, "accuracy": 0, "total_correct": 0}

        with torch.no_grad():
            with tqdm(total=len(test_data_loader), file=sys.stdout) as t:
                for batch_idx, (X_batch, y_batch) in enumerate(test_data_loader):

                    batch_size = X_batch.size(0)
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    tensor_y = torch.cat((tensor_y, y_batch), dim=0)

                    # Forward pass
                    y_pred = self.model.mn_model(X_batch)
                    tensor_y_pred = torch.cat((tensor_y_pred, y_pred), dim=0)
                    outputs = {"prediction_out": y_pred}

                    # Calculate Label losses
                    test_metrics["total_correct"] += get_correct(y_pred, y_batch, self.config["dataset"]["num_classes"])
                    loss_label = self.criterion(outputs, y_batch)
                    test_metrics["target_loss"] += loss_label["target_loss"].detach().cpu().item() * batch_size

                    loss = loss_label["target_loss"]
                    test_metrics["loss"] += loss.detach().cpu().item() * batch_size

                    t.set_postfix(
                        batch_id='{0}'.format(batch_idx + 1))
                    t.update()

        # Update the test metrics
        test_metrics["loss"] /= len(test_data_loader.dataset)
        test_metrics["accuracy"] = test_metrics["total_correct"] / len(test_data_loader.dataset)
        test_metrics["target_loss"] /= len(test_data_loader.dataset)

        # save test metrics in pickle
        with open(os.path.join(self.config.save_dir, f"test_metrics_xtoy.pkl"), "wb") as f:
            pickle.dump(test_metrics, f)

        # print test metrics
        print("Test Metrics:")
        print(f"Loss: {test_metrics['loss']}")
        print(f"Accuracy: {test_metrics['accuracy']}")
        print(f"Target Loss: {test_metrics['target_loss']}")

        # put also in the logger info
        self.logger.info(f"Test Metrics:")
        self.logger.info(f"Loss: {test_metrics['loss']}")
        self.logger.info(f"Accuracy: {test_metrics['accuracy']}")
        self.logger.info(f"Target Loss: {test_metrics['target_loss']}")

        # save the predictions
        tensor_y_pred = tensor_y_pred.cpu()
        tensor_y = tensor_y.cpu()
        if self.config["dataset"]["num_classes"] == 1:
            y_hat = torch.sigmoid(tensor_y_pred)
            y_hat = [1 if y_hat[i] >= 0.5 else 0 for i in range(len(y_hat))]
            correct = [1 if y_hat[i] == tensor_y[i] else 0 for i in range(len(y_hat))]
        else:
            correct = tensor_y_pred.argmax(dim=1).eq(tensor_y)
        with open(os.path.join(self.config.save_dir, f"test_pred_correct_or_not_xtoy.pkl"), "wb") as f:
            pickle.dump(correct, f)

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
