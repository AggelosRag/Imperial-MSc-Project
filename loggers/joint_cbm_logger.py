import os.path
import time

from torch.utils.tensorboard import SummaryWriter

from utils import *
import utils


class JointCBMLogger:
    def __init__(self, config, iteration, tb_path, output_path, train_loader,
                 val_loader, device=None):
        """
        Initialized each parameters of each run.
        """
        self.iteration = iteration
        if iteration is not None:
            self.tb_path = tb_path + '/joint_cbm_logger_expert_' + str(iteration)
            self.output_path = output_path + '/joint_cbm_model_expert_' + str(iteration)
        else:
            self.tb_path = tb_path + '/joint_cbm_logger'
            self.output_path = output_path + '/joint_cbm_model'
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epoch_id = 0
        self.best_epoch_id = 0

        self.n_classes = config['dataset']['num_classes']
        self.n_concepts = config['dataset']['num_concepts']

        self.run_id = 0
        self.run_data = []
        self.run_start_time = None
        self.epoch_duration = None

        self.tb = None
        self.val_best_accuracy = 0
        self.best_val_loss = 1000000
        self.val_auroc = None
        
        self.attributes_per_epoch = {
            "train_loss": 0,
            "val_loss": 0,
            "train_target_loss": 0,
            "train_concept_loss": 0,
            "val_target_loss": 0,
            "val_concept_loss": 0,
            "train_total_correct": 0,
            "val_total_correct": 0,
            "train_selective_loss": 0,
            "train_emp_coverage": 0,
            "train_CE_risk": 0,
            "train_emp_risk": 0,
            "train_cov_penalty": 0,
            "train_aux_loss": 0,
            "val_selective_loss": 0,
            "val_emp_coverage": 0,
            "val_CE_risk": 0,
            "val_emp_risk": 0,
            "val_cov_penalty": 0,
            "val_aux_loss": 0,
            "train_APL": 0,
            "val_APL": 0,
            "train_fidelity": 0,
            "val_fidelity": 0,
            "train_APL_predictions": 0,
            "val_APL_predictions": 0,
        }
        # "train_n_selected": 0,
        # "train_n_rejected": 0,
        # "train_coverage": 0,
        self.train_accuracy = None
        self.val_accuracy = None
        self.val_correct_accuracy = 0
        self.val_incorrect_accuracy = 0
        self.val_correct = 0
        self.val_n_selected = 0
        self.val_n_rejected = 0
        self.val_coverage = 0

        self.tensor_attributes_per_epoch = {
            "val_out_put_sel_proba": torch.FloatTensor().to(self.device),
            "val_out_put_class": torch.FloatTensor().to(self.device),
            "val_out_put_target": torch.FloatTensor().to(self.device),
        }
        self.list_attributes_per_epoch = {
            "train_loss_per_concept": np.zeros(self.n_concepts),
            "val_loss_per_concept": np.zeros(self.n_concepts),
            "train_feature_importance": np.zeros(self.n_concepts),
            "val_feature_importance": np.zeros(self.n_concepts),
        }

        self.all_epoch_attributes = {key: [] for key in self.attributes_per_epoch}
        self.all_epoch_attributes["train_accuracy"] = []
        self.all_epoch_attributes["val_accuracy"] = []

        self.all_epoch_list_attributes = {key: [] for key in self.list_attributes_per_epoch}
        
    def begin_run(self):
        """
        Records all the parameters at the start of each run.

        :param run:

        :return: none
        """
        self.run_start_time = time.time()

        self.run_id += 1
        self.tb = SummaryWriter(f"{self.tb_path}")
        # print("################## TB Log path ###################")
        # print(f"{self.tb_path}")
        # print("################## TB Log path ###################")
    
    def end_run(self):
        """
        Records all the parameters at the end of each run.

        :return: none
        """
        self.tb.close()
        self.epoch_id = 0
        
    def begin_epoch(self):
        for key in self.attributes_per_epoch:
            self.attributes_per_epoch[key] = 0
        for key in self.tensor_attributes_per_epoch:
            self.tensor_attributes_per_epoch[key] = torch.FloatTensor().to(self.device)
        for key in self.list_attributes_per_epoch:
            self.list_attributes_per_epoch[key] = np.zeros(self.n_concepts)

        self.train_accuracy = None
        self.val_accuracy = None
        self.val_correct_accuracy = 0
        self.val_incorrect_accuracy = 0
        self.val_correct = 0
        self.val_n_selected = 0
        self.val_n_rejected = 0
        self.val_coverage = 0

        self.epoch_id += 1

    def update_batch(self, update_dict_or_key, batch_size, value=None, mode='train'):
        prefix = "train_" if mode == 'train' else "val_"
        if isinstance(update_dict_or_key, dict):
            for key, val in update_dict_or_key.items():
                full_key = prefix + key
                if full_key in self.attributes_per_epoch:
                    val = val.detach().item()
                    self.attributes_per_epoch[full_key] += val * batch_size
                elif full_key in self.tensor_attributes_per_epoch:
                    val = val.detach()
                    self.tensor_attributes_per_epoch[full_key] = torch.cat((self.tensor_attributes_per_epoch[full_key], val), dim=0)
                elif full_key in self.list_attributes_per_epoch:
                    val = val.detach()
                    self.list_attributes_per_epoch[full_key] += np.array([x * batch_size for x in val])
        elif isinstance(update_dict_or_key, str) and value is not None:
            full_key = prefix + update_dict_or_key
            if full_key in self.attributes_per_epoch:
                self.attributes_per_epoch[full_key] += value * batch_size
            elif full_key in self.tensor_attributes_per_epoch:
                self.tensor_attributes_per_epoch[full_key] = torch.cat((self.tensor_attributes_per_epoch[full_key], value), dim=0)
            elif full_key in self.list_attributes_per_epoch:
                self.list_attributes_per_epoch[full_key] += np.array([x * batch_size for x in value])
        else:
            raise ValueError("Invalid input: expected a dictionary or a key-value pair")

    def end_epoch(self, selectivenet=False):

        # for multiclass classification
        self.train_accuracy = (self.attributes_per_epoch['train_total_correct'] /
                               len(self.train_loader.dataset)) * 100
        self.val_accuracy = (self.attributes_per_epoch['val_total_correct'] /
                             len(self.val_loader.dataset)) * 100
        self.all_epoch_attributes["train_accuracy"].append(self.train_accuracy)
        self.all_epoch_attributes["val_accuracy"].append(self.val_accuracy)

        for key in self.list_attributes_per_epoch:
            if key.startswith("train_"):
                dataset_length = len(self.train_loader.dataset)
            elif key.startswith("val_"):
                dataset_length = len(self.val_loader.dataset)
            else:
                raise ValueError("Invalid key")
            self.list_attributes_per_epoch[key] = self.list_attributes_per_epoch[key]/dataset_length
            self.all_epoch_list_attributes[key].append((self.list_attributes_per_epoch[key]).tolist())
        for key in self.attributes_per_epoch:
            if key.startswith("train_"):
                dataset_length = len(self.train_loader.dataset)
            elif key.startswith("val_"):
                dataset_length = len(self.val_loader.dataset)
            else:
                raise ValueError("Invalid key: ", key)
            self.attributes_per_epoch[key] /= dataset_length
            self.all_epoch_attributes[key].append(self.attributes_per_epoch[key])

        self.tb.add_scalar("Epoch_stats_model/Train_accuracy", self.train_accuracy, self.epoch_id)
        self.tb.add_scalar("Epoch_stats_model/Val_accuracy", self.val_accuracy, self.epoch_id)

        self.tb.add_scalar("Epoch_Loss/Train_Loss", self.attributes_per_epoch['train_loss'], self.epoch_id)
        self.tb.add_scalar("Epoch_Loss/Val_Loss", self.attributes_per_epoch['val_loss'], self.epoch_id)

        self.tb.add_scalar("Epoch_stats_model/Train_APL", self.attributes_per_epoch['train_APL'], self.epoch_id)
        self.tb.add_scalar("Epoch_stats_model/Val_APL", self.attributes_per_epoch['val_APL'], self.epoch_id)

        self.tb.add_scalar("Epoch_stats_model/Train_Fidelity", self.attributes_per_epoch['train_fidelity'], self.epoch_id)
        self.tb.add_scalar("Epoch_stats_model/Val_Fidelity", self.attributes_per_epoch['val_fidelity'], self.epoch_id)

        self.tb.add_scalar("Epoch_stats_model/Train_APL_predictions", self.attributes_per_epoch['train_APL_predictions'], self.epoch_id)
        self.tb.add_scalar("Epoch_stats_model/Val_APL_predictions", self.attributes_per_epoch['val_APL_predictions'], self.epoch_id)

        if selectivenet:
            self.track_selectivenet_stats()

    def track_selectivenet_stats(self):

        self.tb.add_scalar("Loss_train/Empirical_Coverage ",
                           self.attributes_per_epoch["train_emp_coverage"], self.epoch_id)
        self.tb.add_scalar("Loss_train/CE_Risk",
                           self.attributes_per_epoch["train_CE_risk"], self.epoch_id)
        self.tb.add_scalar("Loss_train/Emp_Risk (KD + Entropy)",
                           self.attributes_per_epoch["train_emp_risk"], self.epoch_id)
        self.tb.add_scalar("Loss_train/Cov_Penalty",
                           self.attributes_per_epoch["train_cov_penalty"], self.epoch_id)
        self.tb.add_scalar(
            "Loss_train/Selective_Loss (Emp + Cov)",
            self.attributes_per_epoch["train_selective_loss"], self.epoch_id
        )
        self.tb.add_scalar("Loss_train/Aux_Loss",
                           self.attributes_per_epoch["train_aux_loss"], self.epoch_id)
        self.tb.add_scalar("Loss_val/Empirical_Coverage ",
                           self.attributes_per_epoch["val_emp_coverage"], self.epoch_id)
        self.tb.add_scalar("Loss_val/CE_Risk",
                           self.attributes_per_epoch["val_CE_risk"], self.epoch_id)
        self.tb.add_scalar("Loss_val/Emp_Risk (KD + Entropy)",
                           self.attributes_per_epoch["val_emp_risk"], self.epoch_id)
        self.tb.add_scalar("Loss_val/Cov_Penalty",
                           self.attributes_per_epoch["val_cov_penalty"], self.epoch_id)
        self.tb.add_scalar(
            "Loss_val/Selective_Loss (Emp + Cov)",
            self.attributes_per_epoch["val_selective_loss"], self.epoch_id
        )
        self.tb.add_scalar("Loss_val/Aux_Loss",
                           self.attributes_per_epoch["val_aux_loss"], self.epoch_id)

        self.tb.add_scalar(
            "Epoch_stats/Accuracy_Correctly_Selected (pi >= 0.5)",
            self.val_correct_accuracy, self.epoch_id
        )
        self.tb.add_scalar(
            "Epoch_stats/Accuracy_Correctly_Rejected (pi < 0.5)",
            self.val_incorrect_accuracy, self.epoch_id
        )

        self.tb.add_scalar("Val_Pi_stats/N_Selected", self.val_n_selected,
                           self.epoch_id)
        self.tb.add_scalar("Val_Pi_stats/N_Rejected", self.val_n_rejected,
                           self.epoch_id)
        self.tb.add_scalar("Val_Pi_stats/coverage", self.val_coverage,
                           self.epoch_id)

    def __str__(self):
        return f"Attributes per epoch: {self.attributes_per_epoch}\nTensor Attributes per epoch: {self.tensor_attributes_per_epoch}\nList Attributes per epoch: {self.list_attributes_per_epoch}"

    def evaluate_correctly(self, selection_threshold):
        prediction_result = self.tensor_attributes_per_epoch["val_out_put_class"].argmax(dim=1)
        selection_result = None
        if self.tensor_attributes_per_epoch["val_out_put_sel_proba"] is not None:
            condition = self.get_correct_condition_for_selection(selection_threshold)
            selection_result = torch.where(
                condition,
                torch.ones_like(self.tensor_attributes_per_epoch["val_out_put_sel_proba"]),
                torch.zeros_like(self.tensor_attributes_per_epoch["val_out_put_sel_proba"]),
            ).view(-1)

        h_rjc = torch.masked_select(prediction_result, selection_result.bool())
        t_rjc = torch.masked_select(self.tensor_attributes_per_epoch["val_out_put_target"], selection_result.bool())
        t = float(torch.where(h_rjc == t_rjc, torch.ones_like(h_rjc), torch.zeros_like(h_rjc)).sum())
        f = float(torch.where(h_rjc != t_rjc, torch.ones_like(h_rjc), torch.zeros_like(h_rjc)).sum())

        acc = float(t / (t + f + 1e-12)) * 100
        self.val_correct = t
        self.val_correct_accuracy = acc
        
    def get_correct_condition_for_selection(self, selection_threshold):
        return self.tensor_attributes_per_epoch["val_out_put_sel_proba"] >= selection_threshold
    
    def evaluate_incorrectly(self, selection_threshold):
        prediction_result = self.tensor_attributes_per_epoch["val_out_put_class"].argmax(dim=1)
        selection_result = None
        if self.tensor_attributes_per_epoch["val_out_put_sel_proba"] is not None:
            condition = self.get_incorrect_condition_for_selection(selection_threshold)
            selection_result = torch.where(
                condition,
                torch.ones_like(self.tensor_attributes_per_epoch["val_out_put_sel_proba"]),
                torch.zeros_like(self.tensor_attributes_per_epoch["val_out_put_sel_proba"]),
            ).view(-1)
        h_rjc = torch.masked_select(prediction_result, selection_result.bool())
        t_rjc = torch.masked_select(self.tensor_attributes_per_epoch["val_out_put_target"], selection_result.bool())
        t = float(torch.where(h_rjc == t_rjc, torch.ones_like(h_rjc), torch.zeros_like(h_rjc)).sum())
        f = float(torch.where(h_rjc != t_rjc, torch.ones_like(h_rjc), torch.zeros_like(h_rjc)).sum())

        acc = float(t / (t + f + 1e-12)) * 100
        self.val_incorrect_accuracy = acc

    def get_incorrect_condition_for_selection(self, selection_threshold):
        return self.tensor_attributes_per_epoch["val_out_put_sel_proba"] < selection_threshold

    def evaluate_correctly_auroc(self, selection_threshold):
        prediction_result = self.tensor_attributes_per_epoch["val_out_put_class"].argmax(dim=1)
        selection_result = None
        if self.tensor_attributes_per_epoch["val_out_put_sel_proba"] is not None:
            condition = self.get_correct_condition_for_selection(
                selection_threshold
            )
            selection_result = torch.where(
                condition,
                torch.ones_like(self.tensor_attributes_per_epoch["val_out_put_sel_proba"]),
                torch.zeros_like(self.tensor_attributes_per_epoch["val_out_put_sel_proba"]),
            ).view(-1)

        h_rjc = torch.masked_select(prediction_result, selection_result.bool())
        t_rjc = torch.masked_select(self.tensor_attributes_per_epoch["val_out_put_target"], selection_result.bool())
        t = float(torch.where(h_rjc == t_rjc, torch.ones_like(h_rjc), torch.zeros_like(h_rjc)).sum())
        f = float(torch.where(h_rjc != t_rjc, torch.ones_like(h_rjc), torch.zeros_like(h_rjc)).sum())

        acc = float(t / (t + f + 1e-12)) * 100
        self.val_correct = t
        self.val_correct_accuracy = acc

        s = selection_result.view(-1, 1)
        sel = torch.cat((s, s), dim=1)
        h_rjc = torch.masked_select(self.tensor_attributes_per_epoch["val_out_put_class"], sel.bool()).view(-1, 2)
        proba = torch.nn.Softmax()(h_rjc)
        t_rjc = torch.masked_select(self.tensor_attributes_per_epoch["val_out_put_target"], selection_result.bool())
        val_auroc, _ = utils.compute_AUC(gt=t_rjc, pred=proba[:, 1])
        self.val_auroc = val_auroc

    def evaluate_coverage_stats(self, selection_threshold):
        prediction_result = self.tensor_attributes_per_epoch["val_out_put_class"].argmax(dim=1)
        selection_result = None
        if self.tensor_attributes_per_epoch["val_out_put_sel_proba"] is not None:
            condition = self.get_correct_condition_for_selection(
                selection_threshold
            )
            selection_result = torch.where(
                condition,
                torch.ones_like(self.tensor_attributes_per_epoch["val_out_put_sel_proba"]),
                torch.zeros_like(self.tensor_attributes_per_epoch["val_out_put_sel_proba"]),
            ).view(-1)
        condition_true = prediction_result == self.tensor_attributes_per_epoch["val_out_put_target"]
        condition_false = prediction_result != self.tensor_attributes_per_epoch["val_out_put_target"]
        condition_acc = selection_result == torch.ones_like(selection_result)
        condition_rjc = selection_result == torch.zeros_like(selection_result)
        ta = float(
            torch.where(
                condition_true & condition_acc,
                torch.ones_like(prediction_result),
                torch.zeros_like(prediction_result),
            ).sum()
        )
        tr = float(
            torch.where(
                condition_true & condition_rjc,
                torch.ones_like(prediction_result),
                torch.zeros_like(prediction_result),
            ).sum()
        )
        fa = float(
            torch.where(
                condition_false & condition_acc,
                torch.ones_like(prediction_result),
                torch.zeros_like(prediction_result),
            ).sum()
        )
        fr = float(
            torch.where(
                condition_false & condition_rjc,
                torch.ones_like(prediction_result),
                torch.zeros_like(prediction_result),
            ).sum()
        )

        rejection_rate = float((tr + fr) / (ta + tr + fa + fr + 1e-12))

        # rejection precision - not used in our code
        rejection_pre = float(tr / (tr + fr + 1e-12))

        self.val_n_rejected = tr + fr

        self.val_n_selected = len(self.val_loader.dataset) - (tr + fr)
        self.val_coverage = (1 - rejection_rate)

    def track_total_train_correct_per_epoch(self, preds, labels):
        """
        Calculates the correct prediction at the each iteration of batch.

        :param preds: predicted labels
        :param labels: true labels

        :return: the totalcorrect prediction at the each iteration of batch
        """
        self.attributes_per_epoch["train_total_correct"] += get_correct(preds, labels, self.n_classes)

    def track_total_val_correct_per_epoch(self, preds, labels):
        """
        Calculates the correct prediction at the each iteration of batch.

        :param preds: predicted labels
        :param labels: true labels

        :return: the totalcorrect prediction at the each iteration of batch
        """
        self.attributes_per_epoch["val_total_correct"] += get_correct(preds, labels, self.n_classes)

    def result(self):
        performance_dict = {**self.all_epoch_list_attributes, **self.all_epoch_attributes}
        # Add the values to the dictionary
        performance_df = pd.DataFrame(
            dict([(col_name, pd.Series(values)) for col_name, values in performance_dict.items()])
        )
        performance_df.to_csv(os.path.join(self.tb_path, "train_val_stats") + ".csv")
        return performance_dict

    def result_epoch(self):
        performance_dict = {**self.attributes_per_epoch,
                            **self.list_attributes_per_epoch}
        # Add the values to the dictionary
        performance_dict['train_accuracy'] = self.train_accuracy
        performance_dict['val_accuracy'] = self.val_accuracy
        performance_dict['val_correct_accuracy'] = self.val_correct_accuracy
        performance_dict['val_incorrect_accuracy'] = self.val_incorrect_accuracy
        performance_dict['val_correct'] = self.val_correct
        performance_dict['val_n_selected'] = self.val_n_selected
        performance_dict['val_n_rejected'] = self.val_n_rejected
        performance_dict['val_coverage'] = self.val_coverage
        return performance_dict
