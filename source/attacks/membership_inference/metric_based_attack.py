
import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../..')
from source.utility.main_parse import save_dict_to_yaml
from scipy.stats import norm
import numpy as np
from defenses.membership_inference.loss_function import get_loss
from utils import cross_entropy, plot_phi_distribution_together
from attacks.membership_inference.attack_utils import phi_stable_batch_epsilon, likelihood_ratio
from attacks.membership_inference.membership_Inference_attack import MembershipInferenceAttack

class MetricBasedMIA(MembershipInferenceAttack):
    def __init__(
            self,
            args,
            num_class,
            device,
            attack_type,
            attack_train_dataset,
            attack_test_dataset,
            save_path,
            batch_size=128):
        # traget train load
        super().__init__()
        self.args = args
        self.num_class = num_class
        self.device = device
        self.attack_type = attack_type
        self.attack_train_dataset = attack_train_dataset
        self.attack_test_dataset = attack_test_dataset
        """
        self.attack_train_loader = torch.utils.data.DataLoader(
            attack_train_dataset, batch_size=batch_size, shuffle=True)
        self.attack_test_loader = torch.utils.data.DataLoader(
            attack_test_dataset, batch_size=batch_size, shuffle=False)
        """
        self.loss_type = args.loss_type
        self.save_path = save_path
        #self.criterion = get_loss(loss_type="ce", device=self.device, args=self.args)
        if self.attack_type == "metric-based":
            self.metric_based_attacks()
        elif self.attack_type == "white_box":
            self.white_box_grid_attacks()
        else:
            raise ValueError("Not implemented yet")

    def tuple_to_dict(self, name_list, dict):
        new_dict = {}
        ss = len(name_list)
        for key, value_tuple in dict.items():
            for i in range(ss):
                new_key = key + name_list[i]
                new_dict[new_key] = float(value_tuple[i])

        return new_dict

    def white_box_grid_attacks(self):

        self.parse_data_white_box_attacks()
        names = ['l1', 'l2', 'Min', 'Max', 'Mean', 'Skewness', 'Kurtosis']
        # self.s_tr_conf[] = self.attack_train_dataset
        name_list = ["acc", "precision", "recall", "f1", "auc"]
        target_train = self.attack_test_dataset[0]["target_train_data"]
        target_test = self.attack_test_dataset[0]["target_test_data"]
        shadow_train = self.attack_train_dataset[0]["shadow_train_data"]
        shadow_test = self.attack_train_dataset[0]["shadow_test_data"]

        wb_dict = {}

        for name in names:
            train_tuple_x, test_tuple_x, _ = self._mem_inf_thre(
                f"{name} ", -shadow_train[0][name], -shadow_test[0][name], -target_train[0][name],
                -target_test[0][name])
            train_tuple_w, test_tuple_w, _ = self._mem_inf_thre(
                f"{name} ", -shadow_train[1][name], -shadow_test[1][name], -target_train[1][name],
                -target_test[1][name])

            for i in range(len(name_list)):
                key1 = f"grid_x_{name}_train_{name_list[i]}"
                key2 = f"grid_x_{name}_test_{name_list[i]}"
                key3 = f"grid_w_{name}_train_{name_list[i]}"
                key4 = f"grid_w_{name}_test_{name_list[i]}"
                wb_dict[key1] = float(train_tuple_x[i])
                wb_dict[key2] = float(test_tuple_x[i])
                wb_dict[key3] = float(train_tuple_w[i])
                wb_dict[key4] = float(test_tuple_w[i])
                self.print_result(f"{key1}", train_tuple_x)
                self.print_result(f"{key2}", test_tuple_x)
                self.print_result(f"{key3}", train_tuple_w)
                self.print_result(f"{key4}", test_tuple_w)

        save_dict_to_yaml(wb_dict, f"{self.save_path}/white_box_grid_attacks.yaml")

    def metric_based_attacks(self):
        """
        a little bit redundant since we make the data into torch dataset,
        but reverse them back into the original data...
        """
        self.parse_data_metric_based_attacks()

        train_tuple0, test_tuple0, test_results0 = self._mem_inf_via_corr()
        self.print_result("correct train", train_tuple0)
        self.print_result("correct test", test_tuple0)

        train_tuple1, test_tuple1, test_results1 = self._mem_inf_thre(
            'confidence', self.s_tr_conf, self.s_te_conf, self.t_tr_conf, self.t_te_conf)
        self.print_result("confidence train", train_tuple1)
        self.print_result("confidence test", test_tuple1)

        train_tuple2, test_tuple2, test_results2 = self._mem_inf_thre(
            'entropy', -self.s_tr_entr, -self.s_te_entr, -self.t_tr_entr, -self.t_te_entr)
        self.print_result("entropy train", train_tuple2)
        self.print_result("entropy test", test_tuple2)

        train_tuple3, test_tuple3, test_results3 = self._mem_inf_thre(
            'modified entropy', -self.s_tr_m_entr, -self.s_te_m_entr, -self.t_tr_m_entr, -self.t_te_m_entr)
        self.print_result("modified entropy train", train_tuple3)
        self.print_result("modified entropy test", test_tuple3)

        train_tuple4, test_tuple4, test_results4 = self._mem_inf_thre(
            'cross entropy loss', -self.shadow_train_celoss, -self.shadow_test_celoss, -self.target_train_celoss,
            -self.target_test_celoss)
        self.print_result("cross entropy loss train", train_tuple4)
        self.print_result("cross entropy loss test", test_tuple4)

        train_tuple5, test_tuple5, test_results5 = self._mem_inf_thre(
            "likelihood ratio", self.shadow_train_likelihood_ratio, self.shadow_test_likelihood_ratio,
            self.target_train_likelihood_ratio, self.target_test_likelihood_ratio)
        self.print_result("likelihood ratio train", train_tuple5)
        self.print_result("likelihood ratio test", test_tuple5)

        mia_dict = {"correct train": train_tuple0,
                    "correct test": test_tuple0,
                    "confidence train": train_tuple1,
                    "confidence test": test_tuple1,
                    "entropy train": train_tuple2,
                    "entropy test": test_tuple2,
                    "modified entropy train": train_tuple3,
                    "modified entropy test": test_tuple3,
                    "cross entropy loss train": train_tuple4,
                    "cross entropy loss test": test_tuple4,
                    "likelihood ratio train": train_tuple5,
                    "likelihood ratio test": test_tuple5
                    }
        name_list = [" acc", " precision", " recall", " f1", " auc"]
        # print(aa)
        # print(aa["confidence test acc"])

        # print(type(aa["confidence test acc"]))
        # exit()
        save_dict_to_yaml(self.tuple_to_dict(name_list, mia_dict), f"{self.save_path}/mia_metric_based.yaml")

        if self.args.plot_distribution:
            plot_phi_distribution_together(self.phi_target_train, self.phi_target_test, self.save_path)
            plot_phi_distribution_together(self.phi_shadow_train, self.phi_shadow_test, self.save_path,
                                           "phi_distribution_shadow_comparison")
            
            #plot_celoss_distribution_together()
        # mia_dict # phi_distribution_shadow_comparison

    def print_result(self, name, given_tuple):
        print("%s" % name, "acc:%.3f, precision:%.3f, recall:%.3f, f1:%.3f, auc:%.3f" % given_tuple)

    def parse_data_white_box_attacks(self):
        # shadow model
        # For train set of shadow medel, we query shadow model, then obtain the outputs, that is **s_tr_outputs**
        self.s_tr_labels = np.array(self.attack_train_dataset[2]["shadow_train_label"])
        self.s_te_labels = np.array(self.attack_train_dataset[2]["shadow_test_label"])

        self.t_tr_labels = np.array(self.attack_test_dataset[2]["target_train_label"])
        self.t_te_labels = np.array(self.attack_test_dataset[2]["target_train_label"])

        self.s_tr_mem_labels = np.ones(len(self.s_tr_labels))
        self.s_te_mem_labels = np.zeros(len(self.s_te_labels))
        self.t_tr_mem_labels = np.ones(len(self.t_tr_labels))
        self.t_te_mem_labels = np.zeros(len(self.t_te_labels))

    def parse_data_metric_based_attacks(self):
        # shadow model
        # For train set of shadow medel, we query shadow model, then obtain the outputs, that is **s_tr_outputs**
        self.s_tr_outputs, self.s_tr_labels = [], []
        self.s_te_outputs, self.s_te_labels = [], []

        for i in range(len(self.attack_train_dataset)):
            # mem_label: the data is a member or not
            data, mem_label, target_label = self.attack_train_dataset[i]
            data, mem_label, target_label = data.numpy(), mem_label.item(), target_label.item()

            if mem_label == 1:
                self.s_tr_outputs.append(data)
                self.s_tr_labels.append(target_label)
            elif mem_label == 0:
                self.s_te_outputs.append(data)
                self.s_te_labels.append(target_label)

        # target model
        self.t_tr_outputs, self.t_tr_labels = [], []
        self.t_te_outputs, self.t_te_labels = [], []
        for i in range(len(self.attack_test_dataset)):
            data, mem_label, target_label = self.attack_test_dataset[i]
            data, mem_label, target_label = data.numpy(), mem_label.item(), target_label.item()
            if mem_label == 1:
                self.t_tr_outputs.append(data)
                self.t_tr_labels.append(target_label)
            elif mem_label == 0:
                self.t_te_outputs.append(data)
                self.t_te_labels.append(target_label)

        # change them into numpy array
        self.s_tr_outputs, self.s_tr_labels = np.array(
            self.s_tr_outputs), np.array(self.s_tr_labels)
        # print(self.s_tr_outputs.shape)(15000, 10)
        # print(self.s_tr_labels.shape) (15000,)

        self.s_te_outputs, self.s_te_labels = np.array(
            self.s_te_outputs), np.array(self.s_te_labels)
        self.t_tr_outputs, self.t_tr_labels = np.array(
            self.t_tr_outputs), np.array(self.t_tr_labels)
        self.t_te_outputs, self.t_te_labels = np.array(
            self.t_te_outputs), np.array(self.t_te_labels)

        self.s_tr_mem_labels = np.ones(len(self.s_tr_labels))
        self.s_te_mem_labels = np.zeros(len(self.s_te_labels))
        self.t_tr_mem_labels = np.ones(len(self.t_tr_labels))
        self.t_te_mem_labels = np.zeros(len(self.t_te_labels))

        # prediction correctness
        self.s_tr_corr = (np.argmax(self.s_tr_outputs, axis=1)
                          == self.s_tr_labels).astype(int)
        self.s_te_corr = (np.argmax(self.s_te_outputs, axis=1)
                          == self.s_te_labels).astype(int)
        self.t_tr_corr = (np.argmax(self.t_tr_outputs, axis=1)
                          == self.t_tr_labels).astype(int)
        self.t_te_corr = (np.argmax(self.t_te_outputs, axis=1)
                          == self.t_te_labels).astype(int)

        # prediction confidence
        self.s_tr_conf = np.max(self.s_tr_outputs, axis=1)
        self.s_te_conf = np.max(self.s_te_outputs, axis=1)
        self.t_tr_conf = np.max(self.t_tr_outputs, axis=1)
        self.t_te_conf = np.max(self.t_te_outputs, axis=1)

        # prediction entropy
        self.s_tr_entr = self._entr_comp(self.s_tr_outputs)
        self.s_te_entr = self._entr_comp(self.s_te_outputs)
        self.t_tr_entr = self._entr_comp(self.t_tr_outputs)
        self.t_te_entr = self._entr_comp(self.t_te_outputs)

        # prediction modified entropy
        self.s_tr_m_entr = self._m_entr_comp(
            self.s_tr_outputs, self.s_tr_labels)
        self.s_te_m_entr = self._m_entr_comp(
            self.s_te_outputs, self.s_te_labels)
        self.t_tr_m_entr = self._m_entr_comp(
            self.t_tr_outputs, self.t_tr_labels)
        self.t_te_m_entr = self._m_entr_comp(
            self.t_te_outputs, self.t_te_labels)

        # cross entropy loss

        self.shadow_train_celoss = cross_entropy(self.s_tr_outputs, self.s_tr_labels)
        self.shadow_test_celoss = cross_entropy(self.s_te_outputs, self.s_te_labels)
        self.target_train_celoss = cross_entropy(self.t_tr_outputs, self.t_tr_labels)
        self.target_test_celoss = cross_entropy(self.t_te_outputs, self.t_te_labels)

        # likelihood ratio attack
        self._likelihood_ratio_data()

    def _log_value(self, probs, small_value=1e-30):
        return -np.log(np.maximum(probs, small_value))

    def _entr_comp(self, probs):
        return np.sum(np.multiply(probs, self._log_value(probs)), axis=1)

    def _m_entr_comp(self, probs, true_labels):
        log_probs = self._log_value(probs)
        reverse_probs = 1 - probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(
            true_labels.size), true_labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(
            true_labels.size), true_labels]
        return np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)


    def _likelihood_ratio_data(self):
        self.phi_shadow_train = phi_stable_batch_epsilon(self.s_tr_outputs, self.s_tr_labels)
        self.phi_shadow_test = phi_stable_batch_epsilon(self.s_te_outputs, self.s_te_labels)
        self.phi_target_train = phi_stable_batch_epsilon(self.t_tr_outputs, self.t_tr_labels)
        self.phi_target_test = phi_stable_batch_epsilon(self.t_te_outputs, self.t_te_labels)

        mean_shadow_train = np.mean(self.phi_shadow_train)
        sigma_shadow_train = np.sqrt(np.var(self.phi_shadow_train))
        mean_shadow_test = np.mean(self.phi_shadow_test)
        sigma_target_test = np.sqrt(np.var(self.phi_shadow_test))

        self.shadow_train_likelihood_ratio = likelihood_ratio(self.phi_shadow_train, mean_shadow_train,
                                                                   sigma_shadow_train, mean_shadow_test,
                                                                   sigma_target_test)
        self.shadow_test_likelihood_ratio = likelihood_ratio(self.phi_shadow_test, mean_shadow_train,
                                                                  sigma_shadow_train, mean_shadow_test,
                                                                  sigma_target_test)
        self.target_train_likelihood_ratio = likelihood_ratio(self.phi_target_train, mean_shadow_train,
                                                                   sigma_shadow_train, mean_shadow_test,
                                                                   sigma_target_test)
        self.target_test_likelihood_ratio = likelihood_ratio(self.phi_target_test, mean_shadow_train,  sigma_shadow_train, mean_shadow_test,
                                                                  sigma_target_test)

    def _thre_setting(self, tr_values, te_values):
        value_list = np.concatenate((tr_values, te_values))
        thre, max_acc = 0, 0
        for value in value_list:
            tr_ratio = np.sum(tr_values >= value) / (len(tr_values) + 0.0)
            te_ratio = np.sum(te_values < value) / (len(te_values) + 0.0)
            acc = 0.5 * (tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
        return thre

    def _mem_inf_via_corr(self):
        # # perform membership inference attack based on whether the input is correctly classified or not
        train_mem_label = np.concatenate(
            [self.s_tr_mem_labels, self.s_te_mem_labels], axis=-1)
        train_pred_label = np.concatenate(
            [self.s_tr_corr, self.s_te_corr], axis=-1)
        train_pred_posteriors = np.concatenate(
            [self.s_tr_corr, self.s_te_corr], axis=-1)  # same as train_pred_label
        train_target_label = np.concatenate(
            [self.s_tr_labels, self.s_te_labels], axis=-1)

        test_mem_label = np.concatenate(
            [self.t_tr_mem_labels, self.t_te_mem_labels], axis=-1)
        test_pred_label = np.concatenate(
            [self.t_tr_corr, self.t_te_corr], axis=-1)
        test_pred_posteriors = np.concatenate(
            [self.t_tr_corr, self.t_te_corr], axis=-1)  # same as train_pred_label
        test_target_label = np.concatenate(
            [self.t_tr_labels, self.t_te_labels], axis=-1)

        train_acc, train_precision, train_recall, train_f1, train_auc = super().cal_metrics(
            train_mem_label, train_pred_label, train_pred_posteriors)
        test_acc, test_precision, test_recall, test_f1, test_auc = super().cal_metrics(
            test_mem_label, test_pred_label, test_pred_posteriors)

        test_results = {"test_mem_label": test_mem_label,
                        "test_pred_label": test_pred_label,
                        "test_pred_prob": test_pred_posteriors,
                        "test_target_label": test_target_label}

        train_tuple = (train_acc, train_precision,
                       train_recall, train_f1, train_auc)
        test_tuple = (test_acc, test_precision,
                      test_recall, test_f1, test_auc)
        # print(train_tuple, test_tuple)
        return train_tuple, test_tuple, test_results

    def _mem_inf_thre(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values):

        # s_tr_values :15000
        # perform membership inference attack by thresholding feature values: the feature can be prediction confidence,
        # (negative) prediction entropy, and (negative) modified entropy

        train_mem_label = []
        # for shadow train label, it is 1, for test sample, it is 0.
        train_pred_label = []
        # by metric based pred, label is 0 or 1
        train_pred_posteriors = []

        train_target_label = []

        test_mem_label = []
        test_pred_label = []
        test_pred_posteriors = []
        test_target_label = []

        thre_list = [self._thre_setting(s_tr_values[self.s_tr_labels == num],
                                        s_te_values[self.s_te_labels == num]) for num in range(self.num_class)]
        # s_tr_values shadow train set value. 15000
        # len(self.s_tr_labels) = 15000
        # len(s_tr_values[self.s_tr_labels == 1]) =1540
        # thre_list  10

        # shadow train
        for i in range(len(s_tr_values)):
            original_label = self.s_tr_labels[i]
            thre = thre_list[original_label]
            pred = s_tr_values[i]
            pred_label = int(s_tr_values[i] >= thre)

            train_mem_label.append(1)
            train_pred_label.append(pred_label)

            # indicator function, so the posterior equals to 0 or 1
            train_pred_posteriors.append(pred)
            train_target_label.append(original_label)

        # shadow test
        for i in range(len(s_te_values)):
            original_label = self.s_te_labels[i]
            thre = thre_list[original_label]
            # choose the threshold for class
            pred = s_te_values[i]
            pred_label = int(s_te_values[i] >= thre)

            train_mem_label.append(0)
            # in test, no sample is  member, so that is 0
            train_pred_label.append(pred_label)
            # indicator function, so the posterior equals to 0 or 1
            train_pred_posteriors.append(pred)
            train_target_label.append(original_label)

        # target train
        for i in range(len(t_tr_values)):
            original_label = self.t_tr_labels[i]
            thre = thre_list[original_label]
            pred = t_tr_values[i]
            pred_label = int(t_tr_values[i] >= thre)

            test_mem_label.append(1)
            test_pred_label.append(pred_label)
            # indicator function, so the posterior equals to 0 or 1
            test_pred_posteriors.append(pred)
            test_target_label.append(original_label)

        # target test
        for i in range(len(t_te_values)):
            original_label = self.t_te_labels[i]
            thre = thre_list[original_label]
            pred = t_te_values[i]
            pred_label = int(t_te_values[i] >= thre)

            test_mem_label.append(0)
            test_pred_label.append(pred_label)
            # indicator function, so the posterior equals to 0 or 1
            test_pred_posteriors.append(pred)
            test_target_label.append(original_label)

        train_acc, train_precision, train_recall, train_f1, train_auc = super().cal_metrics(
            train_mem_label, train_pred_label, train_pred_posteriors)
        test_acc, test_precision, test_recall, test_f1, test_auc = super().cal_metrics(
            test_mem_label, test_pred_label, test_pred_posteriors)

        train_tuple = (train_acc, train_precision,
                       train_recall, train_f1, train_auc)
        test_tuple = (test_acc, test_precision,
                      test_recall, test_f1, test_auc)
        test_results = {"test_mem_label": test_mem_label,
                        "test_pred_label": test_pred_label,
                        "test_pred_prob": test_pred_posteriors,
                        "test_target_label": test_target_label}

        return train_tuple, test_tuple, test_results