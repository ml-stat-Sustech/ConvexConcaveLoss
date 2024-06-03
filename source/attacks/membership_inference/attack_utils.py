import numpy as np
import torch
from scipy.stats import norm
from source.utility.main_parse import save_dict_to_yaml

def phi_stable_batch_epsilon( posterior_probs, labels, epsilon=1e-10):
    posterior_probs = posterior_probs + epsilon
    one_hot_labels = np.zeros_like(posterior_probs)

    one_hot_labels[np.arange(len(labels)), labels] = 1

    # Calculate the log likelihood for the correct labels
    log_likelihood_correct = np.log(posterior_probs[np.arange(len(labels)), labels])

    # Calculate the sum of posterior probabilities for all incorrect labels
    sum_incorrect = np.sum(posterior_probs * (1 - one_hot_labels), axis=1)

    # Replace any zero values with a very small number to prevent division by zero in log
    # Calculate the log likelihood for the incorrect labels
    log_likelihood_incorrect = np.log(sum_incorrect)

    # Calculate phi_stable for each example
    phi_stable = log_likelihood_correct - log_likelihood_incorrect

    return phi_stable



def _log_value( probs, small_value=1e-30):
    return -np.log(np.maximum(probs, small_value))

def default_quantile():
    """Return the default fprs

    Returns:
        arr: Numpy array, indicating the default fprs
    """
    return np.logspace(-5, 0, 100)
def ensure_list(metric):
    """
    Ensure that the metric is in list format.
    If the metric is not a list, convert it into a list containing a single element.
    """
    if not isinstance(metric, list):
        return [metric]
    return metric
def m_entr_comp( probs, true_labels):
    log_probs = _log_value(probs)
    reverse_probs = 1 - probs
    log_reverse_probs = _log_value(reverse_probs)
    modified_probs = np.copy(probs)
    modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(
        true_labels.size), true_labels]
    modified_log_probs = np.copy(log_reverse_probs)
    modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(
        true_labels.size), true_labels]
    return np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)

def likelihood_ratio(x, mu_in, sigma_in, mu_out, sigma_out):
    return norm.pdf(x, loc=mu_in, scale=sigma_in) / norm.pdf(x, loc=mu_out, scale=sigma_out)


def calculate_loss(criterion, outputs, targets):
    return criterion(outputs, targets).cpu().numpy()

def calculate_entropy(posteriors):
    return -np.sum(posteriors * np.log(posteriors + 1e-10), axis=1)

def calculate_confidence(posteriors):
    confidence, _ = torch.max(posteriors, 1)
    return confidence.cpu().numpy()

def calculate_correctness(outputs, targets):
    correct = outputs.argmax(dim=1) == targets
    return correct.cpu().numpy()


def flatten_dict(result_metrics,alphas,keys =None):
    # Initialize an empty dictionary to hold the flattened structure
    flattened = {}

    # Loop through each metric in the result_metrics dictionary
    for metric, values in result_metrics.items():
        # Loop through each key in the inner dictionary

        for key, value in values.items():
            if keys:
                if key not in keys:
                    continue
            
            
            
            if np.isscalar(value):
                new_key = f"{metric}_{key}"
                flattened[new_key] = float(value)
                print(f"{new_key}: {value}")
                continue
            # Create a new key by combining the metric and the original key
            for idx, alpha in enumerate(alphas):
                
                new_key = f"{metric}_{key}_alpha_{alpha}"
                # Add this to the flattened dictionary

                flattened[new_key] = float(value[idx])
                print(f"{new_key}: {value}")
    return flattened

