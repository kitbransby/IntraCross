import numpy as np
import matplotlib.pyplot as plt
import pickle
import numpy as np
from tqdm import tqdm 
from scipy.stats import kstest, norm, wilcoxon, iqr

def ks_test_normality(data, label):
    # Helper: Run K-S test on test set errors for normality
    # Decides whether we use paired t-test or Wilcoxon
    data_std = (data - np.mean(data)) / np.std(data, ddof=1)  # standardize
    stat, p = kstest(data_std, 'norm')
    print(f"{label}: KS statistic = {stat:.4f}, p = {p:.8f}")
    if p > 0.05:
        print(f"  → Likely normal")
    else:
        print(f"  → Not normal")

def longitudinal_stats_analysis(load_folder, data=None, save_viz=None, verbose=False):

    # ---- LOAD RESULTS ---- #

    if data is not None:
        r1_pred, r1star_pred, r2_pred, computer_pred, r2_pred_using_r2_keypts, computer_pred_using_r2_keypts, r1_pred_interpolated, r1star_pred_interpolated, r2_pred_interpolated, computer_pred_interpolated, ids_longi, ids_longi_keypts = data
        r1_pred = np.array(r1_pred) 
        r1star_pred = np.array(r1star_pred) 
        r2_pred = np.array(r2_pred) 
        computer_pred = np.array(computer_pred)
        r2_pred_using_r2_keypts = np.array(r2_pred_using_r2_keypts)
        computer_pred_using_r2_keypts = np.array(computer_pred_using_r2_keypts)
        r1_pred_interpolated = np.array(r1_pred_interpolated)
        r1star_pred_interpolated = np.array(r1star_pred_interpolated)
        r2_pred_interpolated = np.array(r2_pred_interpolated)
        computer_pred_interpolated = np.array(computer_pred_interpolated)
        ids_longi = np.array(ids_longi)
        ids_longi_keypts = np.array(ids_longi_keypts)

    else:
        r1_pred = np.array(load_pkl(load_folder + 'predictions/test_longi_r1_pred.pkl')) 
        r1star_pred = np.array(load_pkl(load_folder + 'predictions/test_longi_r1star_pred.pkl')) 
        r2_pred = np.array(load_pkl(load_folder + 'predictions/test_longi_r2_pred.pkl')) 
        computer_pred = np.array(load_pkl(load_folder + 'predictions/test_longi_computer_pred.pkl'))
        r2_pred_using_r2_keypts = np.array(load_pkl(load_folder + 'predictions/test_longi_r2_pred_using_r2_keypts.pkl'))
        computer_pred_using_r2_keypts = np.array(load_pkl(load_folder + 'predictions/test_longi_computer_pred_using_r2_keypts.pkl'))
        r1_pred_interpolated = np.array(load_pkl(load_folder + 'predictions/test_longi_r1_pred_interpolated.pkl'))
        r1star_pred_interpolated = np.array(load_pkl(load_folder + 'predictions/test_longi_r1star_pred_interpolated.pkl'))
        r2_pred_interpolated = np.array(load_pkl(load_folder + 'predictions/test_longi_r2_pred_interpolated.pkl'))
        computer_pred_interpolated = np.array(load_pkl(load_folder + 'predictions/test_longi_computer_interpolated_pred.pkl'))
        ids_longi = np.array(load_pkl(load_folder + 'predictions/ids_longi_all.pkl'))
        ids_longi_keypts = np.array(load_pkl(load_folder + 'predictions/ids_longi_keypts.pkl'))

    r1_r2_diff = np.abs(r1_pred - r2_pred)
    r1_r1star_diff = np.abs(r1_pred - r1star_pred)
    r1_computer_diff = np.abs(r1_pred - computer_pred)
    r2_computer_diff = np.abs(r2_pred - computer_pred)

    r1_r2_interpolated_diff = np.abs(r1_pred_interpolated - r2_pred_interpolated)
    r1_r1star_interpolated_diff = np.abs(r1_pred_interpolated - r1star_pred_interpolated)
    r1_computer_interpolated_diff = np.abs(r1_pred_interpolated - computer_pred_interpolated)
    r2_computer_interpolated_diff = np.abs(r2_pred_interpolated - computer_pred_interpolated)

    # Perform Shapiro-Wilk test
    print('Are the errors normally distributed?')
    ks_test_normality(r1_r2_interpolated_diff_vessel, "R1 - R2")
    ks_test_normality(r1_computer_interpolated_diff_vessel, "R1 - Computer")
    ks_test_normality(r2_computer_interpolated_diff_vessel, "R2 - Computer")

    if verbose:
        print('\n------ LONGITUDINAL STATS ANALYSIS ------')
        print('\nPer keypoint (keypts only) -----')
        print('R1 vs R2: MEDIAN {:.1f} ({:.1f}) MEAN {:.1f} +/- {:.1f}'.format(
            np.median(r1_r2_diff), iqr(r1_r2_diff), np.mean(r1_r2_diff), np.std(r1_r2_diff)))
        print('R1 vs R1*: MEDIAN {:.1f} ({:.1f}) MEAN {:.1f} +/- {:.1f}'.format(
            np.median(r1_r1star_diff), iqr(r1_r1star_diff), np.mean(r1_r1star_diff), np.std(r1_r1star_diff)))
        print('R1 vs Computer: MEDIAN {:.1f} ({:.1f}) MEAN {:.1f} +/- {:.1f}'.format(
            np.median(r1_computer_diff), iqr(r1_computer_diff), np.mean(r1_computer_diff), np.std(r1_computer_diff)))
        print('R2 vs Computer (using R1 keypts): MEDIAN {:.1f} ({:.1f}) MEAN {:.1f} +/- {:.1f}'.format(
            np.median(r2_computer_diff), iqr(r2_computer_diff), np.mean(r2_computer_diff), np.std(r2_computer_diff)))
        
    r1_r2_ccc = CCC(r1_pred.copy(), r2_pred.copy())
    r1_r2star_ccc = CCC(r1_pred.copy(), r1star_pred.copy())
    r1_computer_ccc = CCC(r1_pred.copy(), computer_pred.copy())
    r2_computer_ccc = CCC(r2_pred.copy(), computer_pred.copy())

    if verbose:
        print('\nConcordance Correlation Coefficient')
        print('Keypts only -----')
        print('R1 vs R2: {:.4f}'.format(r1_r2_ccc))
        print('R1 vs R1*: {:.4f}'.format(r1_r2star_ccc))
        print('R1 vs Computer: {:.4f}'.format(r1_computer_ccc))
        print('R2 vs Computer (using R1 keypts): {:.4f}'.format(r2_computer_ccc))

    mwi, ci_lower, ci_upper = jackknife_mwi(r1_r2_diff, r1_computer_diff, r2_computer_diff, confidence_level=0.95, reduction='mean')
    if verbose:
        print(f"Williams Index (Per key pt, key pts only) [MEAN]: {mwi:.2f} ({ci_lower:.2f}, {ci_upper:.2f})")
    
    stat1, p1 = wilcoxon(r1_computer_diff, r1_r2_diff, alternative='two-sided')
    stat2, p2 = wilcoxon(r2_computer_diff, r1_r2_diff, alternative='two-sided')
    if verbose:
        print('\nPairwise Wilcoxon')
        print('Per keypt, keypts only -----')
        print(f"R1-R2 error vs R1-Computer error: P-value {p1:.4f}")
        print(f"R1-R2 error vs R2-Computer error: P-value {p2:.4f}")


def CCC(y_true, y_pred):
    """
    Calculate the Concordance Correlation Coefficient (CCC).

    Parameters:
        y_true (np.ndarray): Ground truth values (size N).
        y_pred (np.ndarray): Predicted values (size N).

    Returns:
        float: Concordance Correlation Coefficient.
    """
    # Mean of true and predicted values
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    # Variance and covariance
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.cov(y_true, y_pred)[0, 1]

    # Concordance Correlation Coefficient formula
    numerator = 2 * covariance
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    ccc = numerator / denominator

    ccc = np.clip(ccc, -1, 1)
    
    return ccc

def modified_williams_index(expert1_expert2_error: np.ndarray, 
                            expert1_proposed_error: np.ndarray, 
                            expert2_proposed_error: np.ndarray, 
                            reduction) -> float:
    """
    Compute the Modified Williams Index (mWI) using the mean of disagreements across N examples.
    
    Args:
        expert1_expert2_error (np.ndarray): Disagreements between Expert 1 and Expert 2 (size N).
        expert1_proposed_error (np.ndarray): Disagreements between Expert 1 and the model (size N).
        expert2_proposed_error (np.ndarray): Disagreements between Expert 2 and the model (size N).

    Returns:
        float: The computed Modified Williams Index (mWI).
    """
    epsilon = 1e-6  # Prevent division by zero

    # Step 1: Compute the mean disagreement for each comparison
    if reduction == 'median':
        mean_expert1_expert2_error = np.median(expert1_expert2_error)
        mean_expert1_proposed_error = np.median(expert1_proposed_error)
        mean_expert2_proposed_error = np.median(expert2_proposed_error)
    elif reduction == 'mean':
        mean_expert1_expert2_error = np.mean(expert1_expert2_error)
        mean_expert1_proposed_error = np.mean(expert1_proposed_error)
        mean_expert2_proposed_error = np.mean(expert2_proposed_error)
    else:
        raise ValueError("Reduction method must be either 'mean' or 'median'.")

    # Step 2: Compute the inverse disagreements
    inverse_expert1_proposed = 1.0 / (mean_expert1_proposed_error + epsilon)
    inverse_expert2_proposed = 1.0 / (mean_expert2_proposed_error + epsilon)
    inverse_expert1_expert2 = 1.0 / (mean_expert1_expert2_error + epsilon)

    # Step 3: Numerator: Average inverse disagreement between model and each expert
    numerator = (inverse_expert1_proposed + inverse_expert2_proposed) / 2.0

    # Step 4: Denominator: Inverse disagreement between the two experts
    denominator = inverse_expert1_expert2

    # Step 5: mWI: Ratio of the numerator to the denominator
    mwi = numerator / denominator

    return mwi

def jackknife_mwi(expert1_expert2_error: np.ndarray, 
                 expert1_proposed_error: np.ndarray, 
                 expert2_proposed_error: np.ndarray, 
                 confidence_level: float, 
                 reduction) -> tuple:
    """
    Compute the 95% CI for the Modified Williams Index using the jackknife technique.

    Args:
        expert1_expert2_error (np.ndarray): Disagreements between Expert 1 and Expert 2 (size N).
        expert1_proposed_error (np.ndarray): Disagreements between Expert 1 and the model (size N).
        expert2_proposed_error (np.ndarray): Disagreements between Expert 2 and the model (size N).
        confidence_level (float): Confidence level for the CI (default: 0.95).

    Returns:
        mwi_mean (float): Estimated Modified Williams Index.
        ci_lower (float): Lower bound of the confidence interval.
        ci_upper (float): Upper bound of the confidence interval.
    """
    n_samples = len(expert1_expert2_error)
    jackknife_estimates = []

    # Leave-one-out (LOO) approach: Remove one example at a time
    for i in range(n_samples):
        mask = np.arange(n_samples) != i
        mwi_i = modified_williams_index(
            expert1_expert2_error[mask], 
            expert1_proposed_error[mask], 
            expert2_proposed_error[mask], 
            reduction
        )
        jackknife_estimates.append(mwi_i)

    jackknife_estimates = np.array(jackknife_estimates)

    mwi_mean = np.mean(jackknife_estimates)  # ~0.977
    mwi_std = np.std(jackknife_estimates, ddof=1)  # ~0.008
    z = 1.96
    ci_lower, ci_upper  = mwi_mean - z * mwi_std, mwi_mean + z * mwi_std   

    return mwi_mean, ci_lower, ci_upper

def save_pkl(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pkl(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj