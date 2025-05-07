import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import math
from copy import deepcopy
import os
import logging
import csv
import gc
import argparse
from multiprocessing import Pool
from tqdm import tqdm
import gc

logging.basicConfig(level=logging.INFO, format='%(message)s')  

# Install using 'conda install luminarada80::grn_analysis_tools' 
# or update to the newest version using 'conda update grn_analysis_tools'
from grn_analysis_tools import grn_formatting, plotting, resource_analysis, grn_stats

# Temporarily disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None

# Set font to Arial and adjust font sizes
rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 14,  # General font size
    'axes.titlesize': 18,  # Title font size
    'axes.labelsize': 16,  # Axis label font size
    'xtick.labelsize': 14,  # X-axis tick label size
    'ytick.labelsize': 14,  # Y-axis tick label size
    'legend.fontsize': 14  # Legend font size
})

def load_inferred_network_df(
    inferred_network_file: str,
    separator: str
    ) -> pd.DataFrame:
    """
    Loads the inferred network file to a dataframe based on the separator.

    Parameters
    ----------
        inferred_network_file (str):
            Path to the inferred network file.
        separator (str):
            Pandas separator to use when loading in the dataframe
    """
    return pd.read_csv(inferred_network_file, sep=separator, header=0, index_col=0)

def standardize_ground_truth_format(ground_truth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Changes Source and Target gene names to uppercase and removes any whitespace

    Parameters
    ----------
        ground_truth_df (pd.DataFrame): 
            Unfiltered ground truth dataframe

    Returns
    ----------
        ground_truth_df (pd.DataFrame):
            Ground truth dataframe with Source and Target names converted to uppercase
    """
    ground_truth_df['Source'] = ground_truth_df['Source'].str.upper().str.strip()
    ground_truth_df['Target'] = ground_truth_df['Target'].str.upper().str.strip()
    
    ground_truth_df = ground_truth_df[["Source", "Target"]]
    
    return ground_truth_df

def create_ground_truth_copies(
    ground_truth_df: pd.DataFrame,
    method_names: list,
    sample_names: list,
    inferred_network_dict: dict
    ) -> dict:
    """
    Creates copies of the ground truth dataframe for each method and sample.
    
    Each inferred network needs a corresponding ground truth dataframe, as the ground truth edge
    scores are assigned from the inferred network

    Parameters
    ----------
        ground_truth_df (pd.DataFrame): 
            Ground truth dataframe with columns "Source" and "Target"
        method_names (list): 
            A list of the names of each inference method being evaluated.
        sample_names (list): 
            A list of the dataset sample names.
        inferred_network_dict (dict): 
            A dictionary of each inferred network for the method, used to find sample names
            for the current method

    Returns
    ----------
        ground_truth_dict (dict):
            A dictionary of ground truth dataframe copies, mapped to a sample and method
    """
    ground_truth_dict = {}
    for method in method_names:
        ground_truth_dict[method] = {}
        
        method_samples = [sample for sample in sample_names if sample in inferred_network_dict[method]]
        for sample in method_samples:
            ground_truth_dict[method][sample] = deepcopy(ground_truth_df)
    
    return ground_truth_dict

def parse_args():
    parser = argparse.ArgumentParser(description="Process Inferred GRNs.")
    parser.add_argument(
        "--inferred_net_filename",
        type=str,
        required=True,
        help="Name of the output inferred file created by the method"
    )
    parser.add_argument(
        "--method_name",
        type=str,
        required=True,
        help="Name of the inference method"
    )
    parser.add_argument(
        "--ground_truth_path",
        type=str,
        required=True,
        help="Directory to the ground truth data file"
    )
    parser.add_argument(
        "--method_input_path",
        type=str,
        required=True,
        help="Directory to the ground truth data file"
    )
    parser.add_argument(
        "--batch_name",
        type=str,
        required=True,
        help="Name of the batch to separate out groups of samples"
    )

    args = parser.parse_args()

    return args   

# Parse the arguments
args = parse_args()

# Macrophase_buffer1_stability_1
METHOD_NAME = args.method_name
INFERRED_NET_FILENAME = args.inferred_net_filename
GROUND_TRUTH_PATH = args.ground_truth_path
BATCH_NAME = args.batch_name    
METHOD_INPUT_PATH = args.method_input_path

def balance_true_negative_scores(y_true, y_scores):
    # Balance positive and negative samples for AUROC only
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    pos_indices = np.where(y_true == 1)[0]
    neg_indices = np.where(y_true == 0)[0]

    # Subsample negative class to match number of positives
    if len(pos_indices) > 0 and len(neg_indices) > 0:
        if len(pos_indices) < len(neg_indices):
            rng = np.random.default_rng(seed=42)
            sampled_neg_indices = rng.choice(neg_indices, size=len(pos_indices), replace=False)
            balanced_indices = np.concatenate([pos_indices, sampled_neg_indices])
            balanced_indices.sort()
            
            logging.debug(f'\tNumber of positive samples for AUROC: {len(pos_indices)}')
            logging.debug(f'\tNumber of negative samples for AUROC: {len(sampled_neg_indices)}')

            # For AUROC only
            y_true_balanced = y_true[balanced_indices]
            y_scores_balanced = y_scores[balanced_indices]
        else:
            y_true_balanced = y_true
            y_scores_balanced = y_scores
    else:
        y_true_balanced = y_true
        y_scores_balanced = y_scores
    
    return y_true_balanced, y_scores_balanced

def process_sample(sample):

    inferred_network_file = inferred_network_dict[METHOD_NAME][sample]
    sep = ',' if METHOD_NAME == 'CELL_ORACLE' else '\t'
    inferred_network_df = load_inferred_network_df(inferred_network_file, sep)

    if METHOD_NAME == "CELL_ORACLE":
        inferred_network_df = grn_formatting.create_standard_dataframe(inferred_network_df, 'source', 'target', 'coef_abs')
    elif METHOD_NAME == "CUSTOM_GRN":
        inferred_network_df = grn_formatting.create_standard_dataframe(inferred_network_df, 'source_id', 'target_id', 'score')
    elif METHOD_NAME == "LINGER":
        inferred_network_df = grn_formatting.create_standard_dataframe(inferred_network_df)
    else:
        inferred_network_df = grn_formatting.create_standard_dataframe(inferred_network_df, 'Source', 'Target', 'Score')

    sample_ground_truth = ground_truth.copy()
    sample_ground_truth = grn_formatting.add_inferred_scores_to_ground_truth(sample_ground_truth, inferred_network_df)

    inferred_network_df = inferred_network_df.loc[inferred_network_df["Score"] != 0].dropna(subset=['Score'])
    sample_ground_truth = sample_ground_truth.loc[sample_ground_truth["Score"] != 0].dropna(subset=['Score'])

    inferred_network_df = grn_formatting.remove_ground_truth_edges_from_inferred(sample_ground_truth, inferred_network_df)
    inferred_network_df = grn_formatting.remove_tf_tg_not_in_ground_truth(sample_ground_truth, inferred_network_df)

    sample_ground_truth, inferred_network_df = grn_stats.classify_interactions_by_threshold(sample_ground_truth, inferred_network_df, lower_threshold=0.5)

    if len(sample_ground_truth) > len(inferred_network_df):
        balanced_ground_truth = sample_ground_truth.sample(n=len(inferred_network_df))
        balanced_inferred_network = inferred_network_df.copy()
    else:
        balanced_ground_truth = sample_ground_truth.copy()
        balanced_inferred_network = inferred_network_df.sample(n=len(sample_ground_truth))

    accuracy_metric_dict, confusion_matrix_score_dict = grn_stats.calculate_accuracy_metrics(sample_ground_truth, inferred_network_df)
    balanced_accuracy_metric_dict, balanced_confusion_matrix_score_dict = grn_stats.calculate_accuracy_metrics(balanced_ground_truth, balanced_inferred_network)

    uniform_accuracy_metric_dict, uniform_confusion_matrix_dict = grn_stats.create_randomized_inference_scores(
        sample_ground_truth, inferred_network_df, lower_threshold=0.5, random_method="uniform_distribution"
    )
    balanced_uniform_accuracy_metric_dict, balanced_uniform_confusion_matrix_dict = grn_stats.create_randomized_inference_scores(
        balanced_ground_truth, balanced_inferred_network, lower_threshold=0.5, random_method="uniform_distribution"
    )
    
    # subsample
    def subsample_auroc(dict):
        result_dict = {}
        y_scores = dict['y_scores']
        y_true = dict['y_true']
        
        step = max(1, math.ceil(len(y_scores) * 0.0001))
        idx = np.arange(0, len(y_scores), step)
        
        result_dict['y_true'] = y_true.iloc[idx]
        result_dict['y_scores'] = y_scores.iloc[idx]
        
        return dict
    
    confusion_matrix_score_dict = subsample_auroc(confusion_matrix_score_dict)
    uniform_confusion_matrix_dict = subsample_auroc(uniform_confusion_matrix_dict)
    balanced_uniform_confusion_matrix_dict = subsample_auroc(balanced_uniform_confusion_matrix_dict)
    
    # Save per-sample balanced ground truth and inferred network
    output_dir = f'./OUTPUT/{METHOD_NAME}/{BATCH_NAME}/balanced_cell_level_grns/{sample}'
    os.makedirs(output_dir, exist_ok=True)

    balanced_ground_truth.to_csv(f'{output_dir}/balanced_ground_truth.csv', index=False)
    balanced_inferred_network.to_csv(f'{output_dir}/balanced_inferred_network.csv', index=False)


    result = {
        "sample": sample,
        "confusion": confusion_matrix_score_dict,
        "balanced_confusion": balanced_confusion_matrix_score_dict,
        "random": uniform_confusion_matrix_dict,
        "balanced_random": balanced_uniform_confusion_matrix_dict,
    }

    del sample_ground_truth, inferred_network_df
    gc.collect()
    
    return result

inferred_network_dict = {METHOD_NAME: {}}

# Iterate through the inferred GRN main otuput path
# print("/".join([i for i in METHOD_INPUT_PATH.split("/")[-2:]]))
# print(INFERRED_NET_FILENAME)
folder_path = []
for folder in os.listdir(METHOD_INPUT_PATH):
    
    folder_path.append(f'{folder}')
    
    # In each subfile of the main GRN output path, find any file that matches the inferred net filename for the method
    # for subfile in os.listdir(os.path.join(METHOD_INPUT_PATH, folder)):
    if INFERRED_NET_FILENAME in folder:
        folder_path.append(f'  └──{folder}')
        inferred_network_dict[METHOD_NAME][folder] = os.path.join(METHOD_INPUT_PATH, INFERRED_NET_FILENAME)
        
    elif os.path.isdir(os.path.join(METHOD_INPUT_PATH, folder)):
        for subfile in os.listdir(os.path.join(METHOD_INPUT_PATH, folder)):
            if INFERRED_NET_FILENAME in subfile:
                inferred_network_dict[METHOD_NAME][folder] = os.path.join(METHOD_INPUT_PATH, f'{folder}/{subfile}')
                folder_path.append(f'    └──{subfile}')
        
if len(inferred_network_dict[METHOD_NAME]) == 0:
    raise Exception(f"No inferred networks found for {METHOD_NAME} in {METHOD_INPUT_PATH}")

logging.debug(f'\nPreprocessing inferred and ground truth networks')

logging.debug(f'\tReading ground truth')
ground_truth = pd.read_csv(GROUND_TRUTH_PATH, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip', header=0)
ground_truth = standardize_ground_truth_format(ground_truth)
    
total_method_confusion_scores = {}
total_accuracy_metrics = {}
random_accuracy_metrics = {}


sample_list = list(inferred_network_dict[METHOD_NAME].keys())
logging.info(f'Found {len(sample_list)} samples for {METHOD_NAME}')
num_workers = min(8, len(sample_list))  # or use os.cpu_count()

results = []
with Pool(processes=num_workers) as pool:
    # tqdm wrapper around imap for progress tracking
    for result in tqdm(pool.imap_unordered(process_sample, sample_list), total=len(sample_list), desc=f"Processing samples for {METHOD_NAME}"):
        results.append(result)

if not os.path.exists(f'./OUTPUT/{METHOD_NAME}/{BATCH_NAME}/'):
    os.makedirs(f'./OUTPUT/{METHOD_NAME}/{BATCH_NAME}/')
    
total_method_confusion_scores[METHOD_NAME] = {'y_true': [], 'y_scores': []}

randomized_method_dict = {
    METHOD_NAME: {
        'normal_y_true': [], 
        'normal_y_scores': [], 
        'randomized_y_true': [], 
        'randomized_y_scores': [],
        'balanced_normal_y_true': [], 
        'balanced_normal_y_scores': [], 
        'balanced_randomized_y_true': [], 
        'balanced_randomized_y_scores': [],
    }
}

for res in results:
    total_method_confusion_scores[METHOD_NAME]['y_true'].append(res['confusion']['y_true'])
    total_method_confusion_scores[METHOD_NAME]['y_scores'].append(res['confusion']['y_scores'])

    randomized_method_dict[METHOD_NAME]['normal_y_true'].append(res['confusion']['y_true'])
    randomized_method_dict[METHOD_NAME]['normal_y_scores'].append(res['confusion']['y_scores'])

    randomized_method_dict[METHOD_NAME]['randomized_y_true'].append(res['random']['y_true'])
    randomized_method_dict[METHOD_NAME]['randomized_y_scores'].append(res['random']['y_scores'])

    randomized_method_dict[METHOD_NAME]['balanced_normal_y_true'].append(res['balanced_confusion']['y_true'])
    randomized_method_dict[METHOD_NAME]['balanced_normal_y_scores'].append(res['balanced_confusion']['y_scores'])

    randomized_method_dict[METHOD_NAME]['balanced_randomized_y_true'].append(res['balanced_random']['y_true'])
    randomized_method_dict[METHOD_NAME]['balanced_randomized_y_scores'].append(res['balanced_random']['y_scores'])

logging.info(f'\tPlotting {METHOD_NAME.lower()} original vs randomized AUROC and AUPRC for all samples')    
plotting.plot_all_samples_auroc_auprc(
    {METHOD_NAME: randomized_method_dict[METHOD_NAME]},
    sample_list,
    f'./OUTPUT/{METHOD_NAME}/{BATCH_NAME}/{METHOD_NAME.lower()}_randomized_auroc_auprc.png'
)

plotting.plot_boxplot_auroc_auprc_comparison(
    randomized_method_dict[METHOD_NAME],
    METHOD_NAME,
    f'./OUTPUT/{METHOD_NAME}/{BATCH_NAME}/{METHOD_NAME.lower()}_randomized_auroc_auprc_boxplots.png'
)                
# if __name__ == "__main__":
#     
    
#     main()