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
    return pd.read_csv(inferred_network_file, sep=separator, header=0)

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

def main():
    # print_banner()
    
    # Parse the arguments
    args = parse_args()
    # Macrophase_buffer1_stability_1
    METHOD_NAME = args.method_name
    INFERRED_NET_FILENAME = args.inferred_net_filename
    GROUND_TRUTH_PATH = args.ground_truth_path
    BATCH_NAME = args.batch_name    
    METHOD_INPUT_PATH = args.method_input_path
    
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
    
    logging.info(f'\nPreprocessing inferred and ground truth networks')
        
    total_method_confusion_scores = {}
    total_accuracy_metrics = {}
    random_accuracy_metrics = {}
    
    logging.info(f'\tReading ground truth')
    ground_truth = pd.read_csv(GROUND_TRUTH_PATH, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip', header=0)
    ground_truth = standardize_ground_truth_format(ground_truth)
    
    # PROCESSING EACH METHOD
    logging.info(f'\nProcessing samples for {METHOD_NAME}')
    total_method_confusion_scores[METHOD_NAME] = {'y_true':[], 'y_scores':[]}
    randomized_method_dict = {METHOD_NAME: {'normal_y_true':[], 'normal_y_scores':[], 'randomized_y_true':[], 'randomized_y_scores':[]}}
    total_accuracy_metrics[METHOD_NAME] = {}
    random_accuracy_metrics[METHOD_NAME] = {}
    
    # PROCESSING EACH SAMPLE
    sample_list = []
    for i, sample in enumerate(inferred_network_dict[METHOD_NAME]):   
        logging.info(f'\tAnalyzing {sample}')
        sample_list.append(sample)
        
        # ================== READING IN AND STANDARDIZING INFERRED DATAFRAMES ===============
        inferred_network_file = inferred_network_dict[METHOD_NAME][sample]

        sep = ',' if METHOD_NAME == 'CELL_ORACLE' else '\t'
        logging.info("\t\tLoading the inferred network")
        inferred_network_df = load_inferred_network_df(inferred_network_file, sep)
        
        logging.info('\t\tStandardizing the DataFrame to "Source", "Target", "Score" columns')
        if METHOD_NAME == "CELL_ORACLE":
            inferred_network_df = grn_formatting.create_standard_dataframe(
                inferred_network_df, source_col='source', target_col='target', score_col='coef_abs'
            )
            
        elif METHOD_NAME == "CUSTOM_GRN":
            inferred_network_df = grn_formatting.create_standard_dataframe(
                inferred_network_df, source_col='source_id', target_col='target_id', score_col='score'
            )
            
        else:
            # print(inferred_network_df.head())
            inferred_network_df = grn_formatting.create_standard_dataframe(
                inferred_network_df, source_col='Source', target_col='Target', score_col='Score'
            )
        
        # ======================= PREPROCESSING ============================
        sample_ground_truth = ground_truth.copy()
        
        logging.info("\t\tSplitting inferred edges based on if the edges are in the ground truth")
        sample_ground_truth = grn_formatting.add_inferred_scores_to_ground_truth(
            sample_ground_truth, inferred_network_df
        )
        
        # Drop scores with a value of 0 values
        inferred_network_df = inferred_network_df.loc[inferred_network_df["Score"] != 0]
        sample_ground_truth = sample_ground_truth.loc[sample_ground_truth["Score"] != 0]
        
        # # Take the log2 of Score
        # inferred_network_df["Score"] = np.log2(inferred_network_df["Score"])
        # sample_ground_truth["Score"] = np.log2(sample_ground_truth["Score"])
        
        # Drop any NaN values
        inferred_network_df = inferred_network_df.dropna(subset=['Score'])
        sample_ground_truth = sample_ground_truth.dropna(subset=['Score'])
        
        
        inferred_network_df = grn_formatting.remove_ground_truth_edges_from_inferred(
            sample_ground_truth, inferred_network_df
        )
        
        logging.info("\t\tRemoving genes that are not present in both the ground truth and the inferred network")
        inferred_network_df = grn_formatting.remove_tf_tg_not_in_ground_truth(
            sample_ground_truth, inferred_network_df
        )
        

        sample_ground_truth, inferred_network_df = grn_stats.classify_interactions_by_threshold(
            sample_ground_truth, inferred_network_df
        )

        # # Define output paths
        os.makedirs(f"./OUTPUT/{METHOD_NAME}/{BATCH_NAME}/", exist_ok=True)
        
        # ===================== ANALYSIS =======================
        logging.info(f'\t\tCalculating Accuracy Metrics')             
        total_accuracy_metrics[METHOD_NAME][sample] = {}
        random_accuracy_metrics[METHOD_NAME][sample] = {}
        sample_ground_truth = sample_ground_truth
        inferred_network_df = inferred_network_df

        # Calculate the accuracy metrics for the current sample
        accuracy_metric_dict, confusion_matrix_score_dict = grn_stats.calculate_accuracy_metrics(
            sample_ground_truth,
            inferred_network_df
            )
        
        logging.info(f'\t\tPlotting histogram with thresholds') 
        # Plot the threshold histograms of TP, FP, FN, TN
        histogram_ground_truth_dict = {METHOD_NAME: sample_ground_truth}
        histogram_inferred_network_dict = {METHOD_NAME: inferred_network_df}
        plotting.plot_multiple_histogram_with_thresholds(
            histogram_ground_truth_dict,
            histogram_inferred_network_dict,
            save_path=f'./OUTPUT/{METHOD_NAME}/{BATCH_NAME}/histogram_with_threshold'
            )

        uniform_accuracy_metric_dict, uniform_confusion_matrix_dict = grn_stats.create_randomized_inference_scores(
            sample_ground_truth,
            inferred_network_df,
            histogram_save_path=f'./OUTPUT/{METHOD_NAME}/{BATCH_NAME}/randomized_histogram_with_threshold',
            random_method="uniform_distribution"
            )
        
        # Record the y_true and y_scores for the current sample to plot all sample AUROC and AUPRC between methods
        total_method_confusion_scores[METHOD_NAME]['y_true'].append(confusion_matrix_score_dict['y_true'])
        total_method_confusion_scores[METHOD_NAME]['y_scores'].append(confusion_matrix_score_dict['y_scores'])
        
        # Extend the list with values to avoid nesting
        randomized_method_dict[METHOD_NAME]['normal_y_true'].extend(confusion_matrix_score_dict['y_true'])
        randomized_method_dict[METHOD_NAME]['normal_y_scores'].extend(confusion_matrix_score_dict['y_scores'])

        randomized_method_dict[METHOD_NAME]['randomized_y_true'].extend(uniform_confusion_matrix_dict['y_true'])
        randomized_method_dict[METHOD_NAME]['randomized_y_scores'].extend(uniform_confusion_matrix_dict['y_scores'])

        
        sample_randomized_method_dict = {
            METHOD_NAME: {'normal_y_true': confusion_matrix_score_dict['y_true'],
                          'normal_y_scores': confusion_matrix_score_dict['y_scores'],
                          'randomized_y_true': uniform_confusion_matrix_dict['y_true'],
                          'randomized_y_scores': uniform_confusion_matrix_dict['y_scores']}
        }
        
        logging.info('\t\tPlotting AUROC and AUPRC')
        plotting.plot_normal_and_randomized_roc_prc(
                sample_randomized_method_dict,
                f'./OUTPUT/{METHOD_NAME}/{BATCH_NAME}/{METHOD_NAME.lower()}_randomized_auroc_auprc.png'
            )
        
        # Free memory
        del sample_ground_truth, inferred_network_df
        gc.collect()
        
        logging.info('\tDone!')
    
    
    # if not os.path.exists(f'./OUTPUT/{METHOD_NAME}/{BATCH_NAME}/'):
    #     os.makedirs(f'./OUTPUT/{METHOD_NAME}/{BATCH_NAME}/')
    
    # logging.info(f'\tPlotting {METHOD_NAME.lower()} original vs randomized AUROC and AUPRC for all samples')    
    # plotting.plot_all_samples_auroc_auprc(total_method_confusion_scores, sample_list, f'./OUTPUT/{METHOD_NAME}/{BATCH_NAME}/{METHOD_NAME.lower()}_randomized_auroc_auprc.png')
                
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')  
    
    main()