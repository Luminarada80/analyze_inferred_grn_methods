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

def print_banner():
    logging.info("""
                 
 ██████  ██████  ███    ██      █████  ███    ██  █████  ██      ██    ██ ███████ ██ ███████ 
██       ██   ██ ████   ██     ██   ██ ████   ██ ██   ██ ██       ██  ██  ██      ██ ██      
██   ███ ██████  ██ ██  ██     ███████ ██ ██  ██ ███████ ██        ████   ███████ ██ ███████ 
██    ██ ██   ██ ██  ██ ██     ██   ██ ██  ██ ██ ██   ██ ██         ██         ██ ██      ██ 
 ██████  ██   ██ ██   ████     ██   ██ ██   ████ ██   ██ ███████    ██    ███████ ██ ███████ 
                                                                                             
                                                                                             

                 """)

def log_message(message, width=60, fill_char='='):
    """
    Formats a log message with a consistent width by padding it with a fill character.

    Parameters:
    ----------
    message : str
        The message to display in the log.
    width : int
        The total width of the output line (including the message).
    fill_char : str
        The character used to pad the message on both sides.

    Returns:
    ----------
    str
        A formatted log message with consistent width.
    """
    # Calculate padding
    message = f" {message} "
    padding = max(0, width - len(message))
    left_pad = padding // 2
    right_pad = padding - left_pad
    
    return f"\n\n{fill_char * left_pad}{message}{fill_char * right_pad}"


def read_input_files(method_input_path, sample_names) -> tuple[str, list, list, dict]:
    """
    Reads through the current directory to find input files.
    
    Requires a folder called "INPUT", which contains a subfolder for each 
    method. Each method folder should have a subfolder for each sample, containing
    the inferred network for that sample.
    
    The ground truth file should be stored in a separate file within "INPUT" called
    "GROUND_TRUTH".

    Returns
    ----------
        ground_truth_path (str):
            The full path to the ground truth file.
        method_names (list):
            A list containing the method names (names of the subfolders of "INPUT")
        sample_names (list):
            A list containing the sample names (names of the subfolders of each method directory)
        inferred_network_dict (dict):
            A dictionary of the paths to the inferred network files, with the method name and sample
            name corresonding to the file as the keys to the dictionary.
    """
    # Read through the directory to find the input and output files
    method_names = []
    inferred_network_dict = {}
    sample_names = set()
    print(log_message("READING INPUT DIRECTORY STRUCTURE"))
    
    # Looks through the current directory for a folder called "input"
    for folder in os.listdir(method_input_path):
        
        
        if folder.lower() == "input":
            logging.info(folder)
            for subfolder in os.listdir(f'./{folder}'):
                logging.info(f'  └──{subfolder}')
                
                # Finds the folder titled "ground_truth"
                if subfolder.lower() == "ground_truth":
                    ground_truth_dir = f'{os.path.abspath(folder)}/{subfolder}' 
                    
                    # Check to see if there are more than 1 ground truth files
                    num_ground_truth_files = len(os.listdir(ground_truth_dir))
                    if num_ground_truth_files > 1:
                        logging.warning(f'Warning! Multiple files in ground truth input directory. Using the first entry...')
                        
                    # Use the first file in the ground truth directory as the ground truth and set the path
                    ground_truth_filename = os.listdir(ground_truth_dir)[0]
                    ground_truth_path = f"{ground_truth_dir}/{ground_truth_filename}"
                    for file_name in os.listdir(ground_truth_dir):
                        logging.info(f'    └──{file_name}')
                    
                # If the folder is not "ground_truth", assume the folders are the method names
                else:
                    # Create output directories for the samples
                    if not os.path.exists(f'./OUTPUT/{subfolder}'):
                        os.makedirs(f'./OUTPUT/{subfolder}')
                    
                    # Set the method name to the name of the subfolder
                    method_name = subfolder
                    method_names.append(method_name)
                    method_input_dir = f'{os.path.abspath(folder)}/{subfolder}' 
                    
                    # Add the method as a key for the inferred network dictionary
                    if method_name not in inferred_network_dict:
                        inferred_network_dict[method_name] = {}

                    # Iterate through each sample folder in the method directory
                    for sample_name in os.listdir(method_input_dir):
                        
                        sample_dir_path = f'{method_input_dir}/{sample_name}'
                        
                        # If there is a file in the sample folder
                        if len(os.listdir(sample_dir_path)) > 0:
                            logging.info(f'     └──{sample_name}')
                            
                            # Add the name of the folder as a sample name
                            sample_names.add(sample_name)
                            
                            # Create output directories for the samples
                            if not os.path.exists(f'./OUTPUT/{subfolder}/{sample_name}'):
                                os.makedirs(f'./OUTPUT/{subfolder}/{sample_name}')
                            
                            # Add the path to the sample to the inferred network dictionary for the method
                            if sample_name not in inferred_network_dict[method_name]:
                                inferred_network_dict[method_name][sample_name] = None
                            
                            # Find the path to the inferred network file for the current sample
                            num_sample_files = len(os.listdir(sample_dir_path))
                            if num_sample_files > 1:
                                logging.warning(f"Warning! Multiple files in the inferred network directory for {sample_name}. Using the first entry...")
                            for inferred_network_file in os.listdir(sample_dir_path):
                                inferred_network_dict[method_name][sample_name] = f'{method_input_dir}/{sample_name}/{inferred_network_file}'
                                logging.info(f'        └──{inferred_network_file}')
    
    print(log_message("EXTRACTED THE FOLLOWING AS INPUT"))
    logging.info(f'Sample names:')
    for sample_name in sample_names:
        logging.info(f'\t{sample_name}')
        
    logging.info(f'\nMethod names:')
    for method_name in method_names:
        logging.info(f'\t{method_name}')
    
    logging.info(f'\nGround truth file:')
    logging.info(f'\t{ground_truth_filename}')
    
    return ground_truth_path, method_names, sample_names, inferred_network_dict

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

def write_method_accuracy_metric_file(total_accuracy_metric_dict: dict, batch_name) -> None:
    """
    For each inference method, creates a pandas dataframe of the accuracy metrics for each sample and 
    outputs the results to a tsv file.
    
    total_accuracy_metric_dict[metric][sample][accuracy_metric_name] = score

    Parameters
    ----------
        total_accuracy_metric_dict (dict): 
            A dictionary containing the method names, sample names, accuracy metric names, and accuracy 
            metric values.
    """
    for method in total_accuracy_metric_dict.keys():
        total_accuracy_metrics_df = pd.DataFrame(total_accuracy_metric_dict[method]).T
        total_accuracy_metrics_df.to_csv(f'OUTPUT/{method}/{batch_name}/{method.lower()}_{batch_name}_total_accuracy_metrics.tsv', sep='\t') 

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
    print("/".join([i for i in METHOD_INPUT_PATH.split("/")[-2:]]))
    for folder in os.listdir(METHOD_INPUT_PATH):
        
        # In each subfile of the main GRN output path, find any file that matches the inferred net filename for the method
        # for subfile in os.listdir(os.path.join(METHOD_INPUT_PATH, folder)):
        if INFERRED_NET_FILENAME in folder:
                # logging.info(f'  └──{folder}')
                # print(f'Found inferred network file for sample {folder}')
            inferred_network_dict[METHOD_NAME][folder] = os.path.join(METHOD_INPUT_PATH, folder)
    
    print(log_message("INFERENCE METHOD ANALYSIS AND COMPARISON"))
    
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
    randomized_method_dict = {
            f"{METHOD_NAME} Original": {'y_true':[], 'y_scores':[]},
            f"{METHOD_NAME} Randomized": {'y_true':[], 'y_scores':[]}
        }
    total_accuracy_metrics[METHOD_NAME] = {}
    random_accuracy_metrics[METHOD_NAME] = {}
    
    # PROCESSING EACH SAMPLE
    for sample in inferred_network_dict[METHOD_NAME]:   
        logging.info(f'\tAnalyzing {sample}')
        
        # ================== READING IN AND STANDARDIZING INFERRED DATAFRAMES ===============
        inferred_network_file = inferred_network_dict[METHOD_NAME][sample]

        sep = ',' if METHOD_NAME == 'CELL_ORACLE' else '\t'
        inferred_network_df = load_inferred_network_df(inferred_network_file, sep)
        
        if METHOD_NAME == "CELL_ORACLE":
            inferred_network_df = grn_formatting.create_standard_dataframe(
                inferred_network_df, source_col='source', target_col='target', score_col='coef_abs'
            )
            
        elif METHOD_NAME == "CUSTOM_GRN":
            inferred_network_df = grn_formatting.create_standard_dataframe(
                inferred_network_df, source_col='Source', target_col='Target', score_col='Weighted_Score'
            )
            
        else:
            print(inferred_network_df.head())
            inferred_network_df = grn_formatting.create_standard_dataframe(
                inferred_network_df, source_col='Source', target_col='Target', score_col='Score'
            )
        
        if not os.path.exists(f'./OUTPUT/{METHOD_NAME}/{BATCH_NAME}/STANDARDIZED_INFERRED_NETWORKS/'):
            os.makedirs(f'./OUTPUT/{METHOD_NAME}/{BATCH_NAME}/STANDARDIZED_INFERRED_NETWORKS/')
        
        inferred_network_df.to_csv(f'./OUTPUT/{METHOD_NAME}/{BATCH_NAME}/STANDARDIZED_INFERRED_NETWORKS/{sample}_standardized.csv', sep=',', header=True, index=False)
            
        plotting.plot_inference_score_histogram(
            inferred_network_df,
                METHOD_NAME,
                f'./OUTPUT/{METHOD_NAME}/{BATCH_NAME}/{sample}/{METHOD_NAME.lower()}_inferred_network_score_distribution.png')
        
        # ======================= PREPROCESSING ============================
        sample_ground_truth = ground_truth.copy()
        
            # Initialize a dictionary to store sizes
        network_sizes = {
            "Type": ["before_processing", "after_processing"],
            "Ground Truth TFs": [None, None],
            "Ground Truth TGs": [None, None],
            "Ground Truth Edges": [None, None],
            "Inferred TFs": [None, None],
            "Inferred TGs": [None, None],
            "Inferred Edges": [None, None],
        }

        # Compute sizes before processing
        network_sizes["Ground Truth TFs"][0] = len(set(sample_ground_truth["Source"]))
        network_sizes["Ground Truth TGs"][0] = len(set(sample_ground_truth["Target"]))
        network_sizes["Ground Truth Edges"][0] = len(sample_ground_truth)

        network_sizes["Inferred TFs"][0] = len(set(inferred_network_df["Source"]))
        network_sizes["Inferred TGs"][0] = len(set(inferred_network_df["Target"]))
        network_sizes["Inferred Edges"][0] = len(inferred_network_df)

        
        sample_ground_truth = grn_formatting.add_inferred_scores_to_ground_truth(
            sample_ground_truth, inferred_network_df
        )
        
        # inferred_network_df["Score"] = np.log2(inferred_network_df["Score"])
        # sample_ground_truth["Score"] = np.log2(sample_ground_truth["Score"])
        
        inferred_network_df = inferred_network_df.dropna(subset=['Score'])
        sample_ground_truth = sample_ground_truth.dropna(subset=['Score'])
        
        
        inferred_network_df = grn_formatting.remove_ground_truth_edges_from_inferred(
            sample_ground_truth, inferred_network_df
        )
        
        inferred_network_df = grn_formatting.remove_tf_tg_not_in_ground_truth(
            sample_ground_truth, inferred_network_df
        )
        

        sample_ground_truth, inferred_network_df = grn_stats.classify_interactions_by_threshold(
            sample_ground_truth, inferred_network_df
        )
            
        # Update the network size file with the processed network sizes
        # Compute sizes after processing
        network_sizes["Ground Truth TFs"][1] = len(set(sample_ground_truth["Source"]))
        network_sizes["Ground Truth TGs"][1] = len(set(sample_ground_truth["Target"]))
        network_sizes["Ground Truth Edges"][1] = len(sample_ground_truth)

        network_sizes["Inferred TFs"][1] = len(set(inferred_network_df["Source"]))
        network_sizes["Inferred TGs"][1] = len(set(inferred_network_df["Target"]))
        network_sizes["Inferred Edges"][1] = len(inferred_network_df)

        # Convert to DataFrame for easy output
        size_df = pd.DataFrame(network_sizes)

        # Define output paths
        os.makedirs(f"./OUTPUT/{METHOD_NAME}/{BATCH_NAME}/{sample}", exist_ok=True)
        ground_truth_size_path = f"./OUTPUT/{METHOD_NAME}/{BATCH_NAME}/{sample}/{METHOD_NAME.lower()}_ground_truth_size_{sample.lower()}.tsv"
        inferred_network_size_path = f"./OUTPUT/{METHOD_NAME}/{BATCH_NAME}/{sample}/{METHOD_NAME.lower()}_inferred_network_size_{sample.lower()}.tsv"

        # Write the sizes to TSV files
        size_df[["Type", "Ground Truth TFs", "Ground Truth TGs", "Ground Truth Edges"]].to_csv(
            ground_truth_size_path, sep="\t", index=False
        )
        size_df[["Type", "Inferred TFs", "Inferred TGs", "Inferred Edges"]].to_csv(
            inferred_network_size_path, sep="\t", index=False
        )
        
        # ===================== ANALYSIS =======================
        logging.debug(f'\t\tCalculating Accuracy Metrics')             
        total_accuracy_metrics[METHOD_NAME][sample] = {}
        random_accuracy_metrics[METHOD_NAME][sample] = {}
        sample_ground_truth = sample_ground_truth
        inferred_network_df = inferred_network_df

        # Calculate the accuracy metrics for the current sample
        accuracy_metric_dict, confusion_matrix_score_dict = grn_stats.calculate_accuracy_metrics(
            sample_ground_truth,
            inferred_network_df
            )
        
        logging.debug(f'\t\tPlotting histogram with thresholds') 
        # Plot the threshold histograms of TP, FP, FN, TN
        histogram_ground_truth_dict = {METHOD_NAME: sample_ground_truth}
        histogram_inferred_network_dict = {METHOD_NAME: inferred_network_df}
        plotting.plot_multiple_histogram_with_thresholds(
            histogram_ground_truth_dict,
            histogram_inferred_network_dict,
            save_path=f'./OUTPUT/{METHOD_NAME}/{BATCH_NAME}/histogram_with_threshold'
            )

        # Create the randomized inference scores and calculate accuracy metrics
        logging.debug(f'\t\tCreating randomized inference scores') 
        randomized_accuracy_metric_dict, randomized_confusion_matrix_dict = grn_stats.create_randomized_inference_scores(
            sample_ground_truth,
            inferred_network_df,
            histogram_save_path=f'./OUTPUT/{METHOD_NAME}/{BATCH_NAME}/randomized_histogram_with_threshold',
            random_method="random_permutation"
            )
        
        uniform_accuracy_metric_dict, uniform_confusion_matrix_dict = grn_stats.create_randomized_inference_scores(
            sample_ground_truth,
            inferred_network_df,
            random_method="uniform_distribution"
            )
        
        # Write out the accuracy metrics to a tsv file
        logging.debug(f'\t\tWriting accuracy metrics to a tsv file') 
        with open(f'./OUTPUT/{METHOD_NAME.upper()}/{BATCH_NAME}/{sample}/accuracy_metrics.tsv', 'w') as accuracy_metric_file:
            accuracy_metric_file.write(f'Metric\tScore\n')
            for metric_name, score in accuracy_metric_dict.items():
                accuracy_metric_file.write(f'{metric_name}\t{score:.4f}\n')
                total_accuracy_metrics[METHOD_NAME][sample][metric_name] = score
                    
        # Write out the randomized accuracy metrics to a tsv file
        with open(f'./OUTPUT/{METHOD_NAME.upper()}/{BATCH_NAME}/{sample}/randomized_accuracy_method.tsv', 'w') as random_accuracy_file:
            random_accuracy_file.write(f'Metric\tOriginal Score\tRandomized Score\n')
            for metric_name, score in accuracy_metric_dict.items():
                random_accuracy_file.write(f'{metric_name}\t{score:.4f}\t{uniform_accuracy_metric_dict[metric_name]:4f}\n')
                random_accuracy_metrics[METHOD_NAME][sample][metric_name] = uniform_accuracy_metric_dict[metric_name]
        
        # Calculate the normal AUROC and AUPRC
        logging.info(f'\t\tCalculating normal AUROC and AUPRC') 
        auroc = grn_stats.calculate_auroc(confusion_matrix_score_dict)
        auprc = grn_stats.calculate_auprc(confusion_matrix_score_dict)
        
        # Calculate the randomized AUPRC and randomized AUPRC
        logging.info(f'\t\tCalculating randomized AUROC and AUPRC') 
        randomized_auroc = grn_stats.calculate_auroc(uniform_confusion_matrix_dict)
        randomized_auprc = grn_stats.calculate_auprc(uniform_confusion_matrix_dict)
    
        # Plot the normal and randomized AUROC and AUPRC for the individual sample
        confusion_matrix_with_method = {METHOD_NAME: confusion_matrix_score_dict}
        
        # Record the y_true and y_scores for the current sample to plot all sample AUROC and AUPRC between methods
        total_method_confusion_scores[METHOD_NAME]['y_true'].append(confusion_matrix_score_dict['y_true'])
        total_method_confusion_scores[METHOD_NAME]['y_scores'].append(confusion_matrix_score_dict['y_scores'])
        
        # Record the original and randomized y_true and y_scores for each sample to compare against the randomized scores
        randomized_method_dict[f"{METHOD_NAME} Original"]['y_true'].append(confusion_matrix_score_dict['y_true'])
        randomized_method_dict[f"{METHOD_NAME} Original"]['y_scores'].append(confusion_matrix_score_dict['y_scores'])
        
        randomized_method_dict[f"{METHOD_NAME} Randomized"]['y_true'].append(uniform_confusion_matrix_dict['y_true'])
        randomized_method_dict[f"{METHOD_NAME} Randomized"]['y_scores'].append(uniform_confusion_matrix_dict['y_scores'])
        
        sample_randomized_method_dict = {
            f"{METHOD_NAME} Original": {'y_true':confusion_matrix_score_dict['y_true'], 'y_scores':confusion_matrix_score_dict['y_scores']},
            f"{METHOD_NAME} Randomized": {'y_true':uniform_confusion_matrix_dict['y_true'], 'y_scores':uniform_confusion_matrix_dict['y_scores']}
        }
        
        logging.debug(f'\t\tPlotting the normal and randomized AUROC and AUPRC graphs') 
        plotting.plot_auroc_auprc(sample_randomized_method_dict, f'./OUTPUT/{METHOD_NAME}/{BATCH_NAME}/{sample}/randomized_auroc_auprc.png')
        plotting.plot_auroc_auprc(confusion_matrix_with_method, f'./OUTPUT/{METHOD_NAME}/{BATCH_NAME}/{sample}/auroc_auprc.png')

        logging.debug(f'\t\tUpdating the total accuracy metrics for the current method') 
        # Add the auroc and auprc values to the total accuracy metric dictionaries
        total_accuracy_metrics[METHOD_NAME][sample]['auroc'] = auroc
        total_accuracy_metrics[METHOD_NAME][sample]['auprc'] = auprc
        
        random_accuracy_metrics[METHOD_NAME][sample]['auroc'] = randomized_auroc
        random_accuracy_metrics[METHOD_NAME][sample]['auprc'] = randomized_auprc
        
        # Update the accuracy metrics with the confusion matrix keys
        confusion_matrix_keys = ["true_positive", "true_negative", "false_positive", "false_negative"]
        for key in confusion_matrix_keys:
            total_accuracy_metrics[METHOD_NAME][sample][key] = int(confusion_matrix_score_dict[key])
            random_accuracy_metrics[METHOD_NAME][sample][key] = int(uniform_confusion_matrix_dict[key])
        
        # Free memory
        del sample_ground_truth, inferred_network_df
        gc.collect()
    
    logging.info(f'\tPlotting {METHOD_NAME.lower()} original vs randomized AUROC and AUPRC for all samples')
    
    if not os.path.exists(f'./OUTPUT/{METHOD_NAME}/{BATCH_NAME}/'):
        os.makedirs(f'./OUTPUT/{METHOD_NAME}/{BATCH_NAME}/')
        
    plotting.plot_multiple_method_auroc_auprc(randomized_method_dict, f'./OUTPUT/{METHOD_NAME}/{BATCH_NAME}/{METHOD_NAME.lower()}_randomized_auroc_auprc.png')
    
    write_method_accuracy_metric_file(total_accuracy_metrics, BATCH_NAME)
    write_method_accuracy_metric_file(random_accuracy_metrics, BATCH_NAME)
                
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')  
    
    main()