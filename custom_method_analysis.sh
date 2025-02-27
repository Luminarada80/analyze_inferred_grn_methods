#!/bin/bash -l

#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --output=analyze_inferred_grn.log
#SBATCH --error=analyze_inferred_grn.err

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/lib

METHOD_NAME="CUSTOM_GRN"
OUTPUT_FOLDER="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output"

K562_GROUND_TRUTH="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_K562/RN117_ChIPSeq_PMID37486787_Human_K562.tsv"
MACROPHAGE_GROUND_TRUTH="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MACROPHAGE_STABILITY/RN204_macrophage_ground_truth.tsv"
MESC_GROUND_TRUTH="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_SC_DATA/RN111.tsv"

MESC_RN112_LOGOF="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/reference_networks/RN112_mouse_logof_ESC_path.tsv"
MESC_RN114_CHIPX_ESCAPE="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN114_ChIPX_ESCAPE_Mouse_ESC.tsv"
MESC_RN115_LOGOF_ESCAPE="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN115_LOGOF_ESCAPE_Mouse_ESC.tsv"


# # ========== MACROPHAGE ============
# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "inferred_network.tsv" \
#     --method_name "CUSTOM_GRN" \
#     --batch_name "macrophage" \
#     --method_input_path "$OUTPUT_FOLDER/macrophage/macrophage_buffer1_filtered/" \
#     --ground_truth_path "$MACROPHAGE_GROUND_TRUTH"

# # ========== K562 ==============
# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "inferred_network.tsv" \
#     --method_name "CUSTOM_GRN" \
#     --batch_name "K562" \
#     --method_input_path "$OUTPUT_FOLDER/K562/K562_human_filtered/" \
#     --ground_truth_path "$MACROPHAGE_GROUND_TRUTH"

# # ========== mESC ==============
# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "inferred_network.tsv" \
#     --method_name "CUSTOM_GRN" \
#     --batch_name "mESC_RN111" \
#     --method_input_path "$OUTPUT_FOLDER/mESC/filtered_L2_E7.5_rep1/" \
#     --ground_truth_path "$MESC_GROUND_TRUTH"

# # --------- mESC knockout --------
# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "inferred_network.tsv" \
#     --method_name "CUSTOM_GRN" \
#     --batch_name "mESC_RN112_LOGOF" \
#     --method_input_path "$OUTPUT_FOLDER/mESC/filtered_L2_E7.5_rep1/" \
#     --ground_truth_path "$MESC_RN112_LOGOF"

# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "inferred_network.tsv" \
#     --method_name "CUSTOM_GRN" \
#     --batch_name "mESC_RN114_LOGOF_ESCAPE" \
#     --method_input_path "$OUTPUT_FOLDER/mESC/filtered_L2_E7.5_rep1/" \
#     --ground_truth_path "$MESC_RN114_CHIPX_ESCAPE"

# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "inferred_network.tsv" \
#     --method_name "CUSTOM_GRN" \
#     --batch_name "mESC_RN115_LOGOF_ESCAPE" \
#     --method_input_path "$OUTPUT_FOLDER/mESC/filtered_L2_E7.5_rep1/" \
#     --ground_truth_path "$MESC_RN115_LOGOF_ESCAPE"


# mESC vs macrophage
python3 Analyze_Inferred_GRN.py \
    --inferred_net_filename "mESC_vs_macrophage_rf_inferred_network.tsv" \
    --method_name "CUSTOM_GRN" \
    --batch_name "mESC_vs_macrophage" \
    --method_input_path "$OUTPUT_FOLDER/mESC/filtered_L2_E7.5_rep1/" \
    --ground_truth_path "$MACROPHAGE_GROUND_TRUTH"

# mESC vs K562
python3 Analyze_Inferred_GRN.py \
    --inferred_net_filename "mESC_vs_K562_rf_inferred_network.tsv" \
    --method_name "CUSTOM_GRN" \
    --batch_name "mESC_vs_K562" \
    --method_input_path "$OUTPUT_FOLDER/mESC/filtered_L2_E7.5_rep1/" \
    --ground_truth_path "$K562_GROUND_TRUTH"

# Macrophage vs K562
python3 Analyze_Inferred_GRN.py \
    --inferred_net_filename "macrophage_vs_K562_rf_inferred_network.tsv" \
    --method_name "CUSTOM_GRN" \
    --batch_name "macrophage_vs_K562" \
    --method_input_path "$OUTPUT_FOLDER/macrophage/macrophage_buffer1_filtered/" \
    --ground_truth_path "$K562_GROUND_TRUTH"

# Macrophage vs mESC
python3 Analyze_Inferred_GRN.py \
    --inferred_net_filename "macrophage_vs_mESC_rf_inferred_network.tsv" \
    --method_name "CUSTOM_GRN" \
    --batch_name "macrophage_vs_mESC" \
    --method_input_path "$OUTPUT_FOLDER/macrophage/macrophage_buffer1_filtered/" \
    --ground_truth_path "$MESC_GROUND_TRUTH"

# K562 vs macrophage
python3 Analyze_Inferred_GRN.py \
    --inferred_net_filename "K562_vs_macrophage_rf_inferred_network.tsv" \
    --method_name "CUSTOM_GRN" \
    --batch_name "K562_vs_macrophage" \
    --method_input_path "$OUTPUT_FOLDER/K562/K562_human_filtered/" \
    --ground_truth_path "$MACROPHAGE_GROUND_TRUTH"

# K562 vs mESC
python3 Analyze_Inferred_GRN.py \
    --inferred_net_filename "K562_vs_mESC_rf_inferred_network.tsv" \
    --method_name "CUSTOM_GRN" \
    --batch_name "K562_vs_mESC" \
    --method_input_path "$OUTPUT_FOLDER/K562/K562_human_filtered/" \
    --ground_truth_path "$MESC_GROUND_TRUTH"