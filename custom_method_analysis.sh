#!/bin/bash -l

#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 1
#SBATCH --mem=32G

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/lib

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/STATISTICAL_ANALYSIS"
OUTPUT_FOLDER="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output"
METHOD_NAME="CUSTOM_GRN"

# Run the Python script with the selected parameters.
echo "python3 $PROJECT_DIR/Analyze_Inferred_GRN.py "
echo "    --inferred_net_filename ${INFERRED_NET_FILE} "
echo "    --method_name $METHOD_NAME "
echo "    --batch_name ${CELL_TYPE}/${SAMPLE_NAME}/${GROUND_TRUTH_NAME}/${INFERRED_NET_NAME} "
echo "    --method_input_path $INFERRED_NET_DIR "
echo "    --ground_truth_path $GROUND_TRUTH_FILE"

# Run the Python script with the selected parameters.
python3 "$PROJECT_DIR/Analyze_Inferred_GRN.py" \
    --inferred_net_filename "$INFERRED_NET_FILE" \
    --method_name "$METHOD_NAME" \
    --batch_name "${CELL_TYPE}/${SAMPLE_NAME}/${GROUND_TRUTH_NAME}/${INFERRED_NET_NAME}" \
    --method_input_path "$INFERRED_NET_DIR" \
    --ground_truth_path "$GROUND_TRUTH_FILE"



# PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/STATISTICAL_ANALYSIS"
# METHOD_NAME="CUSTOM_GRN"
# OUTPUT_FOLDER="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output"

# # ----- GROUND TRUTH FILE PATHS -----
# K562_RN117_ChIPSeq="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN117_ChIPSeq_PMID37486787_Human_K562.tsv"
# K562_RN118_KO_KNOCK_TF="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN118_KO_KnockTF_Human_K562.tsv"
# K562_RN119_CHIP_AND_KO="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN119_ChIPSeqandKO_PMID37486787andKnockTF_Human_K562.tsv"

# MACROPHAGE_RN204_ChIPSeq="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN204_ChIPSeq_ChIPAtlas_Human_Macrophages.tsv"

# MESC_RN111_ChIPSeq_BEELINE="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN111_ChIPSeq_BEELINE_Mouse_ESC.tsv"
# MESC_RN112_LOGOF="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/reference_networks/RN112_mouse_logof_ESC_path.tsv"
# MESC_RN114_CHIPX_ESCAPE="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN114_ChIPX_ESCAPE_Mouse_ESC.tsv"
# MESC_RN115_LOGOF_ESCAPE="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN115_LOGOF_ESCAPE_Mouse_ESC.tsv"

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


# # mESC vs macrophage
# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "mESC_vs_macrophage_rf_inferred_network.tsv" \
#     --method_name "CUSTOM_GRN" \
#     --batch_name "mESC_vs_macrophage" \
#     --method_input_path "$OUTPUT_FOLDER/mESC/filtered_L2_E7.5_rep1/" \
#     --ground_truth_path "$MACROPHAGE_GROUND_TRUTH"

# # mESC vs K562
# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "mESC_vs_K562_rf_inferred_network.tsv" \
#     --method_name "CUSTOM_GRN" \
#     --batch_name "mESC_vs_K562" \
#     --method_input_path "$OUTPUT_FOLDER/mESC/filtered_L2_E7.5_rep1/" \
#     --ground_truth_path "$K562_GROUND_TRUTH"

# # Macrophage vs K562
# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "macrophage_vs_K562_rf_inferred_network.tsv" \
#     --method_name "CUSTOM_GRN" \
#     --batch_name "macrophage_vs_K562" \
#     --method_input_path "$OUTPUT_FOLDER/macrophage/macrophage_buffer1_filtered/" \
#     --ground_truth_path "$K562_GROUND_TRUTH"

# # Macrophage vs mESC
# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "macrophage_vs_mESC_rf_inferred_network.tsv" \
#     --method_name "CUSTOM_GRN" \
#     --batch_name "macrophage_vs_mESC" \
#     --method_input_path "$OUTPUT_FOLDER/macrophage/macrophage_buffer1_filtered/" \
#     --ground_truth_path "$MESC_GROUND_TRUTH"

# # K562 vs macrophage
# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "K562_vs_macrophage_rf_inferred_network.tsv" \
#     --method_name "CUSTOM_GRN" \
#     --batch_name "K562_vs_macrophage" \
#     --method_input_path "$OUTPUT_FOLDER/K562/K562_human_filtered/" \
#     --ground_truth_path "$MACROPHAGE_GROUND_TRUTH"

# # K562 vs mESC
# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "K562_vs_mESC_rf_inferred_network.tsv" \
#     --method_name "CUSTOM_GRN" \
#     --batch_name "K562_vs_mESC" \
#     --method_input_path "$OUTPUT_FOLDER/K562/K562_human_filtered/" \
#     --ground_truth_path "$MESC_GROUND_TRUTH"

# Sample vs Sample
# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "mESC1_vs_mESC2_rf_inferred_grn.tsv" \
#     --method_name "CUSTOM_GRN" \
#     --batch_name "mESC1_vs_mESC2" \
#     --method_input_path "$OUTPUT_FOLDER/mESC/filtered_L2_E7.5_rep1/" \
#     --ground_truth_path "$MESC_GROUND_TRUTH"

# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "macrophage1_vs_macrophage2_rf_inferred_grn.tsv" \
#     --method_name "CUSTOM_GRN" \
#     --batch_name "macrophage1_vs_macrophage2" \
#     --method_input_path "$OUTPUT_FOLDER/macrophage/macrophage_buffer1_filtered/" \
#     --ground_truth_path "$MACROPHAGE_GROUND_TRUTH"

# ----- Applying trained model to the same sample it was trained on -----
# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "mESC1_rf_inferred_grn.tsv" \
#     --method_name "CUSTOM_GRN" \
#     --batch_name "mESC1" \
#     --method_input_path "$OUTPUT_FOLDER/mESC/filtered_L2_E7.5_rep1/" \
#     --ground_truth_path "$MESC_GROUND_TRUTH"

# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "macrophage1_rf_inferred_grn.tsv" \
#     --method_name "CUSTOM_GRN" \
#     --batch_name "macrophage1" \
#     --method_input_path "$OUTPUT_FOLDER/macrophage/macrophage_buffer1_filtered/" \
#     --ground_truth_path "$MACROPHAGE_GROUND_TRUTH"

# ----- Analyzing the cell-level GRNs -----
# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "rf_inferred_grn.tsv" \
#     --method_name "CUSTOM_GRN" \
#     --batch_name "cell_level" \
#     --method_input_path "$OUTPUT_FOLDER/mESC/filtered_L2_E7.5_rep1/cell_networks_rf/" \
#     --ground_truth_path "$MESC_GROUND_TRUTH"

# ======== AGGREGATED FEATURES XGBOOST MODEL ==========
# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "xgb_inferred_agg_grn.tsv" \
#     --method_name "CUSTOM_GRN" \
#     --batch_name "K562_AGG_SAME_GROUND_TRUTH" \
#     --method_input_path "$OUTPUT_FOLDER/K562/K562_human_filtered/" \
#     --ground_truth_path "$K562_GROUND_TRUTH"

# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "xgb_inferred_agg_grn.tsv" \
#     --method_name "CUSTOM_GRN" \
#     --batch_name "K562_AGG_KNOCK_TF" \
#     --method_input_path "$OUTPUT_FOLDER/K562/K562_human_filtered/" \
#     --ground_truth_path "$K562_RN118_KO_KNOCK_TF"

# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "xgb_inferred_agg_grn.tsv" \
#     --method_name "CUSTOM_GRN" \
#     --batch_name "K562_AGG_CHIP_AND_KO" \
#     --method_input_path "$OUTPUT_FOLDER/K562/K562_human_filtered/" \
#     --ground_truth_path "$K562_RN119_CHIP_AND_KO"

