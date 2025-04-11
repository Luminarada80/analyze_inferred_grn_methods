#!/bin/bash -l

# For each cell type (lets say mESC), I need the following directories:
# CUSTOM_GRN
#     - CELL_TYPE (i.e. "mESC")
#         - SAMPLE_NAME (i.e. "filtered_L2_E7.5_rep1")
#             - GROUND_TRUTH (i.e. "MESC_RN111_ChIPISeq_BEELINE")
#                 - FEATURE_SET (i.e. "full_network_features_raw")

# So I need to pass in:
# 1) Cell type
# 2) Sample name
# 3) Ground truth name
# 3) Inferred network name

MACROPHAGE_GROUND_TRUTH="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN204_ChIPSeq_ChIPAtlas_Human_Macrophages.tsv"

K562_RN117_CHIPSEQ="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN117_ChIPSeq_PMID37486787_Human_K562.tsv"
K562_RN118_KNOCKTF="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN118_KO_KnockTF_Human_K562.tsv"
K562_RN119_CHIP_AND_KO="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN119_ChIPSeqandKO_PMID37486787andKnockTF_Human_K562.tsv"


submit_job() {
    local INFERRED_NET_FILE=$1
    local CELL_TYPE=$2
    local TARGET_NAME=$3
    local SAMPLE_NAME=$4
    local GROUND_TRUTH_NAME=$5
    local FEATURE_SET=$6
    local INFERRED_NET_DIR=$7
    local GROUND_TRUTH_FILE=$8

    # Ensure the log directory exists
    mkdir -p "LOGS/CUSTOM_GRN_METHOD/${CELL_TYPE}_logs/${SAMPLE_NAME}_logs/${CELL_TYPE}_vs_${TARGET_NAME}/"

    # Submit the job
    sbatch \
        --export=ALL,INFERRED_NET_FILE="$INFERRED_NET_FILE",CELL_TYPE="$CELL_TYPE",TARGET_NAME="$TARGET_NAME",SAMPLE_NAME="$SAMPLE_NAME",GROUND_TRUTH_NAME="$GROUND_TRUTH_NAME",FEATURE_SET="$FEATURE_SET",INFERRED_NET_DIR="$INFERRED_NET_DIR",GROUND_TRUTH_FILE="$GROUND_TRUTH_FILE" \
        --output="LOGS/CUSTOM_GRN_METHOD/${CELL_TYPE}_logs/${SAMPLE_NAME}_logs/${CELL_TYPE}_vs_${TARGET_NAME}/${GROUND_TRUTH_NAME}/${FEATURE_SET}.log" \
        --error="LOGS/CUSTOM_GRN_METHOD/${CELL_TYPE}_logs/${SAMPLE_NAME}_logs/${CELL_TYPE}_vs_${TARGET_NAME}/${GROUND_TRUTH_NAME}/${FEATURE_SET}.log" \
        --job-name="custom_grn_method_${SAMPLE_NAME}_vs_${TARGET_NAME}_${GROUND_TRUTH_NAME}_${FEATURE_SET}" \
        /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/STATISTICAL_ANALYSIS/custom_method_analysis.sh
}

# Change TARGET_NAMES
run_macrophage() {
    local CELL_TYPE="macrophage"

    local SAMPLE_NAMES=(
        "macrophage_buffer1_filtered"
        # "macrophage_buffer2_filtered"
        # "macrophage_buffer3_filtered"
        # "macrophage_buffer4_filtered"
        # "macrophage_buffer1_stability1"
        # "macrophage_buffer1_stability2"
        # "macrophage_buffer1_stability3"
        # "macrophage_buffer1_stability4"
        # "macrophage_buffer1_stability5"
        # "macrophage_buffer1_stability6"
        # "macrophage_buffer1_stability7"
        # "macrophage_buffer1_stability8"
        # "macrophage_buffer1_stability9"
        # "macrophage_buffer1_stability10"
        # "macrophage_buffer2_stability1"
        # "macrophage_buffer2_stability2"
        # "macrophage_buffer2_stability3"
        # "macrophage_buffer2_stability4"
        # "macrophage_buffer2_stability5"
        # "macrophage_buffer2_stability6"
        # "macrophage_buffer2_stability7"
        # "macrophage_buffer2_stability8"
        # "macrophage_buffer2_stability9"
        # "macrophage_buffer2_stability10"
        # "macrophage_buffer3_stability1"
        # "macrophage_buffer3_stability2"
        # "macrophage_buffer3_stability3"
        # "macrophage_buffer3_stability4"
        # "macrophage_buffer3_stability5"
        # "macrophage_buffer3_stability6"
        # "macrophage_buffer3_stability7"
        # "macrophage_buffer3_stability8"
        # "macrophage_buffer3_stability9"
        # "macrophage_buffer3_stability10"
        # "macrophage_buffer4_stability1"
        # "macrophage_buffer4_stability2"
        # "macrophage_buffer4_stability3"
        # "macrophage_buffer4_stability4"
        # "macrophage_buffer4_stability5"
        # "macrophage_buffer4_stability6"
        # "macrophage_buffer4_stability7"
        # "macrophage_buffer4_stability8"
        # "macrophage_buffer4_stability9"
        # "macrophage_buffer4_stability10"
    )

    # Specify the target name for the model prediction 
    # (i.e. for file "macrophage_vs_K562_inferred_network_raw_xgb_pred.tsv" enter "K562")
    local TARGET_NAME=( \
        "macrophage" \
        # "K562"
    )

    # Select the name of the feature sets to analyze
    local FEATURE_SETS=( \
        # "inferred_network_raw" \
        "inferred_network_w_string"
        # "inferred_network_method_combos_summed" \
        # "inferred_network_method_combos_raw" \
        # "inferred_network_string_scores_only"
    )

    # Select the ground truth based on the cell type of the TARGET_NAME
    local GROUND_TRUTHS=( \
        "${MACROPHAGE_GROUND_TRUTH}" \
        # "${K562_RN117_CHIPSEQ}" \
        # "${K562_RN118_KNOCKTF}" \
        # "${K562_RN119_CHIP_AND_KO}"
    )

    # Select the name of the ground truths uncommented above
    local GROUND_TRUTH_NAMES=( \
        "RN204_ChIPSeq" \
        # "RN117_ChIPSeq" \
        # "RN118_KO_KNOCK_TF" \
        # "RN119_CHIP_AND_KO" \
    )

    # Run for each selected sample
    for SAMPLE_NAME in "${SAMPLE_NAMES[@]}"; do

        local INFERRED_NET_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/${CELL_TYPE}/${SAMPLE_NAME}/model_predictions"

        # Run for each feature set for the current sample
        for FEATURE_SET in "${FEATURE_SETS[@]}"; do

            # Compare the inferred network against each ground truth file for the cell type
            for i in "${!GROUND_TRUTHS[@]}"; do
                local GROUND_TRUTH_FILE=${GROUND_TRUTHS[$i]}
                local GROUND_TRUTH_NAME=${GROUND_TRUTH_NAMES[$i]}
                local INFERRED_NET_FILE="${CELL_TYPE}_vs_${TARGET_NAME}_${FEATURE_SET}_xgb_pred.tsv"

                # Submit the job for each sample
                submit_job \
                    "$INFERRED_NET_FILE" \
                    "$CELL_TYPE" \
                    "$TARGET_NAME" \
                    "$SAMPLE_NAME" \
                    "$GROUND_TRUTH_NAME" \
                    "$FEATURE_SET" \
                    "$INFERRED_NET_DIR" \
                    "$GROUND_TRUTH_FILE"
            done
        done
    done
}

run_mESC(){
    local CELL_TYPE="mESC"

    local SAMPLE_NAMES=(
        # "1000_cells_E7.5_rep1"
        # "1000_cells_E7.5_rep2"
        # "1000_cells_E7.75_rep1"
        # "1000_cells_E8.0_rep1"
        # "1000_cells_E8.0_rep2"
        # "1000_cells_E8.5_CRISPR_T_KO"
        # "1000_cells_E8.5_CRISPR_T_WT"
        # "2000_cells_E7.5_rep1"
        # "2000_cells_E8.0_rep1"
        # "2000_cells_E8.0_rep2"
        # "2000_cells_E8.5_CRISPR_T_KO"
        # "2000_cells_E8.5_CRISPR_T_WT"
        # "3000_cells_E7.5_rep1"
        # "3000_cells_E8.0_rep1"
        # "3000_cells_E8.0_rep2"
        # "3000_cells_E8.5_CRISPR_T_KO"
        # "3000_cells_E8.5_CRISPR_T_WT"
        # "4000_cells_E7.5_rep1"
        # "4000_cells_E8.0_rep1"
        # "4000_cells_E8.0_rep2"
        # "4000_cells_E8.5_CRISPR_T_KO"
        # "4000_cells_E8.5_CRISPR_T_WT"
        # "5000_cells_E7.5_rep1"
        # "5000_cells_E8.5_CRISPR_T_KO"
        # "70_percent_subsampled_1"
        # "70_percent_subsampled_2"
        # "70_percent_subsampled_3"
        # "70_percent_subsampled_4"
        # "70_percent_subsampled_5"
        # "70_percent_subsampled_6"
        # "70_percent_subsampled_7"
        # "70_percent_subsampled_8"
        # "70_percent_subsampled_9"
        # "70_percent_subsampled_10"
        "filtered_L2_E7.5_rep1"
        "filtered_L2_E7.5_rep2"
        # "filtered_L2_E7.75_rep1"
        # "filtered_L2_E8.0_rep1"
        # "filtered_L2_E8.0_rep2"
        # "filtered_L2_E8.5_CRISPR_T_KO"
        # "filtered_L2_E8.5_CRISPR_T_WT"
        # "filtered_L2_E8.5_rep1"
        # "filtered_L2_E8.5_rep2"
        # "filtered_L2_E8.75_rep1"
        # "filtered_L2_E8.75_rep2"
    )
    local FEATURE_SETS=( \
        "inferred_network_raw" \
        "inferred_network_w_string"
        "inferred_network_method_combos_summed" \
        "inferred_network_method_combos_raw" \
        "inferred_network_string_scores_only"
    )

    local GROUND_TRUTHS=( \
        "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN111_ChIPSeq_BEELINE_Mouse_ESC.tsv"
        "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/reference_networks/RN112_mouse_logof_ESC_path.tsv"
        "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN114_ChIPX_ESCAPE_Mouse_ESC.tsv"
        "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN115_LOGOF_ESCAPE_Mouse_ESC.tsv"
    )

    local GROUND_TRUTH_NAMES=( \
        "RN111_ChIPSeq_BEELINE" \
        "RN112_LOGOF" \
        "RN114_CHIPX_ESCAPE" \
        "RN115_LOGOF_ESCAPE" \
    )

    # Run for each selected sample
    for SAMPLE_NAME in "${SAMPLE_NAMES[@]}"; do

        local INFERRED_NET_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/${CELL_TYPE}/${SAMPLE_NAME}/"

        # Run for each kind of inferred network for the current sample
        for FEATURE_SET in "${FEATURE_SETS[@]}"; do

            # Compare the inferred network against each ground truth file for the cell type
            for i in "${!GROUND_TRUTHS[@]}"; do
                local GROUND_TRUTH_FILE=${GROUND_TRUTHS[$i]}
                local GROUND_TRUTH_NAME=${GROUND_TRUTH_NAMES[$i]}
                local INFERRED_NET_FILE="${FEATURE_SET}_xgb_inferred_grn.tsv"


                # Submit the job for each sample
                submit_job \
                    "$INFERRED_NET_FILE" \
                    "$CELL_TYPE" \
                    "$SAMPLE_NAME" \
                    "$GROUND_TRUTH_NAME" \
                    "$FEATURE_SET" \
                    "$INFERRED_NET_DIR" \
                    "$GROUND_TRUTH_FILE"
            done
        done
    done
}

run_K562(){
    local CELL_TYPE="K562"

    
    local SAMPLE_NAMES=( \
        "K562_human_filtered"
    )

    # Specify the target name for the model prediction 
    # (i.e. for file "K562_vs_macrophage_inferred_network_raw_xgb_pred.tsv" enter "macrophage")
    local TARGET_NAME=( \
        # "macrophage" \
        "K562" \
    )

    # Select the name of the feature sets to analyze
    local FEATURE_SETS=( \
        "inferred_network_raw" \
        "inferred_network_w_string"
        # "inferred_network_method_combos_summed" \
        # "inferred_network_method_combos_raw" \
        # "inferred_network_string_scores_only"
    )

    # Select the ground truth based on the cell type of the TARGET_NAME
    local GROUND_TRUTHS=( \
        # "${MACROPHAGE_GROUND_TRUTH}" \
        # "${K562_RN117_CHIPSEQ}" \
        "${K562_RN118_KNOCKTF}" \
        "${K562_RN119_CHIP_AND_KO}"

    )

    # Select the name of the ground truths uncommented above
    local GROUND_TRUTH_NAMES=( \
        # "RN204_ChIPSeq" \
        # "RN117_ChIPSeq" \
        "RN118_KO_KNOCK_TF" \
        "RN119_CHIP_AND_KO" \
    )

    # Run for each sample of the cell type
    for SAMPLE_NAME in "${SAMPLE_NAMES[@]}"; do

        # Specify the location fo the model predictions for the current sample
        local INFERRED_NET_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/${CELL_TYPE}/${SAMPLE_NAME}/model_predictions"

        # Run for each feature set
        for FEATURE_SET in "${FEATURE_SETS[@]}"; do

            # Compare the model predictions for the cell type against each ground truth file for the target
            for i in "${!GROUND_TRUTHS[@]}"; do
                local GROUND_TRUTH_FILE=${GROUND_TRUTHS[$i]}
                local GROUND_TRUTH_NAME=${GROUND_TRUTH_NAMES[$i]}
                
                local INFERRED_NET_FILE="${CELL_TYPE}_vs_${TARGET_NAME}_${FEATURE_SET}_xgb_pred.tsv"

                # Submit the job for each sample
                submit_job \
                    "$INFERRED_NET_FILE" \
                    "$CELL_TYPE" \
                    "$TARGET_NAME" \
                    "$SAMPLE_NAME" \
                    "$GROUND_TRUTH_NAME" \
                    "$FEATURE_SET" \
                    "$INFERRED_NET_DIR" \
                    "$GROUND_TRUTH_FILE"
            done
        done
    done
}

# run_mESC
run_K562
run_macrophage