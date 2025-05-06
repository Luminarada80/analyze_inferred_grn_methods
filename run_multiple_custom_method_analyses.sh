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
REF_NET_DIR="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS"

MACROPHAGE_RN204_CHIPSEQ="$REF_NET_DIR/RN204_ChIPSeq_ChIPAtlas_Human_Macrophages.tsv"

K562_RN117_CHIPSEQ="$REF_NET_DIR/RN117_ChIPSeq_PMID37486787_Human_K562.tsv"
K562_RN118_KNOCKTF="$REF_NET_DIR/RN118_KO_KnockTF_Human_K562.tsv"
K562_RN119_CHIP_AND_KO="$REF_NET_DIR/RN119_ChIPSeqandKO_PMID37486787andKnockTF_Human_K562.tsv"

MESC_RN111_CHIPSEQ="$REF_NET_DIR/RN111_ChIPSeq_BEELINE_Mouse_ESC.tsv"
MESC_RN112_LOGOF="$REF_NET_DIR/RN112_LOGOF_BEELINE_Mouse_ESC.tsv"
MESC_RN114_CHIPX_ESCAPE="$REF_NET_DIR/RN114_ChIPX_ESCAPE_Mouse_ESC.tsv"
MESC_RN115_LOGOF_ESCAPE="$REF_NET_DIR/RN115_LOGOF_ESCAPE_Mouse_ESC.tsv"


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

    local SAMPLE_NAME=(
        "macrophage_buffer1_filtered"
        # "macrophage_buffer2_filtered"
        # "macrophage_buffer3_filtered"
        # "macrophage_buffer4_filtered"
    )

    # Specify the target name for the model prediction 
    # (i.e. for file "macrophage_vs_K562_inferred_network_raw_xgb_pred.tsv" enter "K562")
    local TARGET_NAME=( \
        # "macrophage" \
        # "K562" \
        "mESC"
    )

    # Select the name of the feature sets to analyze
    local FEATURE_SETS=( \
        "inferred_score_df.parquet" \
        # "inferred_network_w_string" \
        # "inferred_network_string_scores_only" \
        # "inferred_network_w_string_no_tf"
    )

    # Select the ground truth based on the cell type of the TARGET_NAME
    local GROUND_TRUTHS=( \
        # Macrophage Ground Truths
        # "${MACROPHAGE_RN204_CHIPSEQ}" \

        # K562 Ground Truths
        # "${K562_RN117_CHIPSEQ}" \
        # "${K562_RN118_KNOCKTF}" \
        # "${K562_RN119_CHIP_AND_KO}" \

        # mESC Ground Truths
        "${MESC_RN111_CHIPSEQ}" \
        # "${MESC_RN112_LOGOF}" \
        # "${MESC_RN114_CHIPX_ESCAPE}" \
        # "${MESC_RN115_LOGOF_ESCAPE}"
    )

    local GROUND_TRUTH_NAMES=( \
        # Macrophage Ground Truths
        # "RN204_ChIPSeq" \

        # K562 Ground Truths
        # "RN117_ChIPSeq" \
        # "RN118_KO_KNOCK_TF" \
        # "RN119_CHIP_AND_KO" \

        # mESC Ground Truths
        "RN111_CHIPSEQ" \
        "RN112_LOGOF" \
        "RN114_CHIPX_ESCAPE" \
        "RN115_LOGOF_ESCAPE"
    )

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
}

run_mESC(){
    local CELL_TYPE="mESC"

    local SAMPLE_NAME=(
        "filtered_L2_E7.5_rep1"
        # "filtered_L2_E7.5_rep2"
        # "filtered_L2_E7.75_rep1"
        # "filtered_L2_E8.0_rep1"
        # "filtered_L2_E8.0_rep2"
        # "filtered_L2_E8.5_rep1"
        # "filtered_L2_E8.5_rep2"
        # "filtered_L2_E8.75_rep1"
        # "filtered_L2_E8.75_rep2"
    )

    # Specify the target name for the model prediction 
    # (i.e. for file "macrophage_vs_K562_inferred_network_raw_xgb_pred.tsv" enter "K562")
    local TARGET_NAME=( \
        # "macrophage" \
        # "K562" \
        "mESC"
    )

    local FEATURE_SETS=( \
        "inferred_score_df" \
        # "inferred_network_enrich_feat" \
        # "inferred_network" \
        # "inferred_network_raw" \
        # "inferred_network_w_string" \
        # "inferred_network_string_scores_only"
    )

    local GROUND_TRUTHS=( \
        # Macrophage Ground Truths
        # "${MACROPHAGE_RN204_CHIPSEQ}" \

        # K562 Ground Truths
        # "${K562_RN117_CHIPSEQ}" \
        # "${K562_RN118_KNOCKTF}" \
        # "${K562_RN119_CHIP_AND_KO}" \

        # mESC Ground Truths
        "${MESC_RN111_CHIPSEQ}" \
        # "${MESC_RN112_LOGOF}" \
        # "${MESC_RN114_CHIPX_ESCAPE}" \
        # "${MESC_RN115_LOGOF_ESCAPE}"
    )

    local GROUND_TRUTH_NAMES=( \
        # Macrophage Ground Truths
        # "RN204_ChIPSeq" \

        # K562 Ground Truths
        # "RN117_ChIPSeq" \
        # "RN118_KO_KNOCK_TF" \
        # "RN119_CHIP_AND_KO" \

        # mESC Ground Truths
        "RN111_CHIPSEQ" \
        # "RN112_LOGOF" \
        # "RN114_CHIPX_ESCAPE" \
        # "RN115_LOGOF_ESCAPE"
    )

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

}

run_K562(){
    local CELL_TYPE="K562"

    
    local SAMPLE_NAME=( \
        "K562_human_filtered"
    )

    # Specify the target name for the model prediction 
    # (i.e. for file "K562_vs_macrophage_inferred_network_raw_xgb_pred.tsv" enter "macrophage")
    local TARGET_NAME=( \
        # "macrophage" \
        # "K562" \
        "mESC"
    )

    # Select the name of the feature sets to analyze
    local FEATURE_SETS=( \
        "inferred_network_raw" \
        "inferred_network_w_string"
        # "inferred_network_string_scores_only" \
        # "inferred_network_w_string_no_tf"
    )

    # Select the ground truth based on the cell type of the TARGET_NAME
    local GROUND_TRUTHS=( \
        # Macrophage Ground Truths
        # "${MACROPHAGE_RN204_CHIPSEQ}" \

        # K562 Ground Truths
        # "${K562_RN117_CHIPSEQ}" \
        # "${K562_RN118_KNOCKTF}" \
        # "${K562_RN119_CHIP_AND_KO}" \

        # mESC Ground Truths
        "${MESC_RN111_CHIPSEQ}" \
        "${MESC_RN112_LOGOF}" \
        "${MESC_RN114_CHIPX_ESCAPE}" \
        "${MESC_RN115_LOGOF_ESCAPE}"
    )

    local GROUND_TRUTH_NAMES=( \
        # Macrophage Ground Truths
        # "RN204_ChIPSeq" \

        # K562 Ground Truths
        # "RN117_ChIPSeq" \
        # "RN118_KO_KNOCK_TF" \
        # "RN119_CHIP_AND_KO" \

        # mESC Ground Truths
        "RN111_CHIPSEQ" \
        "RN112_LOGOF" \
        "RN114_CHIPX_ESCAPE" \
        "RN115_LOGOF_ESCAPE"
    )

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
}


run_combined_model(){
    local CELL_TYPE="combined_inferred_dfs"

    local SAMPLE_NAME=(
        "filtered_L2_E7.5_rep1"
        # "filtered_L2_E7.5_rep2"
        # "filtered_L2_E7.75_rep1"
        # "filtered_L2_E8.0_rep1"
        # "filtered_L2_E8.0_rep2"
        # "filtered_L2_E8.5_rep1"
        # "filtered_L2_E8.5_rep2"
        # "filtered_L2_E8.75_rep1"
        # "filtered_L2_E8.75_rep2"
    )

    # Specify the target name for the model prediction 
    # (i.e. for file "macrophage_vs_K562_inferred_network_raw_xgb_pred.tsv" enter "K562")
    local TARGET_NAME=( \
        # "macrophage" \
        # "K562" \
        "mESC"
    )

    local FEATURE_SETS=( \
        "inferred_score_df" \
        # "inferred_network_enrich_feat" \
        # "inferred_network" \
        # "inferred_network_raw" \
        # "inferred_network_w_string" \
        # "inferred_network_string_scores_only"
    )

    local GROUND_TRUTHS=( \
        # Macrophage Ground Truths
        # "${MACROPHAGE_RN204_CHIPSEQ}" \

        # K562 Ground Truths
        # "${K562_RN117_CHIPSEQ}" \
        # "${K562_RN118_KNOCKTF}" \
        # "${K562_RN119_CHIP_AND_KO}" \

        # mESC Ground Truths
        # "${MESC_RN111_CHIPSEQ}" \
        # "${MESC_RN112_LOGOF}" \
        # "${MESC_RN114_CHIPX_ESCAPE}" \
        "${MESC_RN115_LOGOF_ESCAPE}"
    )

    local GROUND_TRUTH_NAMES=( \
        # Macrophage Ground Truths
        # "RN204_ChIPSeq" \

        # K562 Ground Truths
        # "RN117_ChIPSeq" \
        # "RN118_KO_KNOCK_TF" \
        # "RN119_CHIP_AND_KO" \

        # mESC Ground Truths
        # "RN111_CHIPSEQ" \
        # "RN112_LOGOF" \
        # "RN114_CHIPX_ESCAPE" \
        "RN115_LOGOF_ESCAPE"
    )

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

}

run_combined_model
# run_mESC
# run_K562
# run_macrophage