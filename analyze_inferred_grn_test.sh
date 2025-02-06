#!/bin/bash -l

#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --mem-per-cpu=16G
#SBATCH --output=analyze_inferred_grn.log
#SBATCH --error=analyze_inferred_grn.err

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/lib

# ========= K562 CellOracle vs LINGER vs SCENIC+ ===========
python3 Compare_Inferred_GRN_Methods.py \
    --input_directory "K562_INPUT" \
    --batch_name "K562"

# ====== MACROPHAGE CellOracle vs LINGER ========
# python3 Compare_Inferred_GRN_Methods.py \
#     --input_directory "MACROPHAGE_INPUT" \
#     --batch_name "macrophage"

# ====== mESC TRIPOD ========
# python3 Compare_Inferred_GRN_Methods.py \
#     --input_directory "TRIPOD_INPUT" \
#     --batch_name "Tripod"

# ====== K562 RN118 KNOCKOUT GROUND TRUTH =======
# python3 Compare_Inferred_GRN_Methods.py \
#     --input_directory "K562_RN118_INPUT" \
#     --batch_name "K562_RN118"

# ====== K562 RN119 KNOCKOUT GROUND TRUTH =======
# python3 Compare_Inferred_GRN_Methods.py \
#     --input_directory "K562_RN119_INPUT" \
#     --batch_name "K562_RN119"

# =========== MACROPHAGE STABILITY ==============
# for ((i=1; i<=4; i++)); do
#     echo Processing macrophage_stability_buffer$i

#     python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "cell_type_specific_trans_regulatory_macrophage.txt" \
#     --method_name "LINGER" \
#     --batch_name "macrophage_stability_buffer$i" \
#     --method_input_path "/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/MACROPHAGE_STABILITY_RESULTS/LINGER_TRAINED_MODELS/Macrophage_buffer$i" \
#     --ground_truth_path "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MACROPHAGE_STABILITY/RN204_macrophage_ground_truth.tsv"

# done

# # =========== K562 Stability ==============
# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "cell_type_specific_trans_regulatory_K562.txt" \
#     --method_name "LINGER" \
#     --batch_name "K562_stability" \
#     --method_input_path "/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/K562_STABILITY_RESULTS/LINGER_TRAINED_MODELS" \
#     --ground_truth_path "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_K562/RN117_ChIPSeq_PMID37486787_Human_K562.tsv"

# # ======= K562 CELL LEVEL LINGER GRN ========
# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "cell_specific_trans_regulatory.txt" \
#     --method_name "LINGER" \
#     --batch_name "K562_cell_level" \
#     --method_input_path "/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/K562_RESULTS/LINGER_TRAINED_MODELS/K562_human_filtered/CELL_SPECIFIC_GRNS" \
#     --ground_truth_path "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_K562/RN117_ChIPSeq_PMID37486787_Human_K562.tsv"

# ============= SCENIC+ mESC GRN ===============
# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "scenic_plus_inferred_grn_mESC.tsv" \
#     --method_name "SCENIC_PLUS" \
#     --batch_name "mESC" \
#     --method_input_path "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/SCENIC_PLUS/mESC/mESC_outs/" \
#     --ground_truth_path "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_SC_DATA/RN111.tsv"

# # ============= SCENIC+ K562 GRN ===============
# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "scenic_plus_inferred_grn_K562.tsv" \
#     --method_name "SCENIC_PLUS" \
#     --batch_name "K562" \
#     --method_input_path "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/SCENIC_PLUS/outs/" \
#     --ground_truth_path "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_K562/RN117_ChIPSeq_PMID37486787_Human_K562.tsv"

# =========== Custom GRN Method ===============
# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "rf_inferred_grn.tsv" \
#     --method_name "SINGLE_CELL_CUSTOM_GRN" \
#     --batch_name "mESC_RN114_ChIPX_ESCAPE_sc_rf" \
#     --method_input_path "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/" \
#     --ground_truth_path "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN114_ChIPX_ESCAPE_Mouse_ESC.tsv"

# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "rf_inferred_grn.tsv" \
#     --method_name "SINGLE_CELL_CUSTOM_GRN" \
#     --batch_name "mESC_RN115_LOGOF_ESCAPE_sc_rf" \
#     --method_input_path "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/" \
#     --ground_truth_path "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN115_LOGOF_ESCAPE_Mouse_ESC.tsv"

# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "rf_inferred_grn.tsv" \
#     --method_name "SINGLE_CELL_CUSTOM_GRN" \
#     --batch_name "mESC_RN112_LOGOF_sc_rf" \
#     --method_input_path "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/" \
#     --ground_truth_path "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/reference_networks/RN112_mouse_logof_ESC_path.tsv"

# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "inferred_grn.tsv" \
#     --method_name "CUSTOM_GRN" \
#     --batch_name "mESC_RN115_LOGOF_ESCAPE" \
#     --method_input_path "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/" \
#     --ground_truth_path "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN115_LOGOF_ESCAPE_Mouse_ESC.tsv"

# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "inferred_grn.tsv" \
#     --method_name "CUSTOM_GRN" \
#     --batch_name "mESC_RN114_ChIPX_ESCAPE" \
#     --method_input_path "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/" \
#     --ground_truth_path "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN114_ChIPX_ESCAPE_Mouse_ESC.tsv"

# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "inferred_grn.tsv" \
#     --method_name "CUSTOM_GRN" \
#     --batch_name "mESC_RN112_LOGOF" \
#     --method_input_path "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/" \
#     --ground_truth_path "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/reference_networks/RN112_mouse_logof_ESC_path.tsv"

# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "inferred_grn.tsv" \
#     --method_name "CUSTOM_GRN" \
#     --batch_name "mESC_RN111_ChIP" \
#     --method_input_path "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/" \
#     --ground_truth_path "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_SC_DATA/RN111.tsv"
    

# python3 Analyze_Inferred_GRN.py \
#     --inferred_net_filename "total_motif_regulatory_scores.tsv" \
#     --method_name "CUSTOM_GRN" \
#     --batch_name "test_mESC" \
#     --method_input_path "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/" \
#     --ground_truth_path "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_SC_DATA/RN111.tsv"

# ======= MACROPHAGE CELL LEVEL LINGER GRN ========
# python3 Analyze_Single_Cell_GRN.py \
#     --inferred_net_filename "cell_specific_trans_regulatory.txt" \
#     --method_name "LINGER" \
#     --batch_name "macrophage_cell_level_1_cell" \
#     --method_input_path "/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/MACROPHAGE_RESULTS/LINGER_TRAINED_MODELS/1/SINGLE_CELL_SPECIFIC_GRN" \
#     --ground_truth_path "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MACROPHAGE_STABILITY/RN204_macrophage_ground_truth.tsv"

# python3 Analyze_Single_Cell_GRN.py \
#     --inferred_net_filename "cell_specific_trans_regulatory.txt" \
#     --method_name "LINGER" \
#     --batch_name "macrophage_cell_level_100_cells" \
#     --method_input_path "/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/MACROPHAGE_RESULTS/LINGER_TRAINED_MODELS/1/100_CELL_SPECIFIC_GRNS" \
#     --ground_truth_path "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MACROPHAGE_STABILITY/RN204_macrophage_ground_truth.tsv"

# python3 Analyze_Single_Cell_GRN.py \
#     --inferred_net_filename "cell_specific_trans_regulatory.txt" \
#     --method_name "LINGER" \
#     --batch_name "macrophage_cell_level_300_cells" \
#     --method_input_path "/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/MACROPHAGE_RESULTS/LINGER_TRAINED_MODELS/1/300_CELL_SPECIFIC_GRNS" \
#     --ground_truth_path "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MACROPHAGE_STABILITY/RN204_macrophage_ground_truth.tsv"