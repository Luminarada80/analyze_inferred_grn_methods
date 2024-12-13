#!/bin/bash -l

#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --mem-per-cpu=16G
#SBATCH --output=analyze_inferred_grn.log
#SBATCH --error=analyze_inferred_grn.err

# ========= K562 CellOracle vs LINGER ===========
# python3 Compare_Inferred_GRN_Methods.py \
#     --input_directory "K562_INPUT" \
#     --batch_name "K562"

# ====== MACROPHAGE CellOracle vs LINGER ========
# python3 Compare_Inferred_GRN_Methods.py \
#     --input_directory "MACROPHAGE_INPUT" \
#     --batch_name "macrophage"

# ====== MACROPHAGE CellOracle vs LINGER ========
python3 Compare_Inferred_GRN_Methods.py \
    --input_directory "TRIPOD_INPUT" \
    --batch_name "Tripod"

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