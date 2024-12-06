#!/bin/bash

# =========== MACROPHAGE STABILITY ==============
for ((i=1; i<=4; i++)); do
    echo Processing macrophage_stability_buffer$i

    python3 Analyze_Inferred_GRN.py \
    --inferred_net_filename "cell_type_specific_trans_regulatory_macrophage.txt" \
    --method_name "LINGER" \
    --batch_name "macrophage_stability_buffer$i" \
    --method_input_path "/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/MACROPHAGE_STABILITY_RESULTS/LINGER_TRAINED_MODELS/Macrophage_buffer$i" \
    --ground_truth_path "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MACROPHAGE_STABILITY/RN204_macrophage_ground_truth.tsv"

done
