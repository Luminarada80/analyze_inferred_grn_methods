# Analyzing Gene Regulatory Network (GRN) Inference Methods
This pipeline compares GRN inference methods against a common ground truth dataset. 
First, each inference method is used to generate an inferred network from the same input dataset of single-cell multiomic
RNAseq and ATACseq data. These inferred networks are then evaluated against an experimentally-derived ground truth dataset from 
either a ChIPseq, HiC, or LOGOF experiment from the same organism and tissue / cell type. For each inference method, interactions between 
each transcription factor (TF) and target gene (TG) in the dataset are scored based on the calculated likelihood of a TF regulating a TG. 

The ground truth dataset contains a list of interactions between a set of TFs and the TGs they interact with. We use these interactions
as true positives when evaluating the inferred networks. We expect interactions between TFs and TGs in both the ground truth 
dataset and the inferred network to have a higher score than interactions between TFs and TGs not in the ground truth. 

For our purposes, we can categorize the information in an inferred GRN into three groups:

**1. Interactions involving genes that are not in the ground truth**

The inferred networks contain edges between genes that are not present in the ground truth. We can only compare the predicted interactions between
genes present in both the inferred network and the ground truth network, because we have no information about the interactions between other genes. 
We remove any predicted interactions that involve TFs or TGs not present in the ground truth from the inferred network, so we are comparing the same
set of TFs and TGs.

**2. Interactions present in the ground truth**

We regard interactions between TFs and TGs found in the ground truth as true positives, as these edges have been shown to exist experimentally.

**3. Interactions not present in the ground truth**

We regard interactions between TFs and TGs not found in the ground truth as true negatives, as these edges are not shown to exist experimentally

We can evaluate the accuracy of the inference methods by comparing edges between TFs and TGs in the ground truth against TFs and TGs not in the ground
truth. As each edge has an inferred score indicating the predicted regulatory interaction, edges in the ground truth should have a higher score than edges not in the ground truth. 

![image](https://github.com/user-attachments/assets/dfb9a535-7d7c-4bc1-b6ea-412fbcb85782)

## Requirements:
Install the GRN analysis toolkit to a conda environment using `conda install luminarada80::grn_analysis_tools`

