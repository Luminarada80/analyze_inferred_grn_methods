# Analyzing Gene Regulatory Network (GRN) Inference Methods

### Gene Regulatory Inference Methods
The goal of single-cell multiomic GRN inference methods is to determine the regulatory relationships between genes
involved in a cellular process. Determining the structure of cell signaling pathways via data-driven approaches will
cut down on time and resource requirements involved with experimentally determining signaling pathway structure. As each
cell type has different behaviors, each cell type also has different cell signaling pathways. These interact in complex
ways to govern the behavior of a cell, and these signaling dynamics are notoriously difficult to determine. Think of this
task as being similar to determining trying to determine what programs are running on a computer by measuring the voltage 
across each transistor. Understanding the structure and function of these networks is key to understanding how diseases perturb
their signaling. The process of piecing together the regulatory interactions in a network is a difficult task, and requires
access to large datasets of genetic information. Single cell RNA and ATAC sequencing methods allow us to determine which genes
were being transcribed at the time of sequencing and which enhancers were open. Researchers are trying to use this information,
paired with information about which transcription factors bind to the open DNA sequence motifs, to infer regulatory relationships
between transcription factors and their target genes. These methods each benchmark how good they are at this task in different ways,
so our goal is to create standardized benchmarking methods to compare each method to one another in an un-biased and fair manner.


To accomplish this, our methods compare multiple GRN inference methods against a common ground truth dataset. First, each inference
method is used to generate an inferred GRN using the same input dataset of single-cell multiomic RNAseq and ATACseq data.

<p align="center">
 <img src="https://github.com/user-attachments/assets/cd9a3dfb-e987-446d-9bbd-9b2c3001ea32" alt="drawing" width="800"/>
</p>


 ### Inferred vs Ground Truth Networks

The ground truth dataset contains a list of interactions between a set of TFs and the TGs they interact with. These interactions are
derived from either ChIPseq, HiC, or LOGOF experiments using the same organism and cell type. We use these interactions
as true positives when evaluating the inferred networks. We expect interactions between TFs and TGs in both the ground truth 
dataset and the inferred network to have a higher score than interactions between TFs and TGs not in the ground truth. 

<p align="center">
 <img src="https://github.com/user-attachments/assets/cf53ad2a-36ce-484e-b707-f595e2698112" alt="drawing" width="800"/>
</p>

## Comparing Inferred Networks to Ground Truth Networks
For our purposes, we can relate the information in an inferred GRN to the ground truth dataset in three main ways:

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

