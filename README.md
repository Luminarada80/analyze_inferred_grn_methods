# Analyzing Gene Regulatory Network (GRN) Inference Methods

## Requirements:
Install the GRN analysis toolkit to a conda environment using `conda install luminarada80::grn_analysis_tools`

## Methods:
### Gene Regulatory Inference Methods
The goal of a GRN inference method is to determine the regulatory relationships between genes
involved in a cellular process. Normally, these regulatory relationships are determined by rigorous molecular experiments that identify regulatory relationships by knocking in or out the gene and studying the changes that occur. Determining the structure of cell signaling pathways via data-driven approaches using large-scale multiomic data will
cut down on the time and resource requirements involved with experimentally determining signaling pathway structure. As each
cell type have different behaviors, they also have different cell signaling pathways. To further complicate the task, multiple pathways interact to integrate and process multiple sources of data to govern the overall behavior of a cell. Trying to determine the structure of a signaling pathway from gene expression is similar to determining trying to determine what programs are running on a computer by measuring the voltage across each transistor, no easy task. The process of piecing together the regulatory interactions in a network is a difficult problem to solve and requires access to large datasets and sophisticated algorithms to make sense of the data. Multiomic data expands on methods that only use single cell RNA sequencing by also including paired ATAC sequencing. This allow us to determine which genes were being transcribed at the time of sequencing along with which enhancers were open. Researchers are trying to use this information to infer regulatory relationships between transcription factors and their target genes. The goal of this project is to create a framework to benchmark how effective they are at this task and to create standardized benchmarking methods to compare future methods to one another in an un-biased and fair manner.

To accomplish this, our methods compare multiple GRN inference methods against a common ground truth dataset. First, each inference
method is used to generate an inferred GRN using the same input dataset of single-cell multiomic RNAseq and ATACseq data.

<p align="center">
 <img src="https://github.com/user-attachments/assets/cd9a3dfb-e987-446d-9bbd-9b2c3001ea32" alt="drawing" width="600"/>
</p>


 ### Inferred vs Ground Truth Networks

The ground truth dataset contains a list of interactions between a set of TFs and the TGs they interact with. These interactions are
derived from either ChIPseq, HiC, or LOGOF experiments using the same organism and cell type. We use these interactions
as true positives when evaluating the inferred networks. We expect interactions between TFs and TGs in both the ground truth 
dataset and the inferred network to have a higher score than interactions between TFs and TGs not in the ground truth. 

<p align="center">
 <img src="https://github.com/user-attachments/assets/cf53ad2a-36ce-484e-b707-f595e2698112" alt="drawing" width="600"/>
</p>

## Comparing Inferred Networks to Ground Truth Networks
For our purposes, we can relate the information in an inferred GRN to the ground truth dataset in three main ways:

<p align="center">
 <img src="https://github.com/user-attachments/assets/2218f503-74ee-482d-a5c6-ffd33ea84071" alt="drawing" width="900"/>
</p>

Each ground truth and non-ground truth edge has an **edge score** calculated by the inference method. We can plot a histogram of these scores and separate the edges by
ground truth (orange) and non-ground truth (blue). We can set a threshold at one standard deviation below the mean ground truth value, and use this to determine if an edge
score should be true or false.
<p align="center">
 <img src="https://github.com/user-attachments/assets/5c1cd3d9-5401-4d57-ac77-73f6bd2a5075" alt="Image 3" width="800"/>
</p>

|True Positive|False Positive|True Negative|False Negative|
|:-----------:|:------------:|:------------:|:-----------:|
|The edge is **above** the threshold and is in the **ground truth**|The edge is **above** the threshold but is in the **non-ground truth**|The edge is **below** the threshold and is in the **ground truth**|The edge is **below** the threshold and is in the **non-ground truth** |


## Comparing Methods

### AUROC: Area Under the Receiver Operating Characteristic Curve
We can evaluate the performce of each method compared to the same ground truth, and evaluate the ratio between the **True Positive Rate (TPR)** and **False Positive Rate (FPR)** across the inferred edge scores. This allows us to compare model performance across the same ground truth dataset. We also compare the performance to a random uniform distribution across the inferred and ground truth edges to ensure that random edge scores would have an **Area Under the Curve (AUC)** of 0.50, which indicates a performance no better than random chance.

<p align="center">
 <img src="https://github.com/user-attachments/assets/eade8349-dad2-479d-bdff-3cefa3d0b5cf" alt="Image 3" width="600"/>
</p>



