# Analyzing Gene Regulatory Network (GRN) Inference Methods

## Requirements:
Install the GRN analysis toolkit to a conda environment using `conda install luminarada80::grn_analysis_tools`

## Methods:
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

<table border="3" style="border-collapse: collapse; text-align: center; width: 100%;">
  <thead>
    <tr>
      <th>Interactions with genes not in the ground truth</th>
      <th>Interactions present in the ground truth</th>
      <th>Interactions not present in the ground truth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        Genes in the inferred network that are not in the ground truth network. These are <b>removed</b> to only compare shared genes.
      </td>
      <td>
        These are the <b>true positive</b> interactions, as they have been shown to exist experimentally.
      </td>
      <td>
        These are the <b>true negatives</b>, as they are not in the ground truth.
      </td>
    </tr>
    <tr>
      <td>
       <p align="center">
        <img src="https://github.com/user-attachments/assets/1d796d06-1306-4756-a016-8db13b035345" alt="Image 1" width="200"/>
       </p>
      </td>
      <td>
       <p align="center">
        <img src="https://github.com/user-attachments/assets/02d6e9bf-d3c3-485d-903a-94bd4a498dcb" alt="Image 2" width="200"/>
        </p>
      </td>
      <td>
       <p align="center">
        <img src="https://github.com/user-attachments/assets/cc2ec3e1-a052-4157-b4c2-3b6061332a4b" alt="Image 3" width="200"/>
        </p>
      </td>
    </tr>
  </tbody>
</table>

Each ground truth and non-ground truth edge has an **edge score** calculated by the inference method. We can plot a histogram of these scores and separate the edges by
ground truth (orange) and non-ground truth (blue). We can set a threshold at one standard deviation below the mean ground truth value, and use this to determine if an edge
score should be true or false.
<p align="center">
 <img src="https://github.com/user-attachments/assets/dc2a7745-0fbb-400b-823b-c59feedfe22e" alt="Image 3" width="1200"/>
</p>

|True Positive|False Positive|True Negative|False Negative|
|:-----------:|:------------:|:------------:|:-----------:|
|The edge is **above** the threshold and is in the **ground truth**|The edge is **above** the threshold but is in the **non-ground truth**|The edge is **below** the threshold and is in the **ground truth**|The edge is **below** the threshold and is in the **non-ground truth** |

