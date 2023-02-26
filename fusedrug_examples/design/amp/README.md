# Peptide Antimicrobial Peptides (AMP) Design 

## Introduction

Antimicrobial peptide (AMP) design with high potency and low toxicity task.
AMPs are part of the innate immune response and drug candidates for tackling antibiotic resistance. AMPs are generally between 12 and 50 amino acids.  

In this work, we offer an implementation for data processing, Wasserstein Autoencoder, and classifiers for both AMP and toxicity, 
which are based (with a few additional extensions) on [Payel Das et al. Accelerated antimicrobial discovery via deep generative models and molecular dynamics simulations](https://www.nature.com/articles/s41551-021-00689-x).
The implementation was broken to easy to use off-the-shelf components that are useful for many sequence-based tasks.

<br/>


## Dataset
The dataset used for the task was built from several publicly available datasets, mixing labeled and unlabeled peptides. Labels are provided for both toxicity and AMP. 
  * [DBAASP - Database of Antimicrobial Activity and Structure of Peptides](https://dbaasp.org/)
  * [SATPdb - structurally annotated therapeutic peptides database](http://crdd.osdd.net/raghava/satpdb/)
  * [TOXINPRED](https://webs.iiitd.edu.in/raghava/toxinpred/dataset.php)
  * [UniProt](https://www.uniprot.org/)

The entire processing code can be found in datasets.py. PeptidesDatasets.dataset() will create the dataset instance. 

<br/>

## Task and Modeling 

<br/>
Antimicrobial peptide (AMP) design with high potency and low toxicity task.
* Implementation of sequence-based Wasserstein Autoencoder.
* Two options for the encoder and decoder: Transformer based and GRU based.
* Several augmentation methods for sequences
* Auxiliary heads are used to either explore the latent space or to encourage it to linearly separate AMPs and toxicity. 

<br/>
Peptide classification (AMP / Toxicity)
* Sequence-based classification 
* Two options for the backbone: Transformer based and GRU based.
* Several augmentation methods for sequences
<br/>



## Usage Instructions

1. Download the data:

  * Download DBAASP data, specifically peptides-complete.csv from: https://dbaasp.org/download-dataset?source=peptides

    And set the environment variable:

    `export DBAASP_DATA_PATH=<full path to peptides-complete.json>`

  * Download UniProt-reviewed peptides. Only two columns file (Entry and Sequence) in uncompressed tsv format downloaded from https://www.uniprot.org/uniprotkb?facets=reviewed%3Atrue%2Clength%3A%5B1%20TO%20200%5D&query=%2A

    And set the environment variable:

    `export UNIPROT_PEPTIDE_REVIEWED_DATA_PATH=< path to tsv file >` 

  * Download UniProt-unreviewed peptides. Only two columns file (Entry and Sequence) in uncompressed tsv format from https://www.uniprot.org/uniprotkb?facets=reviewed%3Atrue%2Clength%3A%5B1%20TO%20200%5D&query=%2A

    And set the environment variable:

    `export UNIPROT_PEPTIDE_NOT_REVIEWED_DATA_PATH=< path to tsv file >` 


  * Download TOXINPRED data from https://webs.iiitd.edu.in/raghava/toxinpred/dataset.php
    And set environment variable:

    `export TOXIN_PRED_DATA_PATH=< path to the folder that contains the downloaded files>`

  * Download 'antimicrobial.fasta' and 'toxic.fasta' from https://webs.iiitd.edu.in/raghava/satpdb/down.php

    And set the environment variable:

    `export SATPDB_DATA_PATH=< path to the folder that contains the downloaded files>` 

  *  Download '*_ne.fasta' and '*_po.fasta' files from https://sourceforge.net/projects/axpep/
  
      And set the environment variable:
  
      `export AXPEP_DATA_PATH=<path to the folder that contains the downloaded files>`

        
2. Train design model

  * Modify the configuration file if necessary `design/config.xml`

  * Run the training script `python design/main_design_train.py`

3. Train classification models

  * Modify the configuration file if necessary `design/config.xml`- including the actual target (amp/toxicity)

  * Run the training script `python classifier/main_classifier_train.py`


