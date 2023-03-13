# Molecular multitask language model 

## Introduction

Multitask molecular model, based on [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu.](https://arxiv.org/pdf/1910.10683.pdf).
The implementation of the T5 architecture is taken from [huggingface transformers](https://huggingface.co/docs/transformers/model_doc/t5).

In this work, we offer an implementation for data processing, and classifiers for the following tasks:
#TODO: Add tasks as they are implemented 
The implementation was broken to easy to use off-the-shelf components that are useful for many sequence-based tasks.

<br/>

#TODO: Change the text below:

## Dataset
The dataset used for the task was built from several publicly available datasets, mixing labeled and unlabeled molecules and proteins and their combinations.  
  * [DBAASP - Database of Antimicrobial Activity and Structure of Peptides](https://dbaasp.org/)
  * 
The entire processing code can be found in the [drug discovery foundation benchmarks repository](https://github.ibm.com/BiomedSciAI-Innersource/DrugDiscoveryFoundationBenchmarks). ????? will create the dataset instance. 

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

  * Set an environment variable for the split file to be generated:  
    Note that this is an output file that the code will generate.

      `export PEPTIDES_DATASETS_SPLIT_FILENAME=<path to a pickle file to be generated>`
            
2. Train design model

  * Modify the configuration file if necessary `design/config.yaml`

  * Run the training script `python design/main_design_train.py`

3. Train classification models

  * Modify the configuration file if necessary `classifier/config.yaml`- including the actual target (amp/toxicity)

  * Run the training script `python classifier/main_classifier_train.py`


