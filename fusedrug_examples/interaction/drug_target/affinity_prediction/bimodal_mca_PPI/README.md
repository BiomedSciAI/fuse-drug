# BimodalMCA binding affinity predictor

This example implements FuseMedML-Drug based training, inference and evaluation of a protein-protein interaction binding affinity prediction model. The model is a Bimodal Multiscale Convolutional Attention network (BiMCA) [[1]](#1), implemented originally here:  [[2]](#2).  

## How to run:
1. Download the data from this [Box link](https://ibm.box.com/v/titan-dataset).
2. Set your local paths to the data and desired results location in `config/bmmca_full_PPI.yaml`, replacing _YOUR_SESSIONS_PATH_, _YOUR_DATA_PATH_ and _YOUR_CACHING_PATH_. 
3. Run `runner.py`.

## References
<a id="1">[1]</a> 
@article{weber2021titan
    author = {Weber, Anna and Born, Jannis and Rodriguez Martinez, Maria},
    title = "{TITAN: T-cell receptor specificity prediction with bimodal attention networks}",
    journal = {Bioinformatics},
    volume = {37},
    number = {Supplement_1},
    pages = {i237-i244},
    year = {2021},
    month = {07},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btab294},
    url = {https://doi.org/10.1093/bioinformatics/btab294}
}

<a id="2">[2]</a>
TITAN - https://github.com/PaccMann/TITAN
