# spCellEval - Benchmarking Cell Phenotyping Methods in Spatial Proteomics 

We present "spCellEval", a quantitative comparison of automated/semi-automated cell phenotyping methods for Spatial Proteomics datasets on a diverse set of 10 curated public datasets. The methods are compared with a list of label transfer metrics divided into 4 categories: classification performance, distribution recovery, stability and scalability. This benchmark acts as a foundation to evaluate and improve automated cell phenotyping. 

![Alt text](img/fig_1.png "Title")

## Current Results Overview: 
![Alt text](img/fig_2.png "Title")

## Getting Started

In order to reproduce the results, the raw datasets currently need to be downloaded from public repositories. Please refer to the public registered Stage 1 manuscript on [figshare]()

Data workflow pipeline:

Raw data --> preprocessing --> run method --> predictions_{fold_id}.csv --> evaluation

### Preprocessing

Preprocessing of each dataset can be found in `src/preprocessing/datasets/<process_dataset.ipynb>`

### Running methods

Scripts to run each method are provided in `src/<method>`.

Datasets and parameter settings can be found in manuscript supplement.

### Evaluation Scripts
The notebooks in  `src/metrics_scripts` can be used to get the complete metrics on all methods for each dataset. 

## Adding your own method
To officially add your own method, please open an issue and provide us with the following to reproduce your method. 
1. GitHub repo for the method
2. List of Parameters used (if any)
3. OPTIONAL: Your predictions (this speeds up the evaluation process)

Folder Structure to add your predictions in 
```
results/
├── Method1/
│   └── predictions_*.csv
├── Method2/
└── Method3/
```


