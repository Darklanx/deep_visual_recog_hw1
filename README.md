# deep_visual_recog_hw1
## Reproducing Submission
To reproduct my submission without retrainig, do the following steps:
1. [Installation](#installation-resnest)
2. [Dataset Preparation](#Dataset-Preparation)


## Installation resnest
`pip3 install resnest --pre.`
`pip3 install -r requirements.txt`

## Dataset Preparation
`kaggle competitions download -c cs-t0828-2020-hw1`

Modify variable 
```
dir_training = "./path-to-training-data-folder"
dir_testing = "./path-to-testing-data-folder"
csv_file = "./path-to-csv-label"
```

