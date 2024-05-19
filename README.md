# INM706 Coursework: Named Entity Recognition

## About:
Type: Named Entity Recognition.

The dataset can be found here: https://www.kaggle.com/datasets/debasisdotcom/name-entity-recognition-ner-dataset


Methods/Models:
- LSTM
- Bidirectional LSTM
- Attempt at Bidirectional LSTM with CRF

# How To Run
Python Version: 3.9.5

## City Hyperion [linux server]
```
sh setup.sh
```

## Windows:
```
.\setup.ps1
```

Sometimes the scripts will not properly install all required libraries and a manual pip install is required for said libraries.

# Checkpoints
Final Model checkpoints are included in the repo under the checkpoints folder.

# Training
Parameters controlable from config.yaml are:
 - Number of epochs
 - Batch size
 - Learning rate

The wand api key should be added to the run_job.sh on linux or entered as requested by Windows.

# Logging

The Wandb logs for training can be found here:

 - https://wandb.ai/ismael-benadjal/INM706_CW

# Inference

There is an inference for each model in the inference.ipynb file.