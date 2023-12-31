# MIMN
Multimodal aspect-category sentiment analysis based on multi-level information


MASAD original data：https://github.com/12190143/MASAD


## Layers directory

This directory stores the definition files of various layers required for building the model, such as multi_ Fusion_ Layer.py

## Models directory

This directory stores model definition files. End2end_ Lstm.py corresponds to the End to End LSTM model and ae_ Lstm.py corresponds to the ATAE-LSTM model and joint_ Model. py corresponds to the Joint Model model and multi_ Joint_ Model. py corresponds to the Multi modal Joint Model and related ablation models proposed in this paper.

## Checkpoint directory
This directory stores training logs and model weight files generated during training.
Note: The hyperparameter configuration used for training can be viewed through training logs
