# MIMN
Multimodal aspect-category sentiment analysis based on multi-level information
# MIMN
Multimodal aspect-category sentiment analysis based on multi-level information


MASAD original data：https://github.com/12190143/MASAD

## datasets

1.The [ChineseWordVector] subdirectory stores pre trained Chinese word vector files
The [EnglishWordVectors] subdirectory store pre-trained Glove word vector files
The [masadDataset] subdirectory contains three types of files:

A. The three files masadTrain/dev/Test.json correspond to the preliminarily integrated training/validation/testing set samples ("Preliminary integration" refers to integrating the source information of each sample, such as ID, aspect, original text, sentiment labels, etc., into a sample_dictionary, which is stored in a JSON file)

B. The img directory contains sample images, img/train_ Dev_ Img memory stores images of training and validation set samples, img/test_ The image of the test set sample stored in IMG memory, named after its corresponding sample ID
Note: The img directory was not originally placed in this masadDataset, so the data_ Utils_ The file path for reading sample images in the masad.py script will no longer be applicable and needs to be modified.
C. The three directories of masadTrain/Test/DevData store the final integrated training set, validation set, and test set samples ("Final Integration" refers to tokenizing and pad processing the original text of each sample to obtain token_id_sequence; extracting image embeddings from the images of each sample; integrating these processed information into a sample_dictionary, stored as a sample_id. pkl file)
Note: The files in the masadTrain/Test/DevData directory are all from data_ Utils_ The masad.py script can be generated using data_ Utils_ Masad. py for understanding.

## Layers directory

This directory stores the definition files of various layers required for building the model, such as multi_ Fusion_ Layer.py

## Models directory

This directory stores model definition files. End2end_ Lstm.py corresponds to the End to End LSTM model and ae_ Lstm.py corresponds to the ATAE-LSTM model and joint_ Model. py corresponds to the Joint Model model and multi_ Joint_ Model. py corresponds to the Multi modal Joint Model and related ablation models proposed in this paper.

## Checkpoint directory
This directory stores training logs and model weight files generated during training.
Note: The hyperparameter configuration used for training can be viewed through training logs
