RSNA Intracranial Hemorrhage Detection (https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection) model

ResNeXt + PCA + BiLSTM for 0.04989 on Private Test Dataset

Sequence Metadata Required: https://www.kaggle.com/mihailburduja/rsna-intracranial-sequence-metadata

Slices are resized to 256x256, embedding vector is resized to 120. 

`models.py` contains the CNN and LSTM model

`datasets.py` contains the torch Datasets for CNN and for LSTM model

`train_cnn.py` trains the CNN and outputs PCA embeddings and predictions

`train_lstm.py` train the LSTM and outputs the submission file 

----------

License: CC BY-NC-ND
