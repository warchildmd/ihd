## Accurate and Efficient Intracranial Hemorrhage Detection and Subtype Classification in 3D CT Scans with Convolutional and Long Short-Term Memory Neural Networks

Mihail Burduja, Radu Tudor Ionescu, Nicolae Verga, Sensors 2020, 20(19), 5611

Official URL: https://www.mdpi.com/1424-8220/20/19/5611/pdf

ArXiv URL: https://arxiv.org/abs/2008.00302

This is the official repository of "Accurate and Efficient Intracranial Hemorrhage Detection and Subtype Classification in 3D CT Scans with Convolutional and Long Short-Term Memory Neural Networks".


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

This software is released un the CC BY-NC-ND license agreement.
The software can be used for non-commercial purposes only.
