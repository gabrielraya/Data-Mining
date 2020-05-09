# AN EVALUATION OF ROTATION-EQUIVARIANT CONVOLUTIONAL NEURAL NETWORKS

This repo contains the Python implementation of rotation-equivariant networks as described in the <a href="https://github.com/gabrielraya/Data-Mining/blob/master/Project/Rotation_Equivariance.pdf" target="_blank">project report</a> 
as part of the final porject of course Data Mining 2019, given at Radboud University. <br/>


We evaluate the performance improvement in a medical image classification task which arises from
using a rotation-equivariant block in a convolutional neural network, as opposed to using traditional
convolutions. We find that performance is not improved much by using rotation equivariance: it
yields an AUC score of 0.8949 while traditional convolutions yield 0.8823. 


<img src="https://github.com/gabrielraya/Data-Mining/blob/master/Project/images/data.PNG?raw=1" align="right">



The data use comes from the [Kaggle competition, Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection)[1], which
is a slightly modified version of the [PatchCamelyon (PCam) benchmark dataset](https://github.com/basveeling/pcam) [2].


[1] Kaggle competition: Histopathologic cancer detection. https://www.kaggle.com/c/histopathologic-cancerdetection/ <br/>
[2] B. S. Veeling, J. Linmans, J. Winkens, T. Cohen, M. Welling. "Rotation Equivariant CNNs for Digital Pathology". arXiv:1806.03962
