
# IntraCross: Cross-Modality Graph Matching for Intravascular Sequence Registration
From  [Kit Mills Bransby](https://kitbransby.github.io/) (QMUL) et al. 

This is the official implementation for the paper IntraCross: Cross-Modality Graph Matching for Intravascular Sequence Registration currently under review (Aug 2025)

Paper PDF: Pre-print available soon. 

## Abstract
Intravascular ultrasound (IVUS) and optical coherence tomography (OCT) are complementary imaging modalities to assess atherosclerosis in vivo. Combining both modalities in a single imaging system has been shown to improve characterization of vulnerable plaque that are likely to cause acute coronary events. However, fundamental differences in tissue sensitivities and acquisition protocols make the registration of sequences challenging. Anatomical landmarks used to align IVUS and OCT sequences can be masked or lack visual similarity between modalities which renders manual alignment time-consuming and prone to observer variability, limiting its clinical use. Existing methods impose strict frame-level correspondences leading to instability in low information regions, and rely on a two-step registration process that compound alignment errors. We propose IntraCross, a novel graph matching framework that learns partial assignments between landmarks rather than enforcing rigid frame-by-frame matching, enabling flexible correspondences while rejecting unmatchable landmarks. This is the first method to perform both temporal and spatial registration simultaneously, aligning with clinical workflows. We extend existing partial matching techniques from 2D to 3D sequences and incorporate a temporal prior to regularize the matching process. Testing in 77 vessels from 22 patients showed a high agreement with expert analysts (Williams Index=1.1; p=0.62, 0.89, 0.07) and our approach outperforms existing methods reported in literature for circumferential registration (p=0.01, 0.04). 

## Usage
Due to licensing restrictions, the PACMAN dataset is not publicly available. Therefore running this code in full (re-training, evaluation etc) is not possible. The uploaded code includes the feature extractors, clustering algorithm and graph matching network and provides a guide to those who want to adapt our work for their registration task and dataset. 

## Adapting the code for your dataset
To train on your own datasets, you will need to change some code. Here are some starters:
* Create a new pytorch dataset and dataloader in ``utils/dataset_<nameofdataset>.py`` and register it in ``utils/load_dataset.py``
* Re-write the validation and evaluation methods found in ``utils/eval_utils.py``, ``utils/train_utils.py`` and ``eval_intracross.ipynb`` based on the new task.  

## Enviroment
* Python 3.11, Pytorch 2.0, Cuda 11.7, Torchinfo 1.8, Scipy 1.13, Scikit-Learn 1.5,  



