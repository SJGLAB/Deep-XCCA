# Deep-XCCA
## Overview
Repository for the application of Deep-XCCA. Messenger RNA (mRNA)-based vaccines have been demonstrated as a promising technology for preventing viral infections due to their safety, rapid production, high potency, and ease of industrialization. There are still some challenges faced by mRNA vaccine design, one of which is how to optimize the coding sequence (CDS) of mRNA to improve its translation efficiency. Here, we proposed a cross-covariance codon attention (XCCA) mechanism to represent codon selection probabilities for each amino acid of the inputted protein, and then based on XCCA we developed a deep learning method called Deep-XCCA for CDS optimization, which was specifically designed to learn the long-term dependencies both in the amino acid and the codon sequences.  

<img width="865" height="649" alt="image" src="https://github.com/user-attachments/assets/73dfccf3-33a3-49c6-b0f6-dc591dadeba2" />


## Functions
_test.py_ codes for predict.  
_model/attention.py_ codes for Deep-XCCA attention.  
_model/wordsequence.py_ codes for Deep-XCCA model.  
_utils/data.py_ codes for input embedding vectors and 64-possible codons as embedding vectors.  
_utils/metric.py_ codes for evaluation metric.   
model weights:https://pan.baidu.com/share/init?surl=tXn8EllT9gwdm8gsebI0PA&pwd=1310

## System_Requirements
### 1.Hardware requorements
Only a standard computer with enough RAM to support the in-memory operations is required.  

### 2.Software requirements
#### OS requirements
#### The codes are tested on the following Oses:
(1)	Linux x64  
(2)	Windows 10 x64  
#### And the following x86_64 version of Python:
Python 3.X  
#### Python dependencies:
(1)	torch  
(2)	Bio 
(3)	torchvision 

## Installation_Guide
### Download the codes
git clone https://github.com/SJGLAB/Deep-XCCA.git
### Prepare the environment
We recommend you to use Anaconda to prepare the environments.  
(1)	conda create -n Deep-XCCA python=3.10  
(2)	conda activate Deep-XCCA  
(3)	pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121    
(4) pip install Bio

## Usage and Demo
### Example of Running Command
python test.py --status dev --batch_size 8 --hidden_dim 3000 --word_emb_dim 3000 --lstm_layer 4 --train_dir ./data/train_cds_2k5_all_label.txt --dev_dir ./data/val_cds_2k5_all_label.txt --test_dir ./data/test_cds_2k5_all_label.txt --load_model_dir w3000_xlxx_num20.model --dset_dir w3000_xlxx_num20.dset --word_emb_dir ./data/train3000.vector

## Contact Us
If you have any questions in using Deep-XCCA, contact us.
