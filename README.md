# NCE-CNN-Tensorflow


Getting Started
-----------
``1.`` Please install the tensorflow library 

``2.`` Checkout our repo:
```
git clone https://github.com/whaleloops/NCE-CNN-Tensorflow.git
```

``3.`` Copy and paste codes into this repo: https://github.com/castorini/NCE-CNN-Torch

``4.``Using following script to download and preprocess the Glove word embedding:
```
$ sh fetch_and_preprocess.sh
``` 
Please make sure your python version >= 2.7, otherwise you will encounter an exception when unzip the downloaded embedding file.

``5.``TODO: add more

Running
--------
 To evaluate on the TrecQA raw dataset, with MAX sampling and number of negative pairs as 8, run:
```
$ python2 PairwiseTrainQA.py --dataset TrecQA --version raw --neg_mode 2 --num_pairs 8
```
To evaluate on the TrecQA clean dataset, simply change -version to clean.
Similarly, if you want to evaluate on the WikiQA dataset, change -dataset to WikiQA (don't need to set the -version).
You can also change the -neg_mode and -num_pairs to select different sampling strategies or negative pairs.
