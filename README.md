# Intermediate Deep Learning Project

This is teamwork of Ruiyu Li and Zhiying Zhu.

## Prerequisites 
fschat==0.2.31
transformers==4.34.1
torch==2.1.0

## Data

### Train/Dev/Test Data
Please download all the data from [here](https://drive.google.com/drive/folders/1dLKpaVktAojeQ7so1_Seao_e3iLQ-Egy?usp=sharing) and put under ```data```. There should be four folders:

 - ```class_binary```
 - ```class_finegrained```
 - ```tag_binary``` 
 - ```tag_finegrained```

Each folder contains a train, dev, and test split.  We provide the detailed data format in the [README file](data/README.md) under ```data```.

## Training Models

### Training the Llama-2 Model

```
sh llama/bias_ft_llama.sh
```

To evaluate the results, run
```
sh llama/bias_eval.sh
```

### Training the BERT model

```
python train_finegrained.py --device [desired cuda device] 
                --batch-size 16 
                --epochs 3 
                --max_len 128 
                --lr 2e-5
                --pretrained-ckpt "ckpt/*saved_ckpt*" [optional]
```

To evaluate the results, run
```
python eval.py/eval_finegrained.py --device 0 [desired cuda device] --eval_ckpt "ckpt/*saved_ckpt*"
```
