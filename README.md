# MIST
As supporting code for the study "[MIST: Multiple Instance Learning Network Based on Swin Transformer for WSI Classification of Colorectal Adenomas](https://pathsocjournals.onlinelibrary.wiley.com/doi/abs/10.1002/path.6027)".

## Install Python-related packages
```
  $ pip install -r requirements.txt
```
## Download Swin Transformer's pre-trained model on ImageNet
```
  $ git clone https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
```
Move the downloaded pre-trained model into the `simclr_swin/pretrain_model` folder.

If you would like to use APEX to speed up your training, go to "https://github.com/NVIDIA/apex" to download the related documents.

## The file structure of the colorectal adenoma WSIs data is as follows

```
root
|-- WSI
|   |-- Ade
|   |   |-- CLASS_1
|   |   |   |-- SLIDE_1.svs
|   |   |   |-- ...
|   |   |-- CLASS_2
|   |   |   |-- SLIDE_1.svs
|   |   |   |-- ...
|   |   |-- CLASS_3
|   |   |   |-- SLIDE_1.svs
|   |   |   |-- ...
|   |   |-- CLASS_4
|   |   |   |-- SLIDE_1.svs
|   |   |   |-- ...
|   |   |-- CLASS_5
|   |   |   |-- SLIDE_1.svs
|   |   |   |-- ...
|   |   |-- CLASS_6
|   |   |   |-- SLIDE_1.svs
|   |   |   |-- ...
```

## Cut WSIs
```
  $ python crop_patches.py -m 0 1 -b 5
```

Once the above script has finished running, `pyramid` folder will appear.
```
root
|-- WSI
|   |-- Ade
|   |   |-- pyramid
|   |   |   |-- CLASS_1
|   |   |   |   |-- SLIDE_1
|   |   |   |   |   |-- PATCH_LOW_1
|   |   |   |   |   |   |-- PATCH_HIGH_1.jpeg
|   |   |   |   |   |   |-- ...
|   |   |   |   |   |-- ...
|   |   |   |   |   |-- PATCH_LOW_1.jpeg
|   |   |   |   |   |-- ...
|   |   |   |   |-- ...
|   |   |   |-- ...
```

## Split the dataset by a ratio of 7:3
```
  $ python split_dataset.py
```
The validation set is divided into the `WSI/Ade_val` folder


## Self-supervised contrastive learning
```
  $ cd simclr_swin
```
### Train the low magnification embedder
```
  $ python run.py --multiscale=1 --level=low
```

### Train the high magnification embedder
```
  $ python run.py --multiscale=1 --level=high
```

Once the self-supervised contrastive learning training is complete, two folders will appear under the./simclr_swin/runs folder for storing the model.We rename them `swinhigh` and `swinlow`.


## Embedding phase
```
  cd ..
  $ python compute_feats.py --dataset Ade --num_classes 6 --backbone swintransformer --magnification tree --weights_high swinhigh --weights_low swinlow
  $ python compute_feats.py --dataset Ade_val --num_classes 6 --backbone swintransformer --magnification tree --weights_high swinhigh --weights_low swinlow
```

Once the above script has been run,`Ade` and `Ade_val` folder will appear inside `datasets` folder.
```
root
|-- datasets
|   |-- DATASET_NAME
|   |   |-- CLASS_1
|   |   |   |-- SLIDE_1.csv
|   |   |   |-- ...
|   |   |-- CLASS_2
|   |   |   |-- SLIDE_1.csv
|   |   |   |-- ...
|   |   |-- ...
|   |   |-- CLASS_1.csv
|   |   |-- CLASS_2.csv
|   |   |-- ...
|   |   |-- Ade.csv
```
An "embedder" folder will also appear to store the embedder.
```
root
|-- embedder
|   |-- DATASET_NAME
|   |   |-- swintransformer-m-embedder-high.pth
|   |   |-- swintransformer-m-embedder-low.pth
```

## Train the multiple instance learning aggregator
```
  $ python train_ade.py
```
If you want to train a model with a weight sampler, run the following script
```
  $ python train_ade_weightedSampler.py
```

Once the above script has been run, a `weights` will appear to store the trained MIST model.

## Test the model
Firstly, the embeddings of the validation set is obtained according to the above steps.Then run the following script.
```
  $ python test_model.py --dataset Ade_val --model_path ./weights/swin/1.pth
```
Once the above script is completed, the files related to the result will appear in the `inference` folder.
```
root
|-- inference
|   |-- test_cm.png
|   |-- test_PRcurve.png
|   |-- test_pred.csv
|   |-- test_ROC.png
```

## Interpretability of the model
```
  $ cd cam
  $ python grad_cam_swin.py
```
Once the above code is run, the result will appear in the `./cam/img` folder.

