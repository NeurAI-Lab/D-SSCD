# Self-supervised pretraining for Scene Change Detection

**This is the official code for NeurIPS 2021 Machine Learning for Autonomous Driving Workshop Paper, ["Self-supervised pretraining for scene change detection"](paper) by [Vijaya Raghavan Thiruvengadathan Ramkumar](www.linkedin.com/in/vijayaraghavan95), [Prashant Bhat](https://www.linkedin.com/in/prashant-s-bhat/), [Elahe Arani](https://www.linkedin.com/in/elahe-arani-630870b2/) and [Bahram Zonooz](https://www.linkedin.com/in/bahram-zonooz-2b5589156/), where we propose a novel self-supervised pretraining architechture based on differenceing called D-SSCD for scene change detection.**


## Abstract


High Definition (HD) maps provide highly accurate details of the surrounding environment that aids in the precise localization of autonomous vehicles. To provide the most recent information, these HD maps must remain up-to-date with the changes present in the real world. Scene Change Detection (SCD) is a critical perception task that helps keep these maps updated by identifying the changes of the scene captured at different time instances. Deep Neural Network (DNNs) based SCD methods hinge on the availability of large-scale labeled images that are expensive to obtain. Therefore, current SCD methods depend heavily on transfer learning from large ImageNet datasets. However, they induce domain shift which results in a drop in change detection performance. To address these challenges, we propose a novel self-supervised pretraining method for the SCD called D-SSCD that learns temporal-consistent representations between the pair of images. The D-SSCD uses absolute feature differencing to learn distinctive representations belonging to the changed region directly from unlabeled pairs of images. Our experimental results on the VL-CMU-CD and Panoramic change detection datasets demonstrate the effectiveness of the proposed method. Compared to the widely used ImageNet pretraining strategy that uses more than a million additional labeled images, D-SSCD can match or surpass it without using any additional data. Our results also demonstrate the robustness of D-SSCD to natural corruptions, out-of-distribution generalization, and its superior performance in limited label scenarios.

![alt text](https://github.com/NeurAI-Lab/D-SSCD/blob/main/images/DSSL_1.png)

## Requirements

- python 3.6+
- opencv 3.4.2+
- pytorch 1.6.0
- torchvision 0.4.0+
- tqdm 4.51.0
- tensorboardX 2.1

## Datasets

Our network is tested on two datasets for street-view scene change detection. 

- 'PCD' dataset from [Change detection from a street image pair using CNN features and superpixel segmentation](http://www.vision.is.tohoku.ac.jp/files/9814/3947/4830/71-Sakurada-BMVC15.pdf). 
  - You can find the information about how to get 'TSUNAMI', 'GSV' and preprocessed datasets for training and test [here](https://kensakurada.github.io/pcd_dataset.html).
- 'VL-CMU-CD' dataset from [Street-View Change Detection with Deconvolutional Networks](http://www.robesafe.com/personal/roberto.arroyo/docs/Alcantarilla16rss.pdf).
  -  'VL-CMU-CD': [[googledrive]](https://drive.google.com/file/d/0B-IG2NONFdciOWY5QkQ3OUgwejQ/view?resourcekey=0-rEzCjPFmDFjt4UMWamV4Eg)

## Dataset Preprocessing

- For DSSCD pretraining - included in the DSSCD--dataset--CMU.py/PCD.py
- For finetuning and evaluation - Please follow the preprocessing method used by the official implementation of [{Dynamic Receptive Temporal Attention Network for Street Scene Change Detection paper}](https://github.com/Herrccc/DR-TANet) 

Dataset folder structure for VL-CMU-CD:
```bash
├── VL-CMU-CD
│   ├── Image_T0
│   ├── Image_T1
│   ├── Ground Truth

```
								
## SSL Training

- For training 'SSCD' on VL-CMU-CD dataset:
```
python3 DSSCD/train.py --ssl_batchsize 16 --ssl_epochs 400 --save_dir /outputs --data_dir /path/to/VL-CMU-CD --img_size 256 --n_proj 256 --hidden_layer 512 --output_stride 8 --pre_train False --m_backbone False --sscd_barlow_twins True --dsscd_barlow_twins False
```

- For training 'D-SSCD' on VL-CMU-CD dataset:
```
python3 DSSCD/train.py --ssl_batchsize 16 --ssl_epochs 400 --save_dir /outputs --data_dir /path/to/VL-CMU-CD --img_size 256 --n_proj 256 --hidden_layer 512 --output_stride 8 --pre_train False --m_backbone False --dsscd_barlow_twins True --sscd_barlow_twins False 	
```
    

## Fine Tuning

We evaluate our SSCD and DSSCD pretraining on DR-TANet.
- Follow the Please follow the train and test procedure used by the official implementation of [{Dynamic Receptive Temporal Attention Network for Street Scene Change Detection paper}](https://github.com/Herrccc/DR-TANet) 

Start training with DR-TANet on 'VL-CMU-CD' dataset.

    python3 train.py --dataset vl_cmu_cd --datadir /path_to_dataset --checkpointdir /path_to_check_point_directory --max-epochs 150 --batch-size 16 --encoder-arch resnet50 --epoch-save 25 --drtam --refinement

Start evaluating with DR-TANet on 'PCD' dataset.

    python3 eval.py --dataset pcd --datadir /path_to_dataset --checkpointdir /path_to_check_point_directory --resultdir /path_to_save_eval_result --encoder-arch resnet50 --drtam --refinement --store-imgs
  
## Evaluating the finetuned model

Start evaluating with DR-TANet on 'PCD' dataset.

    python3 eval.py --dataset pcd --datadir /path_to_dataset --checkpointdir /path_to_check_point_directory --resultdir /path_to_save_eval_result --encoder-arch resnet18 --drtam --refinement --store-imgs
  
## Robustness Evaluation under Natural corruptions

Install dependencies for this official repo:  [{Benchmarking Neural Network Robustness to Common Corruptions and Perturbations}](https://github.com/hendrycks/robustness/tree/master/ImageNet-C/imagenet_c)  

We evaluate our pretraining methods on corrupted VL-CMU-CD test dataset. our model is subjected to 15 types of natural corruption spanning across 4 categories (blur, noise, weather, digital) with 5 severity levels.

- To generate the corrupted VL-CMU-CD data, use the script "generate_corrupted_testdata.py"
![alt text](https://github.com/NeurAI-Lab/D-SSCD/blob/main/images/2021-10-27_17h53_21.png)


- For evaluation of DR-TANet on corrupted test set:
```	
python3 DSSCD/DR-TANet/eval_natural_corrup_vlcmucd.py --dataset vl_cmu_cd --datadir /path_to_dataset --checkpointdir /path_to_check_point_directory --resultdir /path_to_save_eval_result --encoder-arch resnet18 --drtam --refinement --store-imgs
 ```
  
## Evaluation on out of distribution data (OOD) 

Use the same evaluation script by swaping the --dataset name during training and evaluation.
	
- Train in VL-CMU-CD and evaluate in PCD as OOD
```	
--dataset vl_cmu_cd     --dataset pcd
```
- Train in PCD and evaluate in VL-CMU-CD as OOD
```
--dataset pcd          --dataset vl_cmu_cd
```
## Evaluation on limited labels

Use the same train and eval script provided inside "DSSCD/DR-TANet/".

- To finetune the model with different amount of labeled data: change the --data_num args to vary the training data quantity.
```	
python3 train.py --dataset vl_cmu_cd --data_num 0.1 --datadir /path_to_dataset --checkpointdir /path_to_check_point_directory --max-epochs 150 --batch-size 16 --encoder-arch resnet50 --epoch-save 25 --drtam --refinement 
```

## Cite our work
