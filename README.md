# FASTEN Framework

### 1. Overview

FASTEN: Fast Ensemble Learning For Improved Adversarial Robustness

The code is repository for "[FASTEN: Fast Ensemble Learning For Improved Adversarial Robustness](https://ieeexplore.ieee.org/document/10329935)" (IEEE TIFS).


### 2. Prerequisites

python **3.6**  
pytorch **1.4.0**  

### 3. Pipeline 
**3.1 Augmentation Mechinasm**

<img src="/figure/overview.png" width = "700" height = "200" align=center/>

**3.2 Optimization Mechinasm**

<img src="/figure/overview2.png" width = "500" height = "200" align=center/>


### 4. Run the Code  
**4.1 Training Ensembles**

`train.sh`: Train the ensemble models by using different methods. 

For example, you can run the shell in `train.sh`.

Vanilla ensemble：
`python train/train_baseline.py --gpu $GPUID --model-num 3 --batch-size 128 --num-class 10 --arch "ResNet" --depth 20 --epoch 200`

FASTEN ensemble:
`python train/train_fasten.py --gpu $GPUID --model-num 3 --distill-eps 0.06 --distill-alpha 0.06 --distill-steps 1 --beta 3.0 --batch-size 128 --num-class 10 --arch "ResNet" --depth 20 --epoch 200`

DVERGE ensemble:
`python train/train_dverge.py --gpu $GPUID --model-num 3 --distill-eps 0.07 --distill-alpha 0.007 --distill-steps 10 --batch-size 128 --num-class 10 --arch "ResNet" --depth 20 --epoch 200`

**4.2 Testing Ensembles**

`evaluation.sh`: Generate transferable examples / Test robustness under white or black box scenarios.

For example, you can run the `evaluation.sh` to generate black-box adversarial examples:  
`file_baseline="checkpoints/baseline/seed_0/3_ResNet20/epoch_200.pth"`  
`python eval/generate_bbox_mpgd.py --num-steps 10 --model-file $file_baseline`  

After producing the adversarial examples, you can run the shell to test black-box robustness:  
`file="checkpoints/fasten/seed_0/3_ResNet20/eps0.06_steps1_beta3.0/epoch_200.pth"`  
`python eval/eval_bbox.py --gpu $GPUID --model-file $path --folder transfer_adv_examples --steps 100 --num-class 10 --arch "ResNet" --depth 20 --save-to-csv`

Or you can test white-box robustness of FASTEN under the PGD attack in `evaluation.sh`:  
`file="checkpoints/fasten/seed_0/3_ResNet20/eps0.06_steps1_beta3.0/epoch_200.pth"`  
`python eval/eval_wbox_pgd.py --gpu $GPUID --model-file $file --steps 10 --random-start 5 --save-to-csv`  

### 5. Experimental Results

<b>5.1 Black-box Experiment</b>

<img src="/figure/black-box.png" width = "900" height = "250" align=center/>

<b>5.2 White-box Experiment on CIFAR-10</b>

<img src="/figure/white-box.png" width = "900" height = "370" align=center/>

<b>5.3 White-box Experiment on MNIST and CIFAR-100</b>

<img src="/figure/white-box2.png" width = "850" height = "370" align=center/>

<b>5.4 White-box Experiment on AdvFASTEN</b>

<img src="/figure/white-box3.png" width = "900" height = "470" align=center/>



