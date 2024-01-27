GPUID=4

cd ..

# Baseline training
#python train/train_baseline.py --gpu $GPUID --model-num 3

# DVERGE training
#python train/train_dverge.py --gpu $GPUID --model-num 3 --distill-eps 0.07 --distill-alpha 0.007

# ADP training
#python train/train_adp.py --gpu $GPUID --model-num 3

# GAL training
#python train/train_gal.py --gpu $GPUID --model-num 3

# TRS training
#python train/train_trs.py --model-num 3

# FASTEN training
python train/train_fast45.py --gpu $GPUID --model-num 3 --distill-eps 0.07 --distill-alpha 0.07 --distill-steps 1 --beta 3.0 --batch-size 128 --num-class 200 --arch "ResNet" --depth 18 --epoch 120


