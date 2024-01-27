GPUID=4

file="/home/qiupeichao/Accept/DVERGE-main/FASTEN/checkpoints/fast45/seed_0/3_ResNet20/eps0.50_steps1_beta3.0/epoch_36.pth"
file_baseline="/home/qiupeichao/Accept/DVERGE-main/FASTEN/checkpoints/baseline/seed_0/3_ResNet20/epoch_2.pth"
cd ..

# w-box-pgd-test
# python eval/eval_wbox_pgd.py \
#        --gpu $GPUID \
#        --model-file $file\
#        --steps 10 \
#        --random-start 5 \
#        --save-to-csv


# b-box-mpgd-generate
python eval/generate_bbox_mpgd.py \
         --num-steps 10 \
         --model-file $file_baseline