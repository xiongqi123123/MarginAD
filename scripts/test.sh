CUDA_VISIBLE_DEVICES=0\ 
   bash ./tools/dist_test.sh \
    projects/configs/Margin_Stage2_Full.py \
    work_dirs/Margin_Stage2_Full/iter_5860.pth \
    1 \
    --deterministic \
    --eval bbox