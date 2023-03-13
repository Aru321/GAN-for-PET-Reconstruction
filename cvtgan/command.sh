#!/bin/bash

epoch=50
batch_size=5

# other Ex
other_Ex=(1 2 3 4 5 6 7 8)
for i in ${other_Ex[*]}; do
python train_multi.py --Ex_num $i --epochs $epoch --batch_size $batch_size --lr_G 2e-4 --lr_D 2e-5;
done
