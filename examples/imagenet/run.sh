#!/usr/bin/env sh
../../build/bin/imagenet -epoch 40 -lr 0.01 -batchsize 32 -filesize 1280 -ntrain 12900 -ntest 500 -data "/home/xiangrui/imagenet_data.bak" 
