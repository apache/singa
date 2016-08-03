#!/usr/bin/env sh
../../build/bin/imagenet -epoch 90 -lr 0.01 -batchsize 256 -filesize 1280 -ntrain 1281167 -ntest 50000 \
  -data "/home/xiangrui/imagenet_data" -pfreq 100 -nthreads 12
