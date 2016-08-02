#!/usr/bin/env sh
../../build/bin/createdata -trainlist "/data/xiangrui/label/train.txt" -trainfolder "/data/xiangrui/ILSVRC2012_img_train" \
  -testlist "/data/xiangrui/label/val.txt" -testfolder "/data/xiangrui/ILSVRC2012_img_val" -outdata "/home/xiangrui/imagenet_data" -filesize 12800
