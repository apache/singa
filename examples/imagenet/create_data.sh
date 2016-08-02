#!/usr/bin/env sh
../../build/bin/createdata -trainlist "imagenet/label/train.txt" -trainfolder "imagenet/ILSVRC2012_img_train" \
  -testlist "imagenet/label/val.txt" -testfolder "imagenet/ILSVRC2012_img_val" -outdata "imagenet_data" -filesize 1280
