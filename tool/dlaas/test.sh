#!/bin/bash
cd /workspace
wget $1
tar zxf *.tar.gz
cp /workspace/model.py /opt/incubator-singa/tool/dlaas/
cd /opt/incubator-singa/
python tool/dlaas/main.py test $2 $3 
