#!/bin/bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
if [[ $# -ne 1 ]]; then
    echo "usage: $0 model_name"
    echo "   model_name: [vgg16|vgg19], ..."
    exit -1
fi

if [[ $1 == "vgg19" ]]; then
    echo "Downloading label list..."
    if [[ ! -f synset_words.txt ]]; then
        wget -c https://www.dropbox.com/s/qe66xwwc78q7fe5/synset_words.txt?dl=0 -O synset_words.txt
    fi
    echo "Downloading vgg19..."
    if [[ ! -f vgg19.prototxt ]]; then
        wget -c https://www.dropbox.com/s/ehi6dxxp3s1rl8t/vgg19.prototxt?dl=0 -O vgg19.prototxt
    fi

    if [[ ! -f vgg19.caffemodel ]]; then
        wget -c https://www.dropbox.com/s/y8ksbfp3iq0kvdn/vgg19.caffemodel?dl=0 -O vgg19.caffemodel
    fi
    echo "Downloading test images..."
    if [[ ! -d test ]]; then
        wget -c https://www.dropbox.com/s/ch5ktahijwof6ka/test.tar.gz?dl=0 -O test.tar.gz
        tar -zxvf test.tar.gz
    fi
    echo "Converting..."
    python predict.py ./vgg19.prototxt ./vgg19.caffemodel ./synset_words.txt

elif [[ $1 == "vgg16" ]]; then
    echo "Downloading label list..."
    if [[ ! -f synset_words.txt ]]; then
        wget -c https://www.dropbox.com/s/qe66xwwc78q7fe5/synset_words.txt?dl=0 -O synset_words.txt
    fi
    echo "Downloading vgg16..."
    if [[ ! -f vgg16.prototxt ]]; then
        wget -c https://www.dropbox.com/s/ilpt58tle8jqtxj/vgg16.prototxt?dl=0 -O vgg16.prototxt
    fi

    if [[ ! -f vgg16.caffemodel ]]; then
        wget -c https://www.dropbox.com/s/3qidow3qr77ruob/vgg16.caffemodel?dl=0 -O vgg16.caffemodel
    fi
    echo "Downloading test images..."
    if [[ ! -d test ]]; then
        wget -c https://www.dropbox.com/s/ch5ktahijwof6ka/test.tar.gz?dl=0 -O test.tar.gz
        tar -zxvf test.tar.gz
    fi
    echo "Converting..."
    python predict.py ./vgg16.prototxt ./vgg16.caffemodel ./synset_words.txt
else
    echo "unsupported model: $1"
fi
