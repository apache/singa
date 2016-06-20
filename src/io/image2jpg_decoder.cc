/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "singa/io/image2jpg_decoder.h"
#include <vector>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

namespace singa {

namespace io {
  vector<Tensor> Image2JPGDecoder::Decode(string value) {
    vector<Tensor> output;
    RecordProto image;
    image.ParseFromString(value);
    Shape shape(image.shape().begin(), image.shape().end());
    Tensor features(shape), labels(Shape{1});
    
    //string pixel = image.pixel();
    vector<unsigned char> pixel(image.pixel().begin(), image.pixel().end());
    Mat buff(shape[1], shape[2], CV_8UC3, pixel.data());
    Mat mat = imdecode(buff, CV_LOAD_IMAGE_COLOR);
    vector<int> data;
    data.assign(mat.datastart, mat.dataend);
    //for (size_t i = 0; i < image.pixel().size(); i++)
    //  data[i] = static_cast<int>(static_cast<uint8_t>(pixel[i]));
    features.CopyDataFromHostPtr<vector<int>>(&data, data.size());
    int l[1] = {image.label()};
    labels.CopyDataFromHostPtr(l, 1);
    output.push_back(features);
    output.push_back(labels);
    return output;
  }
}
}
