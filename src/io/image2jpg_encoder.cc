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

#include "singa/io/image2jpg_encoder.h"
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

namespace singa {

namespace io{
  string Image2JPGEncoder::Encode(vector<Tensor>& data) {
    // suppose data[0]: data, data[1]: label
    // suppose data[0] has a shape as {channel, height, width}
    CHECK_EQ(data[0].nDim(), 3u);
    string output;
    size_t height = data[0].shape()[1];
    size_t width = data[0].shape()[2];
    Mat mat = Mat(height, width, CV_8UC3);
    Mat resized;
    resize(mat, resized, Size(256, 256));
    Mat test = imread("test/samples/test.jpeg", CV_LOAD_IMAGE_COLOR);
    if (data[0].data_type() == kInt)
      memcpy(mat.data, data[0].data<const int*>(), data[0].Size()*sizeof(int));
    else LOG(FATAL) << "Data type is invalid for an raw image";
    //cout << mat << endl;

    const int* label;
    // suppose each image is attached with only one label
    if (data[1].data_type() == kInt)
      label = data[1].data<const int*>();
    else LOG(FATAL) << "Data type is invalid for image label";

    vector<uchar> buff;
    vector<int> param = vector<int>(2);
    param[0] = CV_IMWRITE_JPEG_QUALITY;
    param[1] = 95; // default is 95
    imencode(".jpg", mat, buff, param);
    string buf(buff.begin(), buff.end());

    RecordProto image;
    image.set_label(label[0]);
    for (size_t i = 0; i < data[0].nDim(); i++)
      image.add_shape(data[0].shape()[i]);
    image.set_pixel(buf);
    image.SerializeToString(&output);

    return output;
  }
}
}
