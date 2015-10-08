/************************************************************
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/

#ifndef SINGA_NEURALNET_INPUT_LAYER_CSV_H_
#define SINGA_NEURALNET_INPUT_LAYER_CSV_H_

#include "./store.h"

namespace singa {

/**
 * Specific layer that parses the value string loaded by Store as a line from
 * a CSV file.
 *
 * It assumes the first column is the label except that has_label_ is configured
 * to false. Or the data is used in deploy mode.
 */
class CSVInputLayer : public SingleLabelRecordLayer {
 public:
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;

 protected:
  bool Parse(int k, int flag, const string& key, const string& val) override;
  void LoadRecord(const string& backend,
                  const string& path,
                  Blob<float>* to) override;

 private:
  std::string sep_;
  bool has_label_;
};

}  // namespace singa

#endif  // SINGA_NEURALNET_INPUT_LAYER_CSV_RECORD_H_
