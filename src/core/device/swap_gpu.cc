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
#include "singa/singa_config.h"
#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <chrono>
#include <thread>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <tuple>        // std::tuple, std::get, std::tie, std::ignore
#include "singa/core/device.h"
#include "singa/utils/cuda_utils.h"


using namespace std;
namespace singa {

const cudaMemcpyKind copyKind[] = {cudaMemcpyHostToHost, cudaMemcpyHostToDevice,
                                   cudaMemcpyDeviceToHost,
                                   cudaMemcpyDeviceToDevice};

struct sort_by_ptr_idx_ascending{
    /*
     sort DeviceOptInfo by ptr and then idx.
     */
    inline bool operator() (const DeviceOptInfo& struct1, const DeviceOptInfo& struct2)
    {
        return ((struct1.ptr<struct2.ptr)||((struct1.ptr==struct2.ptr)&&(struct1.idx<struct2.idx)));
    }
};


struct DeviceOptSimplifiedInfo{
    /*
     members: [idx, operation_type, size_delta]
     */
    size_t size_delta; //size if Malloc, else: delta to last index
    int operation_type;
    int idx;
    DeviceOptSimplifiedInfo(size_t s, int M, int i):size_delta(s),operation_type(M),idx(i){}
};


struct sort_by_DeviceOptSimplifiedInfo_idx_ascending{
    /*
     sort DeviceOptSimplifiedInfo by Idx.
     */
    inline bool operator() (const DeviceOptSimplifiedInfo& struct1, const DeviceOptSimplifiedInfo& struct2)
    {
        return (struct1.idx<struct2.idx);
    }
};


vector<string> SplitOptString(string s, string delimiter) {
  // string delimiter
  size_t pos_start = 0, pos_end, delim_len = delimiter.length();
  string token;
  vector<string> res;
  while ((pos_end = s.find(delimiter, pos_start)) != string::npos) {
    token = s.substr(pos_start, pos_end - pos_start);
    pos_start = pos_end + delim_len;
    res.push_back(token);
  }
  res.push_back(s.substr(pos_start));

  return res;
}


vector<DeviceOptInfo> DeviceOptSeqStrToStruct(vector<string> vec, int &idx_range){
    /*
     convert vector of string into vector of DeviceOptInfo, sorted by ptr 
     and then idx, and update idx_range to pieceMsgVec size.
     format of DeviceOptInfo [ptr, size/-1, flag, idx, timestamp]
     flag: 1 for malloc, -1 for free, 2 for read, 3 for layer,4 for mutable
    */
    vector<DeviceOptInfo>vec_opt_info;

    for (int i=0;i<vec.size();i++) {
      vector<string> v = SplitOptString(vec[i], " ");
      int operation_type;
      if (v[0]=="Malloc"){
        operation_type = 1;
      }else if (v[0]=="Free"){
        operation_type = -1;
      }else if (v[0]=="Mutable"){
        operation_type = 4;
      }else if (v[0]=="Read"){ 
        operation_type = 2;
      }else if (v[0]=="Layer"){
        operation_type = 3;
      }
      //DeviceOptInfo(string p, size_t s, int M, int i):ptr(p),size(s),operation_type(M),idx(i){}
      size_t result;
      stringstream convert(v[2]);
      if (!(convert>>result)){
        result =-1;
        cout<<"error for converting size from str to int."<<endl;
      }
      DeviceOptInfo itm(v[1],result, operation_type, i);
      double temp_time;
      stringstream convert2(v[3]);
      convert2>>temp_time;
      itm.t =temp_time;
      vec_opt_info.push_back(itm);
    }
 
    sort(vec_opt_info.begin(),vec_opt_info.end(),sort_by_ptr_idx_ascending());
    idx_range = static_cast<int>(vec_opt_info.size());

    return vec_opt_info;
}


vector<size_t> DeviceOptSeqRepeatableTestPreProcess(vector<DeviceOptInfo>vec_opt_info){
  /*
  pre process Device Operation Sequence Struct info for repeatable test,
  return a vector of int for fast detection.
  */
  vector<DeviceOptSimplifiedInfo>vec_opt_simplified_info;
  string temp_str;
  int temp_idx=0;
  for (int i=0;i<vec_opt_info.size();i++){
    if (vec_opt_info[i].operation_type==1){
      //update temp_str and idx.
      temp_str = vec_opt_info[i].ptr;
      temp_idx = vec_opt_info[i].idx;
      DeviceOptSimplifiedInfo itm(vec_opt_info[i].size,1,vec_opt_info[i].idx);
      vec_opt_simplified_info.push_back(itm);
    } else {
      DeviceOptSimplifiedInfo itm(vec_opt_info[i].idx-temp_idx,vec_opt_info[i].operation_type,vec_opt_info[i].idx);
      temp_idx = vec_opt_info[i].idx;
      vec_opt_simplified_info.push_back(itm);
    }
  }
    
  sort(vec_opt_simplified_info.begin(),vec_opt_simplified_info.end(),sort_by_DeviceOptSimplifiedInfo_idx_ascending());
  //only after sort then can create vec_rep.
  vector<size_t>vec_rep; // vector of size_delta, name it as vec_rep for simlisity.
  for (int i =0; i<vec_opt_simplified_info.size(); i++){
    vec_rep.push_back(vec_opt_simplified_info[i].size_delta);
  }
  return vec_rep;
}
void RepeatableTest(vector<size_t>rep, int &iteration_length, int &location_of_2nd_iteration, int iteration_length_threshold, int global_index ){
  /*
  repeatable test, input vector of int, 
  in-place update max_legth (length of iteration) 
  and location_of_2nd_iteration (where 2nd iteration starts)
  */
  int idx_range = (int)rep.size();
  int threshold = iteration_length_threshold;
  vector<pair<int,int>>iteration_length_location_of_2nd_iteration;
  
  for (int i=0; i<idx_range;i++){
    if (iteration_length>threshold){
      break;
    }
    for (int len=1; len<(idx_range-i);len++){
      if (iteration_length>threshold){
        break;
      }
      if((equal(rep.begin()+i,rep.begin()+i-1+len,rep.begin()+i+len))&&(iteration_length<len)) {
        iteration_length = len;
        location_of_2nd_iteration = i;
        iteration_length_location_of_2nd_iteration.push_back(make_pair(iteration_length,location_of_2nd_iteration));
      }
    }
  }
}

struct sort_by_idx_ascending{
    /*
     sort DeviceOptInfo by ptr and then idx.
     */
    inline bool operator() (const DeviceOptInfo& struct1, const DeviceOptInfo& struct2)
    {
        return (struct1.idx<struct2.idx);
    }
};


int SwapOutTime(size_t size){
  int ans = 0; 
  //measured in 16 PCIe, pinned memory.
  if (size==0) {ans = 47200;} else {ans = 0.0756 * size + 47200;}
  return ans;
}

int SwapInTime(size_t size){
  int ans = 0; 
  //measured as per ncra ~ ncrd, 16 PCIe, pinned memory.
  if (size==0) {ans = 9700;} else {ans = 0.0823 * size + 9700;}
  return ans;
}

struct sort_by_DOA_origin_descending{
  /*
  sort SwapBlock by DOA_origin, descending
  */
  inline bool operator() (const SwapBlock& struct1, const SwapBlock& struct2)
  {
    return (struct1.DOA_origin>struct2.DOA_origin);
  }
};

struct sort_by_WDOA_descending{
  /*
  sort SwapBlock by weighted DOA_origin, descending
  */
  inline bool operator() (const SwapBlock& struct1, const SwapBlock& struct2)
  {
    return (struct1.WDOA>struct2.WDOA);
  }
};

struct sort_by_AOA_descending{
  /*
   sort SwapBlock by pri, descending
   */
  inline bool operator() (const SwapBlock& struct1, const SwapBlock& struct2)
  {
    return (struct1.AOA>struct2.AOA);
  }
};

struct sort_by_idx_ascending_swap{
  /*
  sort DeviceOptInfo_Swap by idx.
  */
  inline bool operator() (const SwapBlock& struct1, const SwapBlock& struct2)
  {
    return (struct1.r_idx<struct2.r_idx);
  }
};

struct sort_by_idx_descending_swap{
  /*
  sort DeviceOptInfo_Swap by idx. reverse
  */
  inline bool operator() (const SwapBlock& struct1, const SwapBlock& struct2)
  {
    return (struct1.d_idx>struct2.d_idx);
  }
};

struct sort_by_majority_voting_ascending{
  /*
  sort majority voting, ascending
  */
  inline bool operator() (const SwapBlock& struct1, const SwapBlock& struct2)
  {
    return (struct1.majority_voting<struct2.majority_voting);
  }
};


pair<int,int> GetOptIdxAboveLoadLimit(vector<double>vec_load, size_t mem_limit, int start_idx, int end_idx,int iteration_length){
  /*
  get operation index (range) that above the load limit.
  input: vec_load, mem_limit, range [start_idx, end_idx)
  return range overlimit [first_over_limit, first_below_limit)
  */
  int first_over_limit = start_idx;
  int first_below_limit = end_idx;

  for (int i = start_idx+iteration_length; i < end_idx+iteration_length; i++){
    if (vec_load[i] > mem_limit){
      first_over_limit = i-iteration_length;
      break;
    }
  }

  for (int i = end_idx+iteration_length; i > first_over_limit+iteration_length; i--){
    if (vec_load[i] > mem_limit){
      first_below_limit = i-1-iteration_length;
      break;
    }
  }

  if (first_over_limit == start_idx) first_over_limit = -1;
  
  if (first_below_limit == end_idx) first_below_limit = -1;

  return std::make_pair(first_over_limit, first_below_limit);
}


pair<double,int> GetLoadPeak(vector<double>vec_load_test,int iteration_length){
  /*
  return value and index of load peak
  */
  double max_load_test = 0;
  int max_idx_test = 0;
  for (int i = iteration_length; i < iteration_length*2; i++){
    if (max_load_test < vec_load_test[i]){
      max_load_test = vec_load_test[i];
      max_idx_test = i - iteration_length;
    } 
  }
  return std::make_pair(max_load_test,max_idx_test);
}

void UpdateLoad(vector<double>& vec_load,int start_idx, int end_idx, int plus_minus, size_t size,int iteration_length){
  /*
  update load [start_idx, end_idx) by plus_minus*size
  */
  for (int i = start_idx+iteration_length; i<end_idx+iteration_length; i++){
    vec_load[i] = vec_load[i] + static_cast<double>(size) * plus_minus;
  }
}


///define SwapGPU member functions
vector<SwapBlock> SwapGPU::SelectBlock(vector<SwapBlock>vec_swap,vector<double> temp_load,double mem_limit,string mode){
  vector<SwapBlock>vec_swap_selct;
  /*
  select swapping blocks based on a cetain priority score or BO score;
  with load updated
  */
  if (mode == "DOA_origin"){
    sort(vec_swap.begin(),vec_swap.end(),sort_by_DOA_origin_descending());  
  }

  if (mode == "AOA"){
    sort(vec_swap.begin(),vec_swap.end(),sort_by_AOA_descending());  
  }

  if (mode == "WDOA"){
    for (int i = 0; i < vec_swap.size(); i++){
      auto itm = vec_swap[i];
      for (int j = itm.r_idx; j < itm.d_idx; j++){
        itm.WDOA += origin_load[i+iteration_length] - mem_limit;
      }
    }
    sort(vec_swap.begin(),vec_swap.end(),sort_by_WDOA_descending()); 
  }

  if (mode == "majority_voting"){
    //add order for DOA
    sort(vec_swap.begin(),vec_swap.end(),sort_by_DOA_origin_descending()); 
    for (int i = 0; i < vec_swap.size();i++){
      vec_swap[i].majority_voting+=i;
    }
    //add order for AOA
    sort(vec_swap.begin(),vec_swap.end(),sort_by_AOA_descending()); 
    for (int i = 0; i < vec_swap.size();i++){
      vec_swap[i].majority_voting+=i;
    }
    //add order for WDOA
    for (int i = 0; i < vec_swap.size(); i++){
      auto itm = vec_swap[i];
      for (int j = itm.r_idx; j < itm.d_idx; j++){
        itm.WDOA += origin_load[i+iteration_length] - mem_limit;
      }
    }
    sort(vec_swap.begin(),vec_swap.end(),sort_by_WDOA_descending()); 
    for (int i = 0; i < vec_swap.size();i++){
      vec_swap[i].majority_voting+=i;
    }
    sort(vec_swap.begin(),vec_swap.end(),sort_by_majority_voting_ascending()); 
  }



  //select block one by one till updated peak load is no larger than limit.
  for (int i=0; i<vec_swap.size(); i++){
    UpdateLoad(temp_load,vec_swap[i].r_idx_ready,vec_swap[i].d_idx,-1,vec_swap[i].size,iteration_length);
    vec_swap_selct.push_back(vec_swap[i]);
    auto temp_over_limit_ = GetOptIdxAboveLoadLimit(temp_load,mem_limit,0,iteration_length,iteration_length);
    auto max_current = GetLoadPeak(temp_load,iteration_length);
    auto newmax_load = max_current.first;
    if (newmax_load < mem_limit){
      break;
    }
  }
  
  return vec_swap_selct;
}

vector<double> SwapGPU::GetIdealLoad(vector<double>vec_load,vector<SwapBlock> vec_swap_selct){
  /*
  get load_ideal, which is equivalent to load by synchronous swapping.
  */
  auto vec_load_return = vec_load;
  for (int i =0; i<vec_swap_selct.size(); i++){
    int auto_buffer = 0;
    auto itm = vec_swap_selct[i];
    if (itm.cat == "A2") auto_buffer = data_buffer;
    if (itm.cat == "A3") auto_buffer = mutable_data_buffer;
    UpdateLoad(vec_load_return, itm.r_idx+auto_buffer, itm.d_idx, -1, itm.size, iteration_length);
  }
  return vec_load_return;
}

void SwapGPU::Scheduling(vector<SwapBlock>&vec_swap_selct, vector<double>&vec_load_temp,double &overhead,double mem_limit,string mode){
  /*
  Swap Scheduling algo
  update idx_out_end, idx_in_start 
  compute overhead time 
  mode selection: no overhead or stick to limit.
  */ 

  overhead = 0;

  /// mode that stick to the mem_limit
  if (mode == "stick-to-limit"){
    sort(vec_swap_selct.begin(),vec_swap_selct.end(),sort_by_idx_ascending_swap()); 
    for (int i = 0; i<vec_swap_selct.size(); i++){
      auto itm = vec_swap_selct[i];
      int ready_idx = itm.r_idx_ready;

      if (i > 0){
        ready_idx = std::max(ready_idx,vec_swap_selct[i-1].idx_out_end);
      }

      itm.idx_out_start = ready_idx;
      itm.t_out_start = vec_run[ready_idx+iteration_length].t;
      itm.t_out_end = itm.t_out_start + SwapOutTime(itm.size);
      total_swap_out_time+=SwapOutTime(itm.size);
      while (itm.t_out_end > vec_run[ready_idx+iteration_length].t){ 
        //ready means when able to finish swapOut, w/ or w/o overhead.
        ready_idx++; 
      }

      //get min compare with max_idx and ready_idx.
      ready_idx = std::min(max_idx,ready_idx);
      UpdateLoad(vec_load_temp,ready_idx+1,itm.d_idx,-1,itm.size,iteration_length);
      auto temp_over_limit_ = GetOptIdxAboveLoadLimit(vec_load_temp,mem_limit,0,iteration_length,iteration_length);
      if ((temp_over_limit_.first != -1) && (temp_over_limit_.first <= ready_idx)) { 
        UpdateLoad(vec_load_temp,temp_over_limit_.first-1,ready_idx+1,-1,itm.size,iteration_length);
        ready_idx = temp_over_limit_.first - 1; 
        overhead+=(itm.t_out_end-vec_run[ready_idx+iteration_length].t);
      }
      itm.idx_out_end = ready_idx;
      vec_swap_selct[i] = itm; 
    }

    sort(vec_swap_selct.begin(),vec_swap_selct.end(),sort_by_idx_descending_swap());
    for (int i =0; i<vec_swap_selct.size(); i++){
      auto itm = vec_swap_selct[i];
      int need_idx = itm.d_idx;
      if (i > 0){ need_idx = std::min(need_idx,vec_swap_selct[i-1].idx_in_start); }
      itm.idx_in_end = need_idx;
      double prepareTime = vec_run[need_idx+iteration_length].t - SwapInTime(itm.size);
      total_swap_in_time+=SwapInTime(itm.size);
      while (prepareTime < vec_run[need_idx+iteration_length].t){
        need_idx--;
      }
      need_idx = std::max(need_idx,max_idx+1);
      itm.idx_in_start = need_idx;
      itm.t_in_start = prepareTime;
     UpdateLoad(vec_load_temp,itm.idx_in_start,itm.d_idx,1,itm.size,iteration_length); 
      auto temp_over_limit_3 = GetOptIdxAboveLoadLimit(vec_load_temp,mem_limit,0,iteration_length,iteration_length);

      if ((temp_over_limit_3.second != -1) && (vec_run[temp_over_limit_3.second+iteration_length].t > itm.t_in_start)) {
        overhead+=(vec_run[temp_over_limit_3.second+iteration_length].t - itm.t_in_start);
        UpdateLoad(vec_load_temp,itm.idx_in_start,temp_over_limit_3.second+1,-1,itm.size,iteration_length);
        itm.idx_in_start = temp_over_limit_3.second+1;
        auto temp_over_limit_4 = GetOptIdxAboveLoadLimit(vec_load_temp,mem_limit,0,iteration_length,iteration_length);
      }
      vec_swap_selct[i] = itm;
    }
  }///end of first mode.


  ///mode that incurs zero overhead
  if (mode == "no-overhead"){
    //update idx_out_end
    //sort by r_idx for idx_out_end update
    sort(vec_swap_selct.begin(),vec_swap_selct.end(),sort_by_idx_ascending_swap()); 
    for (int i = 0; i<vec_swap_selct.size(); i++){
      auto itm = vec_swap_selct[i];
      int ready_idx = 0;
      if (itm.cat == "A1") { ready_idx = itm.r_idx; }
      if (itm.cat == "A2") { ready_idx = itm.r_idx + data_buffer; }
      if (itm.cat == "A3") { ready_idx = itm.r_idx + mutable_data_buffer; }

      if (i > 0){
        ready_idx = std::max(ready_idx,vec_swap_selct[i-1].idx_out_end);
      }
      itm.idx_out_start = ready_idx;
      itm.t_out_start = vec_run[ready_idx].t;
      itm.t_out_end = itm.t_out_start + SwapOutTime(itm.size);
      while (itm.t_out_end > vec_run[ready_idx].t){
        ready_idx++;
      }
      itm.idx_out_end = ready_idx;
      vec_swap_selct[i] = itm;
    }
    //update idx_in_start
    sort(vec_swap_selct.begin(),vec_swap_selct.end(),sort_by_idx_descending_swap());
    for (int i =0; i<vec_swap_selct.size(); i++){
      auto itm = vec_swap_selct[i];
      int need_idx = itm.d_idx;
      if (i > 0){ need_idx = std::min(need_idx,vec_swap_selct[i-1].idx_in_start); }
      itm.idx_in_end = need_idx;
      double prepareTime = vec_run[need_idx].t - SwapInTime(itm.size);
      while (prepareTime < vec_run[need_idx].t){
        need_idx--;
      }
      itm.idx_in_start = need_idx;
      itm.t_in_start = prepareTime;
      vec_swap_selct[i] = itm;
      UpdateLoad(vec_load_temp,itm.idx_out_end,itm.idx_in_start+1,-1,itm.size,iteration_length);
    }

  }
  
}


void SwapGPU::BuildMetaTables(vector<SwapBlock>vec_swap_selct){
  /*
  construct tables: table_sched, and table_meta
  */
  cudaStream_t stream1;
  cudaStream_t stream2;
  sort(vec_swap_selct.begin(),vec_swap_selct.end(),sort_by_idx_ascending_swap()); 
  //for each swap select, make table_sched and table_meta
  // for (int i = static_cast<int>(vec_swap_selct.size()-1);i>=0; i--){
  for (int i =0; i<vec_swap_selct.size(); i++){
    auto itm = vec_swap_selct[i];

    if (table_sched.find(itm.idx_out_start) == table_sched.end()){
      table_sched[itm.idx_out_start] = std::make_tuple(itm.r_idx,0,-1,-1);
    } else {
      std::get<0>(table_sched.find(itm.idx_out_start)->second) = itm.r_idx;
      std::get<1>(table_sched.find(itm.idx_out_start)->second) = 0;
    }
    //idx_in_start swap
    if (table_sched.find(itm.idx_in_start) == table_sched.end()){
      table_sched[itm.idx_in_start] = std::make_tuple(itm.r_idx,1,-1,-1);      
    } else {
      std::get<0>(table_sched.find(itm.idx_in_start)->second) = itm.r_idx;
      std::get<1>(table_sched.find(itm.idx_in_start)->second) = 1;
    }
    // idx_out_end sync
    if (table_sched.find(itm.idx_out_end) == table_sched.end()){
      table_sched[itm.idx_out_end] = std::make_tuple(-1,-1,itm.r_idx,0);
    } else {
      std::get<2>(table_sched.find(itm.idx_out_end)->second) = itm.r_idx;
      std::get<3>(table_sched.find(itm.idx_out_end)->second) = 0; 
    }
    //i2 sync
    if (table_sched.find(itm.idx_in_end) == table_sched.end()){
      table_sched[itm.idx_in_end] = std::make_tuple(-1,-1,itm.r_idx,1);
    } else {
      std::get<2>(table_sched.find(itm.idx_in_end)->second) = itm.r_idx;
      std::get<3>(table_sched.find(itm.idx_in_end)->second) = 1;
    }

    ///Make table_meta
    void* temp_ptr = nullptr;
    cudaMallocHost(&temp_ptr,itm.size); //pinned memory.
    BlockMeta meta;
    meta.size = itm.size;
    meta.cpu_ptr = temp_ptr;
    meta.out_stream = stream1;
    meta.in_stream = stream2;
    table_meta[itm.r_idx] = meta;
  }

}

void SwapGPU::UpdateMetaTables(Block* block_ptr){
  /*
  update table_meta's block_ and data_; update once atfer swap test is passed.
  enable to update negative r_idx. 
  it's safe in below procedure, as r_global_index and relative_counter should never be the same.
  */

  if (past_test_flag == 1) {
    //update positive r_idx
    int r_global_index = (global_index-location_of_2nd_iteration)%iteration_length;
    if (!(table_meta.find(r_global_index)==table_meta.end())){
     table_meta.find(r_global_index)->second.block_ = block_ptr;
      table_meta.find(r_global_index)->second.data_ = block_ptr->get_data();
    }

    //update negative r_idx
    int relative_counter = r_global_index - iteration_length;
    if (!(table_meta.find(relative_counter)==table_meta.end())){
      table_meta.find(relative_counter)->second.block_ = block_ptr;
      table_meta.find(relative_counter)->second.data_ = block_ptr->get_data();
    }
  }

}

int SwapGPU::Detection(vector<string>vec_block,int &iteration_length, int &location_of_2nd_iteration){
  /*
  test repeatability, detect iteration, and return global_index_threshold.
  */

  ///vec_str (vec_block) to vec_opt_info, sort by ptr and idx.
  int idx_range = 0; 
  vector<DeviceOptInfo> vec_opt_info = DeviceOptSeqStrToStruct(vec_block,idx_range);

  ///rep test
  vector<size_t> vec_rep = DeviceOptSeqRepeatableTestPreProcess(vec_opt_info);
  RepeatableTest(vec_rep,iteration_length,location_of_2nd_iteration,iteration_length_threshold,global_index);

  //Note here location_of_2nd_iteration not exactly start of one iteration, 
  //adjust to nearly start of one by restricting "Malloc"
  int shift_counter = 0;
  for (int i=0;i<iteration_length;i++){
    vector<string> v = SplitOptString(vec_block[location_of_2nd_iteration+i], " ");
    if (v[0]=="Malloc"){
      shift_counter = i; 
      break;
    }
  }
  location_of_2nd_iteration =location_of_2nd_iteration+shift_counter;

  if (iteration_length<iteration_length_threshold) {return -1;}

  return global_index+iteration_length-(global_index-location_of_2nd_iteration)%iteration_length;
} 

void SwapGPU::Plan(){
  /*
  major stream of functions: from make candidate blocks, selection swaps, make tables, etc.
  */

  int idx_range = 0;
  vector<DeviceOptInfo> vec_opt_info = DeviceOptSeqStrToStruct(vec_block,idx_range);
  sort(vec_opt_info.begin(),vec_opt_info.end(),sort_by_idx_ascending());
  
  // scale down idx, to middle iteration.
  temp_time_baseline = vec_opt_info[location_of_5th_iteration].t;
  for (int i=0; i<vec_opt_info.size();i++){
    vec_opt_info[i].idx = vec_opt_info[i].idx - location_of_5th_iteration - iteration_length;
    vec_opt_info[i].t = vec_opt_info[i].t - temp_time_baseline;
  }

  // build opsSqn, and sizeSqn
  vector<DeviceOptInfo>one_itr(&vec_opt_info[location_of_2nd_iteration+4*iteration_length],&vec_opt_info[location_of_2nd_iteration+5*iteration_length]);
  for (int i =0; i<one_itr.size();i++){
    operation_sequence.push_back(one_itr[i].operation_type);
    size_sequence.push_back(one_itr[i].size);
  }
  
  //3 iterations of vec_run and vec_load, max_idx and max_load
  vector<DeviceOptInfo>temp_vec_run(&vec_opt_info[location_of_2nd_iteration+3*iteration_length],&vec_opt_info[location_of_2nd_iteration+6*iteration_length]);
  vec_run = temp_vec_run;

  vector<DeviceOptInfo>temp_vec_run2(&vec_opt_info[location_of_2nd_iteration],&vec_opt_info[location_of_2nd_iteration+3*iteration_length]);
  auto vec_run2 = temp_vec_run2;


  vector<double>vec_load(&global_load[location_of_2nd_iteration],&global_load[location_of_2nd_iteration+3*iteration_length]);
  origin_load = vec_load;

  auto max_current = GetLoadPeak(vec_load,iteration_length);
  max_load = max_current.first;
  max_idx = max_current.second;

  //sort by ptr & idx, sorting the duplicate
  auto vec_run_dup = vec_run;
  sort(vec_run_dup.begin(),vec_run_dup.end(),sort_by_ptr_idx_ascending());
  
  ///formulate swappable items.
  vector<SwapBlock>vec_swap;

  for (int i =1; i<vec_run_dup.size(); i++){
    //SwapBlock(string p, size_t s, int idx_out_start, int i2, double t1, double t2): 
    //ptr(p), size(s), r_idx(idx_out_start),d_idx(i2),r_time(t1), d_time(t2) {}
    if ((vec_run_dup[i].size >= smallest_block) && (vec_run_dup[i-1].idx<max_idx) && (vec_run_dup[i].idx>max_idx) 
      && (vec_run_dup[i-1].ptr ==vec_run_dup[i].ptr) 
      && ((vec_run_dup[i-1].operation_type==3) or (vec_run_dup[i-1].operation_type==2) or (vec_run_dup[i-1].operation_type==4)))
    {
      SwapBlock itm(vec_run_dup[i].ptr, vec_run_dup[i].size, vec_run_dup[i-1].idx, vec_run_dup[i].idx, vec_run_dup[i-1].t, vec_run_dup[i].t);
      itm.DOA_origin = itm.d_time-itm.r_time;
      itm.DOA = itm.d_time-itm.r_time-SwapOutTime(itm.size)-SwapOutTime(itm.size);
      if (itm.DOA>=0){
        itm.AOA = itm.DOA * itm.size;
      } else {
        itm.AOA = itm.DOA * 1/itm.size;
      }
      //cat A
      if (vec_run_dup[i-1].operation_type == 3){ itm.cat = "A1"; itm.r_idx_ready = itm.r_idx; } 
      if (vec_run_dup[i-1].operation_type == 2){ itm.cat = "A2"; itm.r_idx_ready = itm.r_idx + data_buffer;} 
      if (vec_run_dup[i-1].operation_type == 4){ itm.cat = "A3"; itm.r_idx_ready = itm.r_idx + mutable_data_buffer;} 

      vec_swap.push_back(itm);
    } 
  }

  ///load ideal, swap all vec_swap, lest possible memory by one-swap, for data collection only.
  auto vec_load_ideal = GetIdealLoad(vec_load,vec_swap);
  fstream file_load_ideal("load_ideal.csv", ios::in|ios::out|ios::app);
  for (int i=iteration_length; i<iteration_length*2; i++){
    file_load_ideal<<vec_load_ideal[i]<<endl;
  }

  auto max_ideal = GetLoadPeak(vec_load_ideal,iteration_length);
  size_t max_load_ideal = max_ideal.first;
  int max_idx_ideal = max_ideal.second;

  /// majority voting, can specify mode here, can specify load_limit
  auto temp_load = origin_load;
  auto mem_limit_majority_voting = 550<<20;
  auto vec_swap_majority_voting = SelectBlock(vec_swap,temp_load,mem_limit_majority_voting,"majority_voting");
  // vec_swap_selct_global = vec_swap_majority_voting;

  auto vec_load_WDOA = origin_load;
  string mode = "stick-to-limit";

  double overhead_WDOA = 0;
  Scheduling(vec_swap_majority_voting, vec_load_WDOA,overhead_WDOA,mem_limit_majority_voting,mode);

  BuildMetaTables(vec_swap_majority_voting);

}



SwapGPU::~SwapGPU() {
  //print out push-info
  fstream file_block_full("vec_block_full.csv", ios::in|ios::out|ios::app);
  for (int i =0; i<vec_block.size();i++){
    file_block_full<<vec_block[i]<<endl;
  }

  if (ctx_.cublas_handle) CUBLAS_CHECK(cublasDestroy(ctx_.cublas_handle));
  if (ctx_.curand_generator)
    CURAND_CHECK(curandDestroyGenerator(ctx_.curand_generator));
#ifdef USE_CUDNN
  if (ctx_.cudnn_handle) {
    auto status = cudnnDestroy(ctx_.cudnn_handle);
    CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << cudnnGetErrorString(status);
  }
#endif
}
const int kNumCudaStream = 1;

SwapGPU::SwapGPU(int id) : Device(id, kNumCudaStream) {

  MemPoolConf conf;
  conf.add_device(id);
  pool_ = std::make_shared<SwapPool>(conf); 
  Setup();

}

SwapGPU::SwapGPU(int id, std::shared_ptr<DeviceMemPool> pool)
    : Device(id, kNumCudaStream) {
  CHECK(pool != nullptr);
  pool_ = pool;
  Setup();
}

void SwapGPU::Setup() {
  lang_ = kCuda;
  ctx_.stream = NULL;  // use the default sync stream
  // TODO(wangwei) create one handle for each steam?
  CUDA_CHECK(cudaSetDevice(id_));
  // use curandCreateGeneratorHost for CudaHost device
  CURAND_CHECK(
      curandCreateGenerator(&ctx_.curand_generator, CURAND_RNG_PSEUDO_DEFAULT));
  auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  SetRandSeed(seed);
  // TODO(wangwei) if one generator per stream, then need diff offset per gen?
  CURAND_CHECK(curandSetGeneratorOffset(ctx_.curand_generator, 0));
  CUBLAS_CHECK(cublasCreate(&(ctx_.cublas_handle)));

#ifdef USE_CUDNN
  // TODO(wangwei) create one handle for each stream?
  auto status = cudnnCreate(&ctx_.cudnn_handle);
  CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << cudnnGetErrorString(status);
#endif  // USE_CUDNN
}

void SwapGPU::SetRandSeed(unsigned seed) {
  CHECK(ctx_.curand_generator);
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(ctx_.curand_generator, seed));
}

void SwapGPU::DoExec(function<void(Context*)>&& fn, int executor) { fn(&ctx_); }

void SwapGPU::CopyToFrom(void* dst, const void* src, size_t nBytes,
                         CopyDirection direction, Context* ctx) {
  cudaMemcpy(dst, src, nBytes, copyKind[direction]);
  // TODO(wangwei) use async copy
  // cudaMemcpyAsync(dst, src, nBytes,cudaMemcpyDefault, ctx_.stream);
}

size_t SwapGPU::GetAllocatedMem() {
  if (pool_ != nullptr) {
    auto ret = pool_->GetMemUsage();
    return ret.second - ret.first;
  }
  LOG(ERROR) << "The memory pool is not set";
  return 0u;
}

/// Allocate gpu memory.
void* SwapGPU::Malloc(int size) {

  void* ptr = nullptr;
  if (size > 0) {
    CUDA_CHECK(cudaSetDevice(id_));
    pool_->Malloc((void**)&ptr, size);

    ///append vec_block_mf:for swap & pool
    if ((async_swap_flag == 1) && ((global_index - 4*iteration_length) < three_more_iteration_global_index_threshold)
      && ((global_index - iteration_length) >= three_more_iteration_global_index_threshold)){
      string temp_str1 ="Malloc ";
      stringstream strm2;
      strm2<<ptr;
      string temp_str2 = strm2.str();
      stringstream strm3;
      strm3<<size;
      string temp_str3 = strm3.str();
      string temp = temp_str1+temp_str2+" "+temp_str3;
      vec_block_mf.push_back(temp);
    }
    //record mf semantics after swap plan done
    if ((async_swap_flag == 1) && ((global_index - 4*iteration_length) < three_more_iteration_global_index_threshold)){
      fstream file_mf_one_itr("mf_one_itr.csv", ios::in|ios::out|ios::app);
      file_mf_one_itr<<"Malloc "<<ptr<<" "<<size;
      file_mf_one_itr<<endl;
    }
    // TODO(wangwei) remove the memset.
    CUDA_CHECK(cudaMemset(ptr, 0, size));
  }
  return ptr;
}

/// Free gpu memory.
void SwapGPU::Free(void* ptr) {

  if (ptr != nullptr) {
    CUDA_CHECK(cudaSetDevice(id_));
    pool_->Free(ptr);
    ///append vec_block_mf: for swap & pool
    if ((async_swap_flag == 1) && ((global_index - 4*iteration_length) < three_more_iteration_global_index_threshold)
      && ((global_index - iteration_length) >= three_more_iteration_global_index_threshold)){
      string temp_str1 ="Free ";
      stringstream strm2;
      strm2<<ptr;
      string temp_str2 = strm2.str();
      string temp = temp_str1+temp_str2;
      vec_block_mf.push_back(temp);
    }

    if ((async_swap_flag == 1) && ((global_index - 4*iteration_length) < three_more_iteration_global_index_threshold)){
      fstream file_mf_one_itr("mf_one_itr.csv", ios::in|ios::out|ios::app);
      file_mf_one_itr<<"Free "<<ptr<<endl;
    }
  }

}

void SwapGPU::DetectionPlan(){
  /*
    test after every index, at Append. order and index changed.
  */
  ///test iteration
  if (((global_index+1)%(iteration_length_threshold) == 0) && (async_swap_flag == 0) && (past_test_flag == 0)){
    global_index_threshold = Detection(vec_block,iteration_length,location_of_2nd_iteration);
    iteration_length_threshold = std::max(iteration_length_threshold,global_index/10);
    iteration_length_threshold = std::min(2000,iteration_length_threshold);
    if (iteration_length > iteration_length_threshold) {
      past_test_flag = 1;
      three_more_iteration_global_index_threshold = global_index_threshold + 3*iteration_length;
      location_of_5th_iteration = location_of_2nd_iteration + 3*iteration_length;      
   }
 }
 ///switch flag; next idx
 if ((global_index+1) == three_more_iteration_global_index_threshold){
    Plan();
    async_swap_flag = 1;
 }
}

void SwapGPU::AppendAfterMalloc(Block* block_ptr,void* data_ptr,int size){
  /*
  Append info right after Malloc; make block_ptr - data_ptr pair wise table.
  as Block* is not available till Malloc() done.
  */

  //append info
  stringstream strm;
  strm<<block_ptr;
  string temp_str = strm.str();
  DeviceOptInfoToAppend dev_opt_info("Malloc", temp_str,size);
  auto t = (std::chrono::system_clock::now()).time_since_epoch().count();
  dev_opt_info.t = t;
  Append(dev_opt_info);
  
}

void SwapGPU::DeploySwap(){
  /*
  swap and sync as per schedule, at every index, by calling DeploySwapExec()
  */

  int r_global_index = (global_index-location_of_2nd_iteration)%iteration_length; 
  int r_global_index_n = r_global_index - iteration_length;

  if (async_swap_flag == 1){
    if ((global_index < three_more_iteration_global_index_threshold + iteration_length) && (!(table_sched.find(r_global_index_n) == table_sched.end()))) {
      DeploySwapExec(r_global_index_n);
    }
    if ((global_index >= three_more_iteration_global_index_threshold + iteration_length) && (!(table_sched.find(r_global_index_n) == table_sched.end()))) {
      DeploySwapExec(r_global_index_n);
    }
    if ((global_index >= three_more_iteration_global_index_threshold + iteration_length) && (!(table_sched.find(r_global_index) == table_sched.end()))) {
      DeploySwapExec(r_global_index);
    }
  }
}


void SwapGPU::DeploySwapExec(int r_global_index){
  //execute DeploySwap 
  auto swap_idx = std::get<0>(table_sched.find(r_global_index)->second);
  auto swap_dir = std::get<1>(table_sched.find(r_global_index)->second);
  auto sync_idx = std::get<2>(table_sched.find(r_global_index)->second);
  auto sync_dir = std::get<3>(table_sched.find(r_global_index)->second);
  if (swap_dir == 0){ 
    SwapOut(swap_idx); 
  }
  if (swap_dir == 1){ 
    SwapIn(swap_idx); 
  }
  if (sync_dir == 0){
    ///sync swap-out, including sync, update block's data_ to nullptr, free data_, update meta.
    auto last_meta = table_meta.find(sync_idx)->second;
    auto t1 = (std::chrono::system_clock::now()).time_since_epoch().count();
    cudaEventSynchronize(last_meta.in_event);
    auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();

    table_not_at_device[last_meta.block_] = sync_idx;

    last_meta.block_->update_data(nullptr);
    pool_->Free(last_meta.data_);
    ///append vec_block_mf
    if ((async_swap_flag == 1) && ((global_index - 4*iteration_length) < three_more_iteration_global_index_threshold)
      && ((global_index - iteration_length) >= three_more_iteration_global_index_threshold)){
      string temp_str1 ="Free ";
      stringstream strm2;
      strm2<<last_meta.data_;
      string temp_str2 = strm2.str();
      string temp = temp_str1+temp_str2;
      vec_block_mf.push_back(temp);
    }

    if ((async_swap_flag == 1) && ((global_index - 4*iteration_length) < three_more_iteration_global_index_threshold)){
      fstream file_mf_one_itr("mf_one_itr.csv", ios::in|ios::out|ios::app);
      file_mf_one_itr<<"Free "<<last_meta.data_<<" SwapOut(Sync)"<<endl;
    }
    last_meta.data_ = nullptr;
    table_meta.find(sync_idx)->second = last_meta;
  }
  if (sync_dir == 1){
    ///sync swap-in, including sync, update block's data_ to new gpu address, update meta.
    auto last_meta = table_meta.find(sync_idx)->second;
    auto t1 = (std::chrono::system_clock::now()).time_since_epoch().count();
    cudaEventSynchronize(last_meta.out_event);
    auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
    table_not_at_device.erase(last_meta.block_);
    last_meta.block_->update_data(last_meta.data_);
    table_meta.find(sync_idx)->second = last_meta;
  }
}

void SwapGPU::Append(DeviceOptInfoToAppend dev_opt_info){

  //convert block_ptr from string to Block*
  void* temp_ptr;
  stringstream convert(dev_opt_info.block_ptr);
  convert>>temp_ptr;
  auto block_ptr = static_cast<Block*>(temp_ptr);

  // update global load
  if (iteration_length < iteration_length_threshold){
    if (dev_opt_info.operation_type == "Malloc"){
      if (global_load.size()>0){
        global_load.push_back(global_load[global_load.size()-1]+block_ptr->size());
      } else {
        global_load.push_back(block_ptr->size());
      }
    } else if (dev_opt_info.operation_type  == "Free"){
      global_load.push_back(global_load[global_load.size()-1]-block_ptr->size());
    } else {
      global_load.push_back(global_load[global_load.size()-1]);
    }
  }

  //append into vec_block
  stringstream strm1;
  strm1<<dev_opt_info.size;
  string temp_str1 = strm1.str();
  stringstream strm4;
  strm4<<dev_opt_info.t;
  string temp_str4 = strm4.str();
  string block_info = dev_opt_info.operation_type + " " + dev_opt_info.block_ptr + " " +
  temp_str1 + " " + temp_str4;
  //cout<<"1 "<<block_info<<endl;
  vec_block.push_back(block_info);

  //change swap flag on and off
  if (async_swap_flag == 1){
    int r_global_index = (global_index-location_of_2nd_iteration)%iteration_length;
    if (block_ptr->size() != size_sequence[r_global_index]){
      async_swap_flag = 0;
      cout<<"!!!! async_swap_flag changed back to 0"<<endl;
    }
  }

  //update table_meta and table_sched
  UpdateMetaTables(block_ptr);

  //deploy swap at every index.
  DeploySwap();

  //test moved from start of malloc/free to end of append, only global_index+1 changed
  DetectionPlan();

  //NOTE: this global_index includes read/write and AppendLayer as well, in addition to malloc/free.
  global_index++;

  //call PoolOpt to Construct Pool
  if ((async_swap_flag == 1) && ((global_index - 4 * iteration_length) == three_more_iteration_global_index_threshold)){
    pool_->PoolOpt(vec_block_mf);
  }

}



void* SwapGPU::UpdateGpuPtr(const Block* block_ptr){
  /*
  in case that block is not at device memory, swapIn ad hoc.
  used in block class to update ptr after swap in done, if variable is not swapped back yet as expected.
  */ 
  auto r_idx = table_not_at_device.find(block_ptr)->second;
  cudaError_t err;
  BlockMeta meta = table_meta.find(r_idx)->second;
  cudaEventCreate (&meta.in_event);
  void* ptr = nullptr;
  pool_->Malloc((void**)&ptr, meta.size);
  meta.data_ = ptr;
  err = cudaMemcpyAsync(meta.data_,meta.cpu_ptr,meta.size,cudaMemcpyHostToDevice,meta.in_stream);
  cudaEventRecord(meta.in_event,meta.in_stream);
  cudaEventSynchronize(meta.out_event);
  table_meta.find(r_idx)->second = meta;

  return ptr;
}

void SwapGPU::SwapOut(const int idx){
  /*
  memory copy asynchronously GPU -> CPU, and update meta.
  */
  cudaError_t err;
  BlockMeta meta = table_meta.find(idx)->second;
  cudaEventCreate (&meta.out_event);
  err = cudaMemcpyAsync(meta.cpu_ptr,meta.data_,meta.size,cudaMemcpyDeviceToHost,meta.out_stream);
  cudaEventRecord(meta.out_event,meta.out_stream);
  table_meta.find(idx)->second = meta;
}

void SwapGPU::SwapIn(const int idx){
  /*
  memory copy asynchronously CPU -> GPU, and update meta.
  */

  cudaError_t err;
  BlockMeta meta = table_meta.find(idx)->second;
  cudaEventCreate (&meta.in_event);
  void* ptr = nullptr;
  pool_->Malloc((void**)&ptr, meta.size);

  ///append vec_block_mf
  if ((async_swap_flag == 1) && ((global_index - 4*iteration_length) < three_more_iteration_global_index_threshold)
    && ((global_index - iteration_length) >= three_more_iteration_global_index_threshold)){
    string temp_str1 ="Malloc ";
    stringstream strm2;
    strm2<<ptr;
    string temp_str2 = strm2.str();
    stringstream strm3;
    strm3<<meta.size;
    string temp_str3 = strm3.str();
    string temp = temp_str1+temp_str2+" "+temp_str3;
    vec_block_mf.push_back(temp);
  }
  if ((async_swap_flag == 1) && ((global_index - 4*iteration_length) < three_more_iteration_global_index_threshold)){
    fstream file_mf_one_itr("mf_one_itr.csv", ios::in|ios::out|ios::app);
    file_mf_one_itr<<"Malloc "<<ptr<<" "<<meta.size<<" swapIn"<<endl;
  }

  meta.data_ = ptr;
  err = cudaMemcpyAsync(meta.data_,meta.cpu_ptr,meta.size,cudaMemcpyHostToDevice,meta.in_stream);
  cudaEventRecord(meta.in_event,meta.in_stream);
  table_meta.find(idx)->second = meta;
}

void SwapGPU::SwapOutSynchronous(const Block* block_ptr){
  /*
  for synchronous swap, collect speed info
  */
  if (global_index < 1000 && block_ptr->size() > 1<<20) {
    fstream file_block5("speed.csv", ios::in|ios::out|ios::app);
    BlockMeta meta;
    meta.data_ = meta.block_->get_data();
    void* temp_ptr = nullptr;
    cudaMallocHost(&temp_ptr,block_ptr->size()); //pinned memory.
    meta.cpu_ptr = temp_ptr;
    table_block_meta[block_ptr] = meta;
    auto t1 = (std::chrono::system_clock::now()).time_since_epoch().count();
    cudaError_t err;
    err = cudaMemcpy(meta.cpu_ptr, meta.data_,block_ptr->size(),cudaMemcpyDeviceToHost);
    auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
    file_block5<<"Out "<<block_ptr->size()<<' '<<t2-t1<<endl;
  }
}

void SwapGPU::SwapInSynchronous(const Block* block_ptr){
  /*
  for synchronous swap, collect speed info
  */
  if (global_index < 1000 && block_ptr->size() > 1<<20) {
    fstream file_block5("speed.csv", ios::in|ios::out|ios::app);
    BlockMeta meta = table_block_meta.find(block_ptr)->second;
    auto t1 = (std::chrono::system_clock::now()).time_since_epoch().count();
    cudaError_t err;
    err = cudaMemcpy(meta.data_, meta.cpu_ptr,block_ptr->size(),cudaMemcpyHostToDevice);
    auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
    file_block5<<"In "<<block_ptr->size()<<' '<<t2-t1<<endl;
  }
}

}  // namespace singa
#endif  // USE_CUDA