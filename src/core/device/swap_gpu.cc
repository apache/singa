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

///functions to be used
///Section for structs and respective sorting function:
// onePieceMsg, onePairMsg, oneIterMsg, version 11/30 3pm



struct less_than_ptrIdx{
    /*
     sort onePieceMsg by ptr and then idx.
     */
    inline bool operator() (const onePieceMsg& struct1, const onePieceMsg& struct2)
    {
        return ((struct1.ptr<struct2.ptr)||((struct1.ptr==struct2.ptr)&&(struct1.idx<struct2.idx)));
    }
};


struct oneIterMsg{
    /*
     members: [idx, MallocFree, size_delta]
     */
    size_t size_delta;// type as size_t in case size if large.
    int MallocFree;
    int idx;
    oneIterMsg(size_t s, int M, int i):size_delta(s),MallocFree(M),idx(i){}
};


struct less_than_iterIdx{
    /*
     sort oneIterMsg by Idx.
     */
    inline bool operator() (const oneIterMsg& struct1, const oneIterMsg& struct2)
    {
        return (struct1.idx<struct2.idx);
    }
};

struct less_than_lookupIdx{
    /*
     sort lookUpElement by idx.
     */
    inline bool operator() (const lookUpElement& struct1, const lookUpElement& struct2)
    {
        return (struct1.r_idx<struct2.r_idx);
    }
};


/// string delimiter
vector<string> swap_split(string s, string delimiter) {
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

///Section of converting text file -->vector of Sring --> pieceMsg -->pairMsg -->iterMsg
//vector of pairMsg is used in run.
//vector of iterMsg is used in test.

vector<onePieceMsg> swap_strVec_2_pieceMsgVec(vector<string> vec, int &idxRange){
    /*
     convert vector of string into vector of onePieceMsg, sorted by ptr 
     and then idx, and update idxRange to pieceMsgVec size.
     format of onePieceMsg [ptr, size/-1, flag, idx, timestamp]
     flag: 1 for malloc, -1 for free, 2 for read, 3 for layer,4 for mutable
     version on 5/29, with equval blockInfo length: flag, block_, size, t
     */
    vector<onePieceMsg>onePieceMsgVec_;

    for (int i=0;i<vec.size();i++) {
      vector<string> v = swap_split(vec[i], " ");
      int MallocFree;
      if (v[0]=="Malloc"){
        MallocFree = 1;
      }else if (v[0]=="Free"){
        MallocFree = -1;
      }else if (v[0]=="Mutable"){
        MallocFree = 4;
      }else if (v[0]=="Read"){ 
        MallocFree = 2;
      }else if (v[0]=="Layer"){
        MallocFree = 3;
      }
      //onePieceMsg(string p, size_t s, int M, int i):ptr(p),size(s),MallocFree(M),idx(i){}
      size_t result;
      stringstream convert(v[2]);
      if (!(convert>>result)){
          result =-1;
          cout<<"error for converting size from str to int."<<endl;
      }
      onePieceMsg tempMsg(v[1],result, MallocFree, i);
      double tempTime;
      stringstream convert2(v[3]);
      convert2>>tempTime;
      tempMsg.t =tempTime;
      onePieceMsgVec_.push_back(tempMsg);
    }
 
    sort(onePieceMsgVec_.begin(),onePieceMsgVec_.end(),less_than_ptrIdx());
    idxRange = static_cast<int>(onePieceMsgVec_.size());

    return onePieceMsgVec_;
}// end of strVec_2_pieceMsgVec function


vector<size_t> Swap_piece2rep (vector<onePieceMsg>onePieceMsgVec_){
    vector<oneIterMsg>oneIterMsgVec_;
    string tempStr;
    int tempIdx=0;
    for (int i=0;i<onePieceMsgVec_.size();i++){
        if (onePieceMsgVec_[i].MallocFree==1){
            //update tempStr and idx.
            tempStr = onePieceMsgVec_[i].ptr;
            tempIdx = onePieceMsgVec_[i].idx;
            oneIterMsg tempMsg(onePieceMsgVec_[i].size,1,onePieceMsgVec_[i].idx);
            oneIterMsgVec_.push_back(tempMsg);
        } else {
            oneIterMsg tempMsg(onePieceMsgVec_[i].idx-tempIdx,onePieceMsgVec_[i].MallocFree,onePieceMsgVec_[i].idx);
            tempIdx = onePieceMsgVec_[i].idx;
            oneIterMsgVec_.push_back(tempMsg);
        }
        //cout<<oneIterMsgVec_[i].size_delta<<' '<<oneIterMsgVec_[i].MallocFree<<' '<<oneIterMsgVec_[i].idx<<endl;
    }
    
    sort(oneIterMsgVec_.begin(),oneIterMsgVec_.end(),less_than_iterIdx());
    //only after sort then can create rep.
    vector<size_t>rep; // vector of size_delta, name it as rep for simlisity.
    for (int i =0; i<oneIterMsgVec_.size(); i++){
        rep.push_back(oneIterMsgVec_[i].size_delta);
        //cout<<rep[i]<<endl;
    }
    cout<<"rep size: "<<rep.size()<<endl;
    return rep;
}
void repPatternDetector(vector<size_t>rep, int &maxLen, int &location, int maxLen_threshold, int gc ){
    int idxRange = (int)rep.size();
    int threshold = maxLen_threshold;
    vector<pair<int,int>>maxLen_location;
    
    for (int i=0; i<idxRange;i++){
        if (maxLen>threshold){
            break;
        }
        for (int len=1; len<(idxRange-i);len++){
            if (maxLen>threshold){
                break;
            }
            if((equal(rep.begin()+i,rep.begin()+i-1+len,rep.begin()+i+len))&&(maxLen<len)) {
                maxLen = len;
                location = i;
                maxLen_location.push_back(make_pair(maxLen,location));
                // cout<<"maxLen increased, lcoation and maxLen: ("<<location<<","<<maxLen<<")"<<endl;
            }
        }
    }
}// end of repPatternDetector

struct less_than_Idx{
    /*
     sort onePieceMsg by ptr and then idx.
     */
    inline bool operator() (const onePieceMsg& struct1, const onePieceMsg& struct2)
    {
        return (struct1.idx<struct2.idx);
    }
};


int SwapOutTime(size_t size){
    int ans = 0; //TODO(junzhe) used to be 0.29; new param as per vgg
    if (size==0) {ans = 47200;} else {ans = 0.0756 * size + 47200;}
    return ans;
}

int SwapInTime(size_t size){
    //yet to get the formula
    int ans = 0; //TODO(junzhe) used to be 0.13; new param as per vgg
    if (size==0) {ans = 9700;} else {ans = 0.0823 * size + 9700;}
    return ans;
}

struct less_than_dto{
  /*
   sort SwapBlock by dto, descending
   */
  inline bool operator() (const SwapBlock& struct1, const SwapBlock& struct2)
  {
    return (struct1.dto>struct2.dto);
  }
};

struct less_than_wdto{
  /*
   sort SwapBlock by weighted dto, descending
   */
  inline bool operator() (const SwapBlock& struct1, const SwapBlock& struct2)
  {
    return (struct1.wdto>struct2.wdto);
  }
};

// struct less_than_r_idx_ready{
//   /*
//    sort SwapBlock by r_idx_ready, ascending
//    */
//   inline bool operator() (const SwapBlock& struct1, const SwapBlock& struct2)
//   {
//     return (struct1.r_idx_ready<struct2.r_idx_ready);
//   }
// };

struct less_than_pri{
    /*
     sort SwapBlock by pri, descending
     */
    inline bool operator() (const SwapBlock& struct1, const SwapBlock& struct2)
    {
        return (struct1.pri>struct2.pri);
    }
};

struct less_than_Idx_Swap{
    /*
     sort onePieceMsg_Swap by idx.
     */
    inline bool operator() (const SwapBlock& struct1, const SwapBlock& struct2)
    {
        return (struct1.r_idx<struct2.r_idx);
    }
};

struct less_than_Idx_Swap_rvs{
    /*
     sort onePieceMsg_Swap by idx. reverse
     */
    inline bool operator() (const SwapBlock& struct1, const SwapBlock& struct2)
    {
        return (struct1.d_idx>struct2.d_idx);
    }
};


pair<int,int> load_over_limit(vector<double>vec_load, size_t memLimit, int start_idx, int end_idx,int maxLen){
  //input: vec_load, memLimit, range [start_idx, end_idx)
  //return range overlimit [first_over_limit, first_below_limit)
  int first_over_limit = start_idx;
  int first_below_limit = end_idx;

  for (int i = start_idx+maxLen; i < end_idx+maxLen; i++){
    if (vec_load[i] > memLimit){
      first_over_limit = i-maxLen;
      break;
    }
  }

  for (int i = end_idx+maxLen; i > first_over_limit+maxLen; i--){
    if (vec_load[i] > memLimit){
      first_below_limit = i-1-maxLen;
      break;
    }
  }
  if (first_over_limit == start_idx) first_over_limit = -1;
  if (first_below_limit == end_idx) first_below_limit = -1;

  return std::make_pair(first_over_limit, first_below_limit);
}

// pair<int,int> load_below_limit(vector<double>vec_load, size_t memLimit, int start_idx, int end_idx, int maxIdx,int maxLen){
//   //input: vec_load, memLimit, range [start_idx, end_idx]
//   //return range overlimit [first_over_limit, first_below_limit)
//   int first_below_limit = maxIdx;
//   int last_below_limit = maxIdx;

//   for (int i = first_below_limit+maxLen; i > start_idx+maxLen; i--){
//     if (vec_load[i] > memLimit){
//       first_below_limit = i+1-maxLen;
//       break;
//     }
//   }

//   for (int i = last_below_limit+maxLen; i < end_idx+maxLen; i++){
//     if (vec_load[i] > memLimit){
//       last_below_limit = i-1-maxLen;
//       break;
//     }
//   }

//   return std::make_pair(first_below_limit, last_below_limit);
// }

pair<double,int> load_peak(vector<double>vec_load_test,int maxLen){
  double maxLoad_test = 0;
  int maxIdx_test = 0;
  for (int i = maxLen; i < maxLen*2; i++){
    if (maxLoad_test < vec_load_test[i]){
      maxLoad_test = vec_load_test[i];
      maxIdx_test = i - maxLen;
    } 
  }
  return std::make_pair(maxLoad_test,maxIdx_test);
}

void load_update(vector<double>& vec_load,int start_idx, int end_idx, int plusMinus, size_t size,int maxLen){
  //update load [start_idx, end_idx) by plusMinus*size
  for (int i = start_idx+maxLen; i<end_idx+maxLen; i++){
    vec_load[i] = vec_load[i] + static_cast<double>(size) * plusMinus;
  }
}

vector<SwapBlock> SwapGPU::swap_select(vector<SwapBlock>vec_swap,vector<double> tempLoad,double memLimit,string mode){
  vector<SwapBlock>vec_swap_selct;
  //vector<SwapBlock>vec_swap_reject;
  if (mode == "dto"){
    sort(vec_swap.begin(),vec_swap.end(),less_than_dto());  
  }
  if (mode == "pri"){
    sort(vec_swap.begin(),vec_swap.end(),less_than_pri());  
  }
  if (mode == "wdto"){
    //TODO(junzhe) time complexity
    for (int i = 0; i < vec_swap.size(); i++){
      auto itm = vec_swap[i];
      for (int j = itm.r_idx; j < itm.d_idx; j++){
        itm.wdto += origin_load[i+maxLen] - memLimit;
      }
    }
    sort(vec_swap.begin(),vec_swap.end(),less_than_wdto()); 
  }
  cout<<"===============select block one by one================="<<endl;
  for (int i=0; i<vec_swap.size(); i++){
    load_update(tempLoad,vec_swap[i].r_idx_ready,vec_swap[i].d_idx,-1,vec_swap[i].size,maxLen);
    vec_swap_selct.push_back(vec_swap[i]);
    auto tempOverLimit_ = load_over_limit(tempLoad,memLimit,0,maxLen,maxLen);
    cout<<vec_swap[i].r_idx_ready<<","<<vec_swap[i].d_idx<<" ((("<<tempOverLimit_.first<<","<<tempOverLimit_.second<<")))"<<endl;
    auto max_current = load_peak(tempLoad,maxLen);
    auto newMaxLoad = max_current.first;
    if (newMaxLoad < memLimit){
      break;
    }
  }
  cout<<"=============selection done================"<<endl;
  
  return vec_swap_selct;
}

vector<double> SwapGPU::swap_load_ideal(vector<double>vec_load,vector<SwapBlock> vec_swap_selct){
  auto vec_load_return = vec_load;
  for (int i =0; i<vec_swap_selct.size(); i++){
    int auto_buffer = 0;
    auto itm = vec_swap_selct[i];
    if (itm.cat == "A2") auto_buffer = data_buffer;
    if (itm.cat == "A3") auto_buffer = mutable_data_buffer;
    load_update(vec_load_return, itm.r_idx+auto_buffer, itm.d_idx, -1, itm.size, maxLen);
  }
  return vec_load_return;
}

void SwapGPU::swap_sched(vector<SwapBlock>&vec_swap_selct, vector<double>&vec_load_temp,double &overhead,double memLimit,string mode){
  /*
    update i1p, i2p and overhead time based on mode, such as no overhead or stick to limit.
  */ 
  //TODO(junzhe) wordy, can merge in common part.
  overhead = 0;
  cout<<"----------------swap_sched----------------"<<endl;
  if (mode == "stick-to-limit"){
    sort(vec_swap_selct.begin(),vec_swap_selct.end(),less_than_Idx_Swap()); 
    for (int i = 0; i<vec_swap_selct.size(); i++){
      auto itm = vec_swap_selct[i];
      int readyIdx = itm.r_idx_ready;
      cout<<itm.r_idx<<" ["<<itm.size<<"] ";
      cout<<readyIdx;
      if (i > 0){
        readyIdx = std::max(readyIdx,vec_swap_selct[i-1].i1p);
      }
      cout<<" -> "<<readyIdx;
      itm.i1 = readyIdx;
      // cout<<"check t1((("<<itm.t1;
      itm.t1 = vec_run[readyIdx+maxLen].t;
      // cout<<" "<<readyIdx<<" "<<maxLen<<" || "<<vec_run[readyIdx+maxLen].t<<" "<<itm.t1<<")))";
      itm.t1p = itm.t1 + SwapOutTime(itm.size);
      total_swapOutTime+=SwapOutTime(itm.size);
      while (itm.t1p > vec_run[readyIdx+maxLen].t){ //TODO(junzhe) reduce time complexity.
        readyIdx++; //ready means when able to finish swapOut, w/ or w/o overhead.
      }
      //get min compare with maxIdx and readyIdx.
      readyIdx = std::min(maxIdx,readyIdx);
      cout<<" || "<<readyIdx;
      //TODO(junzhe) 1st should be desired.
      load_update(vec_load_temp,readyIdx+1,itm.d_idx,-1,itm.size,maxLen);
      // load_update(vec_load_temp,itm.r_idx_ready,itm.d_idx,-1,itm.size,maxLen);
      auto tempOverLimit_ = load_over_limit(vec_load_temp,memLimit,0,maxLen,maxLen);
      cout<<" ((("<<tempOverLimit_.first<<','<<tempOverLimit_.second<<"))) ";
      if ((tempOverLimit_.first != -1) && (tempOverLimit_.first <= readyIdx)) { 
        load_update(vec_load_temp,tempOverLimit_.first-1,readyIdx+1,-1,itm.size,maxLen);
        // cout<<" ((("<<itm.r_idx<<' '<<itm.d_idx<<"||"<<itm.i1<<' '<<readyIdx<<' '<<tempOverLimit_.first<<")))";
        readyIdx = tempOverLimit_.first - 1; //TODO(junzhe) boundary
        overhead+=(itm.t1p-vec_run[readyIdx+maxLen].t);
        cout<<"==== overhead added "<<itm.t1p-vec_run[readyIdx+maxLen].t<<"... ";
        // cout<<"time spent "<<SwapOutTime(itm.size)<<endl;
        // cout<<"so time "<<itm.t1<<endl;
        // cout<<"eo 1 time "<<itm.t1p<<endl;
        // cout<<"eo 2 time "<<vec_run[readyIdx+maxLen].t<<endl;

      }
      cout<<" -> "<<readyIdx<<endl;   
      itm.i1p = readyIdx;
      vec_swap_selct[i] = itm;
      // auto tempOverLimit_1 = load_over_limit(vec_load_temp,memLimit,0,maxLen,maxLen);
      // cout<<"end: overlimit first and i1p "<<tempOverLimit_1.first<<' '<<itm.i1p<<endl;
      
    }
    cout<<"----------------sched part II-------------"<<endl;
    sort(vec_swap_selct.begin(),vec_swap_selct.end(),less_than_Idx_Swap_rvs());
    for (int i =0; i<vec_swap_selct.size(); i++){
      auto itm = vec_swap_selct[i];
      cout<<itm.r_idx<<" ["<<itm.size<<"] ";
      int needIdx = itm.d_idx;
      cout<<needIdx;
      if (i > 0){ needIdx = std::min(needIdx,vec_swap_selct[i-1].i2p); }
      cout<<" -> "<<needIdx;
      itm.i2 = needIdx;
      double prepareTime = vec_run[needIdx+maxLen].t - SwapInTime(itm.size);
      total_swapInTime+=SwapInTime(itm.size);
      while (prepareTime < vec_run[needIdx+maxLen].t){
        needIdx--;
      }
      needIdx = std::max(needIdx,maxIdx+1);
      cout<<" || "<<needIdx;
      itm.i2p = needIdx;
      itm.t2p = prepareTime;
      // auto tempOverLimit_2 = load_over_limit(vec_load_temp,memLimit,0,maxLen,maxLen);
      // cout<<"(((before come back (right over limit): "<<tempOverLimit_2.second<<endl;
      load_update(vec_load_temp,itm.i2p,itm.d_idx,1,itm.size,maxLen); //TODO(junzhe) range, right boundary
      auto tempOverLimit_3 = load_over_limit(vec_load_temp,memLimit,0,maxLen,maxLen);
      cout<<" ((("<<tempOverLimit_3.first<<","<<tempOverLimit_3.second<<")))";
      // cout<<"|||after come back (right over limit): "<<tempOverLimit_3.second<<endl;
      // if (tempOverLimit_3.second > 0){
      //   cout<<itm.r_idx<<' '<<itm.d_idx<<"||"<<itm.i1<<' '<<itm.i1p<<' '<<itm.i2p<<' '<<itm.i2<<")))"<<endl;
      // }

      if ((tempOverLimit_3.second != -1) && (vec_run[tempOverLimit_3.second+maxLen].t > itm.t2p)) {
        overhead+=(vec_run[tempOverLimit_3.second+maxLen].t - itm.t2p);
        cout<<"==== overhead added "<<vec_run[tempOverLimit_3.second+maxLen].t - itm.t2p<<"... ";
        load_update(vec_load_temp,itm.i2p,tempOverLimit_3.second+1,-1,itm.size,maxLen); //TODO(junzhe) range, right boundary
        itm.i2p = tempOverLimit_3.second+1;
        auto tempOverLimit_4 = load_over_limit(vec_load_temp,memLimit,0,maxLen,maxLen);
      }
      cout<<" -> "<<itm.i2p<<endl;
      //cout<<"after consider overlimit (right over limit): "<<tempOverLimit_4.second<<endl;
      vec_swap_selct[i] = itm;
    }
    cout<<":::::END OF SCHED, overhead is "<<overhead<<endl;
  }///end of first mode.


  ///this mode not really in use, for test purpose only.
  if (mode == "no-overhead"){
    //update i1p
    //sort by r_idx for i1p update
    sort(vec_swap_selct.begin(),vec_swap_selct.end(),less_than_Idx_Swap()); 
    for (int i = 0; i<vec_swap_selct.size(); i++){
      auto itm = vec_swap_selct[i];
      int readyIdx = 0;
      if (itm.cat == "A1") { readyIdx = itm.r_idx; }
      if (itm.cat == "A2") { readyIdx = itm.r_idx + data_buffer; }
      if (itm.cat == "A3") { readyIdx = itm.r_idx + mutable_data_buffer; }

      if (i > 0){
        readyIdx = std::max(readyIdx,vec_swap_selct[i-1].i1p);
      }
      itm.i1 = readyIdx;
      itm.t1 = vec_run[readyIdx].t;
      itm.t1p = itm.t1 + SwapOutTime(itm.size);
      while (itm.t1p > vec_run[readyIdx].t){
        readyIdx++;
      }
      itm.i1p = readyIdx;
      vec_swap_selct[i] = itm;
    }
    //update i2p
    sort(vec_swap_selct.begin(),vec_swap_selct.end(),less_than_Idx_Swap_rvs());
    for (int i =0; i<vec_swap_selct.size(); i++){
      auto itm = vec_swap_selct[i];
      int needIdx = itm.d_idx;
      if (i > 0){ needIdx = std::min(needIdx,vec_swap_selct[i-1].i2p); }
      itm.i2 = needIdx;
      double prepareTime = vec_run[needIdx].t - SwapInTime(itm.size);
      while (prepareTime < vec_run[needIdx].t){
        needIdx--;
      }
      itm.i2p = needIdx;
      itm.t2p = prepareTime;
      vec_swap_selct[i] = itm;
      load_update(vec_load_temp,itm.i1p,itm.i2p+1,-1,itm.size,maxLen); //TODO(junzhe) range, right boundary
    }

  }
  
}


void SwapGPU::swap_construct_tables(vector<SwapBlock>vec_swap_selct){
  cudaStream_t stream1;
  cudaStream_t stream2;
  cout<<"---------------print all 1, 1', 2', 2-----------"<<endl;
  sort(vec_swap_selct.begin(),vec_swap_selct.end(),less_than_Idx_Swap()); 
  //for each swap select, make Table_sched and Table_meta
  // for (int i = static_cast<int>(vec_swap_selct.size()-1);i>=0; i--){
  for (int i =0; i<vec_swap_selct.size(); i++){
    auto itm = vec_swap_selct[i];
    // if (itm.r_idx >= 0){
    //TODO(junzhe) for time being, remove negative r_idx itms.
      cout<<itm.r_idx<<" || "<<itm.i1<<" "<<itm.i1p<<" "<<itm.i2p<<" "<<itm.i2<<endl;
      //i1 swap
      if (Table_sched.find(itm.i1) == Table_sched.end()){
        Table_sched[itm.i1] = std::make_tuple(itm.r_idx,0,-1,-1);
      } else {
        std::get<0>(Table_sched.find(itm.i1)->second) = itm.r_idx;
        std::get<1>(Table_sched.find(itm.i1)->second) = 0;
      }
      //i2p swap
      if (Table_sched.find(itm.i2p) == Table_sched.end()){
        Table_sched[itm.i2p] = std::make_tuple(itm.r_idx,1,-1,-1);      
      } else {
        std::get<0>(Table_sched.find(itm.i2p)->second) = itm.r_idx;
        std::get<1>(Table_sched.find(itm.i2p)->second) = 1;
      }
      // i1p sync
      if (Table_sched.find(itm.i1p) == Table_sched.end()){
        Table_sched[itm.i1p] = std::make_tuple(-1,-1,itm.r_idx,0);
      } else {
        std::get<2>(Table_sched.find(itm.i1p)->second) = itm.r_idx;
        std::get<3>(Table_sched.find(itm.i1p)->second) = 0; 
      }
      //i2 sync
      if (Table_sched.find(itm.i2) == Table_sched.end()){
        Table_sched[itm.i2] = std::make_tuple(-1,-1,itm.r_idx,1);
      } else {
        std::get<2>(Table_sched.find(itm.i2)->second) = itm.r_idx;
        std::get<3>(Table_sched.find(itm.i1p)->second) = 1;
      }

      ///Make Table_meta
      void* tempPtr = nullptr;
      cudaMallocHost(&tempPtr,itm.size); //pinned memory.
      BlockMeta meta;
      meta.size = itm.size;
      meta.cpu_ptr = tempPtr;
      meta.out_stream = stream1;
      meta.in_stream = stream2;
      //meta.last_out_idx = vec_swap_selct[i].last_out_idx;
      //meta.last_in_idx = vec_swap_selct[i].last_in_idx;
      //meta.i2 = vec_swap_selct[i].i2;
      Table_meta[itm.r_idx] = meta;
    // }

  }
  cout<<"---------------print all 1, 1', 2', 2-----------"<<endl;
  cout<<"size of Table_meta: "<<Table_meta.size()<<endl;
  cout<<"size of Table_sched =================="<<Table_sched.size()<<endl;
  cout<<"print Table_sched, idx, r_idx, sync, direction"<<endl;
  for (int i = -500; i<maxLen; i++){
    if (!(Table_sched.find(i) == Table_sched.end())){
      cout<<i<<"-->";
      cout<<std::get<0>(Table_sched.find(i)->second)<<" ";
      cout<<std::get<1>(Table_sched.find(i)->second)<<" ";
      cout<<std::get<2>(Table_sched.find(i)->second)<<" ";
      cout<<std::get<3>(Table_sched.find(i)->second)<<endl;
    }
  }

}

void SwapGPU::swap_update_tables(Block* tempBlock_){
  // update Table_meta's block_ and data_; update once atfer swap test is passed.
  //TODO(junzhe) should not be able to update negative r_idx, as of now.
  if (testFlag == 1) {
    //cout<<gc<<' '<<(gc-location)%maxLen<<' '<<blockInfo<<endl;
    int r_gc = (gc-location)%maxLen;
    if (!(Table_meta.find(r_gc)==Table_meta.end())){
      //cout<<"r_gc, gc and size ot Table_meta "<<r_gc<<' '<<gc<<" "<<Table_meta.size()<<endl;
      //TODO(junzhe) verify the length change, if go in, value update
      // cout<<"To update Block_ at "<<r_gc<<' '<<gc<<' '<<tempBlock_<<' '<<tempBlock_->get_data()<<endl;
      Table_meta.find(r_gc)->second.block_ = tempBlock_;
      Table_meta.find(r_gc)->second.data_ = tempBlock_->get_data();
    }
  }

}

int SwapGPU::swap_test(vector<string>vec_block,int &maxLen, int &location){

  ///vec_str (vec_block) to vec_pieceMsg, sort by ptr and idx.
  int idxRange = 0;
  vector<onePieceMsg> vec_pieceMsg = swap_strVec_2_pieceMsgVec(vec_block,idxRange);
  cout<<"size of vec_pieceMsg & vec_block: "<<vec_pieceMsg.size()<<' '<<vec_block.size()<<endl;
  ///rep test
  vector<size_t> vec_rep = Swap_piece2rep(vec_pieceMsg);
  //int idxRange3=0; //rename TODO(junzhe)
  //int maxLen=0, location =0;
  repPatternDetector(vec_rep,maxLen,location,maxLen_threshold,gc);
  cout<<"maxLen and location are: "<<maxLen<<' '<<location<<endl;
  cout<<"test rep"<<endl;
  //Note here location not exactly start of one iteration, 
  //adjust to nearly start of one by restricting "Malloc"
  int shift_counter = 0;
  for (int i=0;i<maxLen;i++){
    vector<string> v = swap_split(vec_block[location+i], " ");
    if (v[0]=="Malloc"){
      shift_counter = i; 
      break;
    }
  }
  location =location+shift_counter;
  cout<<"shift_counter is "<<shift_counter<<endl;
  cout<<"location changed to "<<location<<endl;

  if (maxLen<maxLen_threshold) {return -1;}

  return gc+maxLen-(gc-location)%maxLen;
} 

void SwapGPU::swap_plan(){
  cout<<":::::::::::::::::::::::::start swap_plan()"<<endl;

  int idxRange = 0;
  vector<onePieceMsg> vec_pieceMsg = swap_strVec_2_pieceMsgVec(vec_block,idxRange);
  cout<<"size of vec_pieceMsg & vec_block: "<<vec_pieceMsg.size()<<' '<<vec_block.size()<<endl;
  sort(vec_pieceMsg.begin(),vec_pieceMsg.end(),less_than_Idx());
  // scale down idx, to middle iteration.
  tempTime_baseline = vec_pieceMsg[three_more_location].t;
  for (int i=0; i<vec_pieceMsg.size();i++){
    vec_pieceMsg[i].idx = vec_pieceMsg[i].idx - three_more_location - maxLen;
    vec_pieceMsg[i].t = vec_pieceMsg[i].t - tempTime_baseline;
  }

  // build opsSqn, and sizeSqn
  // cout<<"------printing sequenc--------"<<endl;
  vector<onePieceMsg>one_itr(&vec_pieceMsg[location+4*maxLen],&vec_pieceMsg[location+5*maxLen]);
  for (int i =0; i<one_itr.size();i++){
    opsSequence.push_back(one_itr[i].MallocFree);
    sizeSequence.push_back(one_itr[i].size);
    // cout<<one_itr[i].MallocFree<<' '<<one_itr[i].size<<endl;
  }
  //3 iterations of vec_run and vec_load, maxIdx and maxLoad
  vector<onePieceMsg>temp_vec_run(&vec_pieceMsg[location+3*maxLen],&vec_pieceMsg[location+6*maxLen]);
  vec_run = temp_vec_run;
  fstream file_vec_run("vec_run36.csv", ios::in|ios::out|ios::app);
  for (int i =0; i<vec_run.size();i++){
    //file_vec_run<<vec_run[i].idx<<' '<<vec_run[i].MallocFree<<' '<<vec_run[i].t<<' '<<vec_run[i].t-tempTime2<<endl;
    // file_vec_run<<i<<' '<<vec_run[i].t<<' '<<vec_run[i].t-tempTime2<<endl;
    // tempTime2 = vec_run[i].t;
    file_vec_run<<i-maxLen<<' '<<vec_run[i].MallocFree<<' '<<vec_run[i].size<<' '<<vec_run[i].t<<endl;
  }

  vector<onePieceMsg>temp_vec_run2(&vec_pieceMsg[location],&vec_pieceMsg[location+3*maxLen]);
  auto vec_run2 = temp_vec_run2;
  fstream file_vec_run2("vec_run03.csv", ios::in|ios::out|ios::app);
  for (int i =0; i<vec_run2.size();i++){
    //file_vec_run<<vec_run[i].idx<<' '<<vec_run[i].MallocFree<<' '<<vec_run[i].t<<' '<<vec_run[i].t-tempTime2<<endl;
    // file_vec_run2<<i<<' '<<vec_run2[i].t<<' '<<vec_run2[i].t-tempTime2<<endl;
    // tempTime2 = vec_run[i].t;
    file_vec_run2<<i<<' '<<vec_run2[i].MallocFree<<' '<<vec_run2[i].size<<endl;
  }

  vector<double>vec_load(&global_load[location],&global_load[location+3*maxLen]);
  origin_load = vec_load;
  //3 iterations
  fstream file_load_origin("load_origin03.csv", ios::in|ios::out|ios::app);
  for (int i=0; i<origin_load.size(); i++){
    file_load_origin<<i<<' '<<origin_load[i]<<endl;
  }
  
  // vector<double>vec_load2(&global_load[location+3*maxLen],&global_load[location+6*maxLen]);
  // auto origin_load2 = vec_load2;
  // //3 iterations
  // fstream file_load_origin2("load_origin36.csv", ios::in|ios::out|ios::app);
  // for (int i=0; i<origin_load2.size(); i++){
  //   file_load_origin2<<i<<" "<<origin_load2[i]<<endl;
  // }

  // //one iteration.
  // fstream file_load_current("load_current.csv", ios::in|ios::out|ios::app);
  // for (int i=0; i<maxLen; i++){
  //   file_load_current<<vec_load[i]<<endl;
  // }
  cout<<"build from scratch:::::::::::::::;"<<endl;
  auto max_current = load_peak(vec_load,maxLen);
  maxLoad = max_current.first;
  maxIdx = max_current.second;
  cout<<"------------------print max_load: (current) "<<maxLoad<<" "<<maxIdx<<endl;


  //sort by ptr & idx, sorting the duplicate
  auto vec_run_dup = vec_run;
  sort(vec_run_dup.begin(),vec_run_dup.end(),less_than_ptrIdx());
  ///formulate swappable items.
  cout<<"==============================print swappable items, with maxIdx "<<maxIdx<<endl;
  vector<SwapBlock>vec_swap;
  // size_t load_swap = 0;
  for (int i =1; i<vec_run_dup.size(); i++){
    //SwapBlock(string p, size_t s, int i1, int i2, double t1, double t2): 
    //ptr(p), size(s), r_idx(i1),d_idx(i2),r_time(t1), d_time(t2) {}
    if ((vec_run_dup[i].size >= smallest_block) && (vec_run_dup[i-1].idx<maxIdx) && (vec_run_dup[i].idx>maxIdx) 
      && (vec_run_dup[i-1].ptr ==vec_run_dup[i].ptr) 
      && ((vec_run_dup[i-1].MallocFree==3) or (vec_run_dup[i-1].MallocFree==2) or (vec_run_dup[i-1].MallocFree==4)))
    {
      SwapBlock itm(vec_run_dup[i].ptr, vec_run_dup[i].size, vec_run_dup[i-1].idx, vec_run_dup[i].idx, vec_run_dup[i-1].t, vec_run_dup[i].t);
      itm.dto = itm.d_time-itm.r_time;
      itm.dt = itm.d_time-itm.r_time-SwapOutTime(itm.size)-SwapOutTime(itm.size);
      if (itm.dt>=0){
        itm.pri = itm.dt * itm.size;
      } else {
        itm.pri = itm.dt * 1/itm.size;
      }
      //cat A
      if (vec_run_dup[i-1].MallocFree == 3){ itm.cat = "A1"; itm.r_idx_ready = itm.r_idx; } 
      if (vec_run_dup[i-1].MallocFree == 2){ itm.cat = "A2"; itm.r_idx_ready = itm.r_idx + data_buffer;} 
      if (vec_run_dup[i-1].MallocFree == 4){ itm.cat = "A3"; itm.r_idx_ready = itm.r_idx + mutable_data_buffer;} 

      vec_swap.push_back(itm);
      // load_swap+=itm.size;
      cout<<itm.size<<" ";
      cout<<"Items Swappable: (r_idx, d_idx, cat, MB, dt/us, PS) || "<<itm.r_idx<<' '<<itm.d_idx;
      cout<<" ||  "<<itm.cat<<"  || "<<(float)(itm.size)/(float)(1024*1024);
      cout<<' || '<<itm.dt/1000<<' '<<itm.pri<<endl;
    } 
  }
  cout<<"size vec_swap: "<<vec_swap.size()<<endl;

  ///load ideal, swap all vec_swap, lest possible memory by one-swap
  auto vec_load_ideal = swap_load_ideal(vec_load,vec_swap);
  fstream file_load_ideal("load_ideal.csv", ios::in|ios::out|ios::app);
  for (int i=maxLen; i<maxLen*2; i++){
    file_load_ideal<<vec_load_ideal[i]<<endl;
  }

  auto max_ideal = load_peak(vec_load_ideal,maxLen);
  size_t maxLoad_ideal = max_ideal.first;
  int maxIdx_ideal = max_ideal.second;
  cout<<"------------------print max_load: (ideal) "<<maxLoad_ideal<<" "<<maxIdx_ideal<<endl;
  //maxLoad_ideal = 400;
  /// select till maxLoad_ideal, dto
  // auto vec_swap_dto = swap_select(vec_swap,maxLoad,maxLoad_ideal,"dto");
  // cout<<"size of vec_swap_dto: "<<vec_swap_dto.size()<<endl;
  // auto vec_load_dto_ideal = swap_load_ideal(vec_load,vec_swap_dto);
  //   fstream file_load_dto_ideal("load_dto_ideal.csv", ios::in|ios::out|ios::app);
  // for (int i=maxLen; i<maxLen*2; i++){
  //   file_load_dto_ideal<<vec_load_dto_ideal[i]<<endl;
  // }
  // auto tempMax_ = load_peak(vec_load_dto_ideal,maxLen);
  // cout<<"------------------print max_load: (dto ideal) "<<tempMax_.first<<" "<<tempMax_.second<<endl;

  // /// select till maxLoad_ideal, pri
  // auto vec_swap_pri = swap_select(vec_swap,maxLoad,maxLoad_ideal,"pri");
  // cout<<"size of vec_swap_pri: "<<vec_swap_dto.size()<<endl;
  // auto vec_load_pri_ideal = swap_load_ideal(vec_load,vec_swap_pri);
  //   fstream file_load_pri_ideal("load_pri_ideal.csv", ios::in|ios::out|ios::app);
  // for (int i=maxLen; i<maxLen*2; i++){
  //   file_load_pri_ideal<<vec_load_pri_ideal[i]<<endl;
  // }
  // tempMax_ = load_peak(vec_load_pri_ideal,maxLen);
  // cout<<"------------------print max_load: (pri ideal) "<<tempMax_.first<<" "<<tempMax_.second<<endl;

  /// select till maxLoad_ideal, wdto
  auto tempLoad = origin_load;
  auto memLimit_wdto = 550<<20;
  //TODO(junzhe) memLimit = maxLoad_ideal*1.4
  auto vec_swap_wdto = swap_select(vec_swap,tempLoad,memLimit_wdto,"wdto");
  // vec_swap_selct_global = vec_swap_wdto;
  cout<<"size of vec_swap_wdto: "<<vec_swap_wdto.size()<<endl;
  // auto vec_load_wdto_ideal = swap_load_ideal(vec_load,vec_swap_wdto);
    // fstream file_load_wdto_ideal("load_wdto_ideal.csv", ios::in|ios::out|ios::app);
  // for (int i=maxLen; i<maxLen*2; i++){
    // file_load_wdto_ideal<<vec_load_wdto_ideal[i]<<endl;
  // }
  // auto tempMax_ = load_peak(vec_load_wdto_ideal,maxLen);
  // cout<<"------------------print max_load: (wdto ideal) "<<tempMax_.first<<" "<<tempMax_.second<<endl;

  /// load_1 no overhead, based on pri
  //auto vec_swap_pri = vec_swap_pri;
  // auto vec_load_pri = origin_load;
  // auto vec_load_dto = origin_load;
  auto vec_load_wdto = origin_load;
  string mode = "stick-to-limit";
  // double overhead_pri = 0;
  // swap_sched(vec_swap_pri, vec_load_pri,overhead_pri,450<<20, mode);
  
  // double overhead_dto = 0;
  // swap_sched(vec_swap_dto, vec_load_dto,overhead_dto,450<<20,mode);
  
  double overhead_wdto = 0;
  swap_sched(vec_swap_wdto, vec_load_wdto,overhead_wdto,memLimit_wdto,mode);

  swap_construct_tables(vec_swap_wdto);


  // fstream file_block10("load_1_pri.csv", ios::in|ios::out|ios::app);
  // for (int i=maxLen; i<maxLen*2; i++){
  //   file_block10<<vec_load_pri[i]<<endl;
  // }
  // fstream file_block11("load_1_dto.csv", ios::in|ios::out|ios::app);
  // for (int i=maxLen; i<maxLen*2; i++){
  //   file_block11<<vec_load_dto[i]<<endl;
  // }
  // fstream file_block12("load_1_wdto.csv", ios::in|ios::out|ios::app);
  // for (int i=maxLen; i<maxLen*2; i++){
  //   file_block12<<vec_load_wdto[i]<<endl;
  // }
  //TODO(junzhe) below verification to be done later.
  // auto max_1 = load_peak(vec_load_pri,maxLen);
  // size_t maxLoad_1 = max_1.first;
  // int maxIdx_1 = max_1.second;
  // cout<<"------------------print max_load: (1) "<<maxLoad_1<<" "<<maxIdx_1<<endl;
  // //change back order by Idx.
  // sort(vec_run.begin(),vec_run.end(),less_than_Idx());
  // cout<<"done with swap_plan..."<<endl;
  // cout<<"load 2 overhead pri: "<<(float)(overhead_pri)/(float)(1000000)<<endl;
  // cout<<"load 2 overhead dto: "<<(float)(overhead_dto)/(float)(1000000)<<endl;
  // cout<<"load 2 overhead wdto: "<<(float)(overhead_wdto)/(float)(1000000)<<endl;
  // cout<<"total_swapOutTime: "<<(float)(total_swapOutTime)/(float)(3000000)<<endl;
  // cout<<"total_swapInTime: "<<(float)(total_swapInTime)/(float)(3000000)<<endl;
  // auto t1 = vec_run[0].t;
  // auto t2 = vec_run[maxLen].t;
  // auto t3 = vec_run[maxLen*2].t;
  // auto t4 = vec_run[maxLen*3-1].t;
  // cout<<"iteration time spent: "<<(float)(t2-t1)/(float)(1000000)<<" "<<(float)(t3-t2)/(float)(1000000)<<" "<<(float)(t4-t3)/(float)(1000000)<<endl;
  // fstream file_time("itr_time.csv", ios::in|ios::out|ios::app);
  // file_time<<"iteration time spent: "<<(float)(t2-t1)/(float)(1000000)<<" "<<(float)(t3-t2)/(float)(1000000)<<" "<<(float)(t4-t3)/(float)(1000000)<<endl;
  // file_time<<"iteration time spent: "<<t2-t1<<" "<<t3-t2<<" "<<t4-t3<<endl;
  // file_time<<"idx "<<0<<" "<<maxLen<<" "<<maxLen*2<<" "<<maxLen*3-1<<endl;
  // file_time<<"value "<<t1<<" "<<t2<<" "<<t3<<" "<<t4<<endl;
}






SwapGPU::~SwapGPU() {
  //print out push-info TODO(junzhe) can remove
  fstream file_block_full("vec_block_full.csv", ios::in|ios::out|ios::app);
  for (int i =0; i<vec_block.size();i++){
    file_block_full<<vec_block[i]<<endl;
  }

  fstream file_time("itr_time.csv", ios::in|ios::out|ios::app);
  file_time<<"=============================="<<endl;
  //main body
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
  pool_ = std::make_shared<Swap>(conf);
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
    // TODO(wangwei) remove the memset.
    CUDA_CHECK(cudaMemset(ptr, 0, size));
  }
  //cout<<"malloc done"<<endl;
  return ptr;
}

/// Free gpu memory.
void SwapGPU::Free(void* ptr) {

  if (ptr != nullptr) {
    CUDA_CHECK(cudaSetDevice(id_));
    pool_->Free(ptr);
  }

  //cout<<"free done"<<endl; 
}

void SwapGPU::Test_sched_switch_swap(){
  /*
    v1: do Test_sched_switch_swap during (before) Malloc and Free.
    swap removed to DeploySwap
    v2: test after every index, at Append. order and index changed.
  */
  ///test iteration
  if (((gc+1)%(maxLen_threshold) == 0) && (asyncSwapFlag == 0) && (testFlag == 0)){
    //TODO(junzhe) not lean, chances are globeCounter found more than 300 idx ago: redudant test.
    cout<<"gc, GC and vec_len before test: "<<gc<<' '<<globeCounter<<' '<<vec_block.size()<<endl;
    globeCounter = swap_test(vec_block,maxLen,location);
    maxLen_threshold = std::max(maxLen_threshold,gc/10);
    maxLen_threshold = std::min(2000,maxLen_threshold);
    if (maxLen > maxLen_threshold) {
      testFlag = 1;
      three_more_globeCounter = globeCounter + 3*maxLen;
      three_more_location = location + 3*maxLen;
      cout<<"compele test-swap:::::::::::::::::::::::::::::::::::::::::::::::::"<<endl;
      cout<<"impt numbers (maxLen, location, GC) "<<maxLen<<' '<<location<<' '<<globeCounter<<endl;
      
   }
 }
 ///switch flag; next idx
 if ((gc+1) == three_more_globeCounter){
    swap_plan();
    asyncSwapFlag = 1;
    // vector<double>vec_load2(&global_load[three_more_location],&global_load[three_more_location+3*maxLen]);
    // origin_load = vec_load2;
    // //load before swap, write in
    // fstream file_load_origin("load_origin.csv", ios::in|ios::out|ios::app);
    // for (int i=0; i<origin_load.size(); i++){
    //   file_load_origin<<origin_load[i]<<endl;
    // }
    cout<<"switched flag for at "<<three_more_globeCounter<<endl;
    cout<<"maxLen is "<<maxLen<<endl;
    cout<<"globeCounter "<<globeCounter<<endl;
 }
}

void SwapGPU::MakeMetaTable(Block* block_,void* data_,int size){
  /*
  Append info right after Malloc; make block_ - data_ pair wise table.
  */

  //append info
  stringstream strm1;
  strm1<<size;
  string tempStr1 = strm1.str();
  stringstream strm3;
  strm3<<block_;
  string tempStr3 = strm3.str();
  stringstream strm4;
  auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
  strm4<<t2;
  string tempStr4 = strm4.str();
  string blockInfo ="Malloc "+tempStr3+" "+tempStr1+" "+tempStr4;
  Append(blockInfo);

  
}

void SwapGPU::DeploySwap(){
   ///swap and sync as per schedule.
  int r_gc = (gc-location)%maxLen; 

  if ((asyncSwapFlag == 1) && (!(Table_sched.find(r_gc) == Table_sched.end()))){
    cout<<"--------sched action at "<<r_gc<<endl;
    auto swap_idx = std::get<0>(Table_sched.find(r_gc)->second);
    auto swap_dir = std::get<1>(Table_sched.find(r_gc)->second);
    auto sync_idx = std::get<2>(Table_sched.find(r_gc)->second);
    auto sync_dir = std::get<3>(Table_sched.find(r_gc)->second);
    if (swap_dir == 0){ 
      SwapOut_idx(swap_idx); 
      cout<<"----Swap Out "<<swap_idx<<endl;
    }
    if (swap_dir == 1){ 
      SwapIn_idx(swap_idx); 
      cout<<"----Swap In "<<swap_idx<<endl;
    }
    //TODO(junzhe) verify sync what else to be done
    if (sync_dir == 0){
      ///sync swap-out, including sync, update block's data_ to nullptr, free data_, update meta.
      auto last_meta = Table_meta.find(sync_idx)->second;
      auto t1 = (std::chrono::system_clock::now()).time_since_epoch().count();
      cudaEventSynchronize(last_meta.in_event);
      auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();

      // Table_not_at_device[last_meta.block_] = sync_idx; //TODO(junzhe) double check if needed.

      last_meta.block_->update_data(nullptr);
      // cout<<"to free data_"<<last_meta.data_<<endl;
      pool_->Free(last_meta.data_);
      last_meta.data_ = nullptr; //not really needed TODO(junzhe)
      cout<<"----sync out "<<sync_idx<<endl;
      Table_meta.find(sync_idx)->second = last_meta;
    }
    if (sync_dir == 1){
      ///sync swap-in, including sync, update block's data_ to new gpu address, update meta.
      //if (!(Table_not_at_device.find(last_meta.block_)==Table_not_at_device.end())){ TODO(junzhe)
      auto last_meta = Table_meta.find(sync_idx)->second;
      auto t1 = (std::chrono::system_clock::now()).time_since_epoch().count();
      cudaEventSynchronize(last_meta.out_event);
      auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
      // Table_not_at_device.erase(last_meta.block_);
      last_meta.block_->update_data(last_meta.data_);
      cout<<"----sync in "<<sync_idx<<endl;
      Table_meta.find(sync_idx)->second = last_meta;
    }
  } 
}

void SwapGPU::Append(string blockInfo){

  vector<string> v = swap_split(blockInfo, " ");
  void* tempPtr;
  stringstream convert(v[1]);
  convert>>tempPtr;
  auto tempBlock_ = static_cast<Block*>(tempPtr);
  
  // insert size, malloc : flag, block_, size, t; others: insert size t.
  if (v.size() != 4) {
    stringstream strm1;
    strm1<<tempBlock_->size();
    string tempStr1 = strm1.str();
    blockInfo = v[0] + ' ' + v[1] + ' ' + tempStr1 + ' ' + v[2];
  }

  // update global load
  if (maxLen < maxLen_threshold){
    if (v[0] == "Malloc"){
      if (global_load.size()>0){
        global_load.push_back(global_load[global_load.size()-1]+tempBlock_->size());
      } else {
        global_load.push_back(tempBlock_->size());
      }
    } else if (v[0] == "Free"){
      global_load.push_back(global_load[global_load.size()-1]-tempBlock_->size());
    } else {
      global_load.push_back(global_load[global_load.size()-1]);
    }
  }

  //append into vec_block
  vec_block.push_back(blockInfo);


  //cout<<blockInfo<<endl;
  //cout<<tempBlock_->size()<<endl;
  //cout<<"load: "<<global_load[global_load.size()-1]<<" len of blockInfo and global_load "<<vec_block.size()<<' '<<global_load.size()<<endl;
  //std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  
  // if (asyncSwapFlag == 1){
  //   vec_block_fresh.push_back(blockInfo);
  // }
  // if ((maxLen>maxLen_threshold)&&((gc-globeCounter+1)==3*maxLen)){
  //   fstream file_block_fresh("vec_block_fresh.csv", ios::in|ios::out|ios::app);
  //   for (int i =0; i<vec_block_fresh.size();i++){
  //     file_block_fresh<<vec_block_fresh[i]<<endl;
  //   }
  // }
  // fstream file_block5("append.csv", ios::in|ios::out|ios::app);
  // file_block5<<gc<<' '<<blockInfo<<' '<<(gc-1247)%612<<endl;

  //print time duration per iteration
  if ((maxLen>maxLen_threshold) && ((gc-location)%(maxLen) == 0)){
    if (tempTime != 0){
      fstream file_time("itr_time.csv", ios::in|ios::out|ios::app);
      auto t_now = (std::chrono::system_clock::now()).time_since_epoch().count();
      file_time<<(float)(t_now - tempTime)/(float)(1000000)<<endl;
      
    }
    tempTime = (std::chrono::system_clock::now()).time_since_epoch().count();
  }

  //check if last iteration, TODO(junzhe) further verify with MallocFree.
  if (asyncSwapFlag == 1){
    int r_gc = (gc-location)%maxLen;
    if (tempBlock_->size() != sizeSequence[r_gc]){
      asyncSwapFlag = 0;
      cout<<"!!!! asyncSwapFlag changed back to 0"<<endl;
    }
  }

  //update Table_meta
  swap_update_tables(tempBlock_);

  //deploy swap at every index.
  DeploySwap();

  //test moved from start of malloc/free to end of append, only gc+1 changed
  Test_sched_switch_swap();
  //NOTE: this gc++ includes read/write and AppendLayer as well, in addition to malloc/free.
  gc++;

}

void* SwapGPU::GetRealGpuPtr(const Block* block_){
  // //here should be not update_data()
  // auto reading_meta = Table_meta.find(Table_not_at_device.find(block_)->second)->second;
  // auto t1 = (std::chrono::system_clock::now()).time_since_epoch().count();
  // cudaEventSynchronize(reading_meta.in_event);
  // auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
  // //cout<<"GetRealGpuPtr, overhead is: "<<t2-t1<<endl;
  // //cout<<"To update_data swap for (In) "<<Table_not_at_device.find(block_)->second<<" "<<reading_meta.data_<<" 0"<<endl;
  // //reading_meta.block_->update_data(reading_meta.data_);

  // //cout<<"last_meta r_idx::::::malloc due to swapIn ( "<<Table_not_at_device.find(block_)->second<<endl;

  // Table_not_at_device.erase(reading_meta.block_);

  return nullptr; //TODO(junzhe) attention, based on no change here.
}

void SwapGPU::SwapOut_idx(const int r_idx){
  //cout<<"doing asynchrous swapOut of r_idx: "<<r_idx<<' '<<endl;
  auto t1 = (std::chrono::system_clock::now()).time_since_epoch().count();  
  cudaError_t err;
  BlockMeta meta = Table_meta.find(r_idx)->second;
  cudaEventCreate (&meta.out_event);
  //cout<<"right before cudaMemcpyAsync Out"<<endl;
  err = cudaMemcpyAsync(meta.cpu_ptr,meta.data_,meta.size,cudaMemcpyDeviceToHost,meta.out_stream);
  cudaEventRecord(meta.out_event,meta.out_stream);
  //cout<<"right after cudaMemcpyAsync"<<endl;
  auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
  // cout<<"To update_data swap for (Out) "<<r_idx<<" "<<meta.block_<<" 0"<<endl;
  //update meta's out_event
  Table_meta.find(r_idx)->second = meta;
  //cout<<"time for asynchrous: "<<t2-t1<<endl;
  //cudaEventSynchronize(event1);
  //auto t4 = (std::chrono::system_clock::now()).time_since_epoch().count();
  //cout<<"time for asynchrous to complete: "<<t4-t1<<endl;
}

void SwapGPU::SwapIn_idx(const int r_idx){
  //logic: extra meta, swap, update meta in Table
  //TODO(junzhe) to clean up free(), make it in somewhere else.
  auto t1 = (std::chrono::system_clock::now()).time_since_epoch().count();
  cudaError_t err;
  BlockMeta meta = Table_meta.find(r_idx)->second;
  cudaEventCreate (&meta.in_event);
  //cout<<"update block and data of r_idx: "<<r_idx<<' '<<meta.block_<<' '<<meta.data_<<endl;
  void* ptr = nullptr;
  pool_->Malloc((void**)&ptr, meta.size);
  //cout<<"expected results update_data:: "<<meta.block_<<" "<<ptr<<endl;
  //cout<<"malloc due to swapIn ("<<r_idx<<") "<<ptr<<endl;
  //void* to_rm_ptr = meta.data_;
  meta.data_ = ptr;
  // cout<<"right before cudaMemcpyAsync In"<<endl;
  err = cudaMemcpyAsync(meta.data_,meta.cpu_ptr,meta.size,cudaMemcpyHostToDevice,meta.in_stream);
  cudaEventRecord(meta.in_event,meta.in_stream);
  // cout<<"right after cudaMemcpyAsync"<<endl;
  // cout<<"To update_data swap for (In) "<<r_idx<<" "<<meta.block_<<" "<<meta.data_<<' '<<ptr<<endl;
  //upadte meta's new gpu addr, in_event
  Table_meta.find(r_idx)->second = meta;
  //meta.block_->update_data(meta.data_); //TODO(junzhe) debug only, not the right place to update.
  //auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
  //cout<<"time for asynchrous: "<<t2-t1<<endl;
}

void SwapGPU::SwapOut(const Block* block_){
  if (gc < 1000 && block_->size() > 1<<20) {
    fstream file_block5("speed.csv", ios::in|ios::out|ios::app);
    BlockMeta meta;
    meta.data_ = meta.block_->get_data();
    void* tempPtr = nullptr;
    cudaMallocHost(&tempPtr,block_->size()); //pinned memory.
    meta.cpu_ptr = tempPtr;
    Table_block_meta[block_] = meta;
    auto t1 = (std::chrono::system_clock::now()).time_since_epoch().count();
    cudaError_t err;
    err = cudaMemcpy(meta.cpu_ptr, meta.data_,block_->size(),cudaMemcpyDeviceToHost);
    auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
    file_block5<<"Out "<<block_->size()<<' '<<t2-t1<<endl;
    cout<<"swap out done at gc: "<<gc<<endl;
  }
}

void SwapGPU::SwapIn(const Block* block_){
  if (gc < 1000 && block_->size() > 1<<20) {
    fstream file_block5("speed.csv", ios::in|ios::out|ios::app);
    BlockMeta meta = Table_block_meta.find(block_)->second;
    auto t1 = (std::chrono::system_clock::now()).time_since_epoch().count();
    cudaError_t err;
    err = cudaMemcpy(meta.data_, meta.cpu_ptr,block_->size(),cudaMemcpyHostToDevice);
    auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
    file_block5<<"In "<<block_->size()<<' '<<t2-t1<<endl;
    cout<<"swap in done at gc: "<<gc<<endl;
  }
}


}  // namespace singa
#endif  // USE_CUDA