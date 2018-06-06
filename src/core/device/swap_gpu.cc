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
struct onePieceMsg{
    /*
     members: [ptr, size, MallocFree, idx]
     */
    string ptr;
    size_t size;
    int MallocFree;
    int idx;
    double t;
    onePieceMsg(string p, size_t s, int M, int i):ptr(p),size(s),MallocFree(M),idx(i){}
};


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
void repPatternDetector(vector<size_t>rep, int &maxLen, int &location){
    int idxRange = (int)rep.size();
    int threshold =100;
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
    int ans =0; //TODO(junzhe) used to be 0.29; new param as per vgg
    if (size==0) {ans = 47200;} else {ans = 0.0756 * size + 47200;}
    return ans;
}

int SwapInTime(size_t size){
    //yet to get the formula
    int ans =0; //TODO(junzhe) used to be 0.13; new param as per vgg
    if (size==0) {ans = 9700;} else {ans = 0.0823 * size + 9700;}
    return ans;
}


struct SwapBlock{
    // more attributes, name different, use ptr.
    /*
     members: [name (r_idx), size, r_idx, d_idx]
     */
    string ptr;
    string cat;  //A1, A2, A3...
    int name;
    size_t size;
    int r_idx; //out idx
    int d_idx; //in idx
    double r_time; // out time
    double d_time; //in time
    double dto; //t2-t1
    double dt; //delta t: t2'-t1'
    double pri;  //look at here if big enough TODO(junzhe)
    //below as per planned.
    int i1;
    int i1p;
    int i2;
    int i2p;
    double t1;
    double t2;
    double t1p;
    double t2p;

    //int last_out_idx = 0; //last during swapOut
    //int last_in_idx = 0; //next during swapIn
    //onePairMsg(int n,size_t s, int r,int d):name(n),size(s),r_idx(r),d_idx(d){}
    //from LayerAppend (3) - r_idx, to next read/write (2) - d_idx
    SwapBlock(string p, size_t s, int i1, int i2, double t1, double t2): ptr(p), size(s), r_idx(i1),d_idx(i2),r_time(t1), d_time(t2) {}
};

struct less_than_dto{
    /*
     sort SwapBlock by dt, descending
     */
    inline bool operator() (const SwapBlock& struct1, const SwapBlock& struct2)
    {
        return (struct1.dto>struct2.dto);
    }
};

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


pair<int,int> load_over_limit(vector<double>vec_load, size_t memLimit, int start_idx, int end_idx){
  //input: vec_load, memLimit, range [start_idx, end_idx)
  //return range overlimit [first_over_limit, first_below_limit)
  int first_over_limit = start_idx;
  int first_below_limit = end_idx;

  for (int i = start_idx; i < end_idx; i++){
    if (vec_load[i] > memLimit){
      first_over_limit = i;
      break;
    }
  }

  for (int i = end_idx; i > first_over_limit; i--){
    if (vec_load[i] > memLimit){
      first_below_limit = i-1;
      break;
    }
  }

  return std::make_pair(first_over_limit, first_below_limit);
}

pair<int,int> load_below_limit(vector<double>vec_load, size_t memLimit, int start_idx, int end_idx, int maxIdx){
  //input: vec_load, memLimit, range [start_idx, end_idx]
  //return range overlimit [first_over_limit, first_below_limit)
  int first_below_limit = maxIdx;
  int last_below_limit = maxIdx;

  for (int i = first_below_limit; i > start_idx; i--){
    if (vec_load[i] > memLimit){
      first_below_limit = i+1;
      break;
    }
  }

  for (int i = last_below_limit; i < end_idx; i++){
    if (vec_load[i] > memLimit){
      last_below_limit = i-1;
      break;
    }
  }

  return std::make_pair(first_below_limit, last_below_limit);
}

void load_update(vector<double>& vec_load,int start_idx, int end_idx, int plusMinus, size_t size,int maxLen){
  //update load [start_idx, end_idx) by plusMinus*size
  if (start_idx < end_idx){
    cout<<start_idx<<' '<<end_idx<<endl;
    for (int i = start_idx; i<end_idx; i++){
      vec_load[i] = vec_load[i] + static_cast<double>(size) * plusMinus;
    }
  } else {
    for (int i = start_idx; i < maxLen; i++){
      vec_load[i] = vec_load[i] + static_cast<double>(size) * plusMinus;
    }
    for (int i = 0; i < end_idx; i++){ //TODO(junzhe) NOTE, end_idx excluded
      vec_load[i] = vec_load[i] + static_cast<double>(size) * plusMinus;
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
  repPatternDetector(vec_rep,maxLen,location);
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

  if (maxLen<100) {return -1;}

  return gc+maxLen-(gc-location)%maxLen;
} 

void SwapGPU::swap_plan(){
  //swap requirement
  cout<<":::::::::::::::::::::::::start swap_plan()"<<endl;
  float memLimit_ratio = 0.70; 
  size_t smallest_block = 1<<20; //1 MB
  int data_buffer = 4; // used to control readyIdx
  int mutable_data_buffer = 6;

  int idxRange = 0;
  vector<onePieceMsg> vec_pieceMsg = swap_strVec_2_pieceMsgVec(vec_block,idxRange);
  cout<<"size of vec_pieceMsg & vec_block: "<<vec_pieceMsg.size()<<' '<<vec_block.size()<<endl;
  sort(vec_pieceMsg.begin(),vec_pieceMsg.end(),less_than_Idx());
  // scale down idx, to middle iteration.
  cout<<"location, maxLen for scaledown: "<<location<<' '<<maxLen<<endl;
  int max_idx=0, min_idx=0;
  for (int i=0; i<vec_pieceMsg.size();i++){
    vec_pieceMsg[i].idx = vec_pieceMsg[i].idx - location - maxLen;
    max_idx = max(vec_pieceMsg[i].idx, max_idx);
    min_idx = min(vec_pieceMsg[i].idx, min_idx);
  }
  cout<<"max_idx and min_idx "<<max_idx<<' '<<min_idx<<endl;
  //cut into vec_run 0, vec_run, vec_run 2
  // vector<onePieceMsg>vec_run_0(&vec_pieceMsg[location],&vec_pieceMsg[location+maxLen]);
  //vector<onePieceMsg>vec_run(&vec_pieceMsg[location+maxLen],&vec_pieceMsg[location+2*maxLen]);
  // vector<onePieceMsg>vec_run_2(&vec_pieceMsg[location+2*maxLen],&vec_pieceMsg[location+3*maxLen]);
  //3 iterations, now named as vec_run
  vector<onePieceMsg>vec_run(&vec_pieceMsg[location],&vec_pieceMsg[location+3*maxLen]);
  cout<<"time for 3 itr: "<<vec_run[vec_run.size()-1].t-vec_run[0].t<<endl;
  ///get peak and idx of vec_load, updated with global_load
  int maxIdx = 0;
  size_t maxLoad = 0;
  //vec_load is 3 itr runs.
  vector<double>vec_load(&global_load[location],&global_load[location+3*maxLen]);
  // get max and idx from first itr.
  for (int i=0; i<maxLen; i++){
    if (maxLoad<vec_load[i]){
      maxLoad = vec_load[i];
      maxIdx = i;
    } 
  }

  //set limit, get over-limit
  size_t memLimit = 165<<20; //maxLoad - (62<<20);//memLimit_ratio * maxLoad;
  auto overLimit_ = load_over_limit(vec_load,memLimit,0,maxLen);
  cout<<"init_over_limit_range "<<overLimit_.first<<' '<<overLimit_.second<<endl;
  cout<<"memLimit and maxLoad are: "<<memLimit<<' '<<maxLoad<<endl;
  cout<<"range: "<<vec_load[overLimit_.first]<<" "<<vec_load[overLimit_.second]<<endl;
  //sort by ptr & idx
  //sort(vec_run.begin(),vec_run.end(),less_than_ptrIdx());
  sort(vec_run.begin(),vec_run.end(),less_than_ptrIdx());
  //for analysis only. TODO(junzhe)
  //fstream file_block3("vec_run_simple.text", ios::in|ios::out|ios::app);
  fstream file_block4("vec_run.text", ios::in|ios::out|ios::app);
  // fstream file_block4("vec_run_2.text", ios::in|ios::out|ios::app);
  for (int i = 0; i<vec_run.size();i++){
    //file_block3<<vec_run[i].idx<<' '<<vec_block[i+location]<<endl;
    file_block4<<vec_run[i].idx<<' '<<vec_run[i].MallocFree<<' '<<vec_run[i].ptr<<' '<<vec_run[i].size<<endl;
    //file_block4<<i<<' '<<vec_block[i+location+maxLen]<<endl;
  }
  //
  vector<SwapBlock>vec_swap;
  size_t sumSizeSwappAble =0;
  ///formulate swappable items.
  cout<<"==============================print swappable items "<<maxIdx<<endl;
  for (int i =1; i<vec_run.size(); i++){
    //SwapBlock(string p, size_t s, int i1, int i2, double t1, double t2): 
    //ptr(p), size(s), r_idx(i1),d_idx(i2),r_time(t1), d_time(t2) {}
    if ((vec_run[i].size >= smallest_block) && (vec_run[i-1].idx<maxIdx) && (vec_run[i].idx>maxIdx) 
      && (vec_run[i-1].ptr ==vec_run[i].ptr) 
      && ((vec_run[i-1].MallocFree==3) or (vec_run[i-1].MallocFree==2) or (vec_run[i-1].MallocFree==4)))
    {
      SwapBlock itm(vec_run[i].ptr,vec_run[i].size,vec_run[i-1].idx, vec_run[i].idx, vec_run[i-1].t, vec_run[i].t);
      itm.dto = itm.d_time-itm.r_time;
      itm.dt = itm.d_time-itm.r_time-SwapOutTime(itm.size)-SwapOutTime(itm.size);
      if (itm.dt>=0){
        itm.pri = itm.dt * itm.size;
      } else {
        itm.pri = itm.dt * 1/itm.size;
      }
      //cat A
      if (vec_run[i-1].MallocFree == 3){ itm.cat = "A1"; } 
      if (vec_run[i-1].MallocFree == 2){ itm.cat = "A2"; } 
      if (vec_run[i-1].MallocFree == 4){ itm.cat = "A3"; } 

      vec_swap.push_back(itm);
      sumSizeSwappAble+=itm.size;
      cout<<"Items Swappable: (r_idx, d_idx, cat, MB, dt/us, PS) || "<<itm.r_idx<<' '<<itm.d_idx;
      cout<<" ||  ."<<itm.cat<<".    "<<(float)(itm.size)/(float)(1024*1024);
      cout<<' '<<itm.dt/1000<<' '<<itm.pri<<endl;
    } 
  }
  cout<<"size vec_swap: "<<vec_swap.size()<<endl;
  //load before swap, write in
  fstream file_load_current("load_current.text", ios::in|ios::out|ios::app);
  for (int i=0; i<maxLen; i++){
    file_load_current<<vec_load[i]<<endl;
  }

  //update just on vec_load
  for (int i =0; i<vec_swap.size(); i++){
    int auto_buffer = 0;
    auto itm = vec_swap[i];
    if (itm.cat == "A2") auto_buffer = data_buffer;
    if (itm.cat == "A3") auto_buffer = mutable_data_buffer;
    load_update(vec_load,itm.r_idx+auto_buffer+maxLen,itm.d_idx+maxLen,-1,itm.size,maxLen);
    //vec_load = load_update_value(vec_load,itm.r_idx+auto_buffer,itm.d_idx,-1,itm.size,maxLen);
  }
  auto vec_load_ideal = vec_load;
  fstream file_load_ideal("load_ideal.text", ios::in|ios::out|ios::app);
  for (int i=maxLen; i<maxLen*2; i++){
    file_load_ideal<<vec_load_ideal[i]<<endl;
  }
  int maxIdx_ideal = 0;
  size_t maxLoad_ideal = 0;
  // get max and idx from first itr.
  for (int i=maxLen; i<maxLen*2; i++){
    if (maxLoad_ideal<vec_load_ideal[i]){
      maxLoad_ideal = vec_load_ideal[i];
      maxIdx_ideal = i;
    } 
  }
  maxIdx_ideal = maxIdx_ideal - maxLen;
  cout<<"load_ideal done, new max: "<<maxLoad_ideal<<" at "<<maxIdx_ideal<<endl;

  auto swapLoad = maxLoad - maxLoad_ideal;
  sort(vec_swap.begin(),vec_swap.end(),less_than_dto());
  vector<SwapBlock>vec_swap_selct;
  size_t sumSizeToSwap = 0;
  ///SwapBlock selection
  cout<<"============== select top a few to swap"<<endl;
  cout<<"maxIdx and maxLoad are: "<<maxIdx<<' '<<maxLoad<<endl;
  cout<<"sumSizeSwappAble: "<<(float)(sumSizeSwappAble)/(float)(1024*1024)<<' '<<sumSizeSwappAble_2/1024/1024<<endl;
  cout<<"memLimit and smallest_block: "<<memLimit<<' '<<smallest_block<<endl;
  // can upgrade here TODO(junzhe), now changed to select by dto
  for (int i =0; i<vec_swap.size(); i++){
    if ((maxLoad-sumSizeToSwap)>memLimit){
      vec_swap_selct.push_back(vec_swap[i]);
      sumSizeToSwap+=vec_swap[i].size;
      cout<<"Item selected: (r_idx, d_idx, MB, dt, cat) "<<vec_swap[i].r_idx<<' '<<vec_swap[i].d_idx;
      cout<<' '<<(float)(vec_swap[i].size)/(float)(1024*1024)<<' '<<vec_swap[i].dt/1000<<' '<<vec_swap[i].cat<<endl;
    } else {
      break;
    }
  }

  cout<<"number of swap_selct: "<<vec_swap_selct.size()<<endl;
  cout<<"swap size in MB: "<<(float)(sumSizeToSwap)/(float)(1024*1024)<<endl;

  ///SwapBlock scheduling - TO upgrade on 6/1 TODO(junzhe), based on version 3/4
  double overhead = 0;
  sort(vec_run.begin(),vec_run.end(),less_than_Idx());
  ///update swap-out idx //t1 and t1', i1 and i1p, sort by r_idx.
  sort(vec_swap_selct.begin(),vec_swap_selct.end(),less_than_Idx_Swap());
  //print only 
  cout<<"print sorted slect blocks--------r_idx, d_idx, ptr"<<endl;
  for (int i =0; i<vec_swap_selct.size(); i++){
    auto itm = vec_swap_selct[i];
    cout<<itm.r_idx<<' '<<itm.d_idx<<' '<<itm.ptr<<endl;
  }

  //load of ideal TODO(junzhe) solve memory issue
  // auto vec_load_ideal = vec_load;
  // for (int i =0; i<vec_swap.size(); i++){
  //   int auto_buffer = 0;
  //   auto itm = vec_swap[i];
  //   if (itm.cat == "A1") auto_buffer = data_buffer;
  //   if (itm.cat == "A2") auto_buffer = mutable_data_buffer;
  //   load_update(vec_load_ideal,itm.r_idx+auto_buffer,itm.d_idx,-1,itm.size,maxLen);
  // }
  // fstream file_load_ideal("load_ideal.text", ios::in|ios::out|ios::app);
  // for (int i=0; i<maxLen; i++){
  //   file_load_ideal<<vec_load_ideal[i]<<endl;
  // }
  // auto belowLimit_ = load_below_limit(vec_load_ideal,100<<20,0,maxLen,maxIdx);
  // cout<<"init_below_limit_range "<<belowLimit_.first<<' '<<belowLimit_.second<<endl;
  // auto start_idx = belowLimit_.second;
  // auto end_idx = belowLimit_.first + maxLen; //TODO(junzhe) these 2 +1 or not?
  // cout<<"below is for cat_B"<<endl;
  // int start_idx = 400;
  // int end_idx = 2;
 
  // for (int i =1; i<vec_run.size(); i++){
  //   if ((vec_run[i].size >= smallest_block) && (vec_run[i-1].idx<start_idx) && (vec_run[i].idx>end_idx) 
  //       && (vec_run[i-1].ptr ==vec_run[i].ptr) 
  //       && ((vec_run[i-1].MallocFree==3) or (vec_run[i-1].MallocFree==2) or (vec_run[i-1].MallocFree==4))){
  //     SwapBlock itm(vec_run[i].ptr,vec_run[i].size,vec_run[i-1].idx, vec_run[i].idx, vec_run[i-1].t, vec_run[i].t);
  //     itm.dt = itm.d_time-itm.r_time-SwapOutTime(itm.size)-SwapOutTime(itm.size);
  //     // if (itm.dt>=0){
  //     //   itm.pri = itm.dt * itm.size;
  //     // } else {
  //     //   itm.pri = itm.dt * 1/itm.size;
  //     // }
  //     //cat A
  //     if (vec_run[i-1].MallocFree == 3){ itm.cat = "B1"; } 
  //     if (vec_run[i-1].MallocFree == 2){ itm.cat = "B2"; } 
  //     if (vec_run[i-1].MallocFree == 4){ itm.cat = "B3"; } 

  //     vec_swap.push_back(itm);
  //     //sumSizeSwappAble+=itm.size;
  //     //sumSizeSwappAble_2+=vec_run[i].size;
  //     cout<<"Items Swappable_cat_B: (r_idx, d_idx, cat, MB, dt/us, PS) || "<<itm.r_idx<<' '<<itm.d_idx;
  //     cout<<" ||  ."<<itm.cat<<".    "<<(float)(itm.size)/(float)(1024*1024);
  //     cout<<' '<<itm.dt/1000<<' '<<itm.pri<<endl;
  //   }
  // }
  // cout<<"size vec_swap: "<<vec_swap.size()<<endl;
  // //load of ideal_B TODO(junzhe) to trouble shoot
  // auto vec_load_3rd = vec_load;
  // for (int i =0; i<vec_swap.size(); i++){
  //   int auto_buffer = 0;
  //   auto itm = vec_swap[i];
  //   if ((itm.cat == "A2") or (itm.cat == "B2")) auto_buffer = data_buffer;
  //   if ((itm.cat == "A3") or (itm.cat == "B3")) auto_buffer = mutable_data_buffer;
  //   if (itm.r_idx < itm.d_idx){
  //     load_update(vec_load_3rd,itm.r_idx+auto_buffer,itm.d_idx,-1,itm.size,maxLen);
  //   } else{
  //     load_update(vec_load_3rd,itm.r_idx,itm.d_idx+auto_buffer,-1,itm.size,maxLen);
  //   } 
  // }
  // fstream file_load_3rd("load_3rd.text", ios::in|ios::out|ios::app);
  // for (int i=0; i<maxLen; i++){
  //   file_load_3rd<<vec_load_3rd[i]<<endl;
  // }
  cout<<"done with swap_plan..."<<endl;
}






SwapGPU::~SwapGPU() {
  //print out push-info TODO(junzhe) can remove
  fstream file_block1("blockInfo.text", ios::in|ios::out|ios::app);
  for (int i=0; i< vec_block.size();i++){
      file_block1<<i<<' '<<vec_block[i]<<endl;
  }
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

  Test_sched_switch_swap();
  //cout<<"malloc after test"<<endl;
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

  Test_sched_switch_swap();
  //cout<<"free after test"<<endl;

  if (ptr != nullptr) {
    CUDA_CHECK(cudaSetDevice(id_));
    pool_->Free(ptr);
  }

  //cout<<"free done"<<endl; 
}

void SwapGPU::Test_sched_switch_swap(){
  /*
    do Test_sched_switch_swap during Malloc and Free.
    swap removed to DeploySwap
  */
  ///test & schedule
  if (((gc+1)%300 == 0) && (asyncSwapFlag == 0) && (testFlag == 0)){
  //TODO(junzhe) not lean, chances are globeCounter found more than 300 idx ago: redudant test.
  cout<<"gc, GC and vec_len before test: "<<gc<<' '<<globeCounter<<' '<<vec_block.size()<<endl;
  globeCounter = swap_test(vec_block,maxLen,location);
  if (maxLen > 100) {
    testFlag = 1;
    cout<<"compele test-swap:::::::::::::::::::::::::::::::::::::::::::::::::"<<endl;
    cout<<"size of Table_sched: "<<Table_sched.size()<<endl;
    cout<<"size of Table_meta: "<<Table_meta.size()<<endl;
    cout<<"impt numbers: "<<maxLen<<' '<<location<<' '<<globeCounter<<endl;
    
   }
 }
 ///switch flag;
 if (gc == globeCounter){
  swap_plan();
  asyncSwapFlag = 1;
  cout<<"switched flag for at "<<globeCounter<<endl;
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
   ///swap as per schedule: this version: both out and in sync
  int r_gc = (gc-location)%maxLen; 
  if ((asyncSwapFlag == 1) && (!(Table_sched.find(r_gc) == Table_sched.end()))){
    cout<<"--------sched action at "<<r_gc<<endl;
    auto swap_idx = std::get<0>(Table_sched.find(r_gc)->second);
    auto swap_dir = std::get<1>(Table_sched.find(r_gc)->second);
    auto sync_idx = std::get<2>(Table_sched.find(r_gc)->second);
    auto sync_dir = std::get<3>(Table_sched.find(r_gc)->second);
    if (swap_dir == 0){ SwapOut_idx(swap_idx); 
      cout<<"Swap Out "<<swap_idx<<endl;
    }
    if (swap_dir == 1){ SwapIn_idx(swap_idx); 
      cout<<"Swap In "<<swap_idx<<endl;
    }
    //TODO(junzhe) verify sync what else to be done
    if (sync_dir == 0){
      auto last_meta = Table_meta.find(sync_idx)->second;
      auto t1 = (std::chrono::system_clock::now()).time_since_epoch().count();
      cudaEventSynchronize(last_meta.in_event);
      auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
      Table_not_at_device[last_meta.block_] = sync_idx; //TODO(junzhe) double check if needed.
      last_meta.block_->update_data(nullptr);
      cout<<"to free data_"<<last_meta.data_<<endl;
      pool_->Free(last_meta.data_);
      last_meta.data_ = nullptr; //not really needed TODO(junzhe)
      cout<<"sync out "<<sync_idx<<endl;
      Table_meta.find(sync_idx)->second = last_meta;
    }
    if (sync_dir == 1){
      //if (!(Table_not_at_device.find(last_meta.block_)==Table_not_at_device.end())){ TODO(junzhe)
      auto last_meta = Table_meta.find(sync_idx)->second;
      auto t1 = (std::chrono::system_clock::now()).time_since_epoch().count();
      cudaEventSynchronize(last_meta.out_event);
      auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
      Table_not_at_device.erase(last_meta.block_);
      last_meta.block_->update_data(last_meta.data_);
      cout<<"sync in "<<sync_idx<<endl;
      Table_meta.find(sync_idx)->second = last_meta;
    }
    cout<<"-------"<<endl;
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
  if (maxLen < 100){
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
  //cout<<blockInfo<<endl;
  //cout<<tempBlock_->size()<<endl;
  //cout<<"load: "<<global_load[global_load.size()-1]<<" len of blockInfo and global_load "<<vec_block.size()<<' '<<global_load.size()<<endl;

  //std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  vec_block.push_back(blockInfo);
  // fstream file_block5("append.text", ios::in|ios::out|ios::app);
  // file_block5<<gc<<' '<<blockInfo<<' '<<(gc-1247)%612<<endl;

  // update Table_meta's block_ and data_
  if (maxLen > 100) {
    //cout<<gc<<' '<<(gc-location)%maxLen<<' '<<blockInfo<<endl;
    int r_gc = (gc-location)%maxLen;
    if (!(Table_meta.find(r_gc)==Table_meta.end())){
      //cout<<"r_gc, gc and size ot Table_meta "<<r_gc<<' '<<gc<<" "<<Table_meta.size()<<endl;
      //TODO(junzhe) verify the length change, if go in, value update
      cout<<"To update Block_ at "<<r_gc<<' '<<gc<<' '<<tempBlock_<<' '<<tempBlock_->get_data()<<endl;
      Table_meta.find(r_gc)->second.block_ = tempBlock_;
      Table_meta.find(r_gc)->second.data_ = tempBlock_->get_data();
    }
  }
  //deploy swap at every index.
  DeploySwap();
  //NOTE: this gc++ includes read/write and AppendLayer as well, in addition to malloc/free.
  gc++;

}

void* SwapGPU::GetRealGpuPtr(const Block* block_){
  //here should be not update_data()
  auto reading_meta = Table_meta.find(Table_not_at_device.find(block_)->second)->second;
  auto t1 = (std::chrono::system_clock::now()).time_since_epoch().count();
  cudaEventSynchronize(reading_meta.in_event);
  auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
  //cout<<"GetRealGpuPtr, overhead is: "<<t2-t1<<endl;
  //cout<<"To update_data swap for (In) "<<Table_not_at_device.find(block_)->second<<" "<<reading_meta.data_<<" 0"<<endl;
  //reading_meta.block_->update_data(reading_meta.data_);

  //cout<<"last_meta r_idx::::::malloc due to swapIn ( "<<Table_not_at_device.find(block_)->second<<endl;

  Table_not_at_device.erase(reading_meta.block_);

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
  cout<<"To update_data swap for (Out) "<<r_idx<<" "<<meta.block_<<" 0"<<endl;
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
  cout<<"right before cudaMemcpyAsync In"<<endl;
  err = cudaMemcpyAsync(meta.data_,meta.cpu_ptr,meta.size,cudaMemcpyHostToDevice,meta.in_stream);
  cudaEventRecord(meta.in_event,meta.in_stream);
  cout<<"right after cudaMemcpyAsync"<<endl;
  cout<<"To update_data swap for (In) "<<r_idx<<" "<<meta.block_<<" "<<meta.data_<<' '<<ptr<<endl;
  Table_meta.find(r_idx)->second = meta;
  //meta.block_->update_data(meta.data_); //TODO(junzhe) debug only, not the right place to update.
  //auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
  //cout<<"time for asynchrous: "<<t2-t1<<endl;
}

void SwapGPU::SwapOut(const Block* block_){
  if (gc < 1000 && block_->size() > 1<<20) {
    fstream file_block5("speed.text", ios::in|ios::out|ios::app);
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
    fstream file_block5("speed.text", ios::in|ios::out|ios::app);
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
