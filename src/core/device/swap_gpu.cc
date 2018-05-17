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

//TODO(junzhe) redefinition.
// struct lookUpElement{
//     /*
//      for memory pool Malloc look-up table.
//      */
//     int r_idx;
//     int d_idx;
//     size_t size;
//     size_t offset;
//     void* ptr;
//     int Occupied; //0 is free, 1 is occupied.
//     int crossItr; //c.
//     int Occupied_backup; //c.
//     int last_Occupied; //c. 1 means primary allocated, 2 means secondary.
// };

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
     convert vector of string into vector of onePieceMsg, sorted by ptr and then idx, and update idxRange to pieceMsgVec size.
     format of onePieceMsg [ptr, size/-1, flag, idx, timestamp]
     flag: 1 for malloc, -1 for free, 2 for mutable/read, 3 for layer
     */
    
    vector<onePieceMsg>onePieceMsgVec_;
    for (int i=0;i<vec.size();i++) {
        vector<string> v = swap_split(vec[i], " ");
        if (v[0]=="Malloc"){
            //change v[2] from str to size_t
            size_t result;
            stringstream convert(v[2]);
            if (!(convert>>result)){
                result =-1;
                cout<<"error for converting size from str to int."<<endl;
            }
            onePieceMsg tempMsg(v[1],result, 1, i);
            double tempTime;
            stringstream convert2(v[3]);
            convert2>>tempTime;
            tempMsg.t =tempTime;
            onePieceMsgVec_.push_back(tempMsg);
        }else if (v[0]=="Free"){
            onePieceMsg tempMsg(v[1],-1, -1, i);
            double tempTime;
            stringstream convert2(v[2]);
            convert2>>tempTime;
            tempMsg.t =tempTime;
            onePieceMsgVec_.push_back(tempMsg);
        }else if (v[0]=="Mutable" or v[0]=="Read"){
            onePieceMsg tempMsg(v[1],-1, 2, i);
            double tempTime;
            stringstream convert2(v[2]);
            convert2>>tempTime;
            tempMsg.t =tempTime;
            onePieceMsgVec_.push_back(tempMsg);
        }else if (v[0]=="Layer"){
            onePieceMsg tempMsg(v[1],-1, 3, i);
            double tempTime;
            stringstream convert2(v[2]);
            convert2>>tempTime;
            tempMsg.t =tempTime;
            onePieceMsgVec_.push_back(tempMsg);
        } else{
            cout<<"error for process the onePriceMsg."<<endl;
            cout<<vec[i]<<endl;
        }
    }
    
    sort(onePieceMsgVec_.begin(),onePieceMsgVec_.end(),less_than_ptrIdx());
    idxRange = static_cast<int>(onePieceMsgVec_.size());
    ///add size for non-malloc block info
    size_t tempSize=0;
    for (int i=0;i<onePieceMsgVec_.size();i++){
        if (onePieceMsgVec_[i].MallocFree==1){
            tempSize = onePieceMsgVec_[i].size;
        } else {
            onePieceMsgVec_[i].size = tempSize;
        }
    }

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

vector<double> compute_peak(vector<onePieceMsg> vec_run, int& max_Idx, size_t& max_Load ){
    //sort by idx
    int maxIdx =0;
    size_t maxLoad =0;
    size_t load=0;
    vector<double>vec_load;
    for (int i=0; i<vec_run.size(); i++){
        if(vec_run[i].MallocFree==1){
            load=load+vec_run[i].size;
        }else if (vec_run[i].MallocFree==-1){
            load=load-vec_run[i].size; //TODO(junzhe) size
        }
        if (maxLoad<load){
            maxLoad =load;
            maxIdx = i;
        }
        vec_load.push_back(load);
    }
    max_Idx = maxIdx;
    max_Load = maxLoad;
    return vec_load;
}

int SwapInTime(size_t size){
    //yet to get the formula
    int ans =0;
    if (size==0) {ans = 9500;} else {ans = 0.13*size;}
    return ans;
}
int SwapOutTime(size_t size){
    int ans =0;
    if (size==0) {ans = 17000;} else {ans = 0.29*size;}
    return ans;
}


struct onePairMsg_Swap{
    // more attributes, name different, use ptr.
    /*
     members: [name (r_idx), size, r_idx, d_idx]
     */
    string ptr;
    int name;
    size_t size;
    int r_idx; //out idx
    int d_idx; //in idx
    double r_time; // out time
    double d_time; //in time
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
    //onePairMsg(int n,size_t s, int r,int d):name(n),size(s),r_idx(r),d_idx(d){}
    //from LayerAppend (3) - r_idx, to next read/write (2) - d_idx
    onePairMsg_Swap(string p, size_t s, int i1, int i2, double t1, double t2): ptr(p), size(s), r_idx(i1),d_idx(i2),r_time(t1), d_time(t2) {}
};

struct less_than_dt{
    /*
     sort onePairMsg_Swap by dt, descending
     */
    inline bool operator() (const onePairMsg_Swap& struct1, const onePairMsg_Swap& struct2)
    {
        return (struct1.dt>struct2.dt);
    }
};

struct less_than_pri{
    /*
     sort onePairMsg_Swap by pri, descending
     */
    inline bool operator() (const onePairMsg_Swap& struct1, const onePairMsg_Swap& struct2)
    {
        return (struct1.pri>struct2.pri);
    }
};

struct less_than_Idx_Swap{
    /*
     sort onePieceMsg_Swap by idx.
     */
    inline bool operator() (const onePairMsg_Swap& struct1, const onePairMsg_Swap& struct2)
    {
        return (struct1.r_idx<struct2.r_idx);
    }
};

struct less_than_Idx_Swap_rvs{
    /*
     sort onePieceMsg_Swap by idx. reverse
     */
    inline bool operator() (const onePairMsg_Swap& struct1, const onePairMsg_Swap& struct2)
    {
        return (struct1.d_idx>struct2.d_idx);
    }
};

int load_over_limit(vector<double>vec_load, size_t memLimit, int old_idx){
    //TODO(junzhe) reduce time complexity
    for (int i = old_idx; i<vec_load.size();i++){
        if (vec_load[i]>=memLimit){
            return i;
        }
    }
    return static_cast<int>(vec_load.size()); //TODO(junzhe) to reset between comp t1' and t2'
}


void load_update(vector<double>& vec_load,int old_idx, int plusMinus, size_t size){
    for (int i = old_idx; i<vec_load.size();i++){
        //cout<<"plusMinus "<<plusMinus<<endl;
        if (plusMinus ==1){
            //cout<<"yes its 1"<<endl;
            vec_load[i]=vec_load[i]+static_cast<double>(size);
        } else {
            //cout<<"no its not 1"<<endl;
            vec_load[i]=vec_load[i]-static_cast<double>(size);
        }
    }
}

int SwapGPU::swap_test(vector<string>vec_block,int &maxLen, int &location){
  ///vec_str (vec_block) to vec_pieceMsg, sort by ptr and idx.
  int idxRange =0;
  vector<onePieceMsg> vec_pieceMsg = swap_strVec_2_pieceMsgVec(vec_block,idxRange);
  cout<<"size of vec_pieceMsg and vec_block are: "<<vec_pieceMsg.size()<<' '<<vec_block.size()<<endl;
  ///rep test
  vector<size_t> vec_rep = Swap_piece2rep(vec_pieceMsg);
  //int idxRange3=0; //rename TODO
  //int maxLen=0, location =0;
  repPatternDetector(vec_rep,maxLen,location);
  cout<<"maxLen and location are: "<<maxLen<<' '<<location<<endl;
  cout<<"test rep"<<endl;
  //Note here location not exactly start of one iteration, adjust to nearly start of one by restricting "Malloc"
  int shift_counter =0;
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
  //TODO(junzhe) below belong to run() section
  ///cut into one iteration.
  sort(vec_pieceMsg.begin(),vec_pieceMsg.end(),less_than_Idx());
  vector<onePieceMsg>vec_run(&vec_pieceMsg[location],&vec_pieceMsg[location+maxLen]);
  cout<<"time for one itr: "<<vec_run[vec_run.size()-1].t-vec_run[0].t<<endl;
  cout<<" to write vec_run for comparison, nothing wrong till here 5/13."<<endl;
   //print out push-info
  fstream file_block2("vec_run.text", ios::in|ios::out|ios::app);
  for (int i = 0; i<vec_run.size();i++){
    file_block2<<vec_run[i].idx<<' '<<vec_run[i].MallocFree<<' '<<vec_run[i].ptr<<vec_run[i].t<<endl;
  }
  //scale down idx
  for (int i=0; i<maxLen;i++){
      vec_run[i].idx = vec_run[i].idx - location;
  }
  ///get peak and idx
  int maxIdx =0;
  size_t maxLoad =0;
  vector<double>vec_load = compute_peak(vec_run, maxIdx, maxLoad);//no longer valid here.

  //sort by ptr & idx
  sort(vec_run.begin(),vec_run.end(),less_than_ptrIdx());
   cout<<" to write vec_run2 for comparison, nothing wrong till here 5/13."<<endl;
   //print out push-info
  fstream file_block3("vec_run2.text", ios::in|ios::out|ios::app);
  for (int i = 0; i<vec_run.size();i++){
    file_block3<<vec_run[i].idx<<' '<<vec_run[i].MallocFree<<' '<<vec_run[i].ptr<<vec_run[i].t<<endl;
  }
  vector<onePairMsg_Swap>vec_swap;
  size_t sumSizeSwapAble =0;
  ///formulate swap items.
  cout<<"===============================print sorted run "<<maxIdx<<endl;
  for (int i =1; i<vec_run.size(); i++){
      //cout<<vec_run[i-1].ptr<<' '<<vec_run[i-1].idx<<' '<<vec_run[i-1].MallocFree<<' '<<vec_run[i-1].size;
      //condition for selecting condidates: 3->2, cross peak
      //from LayerAppend (3) - r_idx, to next read/write (2) - d_idx
      if ((vec_run[i-1].idx<maxIdx) && (vec_run[i].idx>maxIdx) && (vec_run[i-1].ptr ==vec_run[i].ptr) && (vec_run[i-1].MallocFree==3)&&(vec_run[i].MallocFree==2)){
          //cout<<' '<<"selected"<<endl;
          onePairMsg_Swap tempSwap(vec_run[i].ptr,vec_run[i].size,vec_run[i-1].idx, vec_run[i].idx, vec_run[i-1].t, vec_run[i].t);
          //tempSwap.dt_o = tempSwap.d_time-tempSwap.r_time;
          tempSwap.dt = tempSwap.d_time-tempSwap.r_time-SwapOutTime(tempSwap.size)-SwapOutTime(tempSwap.size);
          //tempSwap.dt_p = tempSwap.dt - L_out - L_in;
          if (tempSwap.dt>=0){
              tempSwap.pri = tempSwap.dt * tempSwap.size;
          } else {
              //Note: to make sense for negative dt.
              tempSwap.pri = tempSwap.dt * 1/tempSwap.size;
          }
          vec_swap.push_back(tempSwap);
          sumSizeSwapAble+=tempSwap.size;
          //onePairMsg_Swap(string p, size_t s, int i1, int i2, double t1, double t2): ptr(p), size(s), r_idx(i1),d_idx(i2),r_time(t1), d_time(t2) {}

          cout<<"SwapItem: "<<tempSwap.r_idx<<' '<<tempSwap.d_idx<<endl;
      } else {//cout<<endl;
          
      }
  }
  cout<<"swapable items: "<<vec_swap.size()<<endl;

  ///select the top a few that can meet swap load
  //TODO(junzhe) optimize the for loop.
  cout<<"============== select top a few to swap"<<endl;
  cout<<"maxIdx and maxLoad are: "<<maxIdx<<' '<<maxLoad<<endl;
  cout<<"sumSizeSwapAble: "<<sumSizeSwapAble<<endl;
  size_t memLimit = 20000000; //75% of the maxLoad, for example.
  sort(vec_swap.begin(),vec_swap.end(),less_than_pri());
  vector<onePairMsg_Swap>vec_swap_selct;
  size_t sumSizeToSwap=0;
  for (int i =0; i<vec_swap.size(); i++){
      if ((maxLoad-sumSizeToSwap)>memLimit){
          vec_swap_selct.push_back(vec_swap[i]);
          sumSizeToSwap+=vec_swap[i].size;
      } else {
          break;
      }
  }
  cout<<"number of swap_selct: "<<vec_swap_selct.size()<<endl;
  cout<<"swap size: "<<sumSizeToSwap<<endl;

  ///planing swap in and swap out. version 3/4
    cout<<"swap scheduling.===================="<<endl;
    double overhead=0;
    int old_idx=0; //book keep where the load track is.
    sort(vec_run.begin(),vec_run.end(),less_than_Idx());
    ///update swap-out idx
    //t1 and t1', i1 and i1', sort by r_idx.
    sort(vec_swap_selct.begin(),vec_swap_selct.end(),less_than_Idx_Swap());
    for (int i =0; i<vec_swap_selct.size(); i++){
        int tempIdx=vec_swap_selct[i].r_idx;//idx ready to swapOut, pesudo code use time.
        if ((i>0) and (tempIdx<vec_swap_selct[i-1].i1p)){
            //last t1' bigger than this t1
            tempIdx = vec_swap_selct[i-1].i1p; //alr at M/F idx, diff from pseudo code.
        } else {
            //round to next Malloc/Free
            while ((vec_run[tempIdx].MallocFree!=1) and (vec_run[tempIdx].MallocFree!=-1)){
                tempIdx++;
            }
        }
        //update t1, t1p, i1
        vec_swap_selct[i].i1=vec_run[tempIdx].idx;
        vec_swap_selct[i].t1=vec_run[tempIdx].t;
        vec_swap_selct[i].t1p = vec_swap_selct[i].t1+SwapOutTime(vec_swap_selct[i].size);
        //update i1p, compare with last swap and load
        while ((vec_swap_selct[i].t1p>=vec_run[tempIdx].t) or ((vec_run[tempIdx].MallocFree!=1) and (vec_run[tempIdx].MallocFree!=-1))) {
            tempIdx++; //TODO(junzhe) can speed up
        }
        //update i1p again, with condideration of over limit.
        old_idx = load_over_limit(vec_load,memLimit,old_idx);//TODO(juznhe) worse case is overlimit even before the first swap item.
        if (old_idx<tempIdx){
            //over limit before tempIdx, got overhead
            vec_swap_selct[i].i1p = old_idx;
            overhead+=(vec_swap_selct[i].t1p-vec_run[old_idx].t);
            load_update(vec_load,old_idx,-1,vec_swap_selct[i].size);
        } else {
            vec_swap_selct[i].i1p = tempIdx;//Note: here i1' is immediately at Malloc/Free.
            load_update(vec_load,tempIdx,-1,vec_swap_selct[i].size);
        }
        cout<<"old_idx and i1p: "<<old_idx<<' '<<tempIdx<<' '<<vec_swap_selct[i].r_idx<<' '<<vec_run[old_idx].MallocFree<<" overhead "<<overhead<<endl;
        cout<<"--------------size: "<<vec_swap_selct[i].size<<endl;
    } //for loop
    //cout<<"total overhead: "<<overhead<<endl;
    ///update swap-in index
    //t2 and t2', i2 and i2'.
    sort(vec_swap_selct.begin(),vec_swap_selct.end(),less_than_Idx_Swap_rvs());
    ///step 1: overlap with next swapIn.
    for (int i =0; i<vec_swap_selct.size(); i++){
        int tempIdx=vec_swap_selct[i].d_idx; //idx at which to be used.
        double tempTime; //time need to be swapped in start.
        //condition, if overlap tempIdx later than next i2p, pull in swap in.
        if ((i>0) and (tempIdx>vec_swap_selct[i-1].i2p)){
            tempIdx = vec_swap_selct[i-1].i2p;
            tempTime = vec_run[tempIdx].t - SwapInTime(vec_swap_selct[i].size);
        } else{
            tempTime = vec_swap_selct[i].d_time - SwapInTime(vec_swap_selct[i].size);
        }
        
        //update i2p, t2p; not used for i2 and t2, with wait_till_aval function at data()
        vec_swap_selct[i].i2 = tempIdx;
        while ((tempTime<=vec_run[tempIdx].t) or ((vec_run[tempIdx].MallocFree!=1) and (vec_run[tempIdx].MallocFree!=-1))) {
            tempIdx--; //TODO(junzhe) can speed up
        }
        vec_swap_selct[i].i2p = tempIdx;
        vec_swap_selct[i].t2p = tempTime; //Note here use tempTime
    }
    
    ///step 2: change i2p to load exceeds limit, with overhead.
    //verify TODO(junzhe)
    for (int i = static_cast<int>(vec_swap_selct.size()-1);i>=0; i--){
        old_idx = vec_swap_selct[i].i2p;
        load_update(vec_load,vec_swap_selct[i].i2p, 1,vec_swap_selct[i].size);
        old_idx = load_over_limit(vec_load,memLimit,old_idx);
        if (old_idx< vec_run.size()) {
            while (vec_load[old_idx]>memLimit){
                old_idx++;
            }
            //restore the update if there is over the limit
            load_update(vec_load,vec_swap_selct[i].i2p,-1,vec_swap_selct[i].size);
            //reapply to the just nice idx.
            load_update(vec_load,old_idx,1,vec_swap_selct[i].size);
            vec_swap_selct[i].i2p = old_idx;
            overhead+=(vec_run[old_idx].t-vec_swap_selct[i].t2p);
            //cout<<"overhead "<<vec_run[old_idx].t-vec_swap_selct[i].t2p<<endl;
        }
    }
    //step 3: overhead due to potential overlapping. verify
    cout<<"verify if self overlapping"<<endl;
    for (int i =0; i<vec_swap_selct.size(); i++){
        cout<<vec_swap_selct[i].t2p-vec_swap_selct[i].t1p<<endl;
    }
    cout<<"total overhead: "<<overhead<<endl;
    cout<<"done"<<endl;
    ///make the Table_sched
    //map<int,std::tuple<int,size_t,int>>Table_sched; //schedule, int 0 means D2H, 1 means H2D.
    for (int i = static_cast<int>(vec_swap_selct.size()-1);i>=0; i--){
        //for each selct block, i1 is start swapOut, i2p is start swapIn. junzhe on 5.4 
        //TODO(junzhe) to verify above statement.
        Table_sched[vec_swap_selct[i].i1] = std::make_tuple(vec_swap_selct[i].r_idx, vec_swap_selct[i].size,0);
        Table_sched[vec_swap_selct[i].i2p] = std::make_tuple(vec_swap_selct[i].r_idx,vec_swap_selct[i].size,1);
        //init only, update at MakeTableMeta.
        //Table_Block_[vec_swap_selct[i].r_idx] = "nullptr";
        Table_Block_ptr[vec_swap_selct[i].r_idx] = nullptr;
        cout<<"Table_Block_ptr "<<vec_swap_selct[i].r_idx<<" "<<Table_Block_ptr.find(vec_swap_selct[i].r_idx)->second<<endl;
        //cout<<"***Table_sched "<<vec_swap_selct[i].i1<<' '<<vec_swap_selct[i].r_idx<<' '<<vec_swap_selct[i].size<<endl;
        //cout<<"***Table_sched "<<vec_swap_selct[i].i2p<<' '<<vec_swap_selct[i].r_idx<<' '<<vec_swap_selct[i].size<<endl;
        
      void* tempPtr = nullptr;
      cudaMallocHost(&tempPtr,vec_swap_selct[i].size); //pinned memory.
      BM_new meta;
      meta.size = vec_swap_selct[i].size;
      meta.cpu_ptr = tempPtr;
      Table_new[vec_swap_selct[i].r_idx] = meta;
    }
    cout<<"size of Table_Block_ptr "<<Table_Block_ptr.size()<<endl;

  //update globeCounter
  return gc+maxLen-(gc-location)%maxLen;
}




SwapGPU::~SwapGPU() {
  //print out push-info
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
  cudaStream_t stream1;

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

  //Apend info done at device.cc, TODO(junzhe) if remove below item.
  Table_meta.erase(Table_data_block_.find(ptr)->second);
  //vC12
  Table_block_data_.erase(Table_data_block_.find(ptr)->second);
  Table_data_block_.erase(ptr);
  //cout<<"free done"<<endl;
 
}

void SwapGPU::Test_sched_switch_swap(){
  /*
    do Test_sched_switch_swap during Malloc and Free.
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
    cout<<"size of Table_new: "<<Table_new.size()<<endl;
    cout<<"impt numbers: "<<maxLen<<' '<<location<<' '<<globeCounter<<endl;
   }
 }

 ///switch flag;
 if (gc == globeCounter){
   asyncSwapFlag = 1;
   cout<<"switched flag for at "<<globeCounter<<endl;
 }

 ///swap as per schedule
  int relative_gc = (gc-location)%maxLen; //verified
  //map<int,std::tuple<int,size_t,int>>Table_sched; //schedule, int 0 means D2H, 1 means H2D.
  if ((asyncSwapFlag == 1) && (!(Table_sched.find((gc-location)%maxLen) == Table_sched.end()))){
    //cout<<"std::get<2>(Table_sched.find((gc-location)%maxLen)->second) "<<std::get<2>(Table_sched.find((gc-location)%maxLen)->second)<<endl;
    if (std::get<2>(Table_sched.find((gc-location)%maxLen)->second) == 0) {
      int r_idx = std::get<0>(Table_sched.find((gc-location)%maxLen)->second);
      SwapOut_idx(r_idx);
      //SwapOut(Table_Block_ptr.find(r_idx)->second);
      cout<<"swapOut - print from Malloc"<<endl;
    } else {
      int r_idx = std::get<0>(Table_sched.find((gc-location)%maxLen)->second);
      SwapIn_idx(r_idx);
      //SwapIn(Table_Block_ptr.find(r_idx)->second);
      cout<<"swapIn - print from Malloc"<<endl;
      //cout<<"=========== update data_"<<endl;

    }
  }

}

void SwapGPU::MakeMetaTable(Block* block_,void* data_,int size){
  /*make table and append vec_block.
  this is only called once, right after Malloc.
  Hence the malloc info is pushed here.
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
  
  //Table_meta
  BlockMeta cpu,gpu;
  cpu.size = static_cast<size_t>(size);
  gpu.size = static_cast<size_t>(size);
  gpu.ptr = data_;
  pair<BlockMeta,BlockMeta>meta = std::make_pair(cpu, gpu);
  Table_meta[block_] = meta;

  //TODO(junzhe) below causes core dumped.
  // if (((gc-location)%maxLen==69) or ((gc-location)%maxLen==60)or((gc-location)%maxLen==50)or((gc-location)%maxLen==36)){
  //   cout<<"should update Block_ "<<(gc-location)%maxLen<<endl;
  // }
  //TODO(junzhe) to remove it.
  if (asyncSwapFlag == 1) {
    int r_gc = (gc-location)%maxLen;
    if (!(Table_Block_ptr.find(r_gc)==Table_Block_ptr.end())){
      cout<<"good at "<<r_gc<<" size of Table_Block_ptr "<<Table_Block_ptr.size()<<endl;
      //Table_Block_ptr.at((gc-location)%maxLen) = block_;
    //Table_meta_ridx[(gc-location)%maxLen] = meta;
    }
  }

  Table_data_block_[data_] = block_;
  Table_block_data_[block_] = data_;

}

void SwapGPU::Append(string blockInfo){
  vec_block.push_back(blockInfo);
  //NOTE: this gc++ includes read/write and AppendLayer as well, in addition to malloc/free.
  //vC12 part
  if (maxLen > 100) {
    int r_gc = (gc-location)%maxLen;

    if (!(Table_new.find(r_gc)==Table_new.end())){
      vector<string> v = swap_split(blockInfo, " ");
      void* result;
      stringstream convert(v[1]);
      convert>>result;
      //cout<<"r_gc, gc and size ot Table_new "<<r_gc<<' '<<gc<<" "<<Table_new.size()<<endl;
      //TODO(junzhe) verify the length change, if go in, value update
      Table_new.find(r_gc)->second.block_ = static_cast<Block*>(result);
      Table_new.find(r_gc)->second.data_ = Table_block_data_.find(static_cast<Block*>(result))->second;
    }
  }

  gc++;

}

void* SwapGPU::GetRealGpuPtr(const Block* block_){
  // //void* data_ = Table_meta.find(block_)->second.second.ptr;
  // 
  // cout<<"NOTE--: from SwapGPU, data_, block_ device_: "<<Table_block_data_.find(block_)->second<<' '<<block_<<' '<<this<<endl;
  // if (!(Table_block_data_.find(block_) == Table_block_data_.end())){
    
  //   return Table_block_data_.find(block_)->second;

  // } else {
  //   cout<<"return nullptr"<<endl; 
  // }
  return nullptr; //TODO(junzhe) attention, based on no change here.
}

void SwapGPU::SwapOut_idx(const int r_idx){
  cout<<"doing asynchrous swapOut of r_idx: "<<r_idx<<' '<<endl;
  auto t1 = (std::chrono::system_clock::now()).time_since_epoch().count();  
  //cout<<"before cudaMemcpyAsync"<<endl;
  //cudaStream_t stream3;
  //cudaEvent_t event1;
  
  cudaError_t err;
  //TODO(junzhe) not sure why not working.
  BM_new meta = Table_new.find(r_idx)->second;
  cudaEventCreate (&meta.out_event);
  cout<<"right before cudaMemcpyAsync"<<endl;
  err = cudaMemcpyAsync(meta.cpu_ptr,meta.data_,meta.size,cudaMemcpyDeviceToHost,stream1);
  cout<<"right after cudaMemcpyAsync"<<endl;
  //cudaEventRecord(event1,stream1);
  auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
  //cout<<"time for asynchrous: "<<t2-t1<<endl;
  //cudaEventSynchronize(event1);
  //auto t4 = (std::chrono::system_clock::now()).time_since_epoch().count();
  //cout<<"time for asynchrous to complete: "<<t4-t1<<endl;
}

void SwapGPU::SwapIn_idx(const int r_idx){
  
  //TODO(junzhe) not lean, as size is stored in Table_sched as well.
  //cout<<"doing asynchrous swapIn"<<endl;
  auto t1 = (std::chrono::system_clock::now()).time_since_epoch().count();
  cudaError_t err;
  cudaStream_t stream2;
  cudaEvent_t event2;
  cudaEventCreate (&event2);
  BM_new meta = Table_new.find(r_idx)->second;
  //update gpu ptr. //TODO(junzhe) test funcionality of change data_
  cout<<"update block and data of r_idx: "<<r_idx<<' '<<meta.block_<<' '<<meta.data_<<endl;
  void* ptr = nullptr;
  pool_->Malloc((void**)&ptr, meta.size);
  void* to_rm_ptr = meta.data_;
  meta.data_ = ptr;
  //auto tempPtr = Malloc(meta.size);
  //meta.data_ = Malloc(Table_new.find(r_idx)->second.size);
  err = cudaMemcpyAsync(meta.data_,meta.cpu_ptr,meta.size,cudaMemcpyHostToDevice,stream2);
  if (tempCounter <3){
    meta.block_->update_data(meta.data_);
    pool_->Free(to_rm_ptr);
    tempCounter++;
    cout<<"---========got real update:"<<meta.block_<<" "<<meta.data_<<endl;
  }
  
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  //err = cudaMemcpyAsync(Table_meta.find(r_idx)->second.data_,Table_meta.find(r_gc)->second.cpu_ptr,Table_meta.find(r_gc)->second.size,cudaMemcpyHostToDevice,stream2);

  cudaEventRecord(event2,stream2);
  auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
  //cout<<"time for asynchrous: "<<t2-t1<<endl;
}

void SwapGPU::SwapOut(const Block* block_){
	// 0 is sync, 1 async: asyncSwapFlag
  if (asyncSwapFlag == 0){
      //TODO(junzhe)
     
      printf("A. to swapOut\n");
      auto t1 = (std::chrono::system_clock::now()).time_since_epoch().count();
      size_t swapSize = Table_meta_ridx.find(std::get<0>(Table_sched.find((gc-location)%maxLen)->second))->second.second.size;
      Table_meta_ridx.find(std::get<0>(Table_sched.find((gc-location)%maxLen)->second))->second.first.ptr = malloc(swapSize);
      BlockMeta cpu, gpu;
      cpu = Table_meta_ridx.find(std::get<0>(Table_sched.find((gc-location)%maxLen)->second))->second.first;
      gpu = Table_meta_ridx.find(std::get<0>(Table_sched.find((gc-location)%maxLen)->second))->second.second;
      cudaError_t err;
      cout<<"to Copy: "<<cpu.ptr<<' '<<gpu.ptr<<' '<<gpu.size<<endl;
      err=cudaMemcpy(cpu.ptr,gpu.ptr,gpu.size,cudaMemcpyDeviceToHost);
      if (err != cudaSuccess)
        {
        fprintf(stderr, "SwapOut err (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
        }
      auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
      //without free here.
      //cudaFree(gpu.ptr); //TODO(junzhe) not able to free, work on it.
      //Table_meta_ridx.find(std::get<0>(Table_sched.find((gc-location)%maxLen)->second))->second.second.ptr=nullptr;
  } else {
      //TODO(junzhe) not lean, as size is stored in Table_sched as well.
      cout<<"doing asynchrous swapOut"<<endl;
      size_t swapSize = Table_meta_ridx.find(std::get<0>(Table_sched.find((gc-location)%maxLen)->second))->second.second.size;
      void* tempPtr=nullptr;
      cout<<"tempPtr & swapSize "<<tempPtr<<' '<<swapSize<<endl;
      cudaMallocHost(&tempPtr,swapSize); //pinned memory.
      cout<<"cudaMallocHost done"<<endl;
      Table_meta_ridx.find(std::get<0>(Table_sched.find((gc-location)%maxLen)->second))->second.first.ptr = tempPtr;
      BlockMeta cpu, gpu;
      cpu = Table_meta_ridx.find(std::get<0>(Table_sched.find((gc-location)%maxLen)->second))->second.first;
      gpu = Table_meta_ridx.find(std::get<0>(Table_sched.find((gc-location)%maxLen)->second))->second.second;
      auto t1 = (std::chrono::system_clock::now()).time_since_epoch().count();
      cudaError_t err;
      cout<<"before cudaMemcpyAsync"<<endl;
      cudaStream_t stream3;
      //cudaEvent_t event1;
      //cudaEventCreate (&event1);
      err=cudaMemcpyAsync(cpu.ptr,gpu.ptr,gpu.size,cudaMemcpyDeviceToHost,stream3);
      //cudaEventRecord(event1,stream1);
      auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
      cout<<"time for asynchrous: "<<t2-t1<<endl;
      //cudaEventSynchronize(event1);
      //auto t4 = (std::chrono::system_clock::now()).time_since_epoch().count();
      //cout<<"time for asynchrous to complete: "<<t4-t1<<endl;
  }

  ///below switch is not used, previous test only.
	// switch (asyncSwapFlag) {
	// 	case (3) :
	// 	{	
	// 		//asynchrous here.
	// 		size_t swapSize = Table_meta_ridx.find(std::get<0>(Table_sched.find((gc-location)%maxLen)->second))->second.second.size;
	// 		void* tempPtr;
	// 		cudaMallocHost(&tempPtr,swapSize); //pinned memory.
	// 		Table_meta_ridx.find(std::get<0>(Table_sched.find((gc-location)%maxLen)->second))->second.first.ptr = tempPtr;
	// 		BlockMeta cpu, gpu;
	// 		cpu = Table_meta_ridx.find(std::get<0>(Table_sched.find((gc-location)%maxLen)->second))->second.first;
	// 		gpu = Table_meta_ridx.find(std::get<0>(Table_sched.find((gc-location)%maxLen)->second))->second.second;
	// 		auto t1 = (std::chrono::system_clock::now()).time_since_epoch().count();
	// 		cudaError_t err;
	// 		cudaStream_t stream1;
 //      cudaEvent_t event1;
 //      cudaEventCreate (&event1);
	// 		err=cudaMemcpyAsync(cpu.ptr,gpu.ptr,gpu.size,cudaMemcpyDeviceToHost,stream1);
 //      cudaEventRecord(event1,stream1);
	// 		auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
	// 		cout<<"time for asynchrous: "<<t2-t1<<endl;
 //      cudaEventSynchronize(event1);
 //      auto t4 = (std::chrono::system_clock::now()).time_since_epoch().count();
 //      cout<<"time for asynchrous to complete: "<<t4-t1<<endl;
	// 		cudaError_t err2;
	// 		err2=cudaMemcpy(cpu.ptr,gpu.ptr,gpu.size,cudaMemcpyDeviceToHost);
	// 		auto t3 = (std::chrono::system_clock::now()).time_since_epoch().count();
	// 		cout<<"time for sync: "<<t3-t2<<endl;
	// 		break;
	// 	}
	// 	case (4) :{
	// 	  break;
	// 	}
	// }

}

void SwapGPU::SwapIn(const Block* block_){
	// 0 is sync, 1 async: asyncSwapFlag
  if (asyncSwapFlag == 0){
      printf("1. to swapIn.\n");
      auto t1 = (std::chrono::system_clock::now()).time_since_epoch().count();
      BlockMeta cpu, gpu;
      cpu = Table_meta_ridx.find(std::get<0>(Table_sched.find((gc-location)%maxLen)->second))->second.first;
      gpu = Table_meta_ridx.find(std::get<0>(Table_sched.find((gc-location)%maxLen)->second))->second.second;
      //without free here.
      gpu.ptr = nullptr;
      cudaError_t status = cudaMalloc(&gpu.ptr, gpu.size);
      CHECK_EQ(status, cudaError_t::cudaSuccess);
      //update tables
      Table_meta_ridx.find(std::get<0>(Table_sched.find((gc-location)%maxLen)->second))->second.second.ptr = gpu.ptr;
      //cout<<"after alloc:1 "<<Table_meta.find(data_)->second.second.ptr<<endl;
      cudaError_t err;
      err = cudaMemcpy(gpu.ptr, cpu.ptr ,cpu.size,cudaMemcpyHostToDevice);
      //printf("2. swapIn done.\n");
      free(cpu.ptr);
      //update tables
      Table_meta_ridx.find(std::get<0>(Table_sched.find((gc-location)%maxLen)->second))->second.first.ptr = nullptr;
      //
      auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
      fstream file_block3("blockInfo_swapIn.text", ios::in|ios::out|ios::app);
      file_block3<<t2-t1<<" "<<gpu.size<<endl;
  } else {
      //TODO(junzhe) not lean, as size is stored in Table_sched as well.
      cout<<"doing asynchrous swapIn"<<endl;
      BlockMeta cpu, gpu;
      cpu = Table_meta_ridx.find(std::get<0>(Table_sched.find((gc-location)%maxLen)->second))->second.first;
      gpu = Table_meta_ridx.find(std::get<0>(Table_sched.find((gc-location)%maxLen)->second))->second.second;
      //without free here.
      gpu.ptr = nullptr;
      cudaError_t status = cudaMalloc(&gpu.ptr, gpu.size);
      CHECK_EQ(status, cudaError_t::cudaSuccess);
      //update tables
      Table_meta_ridx.find(std::get<0>(Table_sched.find((gc-location)%maxLen)->second))->second.second.ptr=gpu.ptr;
      auto t1 = (std::chrono::system_clock::now()).time_since_epoch().count();
      cudaError_t err;
     // cudaStream_t stream2;
      cudaEvent_t event2;
      cudaEventCreate (&event2);
      err = cudaMemcpyAsync(gpu.ptr, cpu.ptr ,cpu.size,cudaMemcpyHostToDevice,stream2);
      cudaEventRecord(event2,stream2);
      auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
      cout<<"time for asynchrous: "<<t2-t1<<endl;
  }

}


}  // namespace singa
#endif  // USE_CUDA
