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
#ifndef DISABLE_WARNINGS

#include "singa/core/memory.h"
#include "singa/utils/logging.h"
#include "singa/proto/core.pb.h"
#include <iostream>
#include <fstream> //a.
#include <chrono>
//for SmartMemoryPool
using namespace std;

#ifdef USE_CUDA

namespace singa {
std::atomic<int> CnMemPool::pool_count(0);
std::pair<size_t, size_t> CnMemPool::GetMemUsage() {
  size_t free, total;
  auto status = cnmemMemGetInfo(&free, &total, NULL);
  CHECK_EQ(status, cnmemStatus_t::CNMEM_STATUS_SUCCESS)
    << cnmemGetErrorString(status);
  return std::make_pair(free, total);
}

CnMemPool::CnMemPool(int numDevices, size_t init_size, size_t max_size) {
  for (int i = 0; i < numDevices; i++)
    conf_.add_device(i);
  conf_.set_init_size(init_size);
  conf_.set_max_size(max_size);
  CHECK_LT(++pool_count, 2) << "CnMemPool must be used as a singleton.";
}

CnMemPool::CnMemPool(const MemPoolConf &conf) {
  conf_ = conf;
  CHECK_LT(++pool_count, 2) << "CnMemPool must be used as a singleton.";
}

void CnMemPool::Init() {
  mtx_.lock();
  if (!initialized_) {
    const size_t kNBytesPerMB = (1u << 20);
    CHECK_GE(conf_.device_size(), 1);
    cnmemDevice_t *settingPtr = new cnmemDevice_t[conf_.device_size()];
    CHECK_GT(conf_.init_size(), 0u);
    int i = 0;
    for (auto device : conf_.device()) {
      settingPtr[i].device = device;
      settingPtr[i].size = conf_.init_size() * kNBytesPerMB;
      settingPtr[i].numStreams = 0;
      settingPtr[i].streams = NULL;
      settingPtr[i].streamSizes = 0;
      i++;
    }
    auto status = cnmemInit(conf_.device_size(), settingPtr, conf_.flag());
    CHECK_EQ(status, cnmemStatus_t::CNMEM_STATUS_SUCCESS)
        << " " << cnmemGetErrorString(status);
    delete[] settingPtr;
    initialized_ = true;
  }
  mtx_.unlock();
}

CnMemPool::~CnMemPool() {
  mtx_.lock();
  if (initialized_) {
    cnmemStatus_t status = cnmemFinalize();
    CHECK_EQ(status, cnmemStatus_t::CNMEM_STATUS_SUCCESS)
        << " " << cnmemGetErrorString(status);
    initialized_ = false;
    --pool_count;
  }
  mtx_.unlock();
}

void CnMemPool::Malloc(void **ptr, const size_t size) {
  if (!initialized_)
    Init();
  cnmemStatus_t status = cnmemMalloc(ptr, size, NULL);
  CHECK_EQ(status, cnmemStatus_t::CNMEM_STATUS_SUCCESS)
      << " " << cnmemGetErrorString(status);
}

void CnMemPool::Free(void *ptr) {
  CHECK(initialized_) << "Cannot free the memory as the pool is not initialzied";
  // cout<<"to free ptr "<<ptr<<endl;
  cnmemStatus_t status = cnmemFree(ptr, NULL);
  // cout<<"done cnmemFree "<<endl;
  CHECK_EQ(status, cnmemStatus_t::CNMEM_STATUS_SUCCESS)
      << " " << cnmemGetErrorString(status);
  // cout<<"done status check"<<endl;
}

// ===========================================================================
void CudaMemPool::Malloc(void **ptr, const size_t size) {
  cudaError_t status = cudaMalloc(ptr, size);
  CHECK_EQ(status, cudaError_t::cudaSuccess);
}

void CudaMemPool::Free(void *ptr) {
  cudaError_t status = cudaFree(ptr);
  CHECK_EQ(status, cudaError_t::cudaSuccess);
}

//for SmartMemPool

///Section for structs and respective sorting function:
// PoolOptInfo, PoolBlockLifeTime, PoolOptSimplifiedInfo
struct PoolOptInfo{
    /*
     members: [ptr, size, operation_type, idx]
     */
    string ptr;
    size_t size;
    int operation_type;
    int idx;
    PoolOptInfo(string p, size_t s, int M, int i):ptr(p),size(s),operation_type(M),idx(i){}
};


struct sort_by_ptr_idx_ascending{
  /*
   sort PoolOptInfo by ptr and then idx.
   */
  inline bool operator() (const PoolOptInfo& struct1, const PoolOptInfo& struct2)
  {
    return ((struct1.ptr<struct2.ptr)||((struct1.ptr==struct2.ptr)&&(struct1.idx<struct2.idx)));
  }
};


struct PoolOptSimplifiedInfo{
    /*
     members: [idx, operation_type, size_delta]
     */
    size_t size_delta;// type as size_t in case size if large.
    int operation_type;
    int idx;
    PoolOptSimplifiedInfo(size_t s, int M, int i):size_delta(s),operation_type(M),idx(i){}
};


struct sort_by_itr_idx_ascending{
  /*
   sort PoolOptSimplifiedInfo by Idx.
   */
  inline bool operator() (const PoolOptSimplifiedInfo& struct1, const PoolOptSimplifiedInfo& struct2)
  {
    return (struct1.idx<struct2.idx);
  }
};


struct PoolBlockLifeTime{
    /*
     members: [name (r_idx), size, r_idx, d_idx]
     */
    int name;
    size_t size;
    int r_idx;
    int d_idx;
    PoolBlockLifeTime(int n,size_t s, int r,int d):name(n),size(s),r_idx(r),d_idx(d){}
};


struct sort_by_size_descending{
  /*
  sort PoolBlockLifeTime by descending size.
  */
  inline bool operator() (const PoolBlockLifeTime& struct1, const PoolBlockLifeTime& struct2)
  {
    return (struct1.size>struct2.size);
  }
};

struct sort_by_size_r_idx_descending{
  /*
  sort PoolBlockLifeTime by descending size and r_idx
  */
  inline bool operator() (const PoolBlockLifeTime& struct1, const PoolBlockLifeTime& struct2)
  {
    return ((struct1.size>struct2.size)||((struct1.size==struct2.size)&&(struct1.r_idx<struct2.r_idx)));
  }
};


vector<string> SplitString(string s, string delimiter) {
  /// string delimiter
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


vector<PoolOptInfo> PoolOptSeqStrToStruct(vector<string> vec, int &idx_range){
    /*
     convert vector of string into vector of PoolOptInfo, 
     sorted by ptr and then idx, and update idx_range to pieceMsgVec size.
     */
    vector<PoolOptInfo>vec_pool_opt_info;
    for (int i=0;i<vec.size();i++) {
        vector<string> v = SplitString(vec[i], " ");
        if (v[0]=="Malloc"){
            //convert v[2] from str to size_t
            size_t result;
            stringstream convert(v[2]);
            if (!(convert>>result)){
                result =-1;
                cout<<"error for converting size from str to int."<<endl;
            }
            PoolOptInfo tempMsg(v[1],result, 1, i);
            vec_pool_opt_info.push_back(tempMsg);
        }else if (v[0]=="Free"){
            PoolOptInfo tempMsg(v[1],-1, -1, i);
            vec_pool_opt_info.push_back(tempMsg);
        }else {
            cout<<"error for process the onePriceMsg."<<endl;
        }
    }
    
    sort(vec_pool_opt_info.begin(),vec_pool_opt_info.end(),sort_by_ptr_idx_ascending());
    idx_range = static_cast<int>(vec_pool_opt_info.size());

    return vec_pool_opt_info;
}


pair<vector<PoolBlockLifeTime>,vector<PoolBlockLifeTime>> PoolOptInfoToBlockLifeTime(vector<PoolOptInfo>vec_pool_opt_info, int idx_range){
  /*
  convert vector of opt info into vector of block life time
  return a pair of vectors: 1. normal blocks 2. cross-iteration blocks.
  */
  vector<PoolBlockLifeTime>vec_block_life_time1;
  vector<PoolBlockLifeTime>vec_block_life_time2;
  int i=0;
  
  //while loop processes a pair at each time, if got a pair.
  while (i<(vec_pool_opt_info.size()-1)){
    //condition: start with free. do nothing.
    if (vec_pool_opt_info[i].operation_type==-1){
        i+=1;
    }
    //condition: start with Malloc, next item same ptr and is free.
    if ((vec_pool_opt_info[i].operation_type==1)&& (vec_pool_opt_info[i+1].operation_type==-1)&&((vec_pool_opt_info[i].ptr==vec_pool_opt_info[i+1].ptr))){
      PoolBlockLifeTime temp_block_life_time(vec_pool_opt_info[i].idx,vec_pool_opt_info[i].size,vec_pool_opt_info[i].idx,vec_pool_opt_info[i+1].idx);
      vec_block_life_time1.push_back(temp_block_life_time);
      i+=2;
    }
    // condition: start with Malloc, no free.
    if ((vec_pool_opt_info[i].operation_type==1)&&(vec_pool_opt_info[i].ptr!=vec_pool_opt_info[i+1].ptr)){
      PoolBlockLifeTime temp_block_life_time(vec_pool_opt_info[i].idx,vec_pool_opt_info[i].size,vec_pool_opt_info[i].idx,idx_range);
      vec_block_life_time2.push_back(temp_block_life_time);
      i+=1;
    }
  }//end of while
  //condition: if still left with the last item
  if ((i<vec_pool_opt_info.size())&&(vec_pool_opt_info[i+1].operation_type==1)){
    PoolBlockLifeTime temp_block_life_time(vec_pool_opt_info[i].idx,vec_pool_opt_info[i].size,vec_pool_opt_info[i].idx,idx_range);
    vec_block_life_time2.push_back(temp_block_life_time);
    i+=1;
  }

  //sort both pair
  sort(vec_block_life_time1.begin(),vec_block_life_time1.end(),sort_by_size_r_idx_descending());
  sort(vec_block_life_time2.begin(),vec_block_life_time2.end(),sort_by_size_r_idx_descending());
  pair<vector<PoolBlockLifeTime>,vector<PoolBlockLifeTime>>pair_vec_block_life_time(vec_block_life_time1,vec_block_life_time2);
  
  return pair_vec_block_life_time;
}

///Section implementing coloring algorithm.
vector<pair<size_t, size_t>>  MergeColoredSegments(vector<pair<size_t,size_t>> vec_color_preoccupied){
  /*
  merge consecutive/overlapping segments of vec_color_preoccupied
  input:the collection of color ranges that is once occupied by some block during a block's life time.
  output: merged segments in ascending order.
  time complexity: O(n) for run, O(n^2) for verify section(optional), where n is size of vec_color_preoccupied.
  */
  sort(vec_color_preoccupied.begin(), vec_color_preoccupied.end());
  
  if(vec_color_preoccupied.size()<=1){
    return vec_color_preoccupied;
  }
  
  int m = 0;
  while (m<(vec_color_preoccupied.size()-1)){
    if ((vec_color_preoccupied[m].second +2)> vec_color_preoccupied[m+1].first){
      pair<int,int>tempItem(vec_color_preoccupied[m].first,max(vec_color_preoccupied[m].second,vec_color_preoccupied[m+1].second));
      //remove m+1 and m
      vec_color_preoccupied.erase(vec_color_preoccupied.begin()+m+1);
      vec_color_preoccupied.erase(vec_color_preoccupied.begin()+m);
      //insert the combined range
      vec_color_preoccupied.insert(vec_color_preoccupied.begin()+m,tempItem);
    }else{
        m+=1;
    }
  }//end of while loop
   
  return vec_color_preoccupied;
}


pair<size_t,size_t> FirstFitAllocation(vector<pair<size_t,size_t>> vec_color_merged,size_t size, size_t local_offset){
  /*
   First Fit weighted coloring
   return a pair standing for color_range.
   local_offset shifts the returned color_range, allowing multiple Plan().
   local_offset not changable, whereas offset is changable.
   */
  // condition: if no occupied, put after the local_offset
  if (vec_color_merged.size()==0){
    return pair<size_t,size_t>(0+local_offset,size-1+local_offset);
  }
  
  // condition: able to fit before first block, after the local_offset
  if ((size+local_offset)<(vec_color_merged[0].first+1)){
    return pair<size_t,size_t>(0+local_offset,size-1+local_offset);
  }
  
  size_t y_location= -1;
  if (vec_color_merged.size()>1) {
    int n = 0;
    while (n<(vec_color_merged.size()-1)){
      // condition: able to fit in between middle blocks.
      if ((vec_color_merged[n+1].first-vec_color_merged[n].second-1)>=size){
        y_location = vec_color_merged[n].second+1;
        break;
      }
      n+=1;
    }//end of while loop.
    // condition: allocate after the last block.
    if (y_location == -1){
      y_location = vec_color_merged[vec_color_merged.size()-1].second+1;
    }
  }// end of if loop, conditon C and D.
  
  // condition: colorMeger len =1, allocate after the last block.
  if (vec_color_merged.size()==1){
    y_location = vec_color_merged[0].second+1;
  }
  
  if (y_location==-1){
    cout<<"error in FirstFitAllocation!!!"<<endl;
  }
  
  return pair<size_t,size_t>(y_location,y_location+size-1);
}


pair<size_t,size_t> BestFitAllocation(vector<pair<size_t,size_t>> vec_color_merged,size_t size, size_t local_offset){
  /*
   Best Fit allocation, input and output same as FirstFitAllocation
  */
  // condition: if no occupied, put after the local_offset
  if (vec_color_merged.size()==0){
    return pair<size_t,size_t>(0+local_offset,size-1+local_offset);
  }
  //condition: if size=1, able to fit before the first block
  if ((vec_color_merged.size()==1)&&((size+local_offset)<(vec_color_merged[0].first+1))){
    return pair<size_t,size_t>(0+local_offset,size-1+local_offset);
  }
  
  //condition: lese of second condition
  if ((vec_color_merged.size()==1)&&((size+local_offset)>=(vec_color_merged[0].first+1))){
    return pair<size_t,size_t>(vec_color_merged[0].second+1,vec_color_merged[0].second+size);
  }
  
  size_t y_location=-1;
  pair<int, size_t>temp_hole(-1,-1); // n, hole size between n and n+1
  if (vec_color_merged.size()>1) {
    int n = 0;
    while (n<(vec_color_merged.size()-1)){
      // condition: able to fit in between middle blocks. select smallest.
      if (((vec_color_merged[n+1].first-vec_color_merged[n].second-1)>=size)&&((vec_color_merged[n+1].first-vec_color_merged[n].second-1)<temp_hole.second)){
        temp_hole.first=n;
        temp_hole.second=vec_color_merged[n+1].first-vec_color_merged[n].second-1;
      }
      n+=1;
    }//end of while loop.
    
    if(temp_hole.first==-1){
      // condition: allocate after the last block.
      y_location = vec_color_merged[vec_color_merged.size()-1].second+1;
    }else{
      //condition: best fit in the smallest hole.
      y_location = vec_color_merged[temp_hole.first].second+1;       
    }
  }// end of if loop, conditon D and E.
  
  if (y_location==-1){
    cout<<"error in BestFitAllocation!"<<endl;
  }
  
  return pair<size_t,size_t>(y_location,y_location+size-1);
}

vector<Vertex> AssignColorToVertices(vector<PoolBlockLifeTime> vec_block_life_time, size_t &offset,string color_method){
  /*
   color all or 1/2 vertices using MergeColoredSegments() and FirstFitAllocation(), with updated offset.
   time complexity: O(n^2).
  */
  size_t local_offset = offset; //feed into FirstFitAllocation, shall never change.
  int m = static_cast<int>(vec_block_life_time.size());
  //init all vertices
  vector<Vertex>vertices;
  for (int i=0; i<m;i++){
    Vertex temp_vertex(vec_block_life_time[i].name,vec_block_life_time[i].size,vec_block_life_time[i].r_idx,vec_block_life_time[i].d_idx);
    vertices.push_back(temp_vertex);
  }

  int **adj;
  adj = new int*[m];
  // build edges with values 1 and 0; combine with mergeSeg and FirstFitAllocation in the loop.
  for (int i=0; i<m;i++){
    adj[i] = new int[m];
    for (int j=0; j<m;j++){
      if ((max(vertices[i].r,vertices[j].r))<(min(vertices[i].d,vertices[j].d))){
        adj[i][j]=1;
        if (vertices[j].color_range.second){ //as second never be 0, if not empty.
          vertices[i].vec_color_preoccupied.push_back(vertices[j].color_range);
        }
      }
      else { adj[i][j]=0; }
    }
    
    vector<pair<size_t,size_t>>vec_color_merged = MergeColoredSegments(vertices[i].vec_color_preoccupied);
   
    if(color_method=="FF"){
      vertices[i].color_range = FirstFitAllocation(vec_color_merged,vertices[i].size, local_offset);
        
    }else{ //BF
      vertices[i].color_range = BestFitAllocation(vec_color_merged,vertices[i].size, local_offset);
    }

    //update of offset, largest memory footprint as well.
    if (vertices[i].color_range.second >=offset){
      offset = vertices[i].color_range.second+1;
    }
  }//end of for loop.
  
  return vertices;
}


pair<map<int,int>,map<int,int>> GetCrossIterationBlocks(vector<string>vec_double, int location_2nd_iteration, int iteration_length, int &double_range){
  ///get cross-iteration duration blocks
  vector<PoolOptInfo>vec_pool_opt_info2 = PoolOptSeqStrToStruct(vec_double,double_range);
  pair<vector<PoolBlockLifeTime>,vector<PoolBlockLifeTime>>pair_vec_block_life_time2=PoolOptInfoToBlockLifeTime(vec_pool_opt_info2,double_range);
  
  map<int,int>table_ridx_to_didx; //full duration info, cross-iteration duration.
  map<int,int>table_didx_to_ridx;
  for (int i=0;i<pair_vec_block_life_time2.first.size();i++){
    if(pair_vec_block_life_time2.first[i].r_idx<iteration_length){
      table_ridx_to_didx[pair_vec_block_life_time2.first[i].r_idx] =pair_vec_block_life_time2.first[i].d_idx%iteration_length;
      table_didx_to_ridx[pair_vec_block_life_time2.first[i].d_idx%iteration_length]=pair_vec_block_life_time2.first[i].r_idx;
    }
  }
  
  return pair<map<int,int>,map<int,int>>(table_ridx_to_didx,table_didx_to_ridx);
}


///Section of test functions.
vector<size_t> PoolOptSeqRepeatableTestPreProcess(pair<vector<PoolBlockLifeTime>,vector<PoolBlockLifeTime>>pair_vec_block_life_time){
  /*
  pre process pair of vector of block life time info, for ease of repeatable test.
  */
  vector<PoolBlockLifeTime>vec_block_life_time1 = pair_vec_block_life_time.first;
  vector<PoolBlockLifeTime>vec_block_life_time2 = pair_vec_block_life_time.second;
  vector<PoolOptSimplifiedInfo>vec_pool_opt_simplified_info;

  //process Malloc and Free pair, i.e. normal blocks
  for (int i =0; i<vec_block_life_time1.size(); i++){
    PoolOptSimplifiedInfo tempIterM(vec_block_life_time1[i].size,1,vec_block_life_time1[i].r_idx);
    vec_pool_opt_simplified_info.push_back(tempIterM);
    size_t temp_s_d = static_cast<size_t>(vec_block_life_time1[i].d_idx-vec_block_life_time1[i].r_idx);
    PoolOptSimplifiedInfo tempIterF(temp_s_d,-1,vec_block_life_time1[i].d_idx);
    vec_pool_opt_simplified_info.push_back(tempIterF);
  }
  
  //process Malloc-only blocks, i.e. cross-iteration blocks
  for (int i =0; i<vec_block_life_time2.size(); i++){
    PoolOptSimplifiedInfo tempIterM(vec_block_life_time2[i].size,1,vec_block_life_time2[i].r_idx);
    vec_pool_opt_simplified_info.push_back(tempIterM);
  }
  
  //sort then can create vec_rep.
  sort(vec_pool_opt_simplified_info.begin(),vec_pool_opt_simplified_info.end(),sort_by_itr_idx_ascending());
  vector<size_t>vec_rep; // vector of size_delta, name it as vec_rep for simlisity.
  for (int i =0; i<vec_pool_opt_simplified_info.size(); i++){
    vec_rep.push_back(vec_pool_opt_simplified_info[i].size_delta);
  }

  return vec_rep;
}


vector<size_t> PoolRepeatableTest(vector<size_t>rep, int idx_range, int &iteration_length, int &location_2nd_iteration){
  /*
  get max repeated non-overlapping Seg of a vector, return the repeated segment,
  update iteration_length, and location_2nd_iteration of where Seg starts to repeat.
  brtue force method using equal()
  time complexity O(n^2)
  */
  for (int i=0; i<idx_range;i++){
    for (int len=1; len<(idx_range-i);len++){
      if((equal(rep.begin()+i,rep.begin()+i-1+len,rep.begin()+i+len))&&(iteration_length<len)) {
        iteration_length = len;
        location_2nd_iteration = i;
      }
    }
  }
  //obtain sub_sequence based on iteration_length and location_2nd_iteration
  vector<size_t>sub_sequence(&rep[location_2nd_iteration],&rep[location_2nd_iteration+iteration_length]);
  if(!(equal(rep.begin()+location_2nd_iteration,rep.begin()+iteration_length-1+location_2nd_iteration,sub_sequence.begin()) && equal(rep.begin()+location_2nd_iteration+iteration_length,rep.begin()+2*iteration_length-1+location_2nd_iteration,sub_sequence.begin()))){
    cout<<"error in get the maxRep"<<endl;
  }

  return sub_sequence;
}


void VerifyRepeatableTest(vector<size_t>sub_sequence, int &iteration_length, int &location_2nd_iteration){
    /*
     to cut, in case the repeated Segment returned by PoolRepeatableTest contains multiple iterations.
    */
    int temp_iteration_length = 0;
    int temp_location_2nd_iteration = 0;
    int temp_idx_range = iteration_length;
    
    //verify by testing its subsequence again
    vector<size_t>tempsub_sequence = PoolRepeatableTest(sub_sequence,temp_idx_range,temp_iteration_length, temp_location_2nd_iteration);
    
    //tunable threshold.
    int threshold = 50;
    
    if (temp_iteration_length>threshold){
        iteration_length = temp_iteration_length;
        location_2nd_iteration += temp_location_2nd_iteration;
    }
}


///verify if coloring got overlapping
void OverlapVerification(vector<Vertex> vertices){
    size_t s = vertices.size();
    int i,j;
    for (i=0; i<s; i++){
        for (j=i+1; j<s; j++){
            if (((max(vertices[i].r,vertices[j].r))<(min(vertices[i].d,vertices[j].d)))&& ((max(vertices[i].color_range.first,vertices[j].color_range.first))<(1+min(vertices[i].color_range.second,vertices[j].color_range.second)))){
                cout<<"error overlapping"<<endl;
            }
        }
    }
}


SmartMemPool::SmartMemPool(const MemPoolConf &conf){
    color_method = "BF";
    conf_ = conf;
}

void SmartMemPool::Init(){
  mtx_.lock();
  if(!initialized_){
    initialized_ =true;
  }
  mtx_.unlock();
}



int SmartMemPool::Detection(vector<string>vec_string_test, int &iteration_length, int &location_2nd_iteration){
  /*
  Testing repeatability from raw operation sequence
  returns global_index_threshold, which is when flag shall be switched,
  update iteration_length and location_2nd_iteration of where the repeated Seg starts.
  */
  int idx_range_test=0;
  vector<PoolOptInfo>vec_pool_opt_info3 = PoolOptSeqStrToStruct(vec_string_test,idx_range_test);
  pair<vector<PoolBlockLifeTime>,vector<PoolBlockLifeTime>>pair_vec_block_life_time = PoolOptInfoToBlockLifeTime(vec_pool_opt_info3,idx_range_test);
  vector<size_t>vec_rep = PoolOptSeqRepeatableTestPreProcess(pair_vec_block_life_time);
  
  //repeatable test with verification
  vector<size_t>sub_sequence = PoolRepeatableTest(vec_rep,idx_range_test,iteration_length,location_2nd_iteration);
  VerifyRepeatableTest(sub_sequence, iteration_length, location_2nd_iteration);
  
  //update global_index_threshold if test past, i.e. iteration_length exceed certain threshold
  if (iteration_length>100){ //tunable threshold.
    global_index_threshold = idx_range_test+iteration_length-(idx_range_test-location_2nd_iteration)%iteration_length;
  }
  return global_index_threshold;
}


/// main run funtion
vector<Vertex> SmartMemPool::Plan(vector<string>vec, int &idx_range, size_t &offset, size_t &offset_cross_iteration,string color_method){
  /*
  Planning, i.e. Assign Color to Vertices from raw operation sequence info.
  input vector of strings, return colored vertices,
  update idx_range, offset.
  time complexity: O(n^2) where n is iteration_length.
  */

  vector<PoolOptInfo>vec_pool_opt_info = PoolOptSeqStrToStruct(vec,idx_range);
  pair<vector<PoolBlockLifeTime>,vector<PoolBlockLifeTime>>pair_vec_block_life_time=PoolOptInfoToBlockLifeTime(vec_pool_opt_info,idx_range);
  
  //coloring normal blocks and cross-iteration blocks separately, cannot be miss-matched.
  vector<PoolBlockLifeTime>vec_block_life_time1 = pair_vec_block_life_time.first;
  vector<PoolBlockLifeTime>vec_block_life_time2 = pair_vec_block_life_time.second;

  //color cross-iteration blocks
  vector<Vertex>vertices_2 = AssignColorToVertices(vec_block_life_time2,offset,color_method);

  for (int i=0; i<vertices_2.size();i++){
    vertices_2[i].cross_iteration = 1;
  }
  //update offset
  offset_cross_iteration = offset;
  offset = offset_cross_iteration*2;
  //color normal blocks
  vector<Vertex>vertices = AssignColorToVertices(vec_block_life_time1,offset,color_method);
  
  //merge after coloring
  vertices.insert(vertices.end(),vertices_2.begin(),vertices_2.end());

  return vertices;
}


///Malloc
void SmartMemPool::Malloc(void** ptr, const size_t size){
  /*
   1. switch flag when global_index == global_index_threshold, construct lookup table and malloc the whole pool.
   2. if flag=0, malloc/cudaMalloc, collect vec string
   3. if flag=1, look up table, malloc/cudaMalloc if not in the Table
   4. test repeated sequence every 100 blocks, update global_index_threshold.
   */

  if (!initialized_){
    Init();
  }

  void* allocated_ptr = NULL; //ptr to be returned

  /// 1. switch flag when global_index == global_index_threshold, construct lookup table and malloc the whole pool.    
  if (global_index == global_index_threshold){

    malloc_flag = 1;
    vector<string>vec_raw_opt_info(&vec[location_2nd_iteration],&vec[location_2nd_iteration+iteration_length]);
    
    //color vertices
    vector<Vertex>vertices = Plan(vec_raw_opt_info,idx_range,offset,offset_cross_iteration,color_method);

    //here to verify if the coloring got overlapping. for verify purpose only.
    //OverlapVerification(vertices);
    
    //obtain the cross-iteration duration info
    int double_range=0;
    vector<string>vec_double(&vec[location_2nd_iteration],&vec[location_2nd_iteration+2*iteration_length]);
    pair<map<int,int>,map<int,int>>pairs =GetCrossIterationBlocks(vec_double,location_2nd_iteration,iteration_length,double_range);
    table_ridx_to_didx = pairs.first;
    table_didx_to_ridx = pairs.second;
    
    //make pool
    cudaMalloc(&ptr_pool,offset); //poolSize or memory foot print  offset.

    //make vec_block_meta for lookup purpose after pool is constructed
    for (int i=0; i<idx_range; i++){
        PoolBlockMeta tempElement;
        vec_block_meta.push_back(make_pair(i,tempElement));
    }
    for (int i=0; i<vertices.size(); i++){
        PoolBlockMeta temp;
        temp.r_idx =vertices[i].r;
        temp.d_idx =table_ridx_to_didx.find(vertices[i].r)->second;
        temp.size =vertices[i].size;
        temp.offset=vertices[i].color_range.first;
        temp.ptr = (void*)((char*)ptr_pool+temp.offset*sizeof(char));
        temp.occupied =0;
        temp.cross_iteration = vertices[i].cross_iteration;
        temp.occupied_backup =0; 
        //build tables for lookup.
        vec_block_meta[vertices[i].r].second= temp;
    }
  }
  ///  2. if flag=0, malloc/cudaMalloc, accumulate vec_info at the beginning iterations.
  if(malloc_flag ==0){    
    cudaMalloc(ptr, size);
    allocated_ptr = *ptr;
    //update load
    if(load_flag==1){
      if (global_index>0){
        table_load[global_index]=make_pair(table_load.find(global_index-1)->second.first+size,table_load.find(global_index-1)->second.second);
      }else{ //very first block
        table_load[global_index]=make_pair(size,0);
      }
    }
    //push_back the string for later test and run.
    string temp_str1 ="Malloc ";
    stringstream strm2;
    strm2<<allocated_ptr;
    string temp_str2 = strm2.str();
    stringstream strm3;
    strm3<<size;
    string temp_str3 = strm3.str();
    string temp = temp_str1+temp_str2+" "+temp_str3;
    vec.push_back(temp);
  }else{

    /// 3. if flag=1, look up table.
    int lookup_idx = (global_index-location_2nd_iteration)%iteration_length;
    if ((vec_block_meta[lookup_idx].second.size ==size)&&(vec_block_meta[lookup_idx].second.occupied*vec_block_meta[lookup_idx].second.occupied_backup==0)){
     if (vec_block_meta[lookup_idx].second.occupied==0){
        //condition: normal and cross_iteration's primary.
        //assign ptr and mark as occupied, and add in ptr2rIdx
        allocated_ptr = vec_block_meta[lookup_idx].second.ptr;
        vec_block_meta[lookup_idx].second.occupied= 1;
        table_ptr_to_ridx[allocated_ptr]=lookup_idx;                
        //update load
        if(load_flag==1){
          table_load[global_index]=make_pair(table_load.find(global_index-1)->second.first,table_load.find(global_index-1)->second.second+size);
        }
      }else if ((vec_block_meta[lookup_idx].second.cross_iteration==1) && (vec_block_meta[lookup_idx].second.occupied==1) && (vec_block_meta[lookup_idx].second.occupied_backup ==0)) {
        //condition: cross_iteration's backup
        allocated_ptr = (void*)((char*)vec_block_meta[lookup_idx].second.ptr+offset_cross_iteration*sizeof(char));
        vec_block_meta[lookup_idx].second.occupied_backup=1;
        table_ptr_to_ridx[allocated_ptr]=lookup_idx;
        //update load
        if(load_flag==1){
          table_load[global_index]=make_pair(table_load.find(global_index-1)->second.first,table_load.find(global_index-1)->second.second+size);
        }
      }
    }else{  
      //condition: size not proper or both occupied.
      cudaMalloc(ptr, size);
      allocated_ptr = *ptr;       
      //update load
      if(load_flag==1){
        table_load[global_index]=make_pair(table_load.find(global_index-1)->second.first+size,table_load.find(global_index-1)->second.second);
      }
    } 
  } //end of loop for flag=1
    
  ///4. test repeated sequence every 300 index, update global_index_threshold.
  if (((global_index+1)%300==0) && (malloc_flag ==0) && (global_index_threshold==-1)&&(global_index+2>check_point)){
    global_index_threshold = Detection(vec,iteration_length,location_2nd_iteration);
    check_point=check_point*2;
  }
    
  ///get load info, when global_index == global_index+2iteration_length
  if (global_index==(global_index_threshold+2*iteration_length)&& (global_index_threshold>0)){
    GetMaxLoad();
    load_flag=0;
  }
    
  global_index++;
  //update it for load tracking purpose.
  table_ptr_to_size[allocated_ptr]=size; 
 
  //update *ptr
  *ptr = allocated_ptr;
  
  ///update block_RWMF
  string temp_str1 ="Malloc ";
  stringstream strm2;
  strm2<<allocated_ptr;
  string temp_str2 = strm2.str();
  stringstream strm3;
  strm3<<size;
  string temp_str3 = strm3.str();
  string temp = temp_str1+temp_str2+" "+temp_str3;
  int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  stringstream strm4;
  strm4<<now;
  string temp_str4 = strm4.str();
  temp = temp+" "+temp_str4;
  vec_block_rw_mf.push_back(temp);
}

///Free
void SmartMemPool::Free(void* ptr){
    
  size_t deallocatedSize = table_ptr_to_size.find(ptr)->second;
  
  /// at the begining iterations, via cudaFree, accumulate opt info.  
  if ((global_index_threshold==-1)||(global_index<global_index_threshold)){
    //push_back the string for later test and run.
    string temp_str1 ="Free ";
    stringstream strm2;
    strm2<<ptr;
    string temp_str2 = strm2.str();
    string temp = temp_str1+temp_str2;
    vec.push_back(temp);
    
    //update load before free
    if(load_flag==1){
      table_load[global_index]=make_pair(table_load.find(global_index-1)->second.first-deallocatedSize,table_load.find(global_index-1)->second.second);
    }
    // before flag switch, for sure all free shall be done by free()
    cudaFree(ptr);
  }else{
    /// cases that no need accumulating opt info

    /// free a ptr that is in the memory pool
    if (!(table_ptr_to_ridx.find(ptr)==table_ptr_to_ridx.end())){
      int resp_rIdx = table_ptr_to_ridx.find(ptr)->second;
      table_ptr_to_ridx.erase(ptr);
      
      if (ptr == vec_block_meta[resp_rIdx].second.ptr){
        vec_block_meta[resp_rIdx].second.occupied =0; //freed, able to allocate again.
      }else if (ptr == (void*)((char*)vec_block_meta[resp_rIdx].second.ptr+offset_cross_iteration*sizeof(char))){
        vec_block_meta[resp_rIdx].second.occupied_backup =0;
      } else{
        if (((float)((char*)ptr-((char*)ptr_pool+offset_cross_iteration*sizeof(char)))>0) && ((float)((char*)ptr-((char*)ptr_pool+2*offset_cross_iteration*sizeof(char)))<0)){
          vec_block_meta[resp_rIdx].second.occupied_backup =0;
        }else{
          vec_block_meta[resp_rIdx].second.occupied =0;
        }
      }
      //update load
       if(load_flag==1){
           table_load[global_index]=make_pair(table_load.find(global_index-1)->second.first,table_load.find(global_index-1)->second.second-deallocatedSize);
       }
    }else{
      /// free a ptr that is NOT in the memory pool
      
      //update load
      if(load_flag==1){
          table_load[global_index]=make_pair(table_load.find(global_index-1)->second.first-deallocatedSize,table_load.find(global_index-1)->second.second);
      }
      cudaFree(ptr);
    }
            
  }

  global_index++;

  ///update block_RWMF
  string temp_str1 ="Free ";
  stringstream strm2;
  strm2<<ptr;
  string temp_str2 = strm2.str();
  string temp = temp_str1+temp_str2;
  int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  stringstream strm4;
  strm4<<now;
  string temp_str4 = strm4.str();
  temp = temp+" "+temp_str4;
  vec_block_rw_mf.push_back(temp);
}//end of Free.


SmartMemPool::~SmartMemPool(){

  fstream file_block1("blockInfo_RW.text", ios::in|ios::out|ios::app);
  fstream file_block2("blockInfo_RWMF.text", ios::in|ios::out|ios::app);
  for (int i=0; i< vec_block_rw.size();i++){
    file_block1<<vec_block_rw[i]<<endl;
  }
  for (int i=0; i< vec_block_rw_mf.size();i++){
    file_block2<<vec_block_rw_mf[i]<<endl;
  }
  cudaFree(ptr_pool);

}

void SmartMemPool::GetMaxLoad(){
    
  vector<size_t>vec_load_log;
  for (int i=0; i<table_load.size();i++){
      vec_load_log.push_back(table_load.find(i)->second.first);
  }
  size_t max_cuda_load = *max_element(vec_load_log.begin(),vec_load_log.end());
  int idx_max_cuda_load = static_cast<int>(distance(vec_load_log.begin(),max_element(vec_load_log.begin(),vec_load_log.end())));
  
  vector<size_t>vec_color_load;
  for (int i=0; i<table_load.size();i++){
      vec_color_load.push_back(table_load.find(i)->second.second);
  }
  size_t max_color_load = *max_element(vec_color_load.begin(),vec_color_load.end());
  int idx_max_color_load = static_cast<int>(distance(vec_color_load.begin(),max_element(vec_color_load.begin(),vec_color_load.end())));
  size_t offset_color_load = table_load.find(idx_max_color_load)->second.first;
  
  max_total_load = max(max_cuda_load,max_color_load+offset_color_load);
  max_mem_usage = max(max_cuda_load,offset+offset_color_load);
  
}

std::pair<size_t, size_t> SmartMemPool::GetMemUsage() {
  //note here the pair is different from that of CnMemPool.
  return std::make_pair(max_mem_usage, max_total_load);
}
    
void SmartMemPool::Append(string blockInfo) {
  vec_block_rw.push_back(blockInfo);
  vec_block_rw_mf.push_back(blockInfo);
}

///SwapPool
SwapPool::SwapPool(const MemPoolConf &conf){
    conf_ = conf;
}

void SwapPool::Init(){

  mtx_.lock();
  if(!initialized_){
    initialized_ =true;
  }
  mtx_.unlock();
}


void SwapPool::PoolOpt(vector<string> &vec_mf) {

  vector<PoolOptInfo>vec_pool_opt_info;
  iteration_length_mf = vec_mf.size()/3; //cos input vec_mf is of 3 iteration 

  //convert raw opt info into struct: PoolOptInfo
  for (int i = 0;i < vec_mf.size();i++){
    vector<string> v = SplitString(vec_mf[i], " ");

    if (v[0]=="Malloc"){
      size_t result;
      stringstream convert(v[2]);
      if (!(convert>>result)){
        result = -1;
        cout<<"error for converting size from str to int."<<endl;
      }
      PoolOptInfo tempMsg(v[1],result, 1, i-iteration_length_mf);
      vec_pool_opt_info.push_back(tempMsg);
    }else if (v[0]=="Free"){
      PoolOptInfo tempMsg(v[1],-1, -1, i-iteration_length_mf);
      vec_pool_opt_info.push_back(tempMsg);
    }else {
      cout<<"error for process the onePriceMsg."<<endl;
    }
  }
  //sort by ptr and then idx
  sort(vec_pool_opt_info.begin(),vec_pool_opt_info.end(),sort_by_ptr_idx_ascending());
  
  //convert into block lifetime
  vector<PoolBlockLifeTime>vec_block_life_time;
  int i = 0;

  while (i<(vec_pool_opt_info.size()-1)){
    
    if (vec_pool_opt_info[i].operation_type==-1){
      //condition: start with free. do nothing.
      i+=1;
    } else {
      if ((vec_pool_opt_info[i].operation_type==1)&& (vec_pool_opt_info[i+1].operation_type==-1)
        &&((vec_pool_opt_info[i].ptr==vec_pool_opt_info[i+1].ptr))){
        //condition: start with Malloc, next item same ptr and is free.
        if ((vec_pool_opt_info[i].idx >=0 && vec_pool_opt_info[i].idx <iteration_length_mf)
          ||(vec_pool_opt_info[i+1].idx >=0 && vec_pool_opt_info[i+1].idx <iteration_length_mf)){
          //condition: at least one of the index in range [0,iteration_length_mf]
          PoolBlockLifeTime temp_block_life_time(vec_pool_opt_info[i].idx,vec_pool_opt_info[i].size,vec_pool_opt_info[i].idx,vec_pool_opt_info[i+1].idx);
          vec_block_life_time.push_back(temp_block_life_time); 
        }
        i+=2; //no matter in the middle iteration or not, plus 2.
      } else {
        //condiction: not one pair, Malloc-only block, no free..
        i+=1;
      }
    } 
  }
  sort(vec_block_life_time.begin(),vec_block_life_time.end(),sort_by_size_r_idx_descending());


  ///get E, V of the blocks， coloring
  //V
  int m = static_cast<int>(vec_block_life_time.size());
  vector<Vertex>vertices;
  for (int i=0; i<m;i++){
    Vertex temp_vertex(vec_block_life_time[i].name,vec_block_life_time[i].size,vec_block_life_time[i].r_idx,vec_block_life_time[i].d_idx);
    vertices.push_back(temp_vertex);
  }

  //E and coloring  
  int offset = 0;
  int **adj;
  adj = new int*[m];

  // build edges with values 1 and 0; combine with mergeSeg and FirstFitAllocation in the loop.
  for (int i=0; i<m;i++){
    adj[i] = new int[m];
    for (int j=0; j<m;j++){
      if ((max(vertices[i].r,vertices[j].r))<(min(vertices[i].d,vertices[j].d))
        || (min(vertices[i].d,vertices[j].d)<0 && 
        min(vertices[i].d,vertices[j].d)+2*iteration_length_mf< max(vertices[i].r,vertices[j].r))){
        adj[i][j]=1;
        if (vertices[j].color_range.second){ //as second never be 0, if not empty.
          vertices[i].vec_color_preoccupied.push_back(vertices[j].color_range);
        }
      }
      else { 
        adj[i][j]=0; 
      }
    }
    
    vector<pair<size_t,size_t>>vec_color_merged = MergeColoredSegments(vertices[i].vec_color_preoccupied);

    // vertices[i].color_range = FirstFitAllocation(vec_color_merged,vertices[i].size, local_offset);
    vertices[i].color_range = BestFitAllocation(vec_color_merged,vertices[i].size, offset);

    //update of offset, largest memory footprint as well.
    if (vertices[i].color_range.second >=offset){
      offset = vertices[i].color_range.second+1;
    }
  }//end of for loop.

  //delete adj, the edges
  for (int i=0; i<m;i++){
    delete[] adj[i]; 
  }
  delete[] adj;

  //make pool
  cudaMalloc(&ptr_pool,offset); //poolSize or memory foot print  offset.

  //make table
  for (int i=0; i<vertices.size();i++){
    PoolBlockMeta itm;
    itm.r_idx = vertices[i].r;
    itm.d_idx = vertices[i].d;
    itm.size = vertices[i].size;
    itm.offset = vertices[i].color_range.first;
    itm.ptr = (void*)((char*)ptr_pool+itm.offset*sizeof(char));
    itm.occupied = 0;
    table_pool_meta[vertices[i].r] = itm;
  }
  pool_flag = 1;
    
}

void SwapPool::Malloc(void** ptr, const size_t size){
  
  void* allocated_ptr =nullptr;
  
  if (pool_flag == 0) {
    cudaError_t status = cudaMalloc(ptr, size);
    CHECK_EQ(status, cudaError_t::cudaSuccess);
  } else {
    //pool_flag = 1 
    if (pool_index < iteration_length_mf){
      if ((table_pool_meta.find(pool_index - iteration_length_mf) == table_pool_meta.end()) || (!(size == table_pool_meta.find(pool_index - iteration_length_mf)->second.size))){
        //not in table of negative r_idx
        cudaError_t status = cudaMalloc(ptr, size);
        CHECK_EQ(status, cudaError_t::cudaSuccess);
      } else{
        //in the table of negative r_idx
        auto temp_meta = table_pool_meta.find(pool_index - iteration_length_mf)->second;
        allocated_ptr = temp_meta.ptr;
        *ptr = allocated_ptr;
        table_ptr_to_ridx[allocated_ptr]=pool_index - iteration_length_mf; 

      }
    } else{
      //8 9 10th iteration
      int r_pool_index = pool_index%iteration_length_mf;
      if ((table_pool_meta.find(r_pool_index) == table_pool_meta.end()) || (!(size == table_pool_meta.find(r_pool_index)->second.size))){
        //not here, should be abnormal
        cudaError_t status = cudaMalloc(ptr, size);
        CHECK_EQ(status, cudaError_t::cudaSuccess);
      } else{
        //in the table
        auto temp_meta = table_pool_meta.find(r_pool_index)->second;
        allocated_ptr = temp_meta.ptr;
        *ptr = allocated_ptr;
        table_ptr_to_ridx[allocated_ptr]=r_pool_index; 
      }
    }
  }

    pool_index++;     
  }


void SwapPool::Free(void *ptr) {
  if (pool_flag == 0){
    cudaError_t status = cudaFree(ptr);
    CHECK_EQ(status, cudaError_t::cudaSuccess);
  } else{
    if (table_ptr_to_ridx.find(ptr)==table_ptr_to_ridx.end()){
      cudaError_t status = cudaFree(ptr);
      CHECK_EQ(status, cudaError_t::cudaSuccess);
    }
  }

}

void SwapPool::Append(string blockInfo) {
  //NA
}


void GetMaxLoad (){
  //empty
}

std::pair<size_t, size_t> SwapPool::GetMemUsage() {
  //empty
  return std::make_pair(0, 0);
}

SwapPool::~SwapPool(){
  //NA  
}

}

#endif

#endif