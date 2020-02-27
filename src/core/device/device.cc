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

#include "singa/core/device.h"

namespace singa {
Device::Device(int id, int num_executors)
    : id_(id), num_executors_(num_executors) {
  // TODO(wangwei) create scheduler and vm.
  host_ = defaultDevice;
}

void Device::Exec(function<void(Context*)>&& fn,
                  const vector<Block*> read_blocks,
                  const vector<Block*> write_blocks, bool use_rand_generator) {
  // buffer_flag_ = false;
  if (buffer_flag_== true)
  {
    size_t op_index = buffOps.size();
    // printf("EnterEXEC, %d\n", buffOps.size());
    buffOps.push_back(fn);
    printf("OP[%d]:\n", buffOps.size());

    printf("\tRead:\n");
    for (size_t i = 0; i < read_blocks.size(); ++i) {
      auto blk = read_blocks[i];
      printf("\t\t[%d]: %#x\n", i, blk->mutable_data()); 
      auto it1 = blk2dst_.find(blk);
      if (it1 != blk2dst_.end()) {
	it1->second.push_back(op_index);
      } else {
	blk2dst_[blk] = {op_index};
	tensors_.push_back(blk);
      }

      auto it2 = indegree_.find(blk);
      if (it2 == indegree_.end()) {
	indegree_[blk] = 0;
      } else {
	if (it2->second == 3) {
	  indegree_[blk] = 2;
	}
      }
    }

    printf("\tWrite:\n");
    for (size_t i = 0; i < write_blocks.size(); ++i) {
      auto blk = write_blocks[i];
      printf("\t\t[%d]: %#x\n", i, blk->mutable_data());

      auto it = indegree_.find(blk);
      if (it != indegree_.end()) {
	if (it->second == 0) {
	  it->second = 1;
	}
      } else {
	indegree_[blk] = 2;
      }

      auto iter = blk2dst_.find(blk);
      if (iter == blk2dst_.end() && indegree_[blk] == 2) {
	blk2dst_[blk] = {};
	tensors_.push_back(blk);
	indegree_[blk] = 3;
      }
    }

    src2rblk_[op_index] = std::move(read_blocks);
    src2blk_[op_index] = std::move(write_blocks);
  }
  else
  {
    // printf("immediately ops\n");
    DoExec(std::move(fn), 0);
  }
}

void Device::ExecBuffOps() {
  bool previous_state = buffer_flag_;
  buffer_flag_ = false;
  for (size_t i = 0; i < buffOps.size(); ++i) {
    // printf("EnterBuffStart, %d\n", i);
    DoExec(std::move(buffOps[i]), 0);
    // buffOps.erase(buffOps.begin());
    // printf("EnterBuffExit\n");
  }
  buffer_flag_ = previous_state;

  for (auto it = src2blk_.begin(); it != src2blk_.end(); ++it) {
    printf("OP[%2d]: ", it->first);
    printf("Outputs: ");
    for (size_t i = 0; i < it->second.size(); ++i) {
      printf("%#x\t", it->second[i]);
    }
    printf("\n");
  }
  for (size_t i = 0; i < tensors_.size(); ++i) {
    auto blk = tensors_[i];
    printf("Blocks[%2d]: addr[%#x] ", i, blk);
    if (indegree_[blk] == 0) {
      printf("type[input] ");
    } else if (indegree_[blk] == 1) {
      printf("type[param] ");
    } else if (indegree_[blk] == 2) {
      printf("type[inter] ");
    } else if (indegree_[blk] == 3){
      printf("type[_end_] ");
    }
    printf("OPs: \t");
    for (auto it : blk2dst_[blk]) {
      printf("%2d\t", it);
    }
    printf("\n");
  }
  /*
  for (auto it = blk2dst_.begin(); it != blk2dst_.end(); ++it) {
    printf("Blocks[%#x]: ", it->first);
    if (indegree_[it->first] == 0) {
      printf("type[input] ");
    } else if (indegree_[it->first] == 1) {
      printf("type[param] ");
    } else if (indegree_[it->first] == 2) {
      printf("type[inter] ");
    } else if (indegree_[it->first] == 3){
      printf("type[_end_] ");
    }
    printf("OPs: ");
    for (size_t i = 0; i < it->second.size(); ++i) {
      printf("%d\t", it->second[i]);
    }
    printf("\n");
  }
  */
  
  SafeQueue<int> op_queue;
  std::vector<int>  op_ref;
  std::vector<bool> op_exec;
  std::map<Block *, int> blk_ref;
  std::map<Block *, bool> blk_state;
  std::map<Block *, int> blk_index;
  for (int i = 0; i < src2blk_.size(); ++i) {
    op_ref.push_back(0);
    op_exec.push_back(false);
  }
  for (auto it : tensors_) {
    blk_ref[it] = 0;
    int type = indegree_[it];
    blk_state[it] = type == 0 || type == 1;
    blk_index[it] = 0;
  }
  for (auto it1 : blk2dst_) {
    blk_ref[it1.first] = it1.second.size();
    if (!blk_state[it1.first]) {
      for (auto it2 : it1.second) {
	op_ref[it2] += 1;
      }
    }
  }

  printf("finished init\n");
  for (auto it : tensors_) {
    printf("blk[%#x]: %d\n", it, blk_state[it]);
  }

  for (size_t i = 0; i < op_ref.size(); ++i) {
    printf("op[%d]: %d\n", i, op_ref[i]);
  }

  std::vector<int> temp = op_ref;
  std::vector<int> ans;
  for (int i = 0; i < op_ref.size(); ++i) {
    if (temp[i] == 0) {
      op_queue.Push(i);
      ans.push_back(i);
      op_exec[i] = true;
      printf("push op[%d]\n", i);
      for (auto it1 : src2blk_[i]) {
	blk_state[it1] = true;
	for (int j = blk_index[it1]; blk_state[it1] && j < blk2dst_[it1].size(); ++j) {
	  int dstIndex = blk2dst_[it1][j];
	  for (auto it2 : src2blk_[dstIndex]) {
	    if (it2 == it1) {
	      printf("in-place operation\n");
	      blk_state[it1] = false;
	      break;
	    }
	  }
	  op_ref[dstIndex] -= 1;
	  printf("sub op[%d] ref[%d]\n", dstIndex, op_ref[dstIndex]);
	}
      }
    }
  }

  printf("start dag-sort\n");
  int de_count = 0;
  std::vector<std::vector<int> > anss;
  anss.push_back(ans);
  while (op_queue.Size()) {
    // step 1: pop the first element and execute the operation
    int curIndex = -1;
    op_queue.Pop(curIndex);
    printf("pop op[%d]\n", curIndex);
    DoExec(std::move(buffOps[curIndex]), 0);

    // step 2: decrease the ref count of the input tensors
    for (auto it : src2rblk_[curIndex]) {
      blk_ref[it] -= 1;
      if (blk_ref[it] == 0) {
	printf("deallocate block[%#x]\n", it);
	de_count++;
      }
    }

    // step 3: decrease the ref count of operations that use the output tensor as input tensor
    std::vector<int> ans;
    for (auto it : src2blk_[curIndex]) {
      blk_state[it] = true;
      for (int i = 0; blk_state[it] && i < blk2dst_[it].size(); ++i) {
	int dstIndex = blk2dst_[it][i];
	printf("dstIndex %d[%d]\n", dstIndex, op_ref[dstIndex]);
	if (op_ref[dstIndex] == 0 && op_exec[dstIndex] == false) {
	  op_queue.Push(dstIndex);
	  ans.push_back(dstIndex);
	  op_exec[dstIndex] = true;
	  printf("push op[%d]\n", dstIndex);
	  for (auto t : src2blk_[dstIndex]) {
	    if (t != it)  blk_state[t] = true;
	    printf("update blk[%#x]\n", t);
	    for (int j = 0; blk_state[t] && j < blk2dst_[t].size(); ++j) {
	      int dstIndex = blk2dst_[t][j];
	      if (op_exec[dstIndex]) {
		printf("op[%d] has executed\n", dstIndex);
		continue;
	      }
	      for (auto t1 : src2blk_[dstIndex]) {
		if (t1 == t) {
		  printf("in-place operation\n");
		  blk_state[t1] = false;
		  break;
		}
	      }
	      op_ref[dstIndex] -= 1;
	      printf("sub op[%d] ref[%d]\n", dstIndex, op_ref[dstIndex]);
	    }
	  }
	}
      }
    }
    if (!ans.empty()) anss.push_back(ans);
  }

  printf("total released count: %d/%d(without tensor of type _end_)\n", de_count, tensors_.size());
  for (int i = 0; i < anss.size(); i++) {
    printf("group %d: ", i);
    for (auto it : anss[i]) {
      printf("%d ", it);
    }
    printf("\n");
  }
}

// Todo(Wangwei) Get Block From The Memory manager
Block* Device::NewBlock(int size) {
  CHECK_GE(size, 0)
      << "size is negative, could be caused by the type cast "
      << "from size_t to int. In that case, the size is too large.";
  if (size > 0) {
    void* ptr = Malloc(size);
    return new Block(ptr, size);
  } else {
    return nullptr;
  }
}

// TODO(wangwei) return Block to the memory manager
void Device::FreeBlock(Block* block) {
  if (block != nullptr) {
    Free(block->mutable_data());
    delete block;
  }
}

void Device::CopyDataToFrom(Block* dst, Block* src, size_t nBytes,
                            CopyDirection direct, int dst_offset,
                            int src_offset) {
  this->Exec(
      [this, dst, src, nBytes, direct, dst_offset, src_offset](Context* ctx) {
        this->CopyToFrom(
            reinterpret_cast<char*>(dst->mutable_data()) + dst_offset,
            reinterpret_cast<const char*>(src->data()) + src_offset, nBytes,
            direct, ctx);
      },
      {src}, {dst});
}

void Device::CopyDataFromHostPtr(Block* dst, const void* src, size_t nBytes,
                                 size_t dst_offset) {
  auto direct = lang_ == kCpp ? kHostToHost : kHostToDevice;
  void* dstptr = reinterpret_cast<char*>(dst->mutable_data()) + dst_offset;
  Exec([this, dstptr, src, nBytes,
        direct](Context* ctx) { CopyToFrom(dstptr, src, nBytes, direct, ctx); },
       {}, {dst});
}
void Device::Sync() {}
}  // namespace singa
