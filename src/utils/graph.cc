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

#include "singa/utils/graph.h"

#include <glog/logging.h>
#include <algorithm>
#include <queue>
#include <unordered_set>
#include "singa/utils/common.h"

namespace singa {

using std::map;
using std::string;
using std::vector;

/**************************************************************************
 * Implementation for Node class
 *************************************************************************/
Node::Node(string name) {
  this->name = name;
}
Node::Node(string name, const std::map<string, string>& attrs) {
  this->name = name;
  this->attrs = attrs;
}

Node::Node(const string& name, const string& origin, int id, void* proto) {
  this->name = name;
  this->origin = origin;
  this->proto = proto;
  this->partition_id = id;
}

void Node::AddDstNode(Node* dstnode) {
  dstnodes.push_back(dstnode);
}

void Node::AddSrcNode(Node* srcnode) {
  srcnodes.push_back(srcnode);
}

void Node::RemoveDstNode(Node* dst) {
  auto iter = dstnodes.begin();
  while ((*iter)->name != dst->name && iter != dstnodes.end())
    iter++;
  CHECK_STREQ((*iter)->name.c_str(), dst->name.c_str());
  dstnodes.erase(iter);
}

void Node::RemoveSrcNode(Node* src) {
  auto iter = srcnodes.begin();
  while ((*iter)->name != src->name && iter != srcnodes.end())
    iter++;
  CHECK_STREQ((*iter)->name.c_str(), src->name.c_str());
  srcnodes.erase(iter);
}

/****************************************************************************
 * Implementation for Graph class
 ****************************************************************************/

Graph::~Graph() {
  for (Node* node : nodes_)
    delete node;
}

Node* Graph::AddNode(const string& name, const string& origin, int id,
                    void* proto) {
  Node* node = new Node(name, origin, id, proto);
  nodes_.push_back(node);
  CHECK(name2node_.find(node->name) == name2node_.end())
    << "node " << node->name << " already exists";
  name2node_[node->name] = node;
  return node;
}

Node* Graph::AddNode(const string& name,
                     const std::map<string, string>& attrs) {
  Node* node = new Node(name, attrs);
  nodes_.push_back(node);
  CHECK(name2node_.find(node->name) == name2node_.end())
    << "node " << node->name << " already exists";
  name2node_[node->name] = node;
  return node;
}

void Graph::AddEdge(Node* srcnode, Node* dstnode) {
  srcnode->AddDstNode(dstnode);
  dstnode->AddSrcNode(srcnode);
}

void Graph::AddEdge(const string& src, const string& dst) {
  auto srcnode = name2node_.find(src);
  CHECK(srcnode != name2node_.end()) << "can't find src node " << src;
  auto dstnode = name2node_.find(dst);
  CHECK(dstnode != name2node_.end()) << "can't find dst node " << dst;
  AddEdge(srcnode->second, dstnode->second);
}
void Graph::AddEdge(Node* srcnode, Node* dstnode,
      const std::map<string, string>& attrs) {
  AddEdge(srcnode, dstnode);
  edge_attrs_[GetEdgeName(srcnode->name, dstnode->name)] = attrs;
}
void Graph::AddEdge(const string& src, const std::string& dst,
      const std::map<string, string>& attrs) {
  AddEdge(src, dst);
  edge_attrs_[GetEdgeName(src, dst)] = attrs;
}

void Graph::RemoveEdge(Node* src, Node* dst) {
  src->RemoveDstNode(dst);
  dst->RemoveSrcNode(src);
}

void Graph::RemoveEdge(const string &src, const string& dst) {
  auto srcnode = name2node_.find(src);
  CHECK(srcnode != name2node_.end()) << "can't find src node " << src;
  auto dstnode = name2node_.find(dst);
  CHECK(dstnode != name2node_.end()) << "can't find dst node " << dst;
  RemoveEdge(srcnode->second, dstnode->second);
}

// sort to make `bottom' nodes be placed in the front positions
void Graph::Sort() {
  // nodes to be visited
  std::queue<Node*> visiting_nodes;
  // visited node set
  std::unordered_set<Node*> visited_set;
  // visiting_nodes + visted_set
  std::unordered_set<Node*> visit_set;;
  for (auto node : nodes_) {
    // visit nodes without source nodes firstly
    if (node->srcnodes.size() == 0) {
      visiting_nodes.push(node);
      visit_set.insert(node);
    }
  }
  int n = nodes_.size();
  nodes_.clear();
  while (!visiting_nodes.empty()) {
    auto node = visiting_nodes.front();
    visiting_nodes.pop();
    bool visit = true;
    bool bi_direction = false;
    // check if a node has a bi-direction edge with its neighbour
    for (auto src : node->srcnodes)
      for (auto src_of_src : src->srcnodes)
        if (strcmp((src_of_src->name).c_str(), (node->name).c_str()) == 0) {
          bi_direction = true;
          break;
        }
    // check whether its src nodes number greater than 1
    if (bi_direction && (node->srcnodes).size() > 1) {
        auto src = node->srcnodes.at(0);
        if (visited_set.find(src) == visited_set.end()) {
          visit = false;
        }
    } else {
      for (auto src : node->srcnodes)
        if (visited_set.find(src) == visited_set.end()) {
          visit = false;
          break;
        }
    }
    if (visit) {
      nodes_.push_back(node);
      visited_set.insert(node);
      for (auto dst : node->dstnodes) {
        // queueing the dst node if it is not queued before
        if (visit_set.find(dst) == visit_set.end()) {
          visiting_nodes.push(dst);
          visit_set.insert(dst);
        }
      }
    } else {
      visiting_nodes.push(node);
    }
  }
  CHECK_EQ(nodes_.size(), n);
}

const Graph Graph::Reverse() const {
  Graph g;
  for (Node* n : nodes_)
    g.AddNode(n->name, n->attrs);
  for (Node* src : nodes_)
    for (Node* dst : src->dstnodes) {
      map<string, string> attrs;
      const string edge = GetEdgeName(src->name, dst->name);
      if (edge_attrs_.find(edge) != edge_attrs_.end())
        attrs = edge_attrs_.at(edge);
      g.AddEdge(dst->name, src->name, attrs);
    }
  return g;
}
string Graph::ToJson() const {
  map<string, string> label;
  return ToJson(label);
}


string Graph::ToJson(const map<string, string>& label) const {
  string disp = "{\"directed\":1,\n";

  // add nodes
  disp += "\"nodes\":[\n";

  bool first = true;
  map<string, int> node_id;
  int id = 0;
  for (auto node : nodes_) {
    string name = node->name;
    string lbl = name + " -- ";
    if (label.find(name) != label.end())
      lbl += label.at(name);
    if (node->attrs.find("label") != node->attrs.end())
      lbl += node->attrs.at("label");
    disp += StringPrintf("%c{\"id\":\"%s\", \"label\":\"%s\"",
        !first ? ',' : ' ', name.c_str(), lbl.c_str());
    for (const auto& attr : node->attrs)
      if (attr.first != "label")
        disp += StringPrintf(", \"%s\":\"%s\"",
            attr.first.c_str(), attr.second.c_str());
    disp += "}\n";
    first = false;
    node_id[name] = id++;
  }
  disp += "]\n,\n";

  // add edges
  disp += "\"links\":[\n";
  first = true;
  for (auto src : nodes_) {
    for (auto dst : src->dstnodes) {
      const string edge_name = GetEdgeName(src->name, dst->name);
      string lbl = "";
      if (label.find(edge_name) != label.end())
        lbl = label.at(edge_name);
      disp += StringPrintf("%c{\"source\":%d, \"target\":%d, \"label\": \"%s\"",
          !first ? ',' : ' ', node_id[src->name], node_id[dst->name],
          lbl.c_str());
      if (edge_attrs_.find(edge_name) != edge_attrs_.end()) {
        for (const auto& attr : edge_attrs_.at(edge_name))
          disp += StringPrintf(", \"%s\":\"%s\"",
              attr.first.c_str(), attr.second.c_str());
      }
      disp += "}\n";
      first = false;
    }
  }
  return disp + "]}";
}
}  // namespace singa
