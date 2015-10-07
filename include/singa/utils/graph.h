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

#ifndef SINGA_UTILS_GRAPH_H_
#define SINGA_UTILS_GRAPH_H_

#include <stack>
#include <string>
#include <map>
#include <vector>

namespace singa {

class Node {
 public:
  /**
   * Node constructor.
   *
   * @param name name of the corresponding layer
   */
  explicit Node(std::string name);
  /**
   * Node constructor.
   *
   * This node is a partition of some node.
   * @param name node name
   * @param origin  name of the original node
   * @param id partition id of this node
   * @param proto conf of the corresponding layer
   */
  Node(const std::string& name, const std::string& origin, int id, void* proto);
  ~Node() {}  // the proto field is deleted outside by other functions
  void AddDstNode(Node* dstnode);
  void AddSrcNode(Node* srcnode);
  void RemoveDstNode(Node* dst);
  void RemoveSrcNode(Node* src);

  std::string name = "";
  //! name of the origin node/layer from which is node is derived
  std::string origin = "";
  //! partition id
  int partition_id = 0;
  //! proto of the corresponding layer
  void* proto = nullptr;
  std::vector<Node*> srcnodes;
  std::vector<Node*> dstnodes;
};

/**
 * Neuralnet is constructed by creating a graph with each node representing one
 * layer at first. After topology sort for graph nodes, layers are created and
 * connected.
 */
class Graph {
 public:
  Graph() {}
  ~Graph();
  /**
   * @return all nodes of the graph
   */
  inline const std::vector<Node*>& nodes() const {
    return nodes_;
  }
  /**
   * @param name node name
   * @return return the node of given name
   */
  inline Node* node(const std::string& name) const {
    return name2node_.at(name);
  }
  void AddNode(Node* node);
  Node* AddNode(const std::string& name);
  void AddEdge(Node* srcnode, Node* dstnode);
  void AddEdge(const std::string& src, const std::string& dst);
  void RemoveEdge(Node* src, Node* dst);
  void RemoveEdge(const std::string &src, const std::string& dst);
  /**
   * Dump the graph into json string which can be used to draw a picture by
   * graphviz
   */
  std::string ToJson() const;
  /**
   * \copybreif ToJson()
   *
   * @param info info associated with each node
   */
  std::string ToJson(const std::map<std::string, std::string>& info) const;
  /**
   * Do topology sort for all nodes of the graph.
   */
  void Sort();

 private:
  std::vector<Node*> nodes_;
  std::map<std::string, Node*> name2node_;
};

}  // namespace singa

#endif  // SINGA_UTILS_GRAPH_H_
