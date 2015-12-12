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
using std::string;
using std::map;

/**
 * Node class representing a layer in a neural net.
 *
 * TODO remove layer dependent fields, like origin, and partition_id, to make
 * it an independent and simple class.
 */
class Node {
 public:
  /**
   * Node constructor.
   *
   * @param name identifier of the node, e.g, layer name.
   */
  explicit Node(string name);
  /**
   * Construct a node with specified attributes.
   * @param name node identifier
   * @param attrs node attributes for printing, including "shape", "color", etc.
   * Depending on the visulization engine, if using graphviz, then the attribute
   * list is http://www.graphviz.org/content/attrs.
   */
  Node(string name, const std::map<string, string>& attrs);
  /**
   * @deprecated {to make the Graph class an independent class.}
   *
   * Node constructor used for model partitioning.
   *
   * This node is a partition of some node.
   * @param name node name
   * @param origin  name of the original node
   * @param id partition id of this node
   * @param proto conf of the corresponding layer
   */
  Node(const string& name, const std::string& origin, int id, void* proto);
  ~Node() {}  // the proto field is deleted outside by other functions


  void AddDstNode(Node* dst);
  void AddSrcNode(Node* src);
  void RemoveDstNode(Node* dst);
  void RemoveSrcNode(Node* src);

  string name = "";
  //! name of the origin node/layer from which is node is derived
  string origin = "";
  //! partition id
  int partition_id = 0;
  //! proto of the corresponding layer
  void* proto = nullptr;
  std::vector<Node*> srcnodes;
  std::vector<Node*> dstnodes;
  //!< node attribute including shape, color, etc.
  std::map<string, string> attrs;
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
  const Graph Reverse() const;
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
  inline Node* node(const string& name) const {
    return name2node_.at(name);
  }
  /**
   * Add an exiting node into this graph.
   */
  void AddNode(Node* node);
  /**
   * Creat an node with the given name and add it into the graph.
   * @return the newly created node.
   */
  Node* AddNode(const string& name);
  /**
   * Create an node with the given name and attributes.
   */
  Node* AddNode(const string& name, const std::map<string, string>& attrs);
  /**
   * @deprecated {remove layer related info from node attrs}
   * Add a node with given name and other info.
   */
  Node* AddNode(const std::string& name, const std::string& origin, int id,
                void* proto);
  /**
   * Add an edge connecting the two given nodes.
   */
  void AddEdge(Node* srcnode, Node* dstnode);
  /**
   * Add an edge connecting the two nodes with the given name.
   */
  void AddEdge(const string& src, const std::string& dst);
  /**
   * Add an edge connecting the two given nodes, the edge attributes are also
   * given.
   */
  void AddEdge(Node* srcnode, Node* dstnode,
      const std::map<string, string>& attrs);
  /**
   * Add an edge connecting the two nodes with the given names, the edge
   * attributes are also given, which are used for printing.
   * http://www.graphviz.org/content/attrs
   */
  void AddEdge(const string& src, const std::string& dst,
      const std::map<string, string>& attrs);

  /**
   * Remove the edge connecting the two given nodes.
   */
  void RemoveEdge(Node* src, Node* dst);
  /**
   * Remove the edge connecting two nodes with the given names.
   */
  void RemoveEdge(const string &src, const std::string& dst);
  /**
   * Dump the graph into json string which can be used to draw a picture by
   * graphviz.
   *
   * It calls ToJson(const std::map<std::string, std::string>& label) with
   * empty label mapping.
   */
  string ToJson() const;
  /**
   * \copybreif ToJson()
   *
   * @param label information to be displayed as label for each node
   */
  string ToJson(const map<std::string, std::string>& label) const;
  /**
   * Do topology sort for all nodes of the graph.
   */
  void Sort();

 private:
  /**
   *
   * @return the name of the edge connecting src to dst
   */
  const string GetEdgeName(const string& src, const string& dst) const {
    return src + "-->" + dst;
  }

 private:
  std::vector<Node*> nodes_;
  std::map<string, Node*> name2node_;
  std::map<string, std::map<string, string>> edge_attrs_;
};

}  // namespace singa

#endif  // SINGA_UTILS_GRAPH_H_
