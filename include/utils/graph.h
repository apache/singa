#ifndef SINGA_UTILS_GRAPH_H_
#define SINGA_UTILS_GRAPH_H_
#include <vector>
#include <string>
#include <map>
#include <stack>
#include <memory>

namespace singa {
using std::vector;
using std::string;
using std::map;

class Node {
 public:
  /**
   * Node constructor.
   *
   * @param name name of the corresponding layer
   */
  explicit Node(string name);
  /**
   * Node constructor.
   *
   * This node is a partition of some node.
   * @param name node name
   * @param origin  name of the original node
   * @param id partition id of this node
   * @param proto conf of the corresponding layer
   */
  Node(const string& name, const string& origin, int id, void* proto);
  ~Node();
  void AddDstNode(Node* dstnode);
  void AddSrcNode(Node* srcnode);
  void RemoveDstNode(Node* dst);
  void RemoveSrcNode(Node* src);

 public:
  string name;
  //! name of the origin node/layer from which is node is derived
  string origin;
  //! partition id
  int partition_id;
  //! proto of the corresponding layer
  void* proto;

  vector<Node*> srcnodes;
  vector<Node*> dstnodes;
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
  const vector<Node*>& nodes() const {
    return nodes_;
  }
  /**
   * @param name node name
   * @return return the node of given name
   */
  Node* node(const string& name) const {
    return name2node_.at(name);
  }

  void AddNode(Node* node);
  Node* AddNode(const string& name);
  void AddEdge(Node* srcnode, Node* dstnode);
  void AddEdge(const string& src, const string& dst);
  void RemoveEdge(Node* src, Node* dst);
  void RemoveEdge(const string &src, const string& dst);
  /**
   * Dump the graph into json string which can be used to draw a picture by
   * graphviz
   */
  const string ToJson() const;
  /**
   * \copybreif ToJson()
   *
   * @param info info associated with each node
   */
  const string ToJson(const map<string, string>& info) const;
  /**
   * Do topology sort for all nodes of the graph.
   */
  void Sort();

 private:
  vector<Node*> nodes_;
  map<string, Node*> name2node_;
};
}  // namespace singa
#endif  // SINGA_UTILS_GRAPH_H_
