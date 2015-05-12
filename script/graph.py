import sys
import pygraphviz
import networkx as nx
from networkx.readwrite import json_graph
import json


if __name__=='__main__':
  print sys.argv
  if len(sys.argv)<3:
    print 'usage: draw the network graph\npython graph.py JSON_DAT FIG_FILE'
    sys.exit()

  with open(sys.argv[1]) as fd:
    nodelink=json.load(fd)
    G=json_graph.node_link_graph(nodelink)
    A = nx.to_agraph(G)
    A.layout('dot', args='-Nfontsize=10 -Nwidth=".2" -Nheight=".2" -Nmargin=0 \
        -Gfontsize=8')
    A.draw(sys.argv[2])


