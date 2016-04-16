#/**
# * Licensed to the Apache Software Foundation (ASF) under one
# * or more contributor license agreements.  See the NOTICE file
# * distributed with this work for additional information
# * regarding copyright ownership.  The ASF licenses this file
# * to you under the Apache License, Version 2.0 (the
# * "License"); you may not use this file except in compliance
# * with the License.  You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */

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


