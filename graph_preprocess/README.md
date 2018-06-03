The following three files are the network datasets downloaded at https://snap.stanford.edu/data/
1. p2p-Gnutella08.txt
2. soc-sign-epinions.txt
3. Wiki-Vote.txt

In order to get the community structures of a networkWe,we used Gephi which is the leading visualization and exploration software for all kinds of graphs and networks.The software is available at https://gephi.org/ .

The following three files contains the community label of each node in a network.
Data format: Each row represents the (node, community_label) pairs.
1. Wiki_Vote_communities.txt
2. Epinions_after_community_detection.txt
3. Gnutella08_communities.txt

After getting the community structure of the three networks, we assign each node a community label according to the above three files and initialize the activating probability of each edge in the network. Finally, the following tree graph files are created by the python file "preprocessing_datasets.py".
1. Wiki_Vote.gexf
2. Gnutella08.gexf
3. Epinions.gexf

