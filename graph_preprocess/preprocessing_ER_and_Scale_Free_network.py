import networkx as nx
import random

def activating_probability_init(G):
    """
    激活概率的初始化函数
    """
    for edge, edge_attr in G.edges.items():
        # 边的激活概率从0.01到1范围内随机选取
        edge_attr['probability'] = random.uniform(0, 1)
"""
# 生成两个人工网络
G_ER = nx.fast_gnp_random_graph(n=7000,p=0.0005,directed=True)
print(nx.is_directed(G_ER))
nodes_number = G_ER.number_of_nodes()
edges_number = G_ER.number_of_edges()
ave_node_degree = 2 * edges_number / nodes_number

print('ER网络的节点数量为：', nodes_number, 'ER网络的边总数为：', edges_number, "平均节点度数为：", ave_node_degree)
#print('ER网络中的自环个数为', nx.number_of_selfloops(G_ER))  # 没有自环出现
self_loops_temp = list(nx.selfloop_edges(G_ER))
G_ER.remove_edges_from(self_loops_temp) # 若有自环，进行删除
nx.write_gexf(G_ER, 'original_ER_graph.gexf')


G_SF = nx.scale_free_graph(8000,alpha=0.15,beta=0.80,gamma=0.05)
print(nx.is_directed(G_SF))
nodes_number = G_SF.number_of_nodes()
edges_number = G_SF.number_of_edges()
ave_node_degree = 2 * edges_number / nodes_number
print('SF网络的节点数量为：', nodes_number, 'SF网络的边总数为：', edges_number, "平均节点度数为：", ave_node_degree)
print('SF网络中的自环个数为', nx.number_of_selfloops(G_SF))  # 没有自环出现
self_loops_temp = list(nx.selfloop_edges(G_SF))
G_SF.remove_edges_from(self_loops_temp) # 有自环，进行删除
print('SF网络中的自环个数为', nx.number_of_selfloops(G_SF))  # 没有自环出现
#print(nx.is_directed(G_SF))
edge_list = set(list(G_SF.edges()))
G_SF_final = nx.DiGraph()
G_SF_final.add_edges_from(edge_list)
nodes_number = G_SF_final.number_of_nodes()
edges_number = G_SF_final.number_of_edges()
ave_node_degree = 2 * edges_number / nodes_number
print('SF网络的节点数量为：', nodes_number, 'SF网络的边总数为：', edges_number, "平均节点度数为：", ave_node_degree)

nx.write_gexf(G_SF_final, 'original_SF_graph.gexf')

"""

def preprocess_ER():
    DG = nx.read_gexf('original_ER_graph.gexf')
    # 检查划分后的各个社区所包含的节点数量
    f0 = open('ER_communities.txt')  # 第一列是节点，第二列是节点所属的社区标号
    temp = [x.strip().split('\t') for x in f0.readlines()]
    f0.close()  # 关闭文件
    for i in temp:
        node_label_temp = i[0]
        community_label = i[1]
        DG.nodes[node_label_temp]['community_label'] = community_label  # 把每个节点的所属社区标号作为该节点的"community_label"属性

    all_community_label = set([x[1] for x in temp])  # 所有的社区标签
    community_dict = dict.fromkeys(list(all_community_label), 0)  # 字典初始化,键为社区标签，值为该社区的总节点数
    for node, community in temp:
        community_dict[community] += 1
    # print(community_dict, len(community_dict))
    DG.graph['community'] = community_dict  # 图的'community'属性存储了所有的社区标签和对应的社区节点数量
    print(sorted(community_dict.items(),key = lambda x:x[1],reverse=True))
    activating_probability_init(DG)
    nx.write_gexf(DG, 'ER_graph.gexf')


def preprocess_SF():
    DG = nx.read_gexf('original_SF_graph.gexf')
    # 检查划分后的各个社区所包含的节点数量
    f0 = open('SF_communities.txt')  # 第一列是节点，第二列是节点所属的社区标号
    temp = [x.strip().split('\t') for x in f0.readlines()]
    f0.close()  # 关闭文件
    for i in temp:
        node_label_temp = i[0]
        community_label = i[1]
        DG.nodes[node_label_temp]['community_label'] = community_label  # 把每个节点的所属社区标号作为该节点的"community_label"属性

    all_community_label = set([x[1] for x in temp])  # 所有的社区标签
    community_dict = dict.fromkeys(list(all_community_label), 0)  # 字典初始化,键为社区标签，值为该社区的总节点数
    for node, community in temp:
        community_dict[community] += 1
    # print(community_dict, len(community_dict))
    DG.graph['community'] = community_dict  # 图的'community'属性存储了所有的社区标签和对应的社区节点数量
    print(sorted(community_dict.items(), key=lambda x: x[1], reverse=True),len(community_dict))
    activating_probability_init(DG)
    nx.write_gexf(DG, 'SF_graph.gexf')


if __name__ == '__main__':
    """
    获得了两个已经划分好了社区并且激活概率初始化了的图文件，它们将作为后续处理的图对象
    """
    #preprocess_ER()
    preprocess_SF()