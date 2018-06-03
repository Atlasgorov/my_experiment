import networkx as nx
import random

def activating_probability_init(G):
    """
    激活概率的初始化函数
    """
    for edge, edge_attr in G.edges.items():
        # 边的激活概率从0.01到1范围内随机选取
        edge_attr['probability'] = random.uniform(0, 1)


def preprocess_Wiki_Vote():
    """
    由Wiki_vote的网络数据构建一个有向图对象，并且对其每条边赋予边权，表示激活概率，在[0,1]范围内均匀随机选取。
    图中的每个节点均有一个社区标号，表示其所属的社区。社区标号数据来自软件包Gephi中的社区检测算法，Lovain method的结果。
    :return: 返回一个有向加权图
    """
    """
    # Directed graph (each unordered pair of nodes is saved once): Wiki-Vote.txt
    # Wikipedia voting on promotion to administratorship (till January 2008). Directed edge A->B means user A voted on B becoming Wikipedia administrator.
    # Nodes: 7115 Edges: 103689
    # FromNodeId	ToNodeId
    """
    f0 = open('Wiki-Vote.txt')  # 打开图文件
    temp = [x.strip().split('\t') for x in f0.readlines()] # 创建边元组的列表
    f0.close() # 关闭文件
    DG = nx.DiGraph() # 创建一个有向图对象
    DG.add_edges_from(temp) # 添加边列表中的所有边到图DG中
    del temp # 删除临时变量

    nodes_number = DG.number_of_nodes()
    edges_number = DG.number_of_edges()
    print('节点数量为：', nodes_number, '边总数为：', edges_number)
    print('图中的自环个数为',nx.number_of_selfloops(DG))  # 没有自环出现
    #DG.remove_edges_from(nx.selfloop_edges(DG)) # 若有自环，可以删除

    # 检查划分后的各个社区所包含的节点数量
    f0 = open('Wiki_Vote_communities.txt') # 第一列是节点，第二列是节点所属的社区标号
    temp = [x.strip().split('\t') for x in f0.readlines()]
    f0.close()  # 关闭文件
    for i in temp:
        node_label_temp = i[0]
        community_label = i[1]
        DG.nodes[node_label_temp]['community_label'] = community_label # 把每个节点的所属社区标号作为该节点的"community_label"属性

    all_community_label = set([x[1] for x in temp])  # 所有的社区标签
    community_dict = dict.fromkeys(list(all_community_label), 0)  # 字典初始化,键为社区标签，值为该社区的总节点数
    for node,community in temp:
        community_dict[community] += 1
    #print(community_dict, len(community_dict))
    DG.graph['community'] = community_dict # 图的'community'属性存储了所有的社区标签和对应的社区节点数量
    activating_probability_init(DG)
    nx.write_gexf(DG, 'Wiki_Vote.gexf')


def preprocess_Gnutella08():
    # Directed graph (each unordered pair of nodes is saved once): p2p-Gnutella08.txt
    # Directed Gnutella P2P network from August 8 2002
    # Nodes: 6301 Edges: 20777
    f0 = open('p2p-Gnutella08.txt')  # 打开图文件
    temp = [x.strip().split('\t') for x in f0.readlines()]  # 创建边元组的列表
    f0.close()  # 关闭文件
    DG = nx.DiGraph()
    DG.add_edges_from(temp)
    del temp  # 删除中间变量

    nodes_number = DG.number_of_nodes()
    edges_number = DG.number_of_edges()
    print('节点数量为：', nodes_number, '边总数为：', edges_number)
    print('图中的自环个数为', nx.number_of_selfloops(DG))  # 没有自环出现

    # 检查划分后的各个社区所包含的节点数量
    f0 = open('Gnutella08_communities.txt') # 第一列是节点，第二列是节点所属的社区标号
    temp = [x.strip().split('\t') for x in f0.readlines()]
    f0.close()  # 关闭文件
    for i in temp:
        node_label_temp = i[0]
        community_label = i[1]
        DG.nodes[node_label_temp]['community_label'] = community_label

    all_community_label = set([x[1] for x in temp])  # 所有的社区标签
    community_dict = dict.fromkeys(list(all_community_label), 0)  # 字典初始化,键为社区标签，值为该社区的总节点数
    for i in temp:
        community_dict[i[1]] += 1
    DG.graph['community'] = community_dict  # 图的'community'属性存储了所有的社区标签和对应的社区节点数量
    activating_probability_init(DG)
    nx.write_gexf(DG, 'Gnutella08.gexf')


def preprocess_Epinions():
    # Directed graph: soc-sign-epinions
    # Epinions signed social network
    # Nodes: 131828 Edges: 841372
    # Number of Communities: 6872
    f0 = open('soc-sign-epinions.txt')  # 打开图文件
    temp = [x.strip().split('\t') for x in f0.readlines()] # 创建边元组的列表
    temp2 = [x[0:2] for x in temp]
    f0.close()  # 关闭文件
    DG = nx.DiGraph()
    DG.add_edges_from(temp2)

    del temp # 删除中间变量

    nodes_number = DG.number_of_nodes()
    edges_number = DG.number_of_edges()

    print('节点数量为：', nodes_number, '边总数为：', edges_number)
    print('图中的自环个数为', nx.number_of_selfloops(DG))  # 有自环出现
    DG.remove_edges_from(nx.selfloop_edges(DG)) # 有自环，进行删除
    # 检查划分后的各个社区所包含的节点数量
    f0 = open('D:\project\仿真第二阶段\Epinions\Epinions_after_community_detection.txt')
    temp = [x.strip().split('\t') for x in f0.readlines()]
    f0.close()  # 关闭文件
    # print(temp)
    # print(temp[0][0])
    for i in temp:
        node_label_temp = i[0]
        community_label = i[1]
        DG.nodes[node_label_temp]['community_label'] = community_label

    all_community_label = set([x[1] for x in temp])  # 所有的社区标签
    community_dict = dict.fromkeys(list(all_community_label), 0)  # 字典初始化,键为社区标签，值为该社区的总节点数
    for i in temp:
        community_dict[i[1]] += 1
    DG.graph['community'] = community_dict # 图的'community'属性存储了所有的社区标签和对应的社区节点数量
    activating_probability_init(DG)
    nx.write_gexf(DG, 'Epinions.gexf')


if __name__ == '__main__':
    """
    获得了三个已经划分好了社区并且激活概率初始化了的图文件，它们将作为后续处理的图对象
    """
    preprocess_Wiki_Vote()
    preprocess_Gnutella08()
    preprocess_Epinions()


