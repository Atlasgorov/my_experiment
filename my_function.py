import networkx as nx
from collections import deque
import random
import numpy as np
import time

def random_chose_rumor_sources(nodes_in_rumor_community,rumor_sources_num):
    """
    从谣言社区内部随机选取一定数量的节点作为谣言源节点
    :param nodes_in_rumor_community: 谣言社区内部节点列表
    :param rumor_sources_num: 谣言源节点数量
    :return: 谣言源节点集合
    """
    rumor_source_nodes = set(random.sample(nodes_in_rumor_community, rumor_sources_num)) # 随机选取谣言源节点
    return rumor_source_nodes

def bfs_0(G, source_nodes, rumor_community_label0, protection_level = 0.9):
    """
     从已知的谣言源节点出发进行BFS搜索得到所有的谣言社区内部可达的节点以及所有的桥节点
     :param G: 原始的社交图
     :param source_nodes: 谣言源节点（集合类型）
     :param rumor_community_label0: 谣言社区的标号
     :protection_level:预定的谣言保护等级，比如0.9
     :return: 谣言社区内部的谣言可达节点列表A，桥节点列表B-->M0和B0，受到谣言影响的节点最大值k
     """
    # 在每次随机选取谣言源节点之后，要记录A，B，和k的大小，用于求它们各自的均值。
    start_time = time.clock()
    visited = set() # 初始化一个集合，用来存放访问到的节点
    common_bridges = set() # 普通的桥节点
    special_bridges = set() # 父节点中存在谣言源节点的桥节点,返回该集合时记为M0
    for source0 in source_nodes: # 从多个源节点出发进行BFS搜索
        if source0 not in visited:  # 由于有多个源节点在进行搜索，所以先判断该源节点是否已经被搜索过
            Queue = deque()  # 初始化双端队列，将它作为先入先出队列使用
            visited.add(source0)  # 标记源节点的搜索状态为“已访问”
            Queue.append(source0)  # 将源节点压入队列的队尾（最右端）
            while Queue:  # 当Queue非空时，一直搜索
                source_temp = Queue.popleft()  # 弹出队列首元素(最左端)，作为临时起点
                if G.nodes[source_temp]['community_label'] != rumor_community_label0:  # 如果这个临时起点是桥节点
                    if source_nodes.isdisjoint(set(G.predecessors(source_temp))):
                        common_bridges.add(source_temp)  # 若该source_temp的父节点均不是谣言源节点，则标记它为普通桥节点
                        continue # 由于不用再从桥端节点出发继续进行下一层的搜索，所以跳过当前这次循环，直接从Queue中的下一个节点出发再搜索
                    else:
                        special_bridges.add(source_temp) # 如果该source_temp的父节点中存在谣言源节点时，记为特殊桥节点
                        continue  # 由于不用再从桥端节点出发继续进行下一层的搜索，所以跳过当前这次循环，直接从Queue中的下一个节点出发再搜索
                for out_neighbor in G.successors(source_temp):  # 遍历临时起点的所有子节点
                    if out_neighbor not in visited:
                        visited.add(out_neighbor)  # 标记该节点的搜索状态为“已访问”
                        Queue.append(out_neighbor)
    A = visited - source_nodes - common_bridges - special_bridges # 这是谣言社区内部所有的谣言可达节点,不包括谣言源节点
    Length_A = len(A) # 谣言社区内部谣言可达节点的数量
    M0 = special_bridges # 父节点中存在谣言源节点的桥节点集合，这些节点需要直接进行控制
    Length_M0 = len(M0) # 特殊桥节点的数量
    B0 = common_bridges # 父节点中不存在谣言源节点的桥节点集合，它们是普通的桥节点
    Length_B0 = len(B0) # 普通桥节点的数量
    k = int((1 - protection_level) * Length_A) # 可能会受到谣言影响的节点总数量，预先给定的protection_level代表了保护的等级
    end_time = time.clock()
    time0 = end_time - start_time
    #print("最初，谣言社区内部谣言可达节点为：{}，数量为：{}\n,特殊桥节点为{}，数量为：{}\n,普通桥节点为{}，数量为：{}\n".format(A,Length_A,M0,Length_M0,B0,Length_B0))
    #print("最初，谣言社区内部谣言可达节点数量为：{}\n,特殊桥节点为数量为：{}\n,普通桥节点数量为：{}\n，谣言源节点数量为：{}".format(Length_A,Length_M0,Length_B0,len(source_nodes)))
    return A, Length_A, B0, Length_B0, M0, Length_M0, k , time0


def get_subgraph(G,rumor_sources,rumor_reachable_nodes,common_bridges,special_bridges):
    """
     从原始的社交图中以谣言源节点集合、社区内部谣言可达节点集合A、桥节点集合B0导出的子图G1
    :param G: 原始的社交网络图
    :param rumor_sources: 谣言源节点集合
    :param rumor_reachable_nodes: 社区内部的谣言可达节点
    :param common_bridges: 普通桥节点
    :return: G1
    """
    start_time = time.clock()
    G1 = G.copy()
    # 删除G1中所有不在SR、A、B0集合中的节点，得到G的导出子图G1
    G1.remove_nodes_from(set(G1)-rumor_sources-rumor_reachable_nodes-common_bridges-special_bridges)
    end_time = time.clock()
    time1 = end_time - start_time
    #print('已经得到了子图G1, 节点数量为：',len(G1))
    return G1, time1


def protect_bridge_ends_v4(G,common_bridges):
    """
    求二部图的最小顶点覆盖来阻断谣言源节点抵达所有的桥节点，返回需要阻断的节点（包括桥节点）
    :param G:子图G1
    :param common_bridges:普通桥节点集合
    :return: 封锁节点列表M1
    """
    G_bipartite = nx.Graph() # 无向无权二部图
    parents_of_common_bridges = set() # 一个集合：存放普通桥节点的父节点
    for node in common_bridges:
        G_bipartite.add_node(node,bridge=True)
        for in_neighbor in G.predecessors(node):  # 在图G1中遍历这个普通桥节点的所有父节点(非桥节点)
            if in_neighbor not in common_bridges:
                G_bipartite.add_node(in_neighbor, parent=True)
                parents_of_common_bridges.add(in_neighbor)
                G_bipartite.add_edge(node,in_neighbor)
    M1 = set() # 存放最小顶点覆盖的结果
    while G_bipartite.number_of_edges():
        matching = nx.bipartite.eppstein_matching(G_bipartite,top_nodes=common_bridges-M1) # 最大匹配
        min_vertex_cover = nx.bipartite.to_vertex_cover(G_bipartite, matching, top_nodes=common_bridges-M1) # 最小集合覆盖的节点集合
        M1.update(min_vertex_cover) # 这些节点若从原始图G中移除，则从所有的源节点出发均不可达普通的桥节点
        G_bipartite.remove_nodes_from(M1)
    return M1


def bfs_1(G, source_nodes):
    """
    BFS搜索得到从源节点出发所有能够抵达的节点（不包括源节点）
    :param G:
    :param source_nodes:
    :return: 源节点能够抵达的节点的总量
    """
    visited = set()  # 初始化一个集合，用来存放访问到的节点
    for source0 in source_nodes:  # 从多个源节点出发进行BFS搜索
        if source0 not in visited:  # 由于有多个源节点在进行搜索，所以先判断该源节点是否已经被搜索过
            Queue = deque()  # 初始化双端队列，将它作为先入先出队列使用
            visited.add(source0)  # 标记源节点的搜索状态为“已访问”
            Queue.append(source0)  # 将源节点压入队列的队尾（最右端）
            while Queue:  # 当Queue非空时，一直搜索
                source_temp = Queue.popleft()  # 弹出队列首元素(最左端)，作为临时起点
                for out_neighbor in G.successors(source_temp):  # 遍历临时起点的所有子节点
                    if out_neighbor not in visited: # 若source_temp的子节点（出度邻居）没有访问过
                        visited.add(out_neighbor)  # 标记该节点的搜索状态为“已访问”
                        Queue.append(out_neighbor) # 将之加入队列的队尾
    reachable_nodes_from_sources = visited - source_nodes
    return reachable_nodes_from_sources


def bfs_2(G,source_nodes,rumor_community_label0):
    """
    该函数用于确定从源节点出发是否能抵达桥节点
    :param G:
    :param source_nodes:谣言源节点集合
    :return: False:如果从源节点出发无法抵达桥节点，则返回False,否则返回True，这说明能抵达桥节点
    """
    # 利用BFS搜索检测是否从源节点可达普通桥节点
    visited = set()  # 初始化一个集合，用来存放访问到的节点
    for source0 in source_nodes:  # 从多个源节点出发进行BFS搜索
        if source0 not in visited:  # 由于有多个源节点在进行搜索，所以先判断该源节点是否已经被搜索过
            Queue = deque()  # 初始化双端队列，将它作为先入先出队列使用
            visited.add(source0)  # 标记源节点的搜索状态为“已访问”
            Queue.append(source0)  # 将源节点压入队列的队尾（最右端）
            while Queue:  # 当Queue非空时，一直搜索
                source_temp = Queue.popleft()  # 弹出队列首元素(最左端)，作为临时起点
                if G.nodes[source_temp]['community_label'] != rumor_community_label0:  # 如果这个临时起点是桥节点
                    return True # 说明谣言源可达桥节点
                for out_neighbor in G.successors(source_temp):  # 遍历临时起点的所有子节点
                    if out_neighbor not in visited:
                        visited.add(out_neighbor)  # 标记该节点的搜索状态为“已访问”
                        Queue.append(out_neighbor)
    return False



def my_method(subGraph, K, source_nodes, M0_temp,M0_length, M1_temp,IC_simulation_num_my_method):
    """
    从G1中删除M0与M1集合中的节点，这样子之后，谣言源节点必然不能抵达所有的桥节点，
    然后用贪心算法从剩余的节点里选出移除之后会让谣言影响力减小量最大的节点进行删除，
    直到满足给定的谣言影响力的要求。
    :param subGraph: 原始网络图G的子图G1
    :param K: K是预定的谣言抑制要求，在IC过程结束之后，受到谣言影响的节点总量不超过K。
    :param source_nodes: 谣言源节点集合
    :param M0_length: 父节点中存在谣言源的桥节点的总量
    :param M1_temp: protect_bridge_ends所选出的需要删除的节点
    :param IC_simulation_num_my_method: 为了获得一次IC传播模型的期望(平均)结果所作的总的模拟次数
    :return:需要封锁的节点总数量
    """
    print('进入my method方法')
    G1 = subGraph.copy()
    G1.remove_nodes_from(M1_temp)
    G1.remove_nodes_from(M0_temp)
    M1_length = len(M1_temp)

    # 利用BFS搜索得到删除M1中的节点之后从源节点出发可达的节点
    all_reachable_nodes_from_sources = bfs_1(G1, source_nodes) # 新的所有的谣言源可达节点集合
    if len(all_reachable_nodes_from_sources) <= K: # 若删除M0和M1之后，源节点出发的可达节点数量小于K，那么显然源节点的影响力不会超过K
        number_of_blocked_nodes = M0_length + M1_length
        return number_of_blocked_nodes
    else:
        print('进入贪婪循环删除节点的过程')
    G1.remove_nodes_from(set(G1) - all_reachable_nodes_from_sources - source_nodes)  # 从G1中删除所有源节点不可达的节点

    # 利用贪心方法从G1中不断删除能够让源节点集合影响力减小量最大的节点，直到满足K的要求
    greedily_blocked_nodes = list() # 最终选择进行删除的节点集合
    influence_dict = dict().fromkeys(all_reachable_nodes_from_sources) # 记录了删除每个节点之后相对应的谣言影响力
    number_of_infected_nodes = independent_cascade_propagation(G1, source_nodes, IC_simulation_num_my_method) # 原始的影响力——>平均受影响的节点总数
    while number_of_infected_nodes > K: # 当源节点集合的影响力大于预定的k值时，循环
        for u in influence_dict: # 遍历候选节点集
            edges_data = list() # 由于这里是暂时删除节点u，所以记录跟u关联的所有边的数据，便于重新添加回去
            for parent_node, datadict in G1.pred[u].items():
                edges_data.append((parent_node, u, datadict['probability'])) # 记录节点u的入边，边权为激活概率
            for child_node, datadict in G1.succ[u].items():
                edges_data.append((u, child_node, datadict['probability'])) # 记录节点u的出边，边权为激活概率
            G1.remove_node(u) # 暂时删除节点u
            sources_influence_after_removed_u = independent_cascade_propagation(G1, source_nodes, IC_simulation_num_my_method)
            influence_dict[u] = sources_influence_after_removed_u # 记录删除u之后，源节点集合的影响力
            G1.add_node(u)
            G1.add_weighted_edges_from(edges_data, weight='probability') # 把边添加回去,边权为激活概率
        target_node = min(influence_dict.items(),key=lambda x:x[1])[0] # 找到删除它之后能让源节点集合影响力减小量最大的节点
        influence_dict.pop(target_node) # 把选出的节点从候选节点字典里删除
        G1.remove_node(target_node) # 真正删除所选的节点
        greedily_blocked_nodes.append(target_node)
        number_of_infected_nodes = independent_cascade_propagation(G1, source_nodes, IC_simulation_num_my_method)
    M2_length= len(greedily_blocked_nodes) # 贪婪方法选出的封锁节点数量
    number_of_blocked_nodes = M0_length + M1_length + M2_length
    #print('my_method贪婪选出的封锁节点为：',greedily_blocked_nodes)
    print('此次my method方法结束')
    return number_of_blocked_nodes


def k_core_method(subGraph,k_core_dict, K,source_nodes,rumor_community_label0, special_bridges,M0_length,IC_simulation_num_k_core):
    """
    利用k-core方法从图G中不断删除A集合核数最大的节点，
    直到谣言不可达所有的桥节点并且满足受到谣言影响的节点总量不超过K。
    :param subGraph: 原始网络图G的子图G1
    :param k_core_dict:原始网络图G的所有节点的核数（字典）
    :param K: K是预定的谣言抑制要求，在IC过程结束之后，受到谣言影响的节点总量不超过K。
    :param source_nodes:谣言源节点集合
    :param rumor_community_label0:谣言社区的编号
    :param M0_length: 父节点中存在谣言源节点的桥节点的总数
    :param IC_simulation_num_k_core: 为了获得一次IC传播模型的期望(平均)结果所作的总的模拟次数
    :return: 需要封锁的节点总数量——> M0的数量加上这个方法需要删除的节点数量
    """
    print('进入k-core method')
    G1 = subGraph.copy()
    G1.remove_nodes_from(special_bridges)
    core_num_dict = dict().fromkeys(set(G1) - source_nodes)  # 是子图G1的所有谣言可达节点的核数字典，键是节点名，值是它的核数
    for node in core_num_dict:
        core_num_dict[node] = k_core_dict[node]
    candidates_nodes_list = [x for x, y in sorted(core_num_dict.items(), key=lambda x:x[1])]  # 按照核数的大小令节点升序排列，列表末尾的节点核数最大
    blocked_nodes = list()  # 存放选出的封锁节点
    bridges_can_be_reached = bfs_2(G1,source_nodes,rumor_community_label0)
    while bridges_can_be_reached:
        target_node = candidates_nodes_list.pop()  # 选择核数最大的节点作为待封锁的节点,pop方法默认弹出列表的最后一个元素
        G1.remove_node(target_node)  # 删除目标节点
        blocked_nodes.append(target_node)  # 记录目标节点
        bridges_can_be_reached = bfs_2(G1,source_nodes,rumor_community_label0)
    all_reachable_nodes_from_sources = bfs_1(G1, source_nodes)  # 新的所有的谣言源可达节点集合
    blocked_nodes_length = len(blocked_nodes)
    if len(all_reachable_nodes_from_sources) <= K:  # 若保护了所有的桥节点之后，源节点出发的可达节点数量小于K，那么显然源节点的影响力不会超过K
        number_of_blocked_nodes = M0_length + blocked_nodes_length
        #print('除了特殊桥节点，k_core需要封锁的节点为：', blocked_nodes)
        return number_of_blocked_nodes
    else:
        print('进入循环删除最大核数节点的过程')
    G1.remove_nodes_from(set(G1) - all_reachable_nodes_from_sources - source_nodes)  # 从G1中删除所有源节点不可达的节点
    number_of_infected_nodes = independent_cascade_propagation(G1, source_nodes, IC_simulation_num_k_core)  # 原始的影响力——>平均受影响的节点总数
    while number_of_infected_nodes > K:  # 当源节点集合的影响力大于预定的k值时，循环删除核数最大的节点
        target_node = candidates_nodes_list.pop()  # 选择核数最大的节点作为待封锁的节点,pop方法默认弹出列表的最后一个元素
        G1.remove_node(target_node)  # 删除目标节点
        blocked_nodes.append(target_node)  # 记录目标节点
        number_of_infected_nodes = independent_cascade_propagation(G1, source_nodes, IC_simulation_num_k_core)
    blocked_nodes_length = len(blocked_nodes)
    number_of_blocked_nodes = M0_length + blocked_nodes_length
    #print('除了特殊桥节点，k_core需要封锁的节点为：',blocked_nodes)
    print('此次k-core method结束')
    return number_of_blocked_nodes


def max_degree_method(subGraph, all_degree_dict,K, source_nodes, rumor_community_label0, special_bridges,M0_length, IC_simulation_num_max_degree):
    """
    与k核方法类似，只不过删除节点的指标变成了节点的度数,删除A集合中度数(入度加出度)最大的节点。
    :param subGraph:原始网络图G的子图G1
    :param all_degree_dict:原始网络G的度数字典
    :param K:K是预定的谣言抑制要求，在IC过程结束之后，受到谣言影响的节点总量不超过K。
    :param source_nodes:谣言源节点集合
    :param rumor_community_label0:谣言社区的编号
    :param M0_length: 父节点中存在谣言源节点的桥节点的总数
    :param IC_simulation_num_max_degree: 为了获得一次IC传播模型的期望(平均)结果所作的总的模拟次数
    :return:需要封锁的节点总数量——> M0的数量加上这个方法需要删除的节点数量
    """
    print('进入maxDegree method')
    G1 = subGraph.copy()
    G1.remove_nodes_from(special_bridges)
    degree_dict = dict().fromkeys(set(G1) - source_nodes)  # 是子图G1的所有谣言可达节点的度数字典，键是节点名，值是它的度数
    for node in degree_dict:
        degree_dict[node] = all_degree_dict[node]
    candidates_nodes_list = [x for x, y in sorted(degree_dict.items(), key=lambda x:x[1])]  # 按照度数的大小令节点升序排列，列表末尾的节点度数最大
    blocked_nodes = list()  # 存放选出的封锁节点
    bridges_can_be_reached = bfs_2(G1, source_nodes, rumor_community_label0)
    while bridges_can_be_reached:
        target_node = candidates_nodes_list.pop()  # 选择度数最大的节点作为待封锁的节点,pop方法默认弹出列表的最后一个元素
        G1.remove_node(target_node)  # 删除目标节点
        blocked_nodes.append(target_node)  # 记录目标节点
        bridges_can_be_reached = bfs_2(G1, source_nodes, rumor_community_label0)
    all_reachable_nodes_from_sources = bfs_1(G1, source_nodes)  # 新的所有的谣言源可达节点集合
    blocked_nodes_length = len(blocked_nodes)
    if len(all_reachable_nodes_from_sources) <= K:  # 若保护了所有的桥节点之后，源节点出发的可达节点数量小于K，那么显然源节点的影响力不会超过K
        number_of_blocked_nodes = M0_length + blocked_nodes_length
        #print('除了特殊桥节点，max_degree需要封锁的节点为：', blocked_nodes)
        return number_of_blocked_nodes
    else:
        print('进入循环删除最大度节点的过程')
    G1.remove_nodes_from(set(G1) - all_reachable_nodes_from_sources - source_nodes)  # 从G1中删除所有源节点不可达的节点
    number_of_infected_nodes = independent_cascade_propagation(G1, source_nodes, IC_simulation_num_max_degree)  # 原始的影响力——>平均受影响的节点总数
    while number_of_infected_nodes > K:  # 当源节点集合的影响力大于预定的k值时，循环删除度数最大的节点
        target_node = candidates_nodes_list.pop()  # 选择度数最大的节点作为待封锁的节点,pop方法默认弹出列表的最后一个元素
        G1.remove_node(target_node)  # 删除目标节点
        blocked_nodes.append(target_node)  # 记录目标节点
        number_of_infected_nodes = independent_cascade_propagation(G1, source_nodes, IC_simulation_num_max_degree)
    blocked_nodes_length = len(blocked_nodes)
    number_of_blocked_nodes = M0_length + blocked_nodes_length
    #print('除了特殊桥节点，max_degree需要封锁的节点为：', blocked_nodes)
    print('此次maxDegree method方法结束')
    return number_of_blocked_nodes


def independent_cascade_propagation(G, source_nodes, simulation_num=1000):
    # 调用该函数一次，便会返回simulation_num次的IC传播的平均结果
    result = list()
    for i in range(simulation_num):
        result.append(runIC_v1_directed(G, source_nodes))
    average_eventually_infected_nodes = int(sum(result) / simulation_num)
    return average_eventually_infected_nodes


def activateSimulation(probability):  # 根据激活概率probability来随机激活，返回结果为布尔量
    """
    # 验证代码
    def testing0(num):
        probability= 0.67
        res = 0
        for i in range(num):
            if activateSimulation(probability):
                res += 1
        print(res/num)
    testing0(100000)
    """
    random_number = random.uniform(0, 1)  # 生成0-1之间的服从均匀分布的随机数
    if random_number <= probability:
        return True  # 生成的随机数是以probability的概率落在【0，probability】上，满足，则代表节点v被u激活
    else:
        return False


def runIC_v1_directed(G, seed_nodes):  # 有向图的IC传播过程
    """
    进行独立级联模型的传播仿真
    :param G: 图G
    :param seed_nodes: 谣言源节点作为种子节点
    :return: 返回本次传播过程最终受到谣言影响的节点总量
    """
    U = seed_nodes.copy() # U是每个时间步的起始激活节点集合
    activated_nodes = seed_nodes.copy()
    T = [len(U)]  # 存放每个时间戳上被激活的节点数量
    newly_activated_nodes = set()  # 存放下个时刻新激活的节点
    while True:
        for u in U:
            for v in G.successors(u):
                if v not in activated_nodes: # 若节点v并没有被激活，则尝试激活它
                    if activateSimulation(G[u][v]['probability']):
                        newly_activated_nodes.add(v)
                        activated_nodes.add(v)
        number_of_newly_activated_nodes = len(newly_activated_nodes)
        if not number_of_newly_activated_nodes:  # 如果没有新的被激活节点，则传播过程结束
            break
        T.append(number_of_newly_activated_nodes)
        U = newly_activated_nodes
        newly_activated_nodes = set()
    result_at_each_time_stamp = np.array(T).cumsum() # 求累和,此时的result列表中的数据代表每个离散时刻上面的总的被谣言激活的节点数量。
    number_of_infected_nodes = result_at_each_time_stamp[-1] - result_at_each_time_stamp[0] # 最终被谣言影响的总节点数量是t > 0 时刻被激活的节点总数
    return number_of_infected_nodes


# 测试函数
def bfs_0_test(G, source_nodes, rumor_community_label0, protection_level = 0.9):
    """
     从已知的谣言源节点出发进行BFS搜索得到所有的谣言社区内部可达的节点以及所有的桥节点
     :param G: 原始的社交图
     :param source_nodes: 谣言源节点（集合类型）
     :param rumor_community_label0: 谣言社区的标号
     :protection_level:预定的谣言保护等级，比如0.9
     :return: 谣言社区内部的谣言可达节点列表A，桥节点列表B-->M0和B0，受到谣言影响的节点最大值k
     """
    # 在每次随机选取谣言源节点之后，要记录A，B，和k的大小，用于求它们各自的均值。

    visited = set() # 初始化一个集合，用来存放访问到的节点
    common_bridges = set() # 普通的桥节点
    special_bridges = set() # 父节点中存在谣言源节点的桥节点,返回该集合时记为M0
    for source0 in source_nodes: # 从多个源节点出发进行BFS搜索
        if source0 not in visited:  # 由于有多个源节点在进行搜索，所以先判断该源节点是否已经被搜索过
            Queue = deque()  # 初始化双端队列，将它作为先入先出队列使用
            visited.add(source0)  # 标记源节点的搜索状态为“已访问”
            Queue.append(source0)  # 将源节点压入队列的队尾（最右端）
            while Queue:  # 当Queue非空时，一直搜索
                source_temp = Queue.popleft()  # 弹出队列首元素(最左端)，作为临时起点
                if G.nodes[source_temp]['community_label'] != rumor_community_label0:  # 如果这个临时起点是桥节点
                    if source_nodes.isdisjoint(set(G.predecessors(source_temp))):
                        common_bridges.add(source_temp)  # 若该source_temp的父节点均不是谣言源节点，则标记它为普通桥节点
                        continue # 由于不用再从桥端节点出发继续进行下一层的搜索，所以跳过当前这次循环，直接从Queue中的下一个节点出发再搜索
                    else:
                        special_bridges.add(source_temp) # 如果该source_temp的父节点中存在谣言源节点时，记为特殊桥节点
                        continue  # 由于不用再从桥端节点出发继续进行下一层的搜索，所以跳过当前这次循环，直接从Queue中的下一个节点出发再搜索
                for out_neighbor in G.successors(source_temp):  # 遍历临时起点的所有子节点
                    if out_neighbor not in visited:
                        visited.add(out_neighbor)  # 标记该节点的搜索状态为“已访问”
                        Queue.append(out_neighbor)
    A = visited - source_nodes - common_bridges - special_bridges # 这是谣言社区内部所有的谣言可达节点,不包括谣言源节点
    Length_A = len(A) # 谣言社区内部谣言可达节点的数量
    M0 = special_bridges # 父节点中存在谣言源节点的桥节点集合，这些节点需要直接进行控制
    Length_M0 = len(M0) # 特殊桥节点的数量
    B0 = common_bridges # 父节点中不存在谣言源节点的桥节点集合，它们是普通的桥节点
    Length_B0 = len(B0) # 普通桥节点的数量
    k = int((1 - protection_level) * Length_A) # 可能会受到谣言影响的节点总数量，预先给定的protection_level代表了保护的等级

    print("最初，谣言社区内部谣言可达节点为：{}，数量为：{}\n,特殊桥节点为{}，数量为：{}\n,普通桥节点为{}，数量为：{}\n".format(A,Length_A,M0,Length_M0,B0,Length_B0))
    # 检测，删除M1之后是否有效
    G_test = G.copy()
    G_test.remove_nodes_from(M0)
    G_test.remove_nodes_from(B0)
    if not bfs_2(G_test, source_nodes, rumor_community_label0):
        print('恭喜！删除M0和M1之后，从源节点出发不可达任何桥节点')
    else:
        print('程序出错！！！！！请检查。')
        exit()
    return A, Length_A, B0, Length_B0, M0, Length_M0, k


def protect_bridge_ends_TEST(G,G_origin, rumor_sources,common_bridges,special_bridges,rumor_community_label0):
    """
    求二部图的最小顶点覆盖来阻断谣言源节点抵达所有的桥节点，返回需要阻断的节点（包括桥节点）
    :param G:子图G1
    :param G_origin:原始图G
    :param rumor_sources:谣言源节点集合
    :param common_bridges:普通桥节点集合
    :param special_bridges:父节点中存在谣言源的节点集合
    :param rumor_community_label0: 谣言社区编号
    :return: 封锁节点列表M1
    """
    G_bipartite = nx.Graph() # 无向无权二部图
    parents_of_common_bridges = set() # 一个集合：存放普通桥节点的父节点
    G1 = G.copy()
    G1.remove_nodes_from(special_bridges)
    for node in common_bridges:
        G_bipartite.add_node(node,bridge=True)
        for in_neighbor in G.predecessors(node):  # 在图G1中遍历这个普通桥节点的所有父节点(非桥节点)
            if in_neighbor not in common_bridges:
                G_bipartite.add_node(in_neighbor, parent=True)
                parents_of_common_bridges.add(in_neighbor)
                G_bipartite.add_edge(node,in_neighbor)
    #print(nx.is_connected(G_bipartite))
   # print('普通桥节点的谣言可达父节点总数为：',len(parents_of_common_bridges))
    #print('二部图的顶点总数为：',G_bipartite.number_of_nodes())
    M1 = set()
    while G_bipartite.number_of_edges():
        matching = nx.bipartite.eppstein_matching(G_bipartite,top_nodes=common_bridges-M1) # 最大匹配
        min_vertex_cover = nx.bipartite.to_vertex_cover(G_bipartite, matching, top_nodes=common_bridges-M1) # 最小集合覆盖的节点集合
        M1.update(min_vertex_cover) # 这些节点若从原始图G中移除，则从所有的源节点出发均不可达普通的桥节点
        G_bipartite.remove_nodes_from(M1)
    print('已经得到my_method在保护桥节点阶段需要控制的节点,总量为：',len(M1))
   # print('二部图删除最小顶点覆盖之后的边数为：',G_bipartite.number_of_edges())
    #print('剩余节点总量是：',G_bipartite.number_of_nodes(),'\n',G_bipartite.nodes(data=True))
    #print('剩余边是：',G_bipartite.edges())

    rest_of_common_bridges = common_bridges - M1  # 未被封锁的普通桥节点
    #print('剩余普通的桥节点的数量为：', len(rest_of_common_bridges))
    rumor_reachable_parents_count = 0
    G1.remove_nodes_from(M1)
    if len(rest_of_common_bridges):
        for node in rest_of_common_bridges:
            for parent in G1.predecessors(node):
                if G1.nodes[parent]['community_label'] == rumor_community_label0:
                    rumor_reachable_parents_count += 1
    #print('剩余普通的桥节点的谣言可达父节点的数量为：',rumor_reachable_parents_count)


    # 检测，删除M1之后是否有效
    G_test = G_origin.copy()
    G_test.remove_nodes_from(M1)
    G_test.remove_nodes_from(special_bridges)


    if not bfs_2(G_test, rumor_sources, rumor_community_label0):
        print('恭喜！删除M0和M1之后，从源节点出发不可达任何桥节点')
        #exit()
    else:
        print('程序出错！！！！！请检查。')
        exit()
    return M1


def test_for_protection_of_bridge_ends(G,M0_temp,M1_temp,source_nodes,common_bridges,rumor_community_label0):
    """
    从原图的浅拷贝中删除节点集合M0和M1，检查是否从所有的源节点出发均不可达所有的普通桥节点
    :param G: 原始的社交网络图
    :param M0_temp: 父节点中存在谣言源节点的桥节点
    :param M1_temp: protect_bridge_ends所选出的需要删除的节点
    :param source_nodes: 谣言源节点集合
    :param common_bridges: 普通桥节点集合
    :return: 逻辑值，如果均不可达，那么返回真。否则返回假。
    """
    G_test = G.copy()
    G_test.remove_nodes_from(M0_temp)
    G_test.remove_nodes_from(M1_temp)
    reachable_paths_num0 = 0
    reachable_paths_num1 = 0

    # 利用库函数has_path检测是否从谣言源节点可达普通桥节点
    for rumor_source in source_nodes:
        for common_bridge in common_bridges:
            if nx.has_path(G_test,rumor_source,common_bridge):
                reachable_paths_num0 += 1

    # 利用BFS搜索检测是否从源节点可达普通桥节点
    visited = set()  # 初始化一个集合，用来存放访问到的节点
    for source0 in source_nodes:  # 从多个源节点出发进行BFS搜索
        if source0 not in visited:  # 由于有多个源节点在进行搜索，所以先判断该源节点是否已经被搜索过
            Queue = deque()  # 初始化双端队列，将它作为先入先出队列使用
            visited.add(source0)  # 标记源节点的搜索状态为“已访问”
            Queue.append(source0)  # 将源节点压入队列的队尾（最右端）
            while Queue:  # 当Queue非空时，一直搜索
                source_temp = Queue.popleft()  # 弹出队列首元素(最左端)，作为临时起点
                if G_test.nodes[source_temp]['community_label'] != rumor_community_label0:  # 如果这个临时起点是桥节点
                    reachable_paths_num1 += 1
                for out_neighbor in G_test.successors(source_temp):  # 遍历临时起点的所有子节点
                    if out_neighbor not in visited:
                        visited.add(out_neighbor)  # 标记该节点的搜索状态为“已访问”
                        Queue.append(out_neighbor)

    print('reachable_paths_num0 = {}\n,reachable_paths_num1 = {}\n'.format(reachable_paths_num0,reachable_paths_num1))
    if reachable_paths_num0 == 0 and reachable_paths_num1 == 0:
        A_after_control = visited - source_nodes
        print("控制之后，谣言源节点在社区内部能抵达的节点为{}，数量为{}".format(A_after_control,len(A_after_control)))

