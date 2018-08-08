# SF随机网络的实验
# ('5', 1647)
import networkx as nx
import numpy as np
import time
from my_experiment import my_function as mf


def experiment_on_SF(percentage):
    G = nx.read_gexf('./graph_preprocess/SF_graph.gexf')
    # 我选择社区标号为5的社区为谣言社区，因为它规模很大，总节点数为1647个,社区划分之后的总社区数量是28个。
    rumor_community_size = 1647  # 谣言社区包含的节点总数
    rumor_community_label = '5'  # 选择第3号社区作为谣言社区
    rumor_sources_percentage = percentage  # 谣言源节点比例
    rumor_sources_number = int(rumor_community_size * rumor_sources_percentage)
    # print(rumor_sources_number)
    rumor_community_nodes = [x for x in G.nodes if G.nodes[x]['community_label'] == rumor_community_label]
    # print(len(rumor_community_nodes)) # 2191
    simulation_num = 100  # 仿真的次数

    # 用于保存程序运行结果的变量初始化
    # 四个方法的单次结果列表
    my_result = list()
    k_core_result = list()
    max_degree_result = list()
    betweenness_centrality_result = list()
    # 三种节点数量的结果列表
    num_of_reachable_nodes_from_sources = list()  # 谣言社区内部谣言可达的节点数量
    num_of_common_bridges = list()  # 普通桥节点的数量
    num_of_special_bridges = list()  # 特殊桥节点的数量

    # 用于k核方法和maxDegree方法的初始量
    # 核数字典
    all_k_core_dict = nx.core_number(G)
    # 度数字典
    all_degree_dict = G.degree()

    # for 循环 遍历所有的谣言源规模进行仿真，对于每个谣言源规模都随机采样谣言源节点1000次，
    # 得到相应的四种算法所需封锁节点的数量和算法所需的运行时间，最后得到每个谣言源规模所需的平均封锁节点数量和平均运行时间
    simulation_count = 0  # 仿真次数计数器

    # 四个算法的运行时间列表
    my_running_time = list()
    k_core_running_time = list()
    max_degree_running_time = list()
    betweenness_centrality_running_time = list()

    while simulation_count < simulation_num:
        # 预处理步骤，得到子图G1
        rumor_originators = mf.random_chose_rumor_sources(rumor_community_nodes,
                                                          rumor_sources_number)  # 从谣言社区内部随机选择节点作为谣言源节点
        A, Length_A, B0, Length_B0, M0, Length_M0, k, time0 = mf.bfs_0(G, rumor_originators, rumor_community_label,
                                                                       protection_level=0.9)  # 从源节点出发BFS搜索
        num_of_reachable_nodes_from_sources.append(Length_A)  # 添加本次实验中谣言社区内部谣言可达的节点数量
        num_of_common_bridges.append(Length_B0)  # 添加本次实验中普通桥节点的数量
        num_of_special_bridges.append(Length_M0)  # 添加本次实验中特殊桥节点的数量
        G1, time1 = mf.get_subgraph(G, rumor_originators, A, B0, M0)  # 得到原始图G的子图G1
        basic_running_time = time0 + time1  # time0和time1作为三种方法的公用损耗时间，单位是秒
        print("basic_running_time = ", basic_running_time, '(s)')

        # 我的算法
        my_start_time = time.clock()  # 我的算法开始时间
        M1 = mf.protect_bridge_ends_v4(G1, B0)  # 对于从谣言源节点到达所有的普通桥节点的路径求一个集合覆盖，得到封锁节点集合M1
        my_result.append(mf.my_method(G1, k, rumor_originators, M0, Length_M0, M1, 10))  # 记录我的方法的结果
        my_end_time = time.clock()  # 我的算法结束时间
        my_running_time.append(basic_running_time + my_end_time - my_start_time)  # 我的方法所用的时间
        print('我的方法所用的时间:', my_running_time[-1], '(s)')

        # k核算法
        k_core_start_time = time.clock()  # k核算法开始时间
        k_core_result.append(
            mf.k_core_method(G1, all_k_core_dict, k, rumor_originators, rumor_community_label, M0, Length_M0,
                             10))  # 记录k-core方法的结果
        k_core_end_time = time.clock()  # k核算法结束时间
        k_core_running_time.append(basic_running_time + k_core_end_time - k_core_start_time)  # k核方法所用的时间
        print('k核方法所用的时间:', k_core_running_time[-1], '(s)')

        # 最大度算法
        max_degree_start_time = time.clock()  # 最大度算法开始时间
        max_degree_result.append(
            mf.max_degree_method(G1, all_degree_dict, k, rumor_originators, rumor_community_label, M0, Length_M0,
                                 10))  # 记录maxDegree方法的结果
        max_degree_end_time = time.clock()  # 最大度算法结束时间
        max_degree_running_time.append(
            basic_running_time + max_degree_end_time - max_degree_start_time)  # maxDegree方法所需时间
        print('maxDegree方法所用的时间:', max_degree_running_time[-1])

        # 介数中心度算法
        betweenness_centrality_start_time = time.clock()  # 介数中心度算法开始时间
        # 介数中心度字典
        all_betweenness_centrality_dict = nx.betweenness_centrality(G1)
        betweenness_centrality_result.append(
            mf.betweenness_centrality_method(G1, all_betweenness_centrality_dict, k, rumor_originators,
                                             rumor_community_label, M0, Length_M0, 10))
        betweenness_centrality_end_time = time.clock()  # 介数中心度算法结束时间
        betweenness_centrality_running_time.append(
            basic_running_time + betweenness_centrality_end_time - betweenness_centrality_start_time)  # 介数中心度算法所需时间
        print('betweenness_centrality方法所用的时间:', betweenness_centrality_running_time[-1])

        simulation_count += 1  # 四个算法都运行一次之后，仿真次数增加1次
        print('现在的仿真次数是：', simulation_count)

    # 将所有的结果转化为numpy矩阵，便于存储和处理

    my_final_result = np.array(my_result).T  # 每一列是一种谣言源规模对应的结果,矩阵有simulation_num行，10列
    k_core_result = np.array(k_core_result).T  # 每一列是一种谣言源规模对应的结果,矩阵有simulation_num行，10列
    max_degree_result = np.array(max_degree_result).T  # 每一列是一种谣言源规模对应的结果,矩阵有simulation_num行，10列

    betweenness_centrality_result = np.array(betweenness_centrality_result).T  # 每一列是一种谣言源规模对应的结果,矩阵有simulation_num行，10列

    num_of_reachable_nodes_from_sources = np.array(num_of_reachable_nodes_from_sources).T  # ,矩阵有simulation_num行，10列
    num_of_common_bridges = np.array(num_of_common_bridges).T  # 矩阵有simulation_num行，10列
    num_of_special_bridges = np.array(num_of_special_bridges).T  # 矩阵有simulation_num行，10列
    run_time_my_method = np.array(my_running_time).T  # 每一列是一个谣言规模每次仿真所需的运行时间，单位是秒
    run_time_k_core = np.array(k_core_running_time).T  # 每一列是一个谣言规模每次仿真所需的运行时间，单位是秒
    run_time_max_degree = np.array(max_degree_running_time).T  # 每一列是一个谣言规模每次仿真所需的运行时间，单位是秒

    run_time_betweenness_centrality = np.array(betweenness_centrality_running_time).T  # 每一列是一个谣言规模每次仿真所需的运行时间，单位是秒

    # 平均结果

    my_final_average_result = (my_final_result.mean(axis=0)).astype(int)  # 按每列求均值,且取整
    k_core_average_result = (k_core_result.mean(axis=0)).astype(int)  # 按每列求均值,且取整
    max_degree_average_result = (max_degree_result.mean(axis=0)).astype(int)  # 按每列求均值,且取整

    betweenness_centrality_average_result = (betweenness_centrality_result.mean(axis=0)).astype(int)  # 按每列求均值,且取整

    average_num_of_reachable_nodes_from_sources = (num_of_reachable_nodes_from_sources.mean(axis=0)).astype(
        int)  # 按每列求均值,且取整
    average_num_of_common_bridges = (num_of_common_bridges.mean(axis=0)).astype(int)  # 按每列求均值,且取整
    average_num_of_special_bridges = (num_of_special_bridges.mean(axis=0)).astype(int)  # 按每列求均值,且取整
    average_run_time_my_method = (run_time_my_method.mean(axis=0))
    average_run_time_k_core = (run_time_k_core.mean(axis=0))
    average_run_time_max_degree = (run_time_max_degree.mean(axis=0))

    average_run_time_betweenness_centrality = (run_time_betweenness_centrality.mean(axis=0))

    # 存储不同谣言规模下的结果
    f1 = open(r'./experiment_results/Scale_Free_network/results_new_SF.txt', 'a')
    f1.write("******************************************************\n")
    f1.write("for {} rumor sources\n".format(rumor_sources_number))
    f1.write("my results:{}\n".format(my_final_average_result))
    f1.write("k-core results:{}\n".format(k_core_average_result))
    f1.write("max degree results:{}\n".format(max_degree_average_result))
    f1.write("bc results:{}\n".format(betweenness_centrality_average_result))
    f1.write('average |A| :{}\n'.format(average_num_of_reachable_nodes_from_sources))
    f1.write('average |B0|:{}\n'.format(average_num_of_common_bridges))
    f1.write('average |M0|:{}\n'.format(average_num_of_special_bridges))
    f1.write('average run time of my method:{}\n'.format(average_run_time_my_method))
    f1.write('average run time of K-core method:{}\n'.format(average_run_time_k_core))
    f1.write('average run time of MaxDegree:{}\n'.format(average_run_time_max_degree))
    f1.write('average run time of BC method:{}\n'.format(average_run_time_betweenness_centrality))
    f1.write("******************************************************\n\n")
    f1.close()


if __name__ == '__main__':
    percentage_list = [0.015, 0.025, 0.035, 0.045]
    for i in percentage_list:
        experiment_on_SF(i)