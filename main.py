from linear_powergrid import LinearPowerGrid
from decentralized.estimators import Richardson
import extract_config

CENTRAL = True

# 不显示warnings
import warnings
warnings.filterwarnings("ignore")
#

psap_conf = 'ieee118psp.txt'
psap_rule_line = './rules/rule_ieee118psap_line'
psap_rule_bus = './rules/rule_ieee118psap_bus'

cdf_conf = 'ieee118cdf.txt'
cdf_rule_bus = './rules/rule_ieee118cdf_bus'
cdf_rule_branch = './rules/rule_ieee118cdf_branch'

def main():
  ### CDF Format ###
  bus_cdf = extract_config.tools(cdf_conf, cdf_rule_bus, 0)
  branch_cdf = extract_config.tools(cdf_conf, cdf_rule_branch, 1)

  s_bus_code = bus_cdf.get_items(0)
  s_bus_state = bus_cdf.get_items([6, 8])
  s_branch_code = branch_cdf.get_items([0,2])   # extract the bus number of out-in side.

  resistance = branch_cdf.get_items(6)  # line resistance
  reactance = branch_cdf.get_items(7)   # line reactance
  shunt_conductance = bus_cdf.get_items(16)  # Shunt conductance
  shunt_susceptance = bus_cdf.get_items(17)  # Shunt susceptance

  bus_type = bus_cdf.get_items(5)   # Bus type: 0 -> Load
                                    #           2 -> Generation
                                    #           3 -> Reference
  ### 电网建模 ###
  x_operation = s_bus_state
  PMU = [3,5,9,12,15,17,21,25,114,28,40,37,34,70,71,53,56,45,49,62,64,68,105,110,76,79,100,92,96,85,86,89]

  if CENTRAL:
    model = LinearPowerGrid(118)
    model.set_local(s_bus_code, bus_type, PMU)
    model.gen_gbbsh(s_branch_code, resistance, reactance, shunt_conductance, shunt_susceptance)
    model.set_edge(x_operation)
    #model.delete_reference_bus()
    ### 集中式建模完毕 ###
    #model.inject_baddata(moment=5, probability=50)  # 在第5个仿真时刻开始每个仪表有10/10e6的概率产生坏数据
    #model.inject_falsedata(moment=5)  # 在第5个仿真时刻开始注入隐匿虚假数据
    #model.inject_falsedata_PCA(moment=500)  # 在第30个仿真时刻开始注入PCA隐匿虚假数据
    model.estimator(sim_time=10, variance=1)  # 在状态幅值0-1内任意变化的情况下进行估计
    #model.summary()

  else:
    model = Richardson(118)
    model.set_local(s_bus_code, bus_type, PMU)
    model.gen_gbbsh(s_branch_code, resistance, reactance, shunt_conductance, shunt_susceptance)
    model.set_edge(x_operation)
    #model.summary()
    #model.delete_reference_bus()
########## 注入虚假数据 ###########
    #falsedata = model.inject_baddata(sparse_amount=100, amptitude=5)  # 注入5个幅值0-100的虚假数据
    #falsedata = model.inject_falsedata_PCA(sparse_amount=2, amptitude=100)  # 注入5个幅值0-100的虚假数据
    #print(falsedata['measurement_injected'])
    model.detect_baddata(centralized=True)
########## 删除某些测量值 ###########
    '''
    res = model.delete_measurements(busTobeDeleted = falsedata['measurement_injected_sequence'])
    if res is False:
      for i in range(100):
        falsedata = model.inject_falsedata(sparse_amount=falsedata['state_injected_amount'], amptitude=falsedata['amptitude'], measure_tobe_injected=falsedata['measurement_injected'], delete_previous_injected=True)  # 重新注入虚假数据
        res = model.delete_measurements(busTobeDeleted = falsedata['measurement_injected'])
        if res is True:
          break
    # model.delete_measurements(amount=30)
    model.detect_falsedata()
    '''

######## 分布式建模 ###########
    node1 = [1,2,3,4,5,6,7,8,9,10,11,12,14,16,117]
    node2 = [13,15,17,18,19,20,21,22,23,25,26,27,28,29,30,31,32,33,113,114,115]
    node3 = [24,38,70,71,72,73,74]
    node4 = [34,35,36,37,39,40,41,42,43]
    node5 = [44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,77,80,81,100,116]
    node6 = [75,76,78,79,82,95,96,97,98,118]
    node7 = [83,84,85,86,87,88,89,90,91,92,93,94]
    node8 = [99,101,102,103,104,105,106,107,108,109,110,111,112]
    nodes = [node1,node2,node3,node4,node5,node6,node7,node8]
    '''
    node1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,117]
    node2 = [23,25,26,27,28,29,31,32,113,114,115]
    node3 = [5,16,17,18,19,20,21,22,24,30,33,34,35,36,37,39,40,71,72,73]
    node4 = [38,41,42,43,44,45,46,47,48,69,70,74,75,76,77,118]
    node5 = [49,50,51,54,65,66,68,78,79,80,81,82,95,96,97,98,99,116]
    node6 = [52,53,55,56,57,58,59,60,61,62,63,64,67]
    node7 = [83,84,85,86,87,88,89,90,91,92,93,94,100,101,102,103,104,105,106,107,108,109,110,111,112]
    nodes = [node1,node2,node3,node4,node5,node6,node7]
    '''
    #model.set_variance(R)  # 方差矩阵 (Default: 单位矩阵)
    model.set_nodes(nodes, x_operation)
    # model.summary()
  ############# 分布式建模完毕 ###############
    # print(model.gen_graph())
    #model.print_H(1)
    # falsedata = model.inject_falsedata(sparse_amount=10, amptitude=60)  # 注入5个幅值0-100的虚假数据
  ### 分布式估计 ###
    model.estimator(main_period=300 ,gamma_period=100, is_async=True)
    model.detect_baddata(is_plot=True)

if __name__ == '__main__':
  main()
