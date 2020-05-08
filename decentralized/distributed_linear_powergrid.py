from linear_powergrid import LinearPowerGrid
from decentralized.estimators import Richardson,Stocastic

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from scipy.linalg import block_diag
import copy
import time
import json
import os

class ComplexEncoder(json.JSONEncoder):
  def default(self, obj):
    # 如果要转换的对象是复数类型，程序负责处理
    if isinstance(obj, complex):
      return {"__complex__": 'true', 'real': obj.real, 'imag': obj.imag}
    # 对于其他类型，还使用JSONEncoder的默认处理
    return json.JSONEncoder.default(self, obj)

def as_complex(dct):
  if '__complex__' in dct:
    return complex(dct['real'], dct['imag'])
  return dct

class DistributedLinearPowerGrid(LinearPowerGrid):
  def __init__(self, nodes = [], pmu=[],conf_dict={}):
    super().__init__(pmu=pmu, conf_dict=conf_dict)
    self.is_distribute = True
    self.x_est_center_list = []
    self.x_est_distribute = np.zeros((self.size*2,1))
    self.x_est_distribute_lists = []
    self.Phi_distribute = []
    self.cluster = {}
    self.is_distribute = True
    # 簇配置
    if self.model_name == 'PowerGrid':
      self.nodes_num = len(nodes)   # 表示有多少个节点
    elif self.model_name == 'WSNs':
      self.nodes_num = self.size
      nodes = []
      for i in range(1,self.size+1):
        nodes.append([i])
    # 分布式配置
    self.reorder(nodes) # 9.8s
    self.x_real_list, self.z_distribute, self.H_distribute,self.Phi_distribute = self.distribulize(self.x_real, self.z_observed, self.H, self.Phi, graph=True) # 0.018s
    # GUI 配置
    main_period = int(conf_dict['decentralized_main_period'])
    gamma_period = int(conf_dict['decentralized_child_period'])
    self.decentralized_method = conf_dict['decentralized_method']
    is_async = conf_dict['is_asyn']
    if conf_dict['decentralized_method'] == 'Richardson(finite)': is_finite = True; main_period=self.nodes_num-2
    else: is_finite = False
    diff_limit = int(conf_dict['asyn_tolerance_diff'])
    self.estimator_conf_dict = {
      'main_period': main_period,
      'gamma_period': gamma_period,
      'is_async': is_async,
      'is_finite': is_finite,
      'diff_limit': diff_limit,
      'attacked_nodes': None,
      'is_DoS': conf_dict['is_DoS'],
    }
    if conf_dict['is_DoS'] is True:
      self.estimator_conf_dict['DoS_dict'] = conf_dict['DoS_dict']
    self.is_plot = conf_dict['is_plot']

  def jaccobi_H_distribute(self, x_operation):
    # 初始化
    Jaccobi_distribue = []
    Phi_distribute = []
    node_list = list(map(int,self.bus_info_dict.keys()))
    # 配置总线测量
    cnt = 0
    for cluster,cluster_info_dict in self.cluster_info_dict.items():
      Jaccobi_H = np.mat(np.zeros([0, 2*self.size]), dtype=complex)   # 量测矩阵
      for bus in cluster_info_dict['cluster']:
        if self.bus_info_dict[bus]['attr'] == 'PMU':
          a = np.mat(np.zeros([2, 2*self.size]))
          a[0,cnt] = 1
          a[1,cnt+1] = 1
          Jaccobi_H = np.row_stack((Jaccobi_H, a))
        else:
          a = np.mat(np.zeros([1, 2*self.size]))
          a[0,cnt] = 1
          Jaccobi_H = np.row_stack((Jaccobi_H, a))
        # 配置边测量
        for connect_info_dict in self.bus_info_dict[bus]['connect']:
          a = np.mat(np.zeros([2, 2*self.size]), dtype=complex)
          # Substitute operation point x0 and g_ij, b_ij, bsh_ij to jacobbi matrices B_ij and B_ji
          for i,J_i in enumerate(self.jacobi_h_ij):
            for j,J_ij in enumerate(J_i):
              for k,symbol_i in enumerate(self.state_i_symbol):
                J_ij = J_ij.subs(symbol_i, x_operation[cnt+k,0])
              for k,symbol_j in enumerate(self.state_j_symbol):
                J_ij = J_ij.subs(symbol_j, x_operation[2*node_list.index(connect_info_dict['dst'])+k,0])
              for k,symbol_value in enumerate(self.value_symbol):
                J_ij = J_ij.subs(symbol_value, connect_info_dict['para'][k])
              a[i, cnt+j] = J_ij
          for j,J_j in enumerate(self.jacobi_h_ji):
            for i,J_ji in enumerate(J_j):
              for k,symbol_i in enumerate(self.state_i_symbol):
                J_ji = J_ji.subs(symbol_i, x_operation[cnt+k,0])
              for k,symbol_j in enumerate(self.state_j_symbol):
                J_ji = J_ji.subs(symbol_j, x_operation[2*node_list.index(connect_info_dict['dst'])+k,0])
              for k,symbol_value in enumerate(self.value_symbol):
                J_ji = J_ji.subs(symbol_value, connect_info_dict['para'][k])
              a[j, 2*node_list.index(connect_info_dict['dst'])+i] = J_ji
          Jaccobi_H = np.row_stack((Jaccobi_H, a))  # Augment H
        cnt += 2
      Jaccobi_distribue.append(Jaccobi_H)
    Jaccobi_H = np.vstack([x for y in Jaccobi_distribue for x in y])
    Phi = Jaccobi_H.H * self.R_I * Jaccobi_H
    # distribulize H
    H_distribute = []
    col_cnt = 0
    for cluster_num,Jaccobi_row in enumerate(Jaccobi_distribue):
      H_distribute.append([])
      for col in self.node_col_amount:
        H_distribute[cluster_num].append(Jaccobi_row[:, col_cnt:col_cnt+col])
        col_cnt += col
    return Jaccobi_H, H_distribute, Phi

  def reorder(self, cluster_s):
    """
      将系统按节点分割, 记录重新排列后的集中式次序
        并将计算中不变的参数分布化得到分布化后的列表(R,x_real)

    输入
    ----
      cluster_s: [[...nodes...], [...], ...]
 
    得到
    ---- 
      nodes_num: 分割成多少个节点
      node_row_amount[]: 每个节点有多少行(测量值) (顺序为节点顺序)
      node_col_amount[]: 每个节点有多少列(状态值) (顺序为节点顺序)
      R_I_distribute_diag[]: 每个节点的局部协方差矩阵(...)
 
    返回
    ----
    {<cluster_num>:{'cluster':[...buses...], 'connect':[...clusters...], 'row_amount':<>, 'col_amount':<>}, ...}
    { '<cluster_num>':
      { 'cluster': [...nodes...]
        'connect':
          [ { 'dst':'<连接节点号>'
              'para':'<连接参数>'
          } ]
        'row_amount':<量测个数>
        'col_amount':<状态个数>
      }

      ...
    }
    """
    nodes = []
    for cluster in cluster_s:
      nodes += cluster
    ##
    #row_seq = []
    self.node_row_amount=[]   # 表示每个节点有多少测量值
    self.node_col_amount=[]   # 表示每个节点有多少状态值
    self.node_state_split=[]
    self.node_measure_split=[]
    ##
    bus_connection_reorder = {}
    cluster_info_dict = {}
    # 将self.bus_info_dict重新排列, 并产生cluster_info_dict
    for cnt,cluster in enumerate(cluster_s):
      cluster_info_dict[cnt] = {}
      cluster_info_dict[cnt]['connect'] = []
      row_amount = 0
      if self.model_name == 'PowerGrid':
        for node in cluster:
          ## 假设所有边测量的仪表都部署在输出端，若两个节点之间有连接，那么输出总线在哪个节点，哪个节点就包含另一个节点的数据。
          # 不知道怎么写
          ##
          bus_connection_reorder[node] = self.bus_info_dict[node]
          if bus_connection_reorder[node]['attr'] == 'PMU':
            row_amount += 2
          else:
            row_amount += 1
          row_amount += len(bus_connection_reorder[node]['connect'])*len(self.h)
      elif self.model_name == 'WSNs':
        for node in cluster:
          bus_connection_reorder[node] = self.bus_info_dict[node]
          if bus_connection_reorder[node]['attr'] == 'PMU':
            row_amount += 2
          row_amount += len(bus_connection_reorder[node]['connect'])*len(self.h)
      cluster_info_dict[cnt]['cluster'] = cluster
      #cluster_info_dict[cnt]['connect'].append({'dst':?, 'para':?})
      cluster_info_dict[cnt]['row_amount'] = row_amount
      cluster_info_dict[cnt]['col_amount'] = len(cluster)*2
      ##
      self.node_row_amount.append(cluster_info_dict[cnt]['row_amount'])
      self.node_col_amount.append(cluster_info_dict[cnt]['col_amount'])
      if cnt == 0:
        self.node_state_split.append(self.node_col_amount[cnt])
        self.node_measure_split.append(self.node_row_amount[cnt])
      elif cnt != self.nodes_num-1:
        self.node_state_split.append(self.node_col_amount[cnt]+self.node_state_split[cnt-1]) 
        self.node_measure_split.append(self.node_row_amount[cnt]+self.node_measure_split[cnt-1])
      ##
    # cols reordered sequence
    col_seq = []
    for bus in list(bus_connection_reorder.keys()):
      if (bus in self.nodes_ref) and (self.is_reference_deleted is True):
        continue
      col_seq.append(2*(bus-1))
      col_seq.append(2*(bus-1)+1)
    # 排序转换矩阵
    #self.row_reorder_matrix = np.mat(np.eye(self.measure_size)[row_seq,:])
    self.col_reorder_matrix = np.mat(np.eye(self.state_size)[:,col_seq])
    # Reorder
    self.bus_info_dict, self.cluster_info_dict = bus_connection_reorder,cluster_info_dict
    self.x_real = self.col_reorder_matrix.T * self.x_real
    self.set_variance_matrix()
    self.H,self.Phi = self.jaccobi_H(self.x_real)
    self.z_observed = self.create_measurement(self.x_real) + self.R_error * np.mat(np.random.random((self.measure_size,1)))
    #print(self.z_observed-self.create_measurement(self.x_real));exit() # 看看噪声多大
    # 分布化R_I
    R_I_distribute_diag = []
    row_cnt = 0
    for row in self.node_row_amount:
      R_I_distribute_diag.append(self.R_I[row_cnt:row_cnt+row, row_cnt:row_cnt+row])
      row_cnt += row
    self.R_I_distribute_diag = R_I_distribute_diag

  def distribulize(self, state_vec, measure_vec, Jaccobi_mat, Phi_mat, graph=False):
    """
    将H矩阵和z向量的相关参数按节点分块重新排列, 并得到分布化的列表

    得到
    ----
      z_distribute[]    每个节点的测量向量 (顺序为节点顺序)
      x_real_list[]     每个节点上一时刻的状态向量 (...)
      H_distribute[]    每个节点的局部H矩阵 (...)
      nodes_graph(np.mat) 各节点间的连接图矩阵
      Phi_graph(np.mat)   各节点间Phi矩阵的连接图矩阵
      Precondition_center(np.mat) precondition矩阵
      Precondition_distribute{}   空的字典, 在这里先声明, 之后迭代计算时用到 

    返回
    ----
      分布化后的结果
    """
    # 分布化 x_real
    state_vec_distribute = copy.deepcopy(np.array_split(state_vec, self.node_state_split))
    # 分布化 z_observed
    measure_vec_distribute = copy.deepcopy(np.array_split(measure_vec, self.node_measure_split))
    # distribulize H
    Jaccobi_mat_distribute = []
    Jaccobi_tmp = copy.deepcopy(np.array_split(Jaccobi_mat, self.node_measure_split, axis=0)) # 按行分割
    for i in Jaccobi_tmp:
      Jaccobi_mat_distribute.append(np.array_split(i, self.node_state_split, axis=1))  # 按列分割
    # distribulize Phi
    Phi_mat_distribute = []
    Phi_tmp = copy.deepcopy(np.array_split(Phi_mat, self.node_state_split, axis=0)) # 按行分割
    for i in Phi_tmp:
      Phi_mat_distribute.append(np.array_split(i, self.node_state_split, axis=1))  # 按列分割
    if graph is True:
      # real graph: 通过H分块矩阵是否0矩阵来判断
      g=np.mat(np.zeros([self.nodes_num, self.nodes_num]), dtype='int')
      i=0
      for m in Jaccobi_mat_distribute:
        j=0
        for n in m:
          if (len(n.nonzero()[0]) != 0):# and (i!=j):
            g[i,j] = 1
            g[j,i] = 1
          j+=1
        i+=1
      self.nodes_graph = g
      # Phi graph
      #pg = self.nodes_graph*self.nodes_graph
      pg = np.mat(np.zeros([self.nodes_num, self.nodes_num]), dtype='int')
      i=0
      for m in Phi_mat_distribute:
        j=0
        for n in m:
          if (len(n.nonzero()[0]) != 0):# and (i!=j):
            pg[i,j] = 1
          j+=1
        i+=1
      self.Phi_graph = pg
      # cluster连接图
      self.neighbors_dict = {}
      for num in range(self.nodes_num):
        self.neighbors_dict[num] = {'him': self.get_neighbor(num, -1),'me': self.get_neighbor(num, 1),'neighbors': self.get_neighbor(num, 2), 'nn':self.get_neighbor(num, 3)}
      # cluster连接图2
      for num in range(self.nodes_num):
        self.cluster_info_dict[num]['connect'] = self.get_neighbor(num, -1)
    return state_vec_distribute,measure_vec_distribute,Jaccobi_mat_distribute,Phi_mat_distribute

  def estimator(self, is_async=False, plot=[]):
    """
      进行分布式估计

    输入
    ---- 
      self.sim_time: 仿真时间
      is_async: 是否使用异步算法: [False,True]
      plot 第个时刻需要画分布式迭代图: [#,#,#...]
  
    返回
    ----
      NULL
    """
    # 哪些节点遭受FDI攻击
    if self.is_FDI is True or self.is_FDI_PCA is True:
      self.attacked_nodes = []
      for i in self.FDI_conf_dic['which_state']:
        for nodenum,nodedict in self.cluster_info_dict.items():
          if i/2 in nodedict['cluster']:
            self.attacked_nodes.append(nodenum)
      self.attacked_nodes = list(set(self.attacked_nodes)) #去重复
      self.attacked_nodes.sort() #排序
      self.estimator_conf_dict['attacked_nodes'] = self.attacked_nodes
    # DoS攻击配置

    # 初始化分布式估计器
    if self.decentralized_method == 'Richardson' or self.decentralized_method == 'Richardson(finite)':
      child_estimator = Richardson(self.cluster_info_dict, self.neighbors_dict, self.x_real, self.x_est_center, self.estimator_conf_dict)
    elif self.decentralized_method == 'Stocastics':
      child_estimator = Stocastic(self.cluster_info_dict, self.neighbors_dict, self.x_real, self.x_est_center, self.estimator_conf_dict)
    else:
      print(self.decentralized_method+'is not known!')
      exit()
    # 估计器配置
    pass
    # 开始
    res = {'sim_time':self.sim_time, 'state_est':np.empty((self.state_size,0)), 'state_predict':np.zeros((self.state_size,1)), 'state_real':np.empty((self.state_size,0)), 'state_error':np.empty((self.state_size,0))}
    a = np.copy(self.x_real)
    b = np.zeros((self.size*2,1), dtype=complex)
    state_error_mat = np.mat(np.eye(self.state_size))
    self.x_est_distribute = self.x_real # 第一次将真实状态作为上一次的估计结果（也可以使用全0向量，但是需要将next函数放在循环尾部）
    for t in range(self.sim_time):
      self.next() # 进入下一时刻
      if self.is_baddata is True:
        self.__inject_baddata(t)
      if self.is_FDI is True:
        self.__inject_falsedata(t)
      elif self.is_FDI_PCA is True:
        self.z_observed_history = np.column_stack((self.z_observed_history, self.z_observed))
        self.__inject_falsedata_PCA(t)
      # 全局状态估计
      is_bad_centralized,residual_centralized = super().estimator(once=True)
      self.x_est_center_list = copy.deepcopy(np.array_split(self.x_est_center, self.node_state_split))
      # 分布式状态估计
      self.x_est_distribute_lists,self.x_est_distribute = child_estimator.algorithm(self.H_distribute, self.Phi_distribute, self.R_I_distribute_diag, self.z_distribute, is_plot=self.is_plot) 
      is_bad,residual = self.detect_baddata(is_plot=self.is_plot)
      # 预测
      a,b,state_error_mat = self.predict(self.x_est_distribute,[a,b,state_error_mat])
      # 画出中间结果
      if self.is_plot is True:
        # 估计误差(按节点划分)
        plt.figure('估计误差(乱序)')
        # 与全局估计相比
        plt.subplot(211)
        plt.title('与全局估计相比')
        plt.plot(range(self.state_size), self.x_est_distribute - self.x_est_center, 'b.') # 点图
        plt.xlim([0,self.state_size])
        plt.xlabel("状态")
        plt.ylabel("误差")
        # 与真实状态相比
        plt.subplot(212)
        tmp_cnt=0
        plt.title('与真实状态相比')
        plt.plot(range(self.state_size), self.x_est_distribute - self.x_real, 'b.') # 点图
        plt.xlim([0,self.state_size])
        plt.xlabel("状态")
        plt.ylabel("误差")
        plt.draw()
        # 估计误差(按状态划分)
        plt.figure('估计误差(顺序)')
        # 与全局估计相比
        plt.subplot(211)
        plt.title('与全局估计相比')
        plt.plot(range(self.state_size), self.col_reorder_matrix.T * (self.x_est_distribute - self.x_est_center), 'b.') # 点图
        plt.xlim([0,self.state_size])
        plt.xlabel("状态")
        plt.ylabel("误差")
        # 与真实状态相比
        plt.subplot(212)
        tmp_cnt=0
        plt.title('与真实状态相比')
        plt.plot(range(self.state_size), self.col_reorder_matrix.T * (self.x_est_distribute - self.x_real), 'b.') # 点图
        plt.xlim([0,self.state_size])
        plt.xlabel("状态")
        plt.ylabel("误差")
        plt.show()
      # 坏值检测
      if is_bad is True:
        print('分布式估计器在第%i时刻检测到坏值，估计的残差为: %s' % (t, str(residual)))
      if is_bad_centralized is True:
        print('集中式估计器在第%i时刻检测到坏值，估计的残差为: %.3f' % (t, residual_centralized))
      # 记录
      res['state_est'] = np.column_stack((res['state_est'], self.x_est_distribute))
      res['state_real'] = np.column_stack((res['state_real'], self.x_real))
      res['state_error'] = np.column_stack((res['state_error'], np.array(self.x_est_distribute-self.x_real)))
      res['state_predict'] = np.column_stack((res['state_predict'], self.x_predict))
      self.next() # 进入下一时刻
    # 画图
    self.plot(res)

  def next(self, diff=None):
    """
      将分布式系统更新至下一时刻

    输入
    ---- 
      diff: 自定义状态变化量 (array)

    返回
    ----
      NULL
    """
    if diff is None:
      self.x_real += np.random.random((self.state_size,1)) * self.state_variance
    else:
      self.x_real += diff + np.random.random((self.state_size,1)) * self.state_variance
    #self.z_observed = self.H * self.x_real + np.random.random((self.measure_size,1)) # 线性
    self.H, self.Phi = self.jaccobi_H(self.x_est_distribute)
    self.z_observed = self.create_measurement(self.x_real) + self.R_error * np.mat(np.random.random((self.measure_size,1)))
    self.x_real_list,self.z_distribute,self.H_distribute,self.Phi_distribute = self.distribulize(self.x_real, self.z_observed, self.H, self.Phi, graph=True)

  def get_neighbor(self, n, t=2):
    """
      获取节点的邻居
  
    输入
    ---- 
      n 节点号(从0开始计)
      t -> -1: Aji != 0 
            1: Aij != 0
            2: Aij and Aji !=0
            3: n 的邻居的所有邻居
  
    返回
    ----
      邻居节点号 (1d-list)(start from 0)
    """
    res = []
    for i in range(self.nodes_num):
      if t == -1:
        if self.nodes_graph[i, n] != 0:
          res.append(i)
      elif t == 1:
        if self.nodes_graph[n, i] != 0:
          res.append(i)
      elif t == 2:
        if (self.nodes_graph[n, i] != 0) or (self.nodes_graph[i, n] != 0):
          res.append(i)
      elif t == 3:
        if i == n:
          continue
        if (self.Phi_graph[n, i] != 0):
          res.append(i)
    return res

  def plot(self, est_res, choosen_state=9):
    x_single_axis = np.arange(1,self.sim_time+1).T
    x_axis = np.tile(np.arange(1,self.sim_time+1),[self.state_size,1]).T # 向下复制
    plt.figure('各仿真时刻状态估计结果')
    plt.subplot(211)
    plt.title(str(choosen_state) + '状态细节图')
    plt.plot(x_single_axis, est_res['state_est'][choosen_state,:].T, 'g*-')
    plt.plot(x_single_axis, est_res['state_predict'][choosen_state,:-1].T)
    plt.plot(x_single_axis, est_res['state_real'][choosen_state,:].T, 'y*-')
    plt.legend(['估计','预测','真实'], loc='upper right', frameon=False)
    plt.xlabel("时刻")
    plt.ylabel("幅值")
    plt.subplot(212)
    plt.title('非线性算法各状态估计误差')
    plt.plot(x_axis, est_res['state_error'].T, '.-')
    plt.xlabel("时刻")
    plt.ylabel("误差")
    plt.show()
    if self.model_name == 'WSNs':
      for i in range(est_res['sim_time']):
        plt.figure(str(i)+'时刻定位结果')
        plt.plot(est_res['state_real'][0::2,i],est_res['state_real'][1::2,i], 'o')
        plt.plot(est_res['state_est'][0::2,i],est_res['state_est'][1::2,i], 'x')
        plt.legend(['真实位置','估计位置'], loc='upper right', frameon=False)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

  def __inject_falsedata(self, t):
    """
      注入虚假数据
  
    输入
    ----  
      t: 仿真的当前时刻
  
    返回
    ---- 
      None
    """
    # 攻击是否开始
    if self.time_falsedata <= t:
      self.API_inject_falsedata(t)
      # 注入后更新z_distribute
      z_distribute = []
      row_cnt = 0
      for row in self.node_row_amount:
        z_distribute.append(self.z_observed[row_cnt:row_cnt+row, 0])
        row_cnt += row
      self.z_distribute = z_distribute

  def __inject_falsedata_PCA(self, t):
    """
      注入PCA虚假数据
 
    输入
    ----  
      t: 仿真的当前时刻
  
    返回
    ---- 
      None
    """
    if self.time_falsedata <= t:
      self.API_inject_falsedata_PCA(t)
      z_distribute = []
      row_cnt = 0
      for row in self.node_row_amount:
        z_distribute.append(self.z_observed[row_cnt:row_cnt+row, 0])
        row_cnt += row
      self.z_distribute = z_distribute

  def __inject_baddata(self, t):
    """
      注入虚假数据
  
    输入
    ----  
      t: 仿真的当前时刻
  
    返回
    ---- 
      None
    """
    if self.time_baddata <= t:
      self.API_inject_baddata(t)
      z_distribute = []
      row_cnt = 0
      for row in self.node_row_amount:
        z_distribute.append(self.z_observed[row_cnt:row_cnt+row, 0])
        row_cnt += row
      self.z_distribute = z_distribute

  def detect_baddata(self, confidence=0.5, centralized=False, is_plot = True):
    """
      坏值检测
  
    输入
    ---- 
      confidence: 置信区间
      centralized: [True] - 集中式坏值检测结果
  
    返回
    ----
      邻居节点号: (1d-list)(start from 0)
    """
    is_bad = False
    if centralized:
      residual = 0.0
      is_bad,residual = super().estimator(0)
    else:
      if self.x_est_distribute_lists[0].shape[0] != self.state_size: # 每个节点只估计本地状态
        residual = []
        est_z_list = []
        for i in range(self.nodes_num):
          tmp_est_z = np.zeros((self.node_row_amount[i],1), dtype=complex)
          tmp_neighbor = self.get_neighbor(i, 1)
          for j in tmp_neighbor:
            tmp_est_z += self.H_distribute[i][j] * self.x_est_distribute_lists[j]
          est_z_list.append(tmp_est_z)
      else: # 估计全局状态
        residual = []
        est_z_list = []
        for i in range(self.nodes_num):
          tmp_est_z = np.column_stack(self.H_distribute[i]) * self.x_est_distribute_lists[i]
          est_z_list.append(tmp_est_z)
      cnt = 0
      detect_res_list = []
      chi_list = []
      for i in est_z_list:
        detect_res_list.append( round(float(np.sqrt((self.z_distribute[cnt] - i).T * (self.z_distribute[cnt] - i))[0,0]),3) )
        chi_list.append( self.chi2square_val(self.node_row_amount[cnt] - self.node_col_amount[cnt], confidence) )
        if detect_res_list[cnt] > chi_list[cnt]:
          is_bad = True
        cnt += 1
      if is_plot is True:
        plt.figure('坏值检测')
        plt.plot(chi_list, 'r--', marker='.')
        plt.plot(detect_res_list, 'b', marker='.')
        plt.legend(['阈值', '残差'], loc='upper right', frameon=False)
        plt.axis([0,self.nodes_num-1,0,110])
        plt.xlabel("子系统号")
        plt.ylabel("幅值")
        plt.draw()
      return is_bad,detect_res_list

  def gen_graph(self):
    return self.nodes_graph,self.Phi_graph