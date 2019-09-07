from linear_powergrid import LinearPowerGrid

import numpy as np
import matplotlib.pyplot as plt
#plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
import matplotlib.pylab as pylab
from scipy.linalg import block_diag

class DistributedLinearPowerGrid(LinearPowerGrid):
  def __init__(self, size):
    super().__init__(size=size)
    self.is_distribute = True

#############################################################
 # 函数 -- 
 #       set_nodes(): 将系统分布化，分割成多个子系统(节点)
 #                    也就是将H矩阵行列重新排列
 # 输入 -- 
 #       nodes[][]  节点包含的所有总线编号
 #       x_operation 上一时刻估计的状态值 
 # 得到 -- 
 #       nodes_num 分割成多少个节点
 #       node_row_amount[] 每个节点有多少行(测量值) (顺序为节点顺序)
 #       node_col_amount[] 每个节点有多少列(状态值) (顺序为节点顺序)
 #       z_distribute[]    每个节点的测量向量 (顺序为节点顺序)
 #       x_real_list[]     每个节点上一时刻的状态向量 (...)
 #       x_observed_list[] 每个节点在上一时刻状态向量加噪声后 (...)(没啥用)
 #       H_distribute[]    每个节点的局部H矩阵 (...)
 #       R_I_distribute_diag[] 每个节点的局部协方差矩阵(...)
 #       nodes_graph(np.mat) 各节点间的连接图矩阵
 #       Phi_graph(np.mat)   各节点间Phi矩阵的连接图矩阵
 #       Precondition_center(np.mat) precondition矩阵
 #       Precondition_distribute{}   空的字典, 在这里先声明, 之后迭代计算时用到 
 # 返回 --
 #       NULL
#############################################################
  def set_nodes(self, nodes = [], x_operation=None):
    self.nodes = nodes
    self.nodes_num = len(nodes)   # 表示有多少个节点
    
    # draw node connect graph matrix
    '''
    nodes_graph = np.mat(np.zeros([len(nodes), len(nodes)]), dtype='int')
    nout_cnt = 0
    for node in nodes:
      for bus in node:
        br_cnt = 0
        for bra in self.branch[0]:
          if bus == bra:
            if (self.branch[0][br_cnt] in self.pmu) or (self.branch[1][br_cnt] in self.pmu):
              nin_cnt = 0
              for n in nodes:
                if nin_cnt == nout_cnt:
                  nin_cnt+=1
                  continue
                if self.branch[1][br_cnt] in n:
                  # whether neighbor nodes knows each other?
                  nodes_graph[nin_cnt, nout_cnt] = 1
                  nodes_graph[nout_cnt, nin_cnt] = 1
                nin_cnt+=1
          br_cnt+=1
      nout_cnt+=1
    self.nodes_graph = nodes_graph
    '''

    # col and row amount per node
    self.node_row_amount=[]   # 表示每个节点有多少测量值
    self.node_col_amount=[]   # 表示每个节点有多少状态值
    for node in nodes:
      cnt = 0
      self.node_col_amount.append(len(node)*2)
      for bus in node:
        if bus in self.pmu:
          cnt+=2
        else:
          cnt+=1
        for bra in self.branch[0]:
          if bra == bus:
            cnt+=2
      self.node_row_amount.append(cnt)
    
    # reorder rows of H and R
    # 假设所有边测量的仪表都部署在输出端，若两个节点之间有连接，那么输出总线在哪个节点，哪个节点就包含另一个节点的数据。
    H = np.mat(np.empty([0, self.state_size]))
    R = np.mat(np.zeros([0,0]))
    z_order = []
    for node in nodes:
      cnt = 0
      order_cnt = 0
      for i in self.measure_who:
        if type(i) is type([]): # (PMU bus) or edge
          now = i[0]
          if now in node:
            H = np.row_stack((H, self.H[(order_cnt, order_cnt+1),:]))
            R = block_diag(R, self.R[order_cnt:order_cnt+2, order_cnt:order_cnt+2])
            z_order.append(self.measure_who[cnt])
          order_cnt += 2
        else: # SCADA bus
          now = i
          if now in node:
            H = np.row_stack((H, self.H[order_cnt,:]))
            R = block_diag(R, self.R[order_cnt, order_cnt])
            z_order.append(self.measure_who[cnt])
          order_cnt += 1
        cnt+=1
    self.measure_who = z_order
    self.R = np.mat(R)
    self.R_I = self.R.I

    # reorder cols of H
    seq = []
    cnt=0
    for node in nodes:
      for n in node:
        if (n == self.bus_ref) and (self.is_reference_deleted is True):
          continue
        self.x_who[cnt] = n
        seq.append(2*(n-1))
        seq.append(2*(n-1)+1)
        cnt+=1

    H = H[:, seq]
    self.H = H

    # update measurement in the form of distribute
    x = np.mat(np.zeros([self.state_size,1]))
    cnt = 0
    for i, j in zip(x_operation[0], x_operation[1]):
      x[2*cnt, 0] = i
      x[2*cnt+1, 0] = j
      cnt+=1
    # 按节点排序后的状态
    x_reorder = x[seq,:]
    # 真实状态值
    self.x_real = x_reorder
    # 有噪声的状态值
    self.x_observed = x_reorder + np.random.random((self.state_size,1))

    if x_operation is not None:
      self.z_observed = H * self.x_observed
      z_distribute = []
      row_cnt = 0
      for row in self.node_row_amount:
        z_distribute.append(self.z_observed[row_cnt:row_cnt+row, 0])
        row_cnt += row
      self.z_distribute = z_distribute

    # 每个节点的真实状态向量和带噪声的状态向量
    self.x_real_list = []
    self.x_observed_list = []
    tmp_cnt = 0
    for i in range(self.nodes_num):
      self.x_real_list.append(self.x_real[tmp_cnt:tmp_cnt+self.node_col_amount[i],:])
      self.x_observed_list.append(self.x_observed[tmp_cnt:tmp_cnt+self.node_col_amount[i],:])
      tmp_cnt += self.node_col_amount[i]

    # distribulize H
    H_distribute = []
    i = 0
    row_cnt = 0
    for row in self.node_row_amount:
      H_distribute.append([])
      col_cnt = 0
      for col in self.node_col_amount:
        H_distribute[i].append(H[row_cnt:row_cnt+row, col_cnt:col_cnt+col])
        col_cnt += col
      row_cnt += row
      i += 1
    self.is_distribute = True
    self.H_distribute = H_distribute

    # real graph: 通过H分块矩阵是否0矩阵来判断
    g=np.mat(np.zeros([self.nodes_num, self.nodes_num]), dtype='int')
    i=0
    for m in H_distribute:
      j=0
      for n in m:
        if (len(n.nonzero()[0]) != 0):# and (i!=j):
          g[i,j] = 1
        j+=1
      i+=1
    self.nodes_graph = g

    # distribulize R.I
    R_I_distribute_diag = []
    row_cnt = 0
    for row in self.node_row_amount:
      R_I_distribute_diag.append(self.R_I[row_cnt:row_cnt+row, row_cnt:row_cnt+row])
      row_cnt += row
    self.R_I_distribute_diag = R_I_distribute_diag

    # distribulize Phi
    Phi = H.H * self.R.I * H
    Phi_distribute = []
    i=0
    row_cnt = 0
    for row in self.node_col_amount:
      Phi_distribute.append([])
      col_cnt = 0
      for col in self.node_col_amount:
        Phi_distribute[i].append(Phi[row_cnt:row_cnt+row, col_cnt:col_cnt+col])
        col_cnt += col
      row_cnt += row
      i += 1
    self.Phi = Phi
    self.x_est_center = Phi.I*H.H*self.R.I*self.z_observed

    # Precondition matrix
    P = []
    iterator = 0
    for i in range(self.nodes_num):
      P.append(Phi[iterator:self.node_col_amount[i]+iterator, iterator:self.node_col_amount[i]+iterator].I)
      iterator += self.node_col_amount[i]
    self.Precondition_center = P[0]
    for i in range(1,self.nodes_num):
      self.Precondition_center = block_diag(self.Precondition_center,P[i])
    self.Precondition_center = np.mat(self.Precondition_center)

    # Phi graph
    pg=np.mat(np.zeros([self.nodes_num, self.nodes_num]), dtype='int')
    i=0
    for m in Phi_distribute:
      j=0
      for n in m:
        if (len(n.nonzero()[0]) != 0):# and (i!=j):
          pg[i,j] = 1
        j += 1
      i += 1
    self.Phi_graph = pg

    self.Precondition_distribute = {}	#test

  #############################################################
  # 函数 -- 
  #       estimator(): 进行分布式估计
  # 输入 -- 
  #       period: 迭代周期
  #       sync: 是否使用同步算法: [True,False]
  #       is_plot 是否画图: [True,False]
  # 返回 --
  #       NULL
  #############################################################
  def estimator(self, period, sync=True, is_plot=False):
    pass

  #############################################################
  # 函数 -- 
  #       get_neighbor(): 获取节点的邻居
  # 输入 -- 
  #       n 节点号(从0开始计)
  #       t -> -1: Aji != 0 
  #             1: Aij != 0
  #             2: Aij and Aji !=0
  #				     3: n 的邻居的所有邻居
  # 返回 --
  #       邻居节点号 (1d-list)(start from 0)
  #############################################################
  def get_neighbor(self, n, t=2):
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
      	if (self.Phi_graph[n, i] != 0):
      		res.append(i)
    return res

  #############################################################
  # 函数 -- 
  #       inject_falsedata(): 注入虚假数据
  # 输入 --  
  #     * sparse_amount 要攻击多少个状态(若measure_tobe_injected非None, 则以它为准)
  #     * amptitude     攻击的幅值(后面考虑自定义不同幅值，输入列表)
  #     * measure_tobe_injected（还未实现）
  #                     自定义的对测量值攻击向量(None - 随机选择要攻击的测量值)
  #     * delete_previous_injected
  #                     若为False, 则参数measure_tobe_injected表示注入这个攻击
  #                     若为True,  则在删除measure_tobe_injected这个之前注入的攻击向量后
  #                               以同样的形式(稀疏数)再重新注入另一个攻击.
  #                               若measure_tobe_injected=None, 那么就是随机生成, 不建议这么搞
  #                            注: (现在重新注入攻击还只能随机, 以后补充)
  # 返回 --
  #     * falsedata_info_dict [type:dic]
  #                     攻击向量的特性 - state_injected: 注入的状态攻击向量
  #                                  - measurement_injected: 注入的测量攻击向量
  #                                  - state_injected_amount: 注入了多少个状态值
  #                                  - measurement_injected_amount: 注入了多少个测量值
  #############################################################
  def inject_falsedata(self, sparse_amount=0, amptitude=0, measure_tobe_injected=None, delete_previous_injected=False):
    super().inject_falsedata(sparse_amount=sparse_amount, amptitude=amptitude, measure_tobe_injected=measure_tobe_injected, delete_previous_injected=delete_previous_injected)
    z_distribute = []
    row_cnt = 0
    for row in self.node_row_amount:
      z_distribute.append(self.z_observed[row_cnt:row_cnt+row, 0])
      row_cnt += row
    self.z_distribute = z_distribute

  #############################################################
  # 函数 -- 
  #       detect_baddata(): 坏值检测
  # 输入 -- 
  #       centralized: [True] - 集中式坏值检测结果
  # 返回 --
  #       邻居节点号 (1d-list)(start from 0)
  #############################################################
  def detect_baddata(self, centralized=False, is_plot = True):
    if centralized:
      super().detect_baddata()
    else:
      est_z_list = []
      for i in range(self.nodes_num):
        tmp_est_z = np.zeros((self.node_row_amount[i],1), dtype=complex)
        tmp_neighbor = self.get_neighbor(i, 1)
        for j in tmp_neighbor:
          tmp_est_z += self.H_distribute[i][j] * self.x_est_distribute_lists[j]
        est_z_list.append(tmp_est_z)
      
      cnt = 0
      detect_res_list = []
      chi_list = []
      for i in est_z_list:
        detect_res_list.append( np.sqrt((self.z_distribute[cnt] - i).T * (self.z_distribute[cnt] - i))[0,0] )
        chi_list.append( self.chi2square_val(self.node_row_amount[cnt] - self.node_col_amount[cnt], 0.5) )
        if detect_res_list[cnt] < chi_list[cnt]:
          print('测量残差为: %.3f, 小于置信度为0.5的卡方检验值: %.3f, 未检测到攻击.' % (float(detect_res_list[cnt]), chi_list[cnt]))
        else:
          print('测量残差为: %.3f, 大于置信度为0.5的卡方检验值: %.3f, 检测到攻击.' % (float(detect_res_list[cnt]), chi_list[cnt]))
        cnt += 1
      if is_plot is True:
        plt.figure('坏值检测')
        #plt.title('坏值检测')
        plt.plot(chi_list, 'r--', marker='.')
        plt.plot(detect_res_list, 'b', marker='.')
        plt.legend(['阈值', '残差'], loc='upper right')
        plt.xlabel("子系统号")
        plt.ylabel("幅值")
        plt.grid(True)
        plt.show()

  def gen_graph(self):
    return self.nodes_graph