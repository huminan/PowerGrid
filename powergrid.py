# coding=utf-8
import numpy as np
#import scipy as sp
import sympy as sb
import matplotlib.pyplot as plt
#plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
import matplotlib.pylab as pylab
from numpy import linalg
from scipy.linalg import block_diag

import extract_config

# 分布式 循环次数
GAMMA_MAX_EIGEN = 200       # 计算Gamma最大特征值的循环次数
GAMMA_MIN_EIGEN = 500      # 计算Gamma最小特征值的循环次数
MAIN_LOOP_PERIOD = 10000    # 主程序循环次数

# Measure variance
PMU_VOLTAGE_VARIANCE = .1
PMU_ANGLE_VARIANCE = .1
SCADA_VOLTAGE_VARIANCE = 10   # 10
SCADA_POWER_VARIANCE = 10   #10

psap_conf = 'ieee118psp.txt'
psap_rule_line = './rules/rule_ieee118psap_line'
psap_rule_bus = './rules/rule_ieee118psap_bus'

cdf_conf = 'ieee118cdf.txt'
cdf_rule_bus = './rules/rule_ieee118cdf_bus'
cdf_rule_branch = './rules/rule_ieee118cdf_branch'

class POWER_GRID_MODEL(object):
  def __init__(self, size):#, branch, bus, branch_items, bus_items):
    self.size = size  # eliminate reference bus
    G_ij = sb.symbols("G_ij")
    B_ij = sb.symbols("B_ij")
    BSH_ij = sb.symbols("BSH_ij")
    Vi = sb.symbols("V_i")
    Vj = sb.symbols("V_j")
    Theta_i = sb.symbols("Theta_i")
    Theta_j = sb.symbols("Theta_j")
    alpha_ij = sb.symbols("alpha_ij")

    P_ij = G_ij*Vi**2 - G_ij*Vi*Vj*sb.cos(Theta_i - Theta_j) - B_ij*Vi*Vj*sb.sin(Theta_i - Theta_j)
    Q_ij = -(B_ij + BSH_ij)*Vi**2 + B_ij*Vi*Vj*sb.cos(Theta_i-Theta_j) - G_ij*Vi*Vj*sb.sin(Theta_i-Theta_j)
    self.h = [P_ij, Q_ij]
    self.value_symbol = [G_ij, B_ij, BSH_ij]
    self.state_i_symbol = [Vi, Theta_i]
    self.state_j_symbol = [Vj, Theta_j]
    self.jacobi_h_ij = self.__jacob(self.h, self.state_i_symbol)
    self.jacobi_h_ji = self.__jacob(self.h, self.state_j_symbol)

    self.GBBSH = []

    self.branch = []

    self.H = np.mat(np.zeros([0, 2*size]), dtype=complex)
    self.R = np.mat(np.zeros([0, 0]), dtype=complex)
    self.z_observed = np.mat(np.zeros([0,0]))
    self.measure_who = []   # Indicate whose z: single i -> bus i
                      #                   [i, i]   -> pmu i
                      #                   [i, j]   -> branch i -> j

    self.x_who = list(range(1,size+1))   # Indicate whose x: per x has 2 members
                      # Defult the order of Bus

    self.is_distribute = False  # Defult is a centralized system
    self.is_reference_deleted = False

  def gen_gbbsh(self, branch_num, resistance, reactance, shunt_conductance, shunt_susceptance):
    self.branch = tuple(branch_num)

    conductance = tuple([r/(r**2+x**2) for r,x in zip(resistance, reactance)])
    susceptance = tuple([-x/(r**2+x**2) for r,x in zip(resistance, reactance)])
    self.GBBSH.append(conductance)
    self.GBBSH.append(susceptance)

    shunt_admittance_bus =[]
    shunt_admittance_branch = []
    for c, s in zip(shunt_conductance, shunt_susceptance):
      shunt_admittance_bus.append( complex(c, s) )
    for o, i in zip(branch_num[0], branch_num[1]):
      shunt_admittance_branch.append((shunt_admittance_bus[o-1]+shunt_admittance_bus[i-1])/2)
    self.GBBSH.append(shunt_admittance_branch)

    self.GBBSH = tuple(self.GBBSH)

  def set_edge(self, x_operation):
    if len(self.GBBSH) == 0:
      raise Exception('Please call \'gen_gbbsh(branch_num, conductance, susceptance)\' first!')
    tmp_H = self.H
    cnt=0
    for branch_out,branch_in in zip(self.branch[0],self.branch[1]):
      # Substitute operation point x0 and g_ij, b_ij, bsh_ij to jacobbi matrices B_ij and B_ji
      tmp_ij = []
      for i in range(len(self.jacobi_h_ij)):
        tmp_ij.append([])
        for j in range(len(self.jacobi_h_ij[i])):
          tmp_ij[i].append(self.jacobi_h_ij[i][j])
          for k in range(len(self.state_i_symbol)):
            tmp_ij[i][j] = tmp_ij[i][j].subs(self.state_i_symbol[k], x_operation[k][branch_out-1])
          for k in range(len(self.state_j_symbol)):
            tmp_ij[i][j] = tmp_ij[i][j].subs(self.state_j_symbol[k], x_operation[k][branch_in-1])
          for k in range(len(self.value_symbol)):
            tmp_ij[i][j] = tmp_ij[i][j].subs(self.value_symbol[k], self.GBBSH[k][cnt])
      tmp_ji = []
      for i in range(len(self.jacobi_h_ji)):
        tmp_ji.append([])
        for j in range(len(self.jacobi_h_ji[i])):
          tmp_ji[i].append(self.jacobi_h_ji[i][j])
          for k in range(len(self.state_i_symbol)):
            tmp_ji[i][j] = tmp_ji[i][j].subs(self.state_i_symbol[k], x_operation[k][branch_out-1])
          for k in range(len(self.state_j_symbol)):
            tmp_ji[i][j] = tmp_ji[i][j].subs(self.state_j_symbol[k], x_operation[k][branch_in-1])
          for k in range(len(self.value_symbol)):
            tmp_ji[i][j] = tmp_ji[i][j].subs(self.value_symbol[k], self.GBBSH[k][cnt])
      # Flag
      self.measure_who.append([branch_out, branch_in])
      #
      a = np.mat(np.zeros([2, 2*self.size]), dtype=complex)
      for i in range(len(tmp_ij)):
        for j in range(len(tmp_ij[i])):
          a[i, 2*(branch_out-1)+j] = tmp_ij[i][j]
      for i in range(len(tmp_ji)):
        for j in range(len(tmp_ji[i])):
          a[i, 2*(branch_in-1)+j] = tmp_ji[i][j]
      tmp_H = np.row_stack((tmp_H, a))
      # Variance matrix
      self.R = block_diag(self.R, np.eye(2)*SCADA_POWER_VARIANCE)
      cnt += 1
    self.H = tmp_H
    self.R = np.mat(self.R)
    self.R_I = self.R.I

    self.measure_size = tmp_H.shape[0]
    self.state_size = tmp_H.shape[1]

    # calc measurement z
    x = np.mat(np.zeros([self.state_size,1]))
    cnt = 0
    for i, j in zip(x_operation[0], x_operation[1]):
      x[2*cnt, 0] = i
      x[2*cnt+1, 0] = j
      cnt+=1
    # 真实状态
    self.x_real = x
    # 加有噪声的状态
    self.x_observed = x + np.random.random((self.state_size,1))

    self.z_observed = self.H * self.x_observed
    self.x_est_center = (self.H.T*self.R_I*self.H).I*self.H.T*self.R_I*self.z_observed
    self.Phi = self.H.T*self.R_I*self.H

  def set_variance(self, R=None):
    if R is None:
      self.R = np.mat(np.eye(self.measure_size)) * 1
      self.R_I = self.R.I
    elif (R.shape[0]!=self.measure_size) or (R.shape[1]!=self.measure_size):
      raise Exception('set_variance: Wrong size of R!')
    elif (type(R) is type([])) or (type(R) is type(())):
      self.R = R[0]
      for i in R[1:]:
        self.R = block_diag(self.R, i)
      self.R = np.mat(self.R)
    else:
      self.R = R
      self.R_I = R.I
    # x_est = Phi.I * alpha
    self.Phi = self.H.T*self.R_I*self.H

  def set_local(self, bus_num, bus_type, pmu=[]):
    self.pmu = pmu
    for bus in bus_num:
      if bus in pmu:  # PMU
        a = np.mat(np.zeros([2, 2*self.size]))
        a[0,2*(bus-1)] = 1
        a[1,2*(bus-1)+1] = 1
        self.H = np.row_stack((self.H, a))
        self.measure_who.append([bus, bus])
        # Variance matrix
        self.R = block_diag(self.R, np.eye(1)*PMU_VOLTAGE_VARIANCE)
        self.R = block_diag(self.R, np.eye(1)*PMU_ANGLE_VARIANCE)
      else: # SCADA
        a = np.mat(np.zeros([1, 2*self.size]))
        a[0,2*(bus-1)] = 1
        self.H = np.row_stack((self.H, a))
        self.measure_who.append(bus)
        # Variance matrix
        self.R = block_diag(self.R, np.eye(1)*SCADA_VOLTAGE_VARIANCE)
      if bus_type[bus-1] == 3:  # if reference
        self.bus_ref = bus

  def delete_reference_bus(self):
    self.H = np.delete(self.H, (self.bus_ref-1)*2, 1)
    self.H = np.delete(self.H, (self.bus_ref-1)*2, 1)
    self.x_real = np.delete(self.x_real, (self.bus_ref-1)*2, 0)
    self.x_real = np.delete(self.x_real, (self.bus_ref-1)*2, 0)
    self.x_observed = np.delete(self.x_observed, (self.bus_ref-1)*2, 0)
    self.x_observed = np.delete(self.x_observed, (self.bus_ref-1)*2, 0)
    self.z_observed = self.H * self.x_observed
    self.x_est_center = (self.H.T*self.R_I*self.H).I*self.H.T*self.R_I*self.z_observed
    self.Phi = self.H.T*self.R_I*self.H
    self.state_size -= 2
    self.is_reference_deleted = True

  def print_H(self, para):
    try:
      stop = self.measure_who.index(para)
    except ValueError:
      print('wrong input!')
      exit()
      return
    
    t = 0 # type: 0->Bus; 1->PMU; 2->Branch
    cnt = 0
    H_cnt = 0

    '''
    ## eliminate Reference bus's state
    bus_ref = self.x_who.index(self.bus_ref)
    H_real = np.delete(self.H, [bus_ref, bus_ref+1], axis=1)
    # and others
    Phi_real = H_real.T*self.R*H_real

    # eliminate state num
    for i in range(self.nodes_num):
      bus_ref -= self.node_col_amount[i]
      if bus_ref < 0:
        self.node_col_amount[i] -= 2
        break
    node_col_amount = self.node_col_amount
    # Precondition matrix
    P = []
    iterator = 0
    for i in range(self.nodes_num):
      P.append(Phi_real[iterator:node_col_amount[i]+iterator, iterator:node_col_amount[i]+iterator].I)
      iterator += node_col_amount[i]
    Precondition_center = P[0]
    for i in range(1,self.nodes_num):
      Precondition_center = block_diag(Precondition_center,P[i])

    # print(np.linalg.matrix_rank(H_real))
    '''

    ## alpha
    #alpha = self.H.T * self.R * self.z_observed
    #print(alpha)
    print('====================================')

    ## WLS Estimate
    x_est = self.Phi.I * self.H.T * self.R.I * self.z_observed
    # z_est = self.H * x_est

    print('====================================')
    for i in self.measure_who:
      if cnt == stop:
        if type(i) is type([]):
          if i[0] == i[1]:
            t=1
            print('Bus(PMU): '+str(i[0]))
            print('')
            print('------------------------------------')
            print('Estimate state[V, angle]: '+str(x_est[[2*self.x_who.index(i[0]), 2*self.x_who.index(i[0])+1], :].T))
            # prepare for z
            z = self.z_observed[(H_cnt, H_cnt+1), :]
          else:
            t=2
            print('Branch: '+str(i[0])+' -> '+str(i[1]))
            print('')
            print('------------------------------------')
            # prepare for z, z* = z + h(x) - Bij*x - Bji*x
            x_location_i = 2*self.x_who.index(i[0])
            x_location_j = 2*self.x_who.index(i[1])
            print('Estimate Bus '+ str(i[0]) +' state[V, angle]: '+str(x_est[[x_location_i, x_location_i+1], :].T))
            print('Estimate Bus '+ str(i[1]) +' state[V, angle]: '+str(x_est[[x_location_j, x_location_j+1], :].T))

            print('Real Bus '+ str(i[0]) +' state[V, angle]: '+str(self.x_observed[[x_location_i, x_location_i+1], :].T))
            print('Real Bus '+ str(i[1]) +' state[V, angle]: '+str(self.x_observed[[x_location_j, x_location_j+1], :].T))
            
            H_0 = []
            for a in range(len(self.h)):
              H_0.append(self.h[a])
              for k in range(len(self.state_i_symbol)):
                H_0[a] = H_0[a].subs(self.state_i_symbol[k], self.x_observed[x_location_i+k,:])
              for k in range(len(self.state_j_symbol)):
                H_0[a] = H_0[a].subs(self.state_j_symbol[k], self.x_observed[x_location_j+k,:])
              for k in range(len(self.value_symbol)):
                H_0[a] = H_0[a].subs(self.value_symbol[k], self.GBBSH[k][stop])
            
            z = self.z_observed[(H_cnt, H_cnt+1), :] - self.H[(H_cnt, H_cnt+1),(x_location_i,x_location_i+1)]*self.x_observed[(x_location_i,x_location_i+1),:] - self.H[(H_cnt, H_cnt+1),(x_location_j,x_location_j+1)]*self.x_observed[(x_location_j,x_location_j+1),:] + np.mat(H_0, dtype=complex).T
          #print(self.H[(H_cnt, H_cnt+1),:])
        else:
          t=0
          print('Bus: '+str(i))
          print('')
          print('------------------------------------')
          print('Estimate state[V, angle]: '+str(x_est[[2*self.x_who.index(i), 2*self.x_who.index(i)+1], :].T))
          print('Real state[V, angle]: '+str(self.x_observed[[2*self.x_who.index(i), 2*self.x_who.index(i)+1], :].T))
          #print(self.H[H_cnt,:])
          # prepare for z
          z = self.z_observed[H_cnt, :]
        break
      if type(i) is type([]):
        H_cnt+=2
      else:
        H_cnt+=1
      cnt+=1
    print('')
    print('------------------------------------')
    print('Row location: '+str(H_cnt))
    loc = self.H[H_cnt, :].nonzero()
    print('Col location: '+str(loc[1]))
    if self.is_distribute is True:
      r=1
      for i in self.node_row_amount:
        H_cnt-=i
        if H_cnt<0:
          break
        r+=1
      print('Row in node: '+str(r))
      print('Col in node:', end=' ')
      for j in range(len(loc[1])):
        c=1
        Hc_cnt = loc[1][j]
        for i in self.node_col_amount:
          Hc_cnt-=i
          if Hc_cnt<0:
            break
          c+=1
        print(str(c), end=' ')
      print('')
    print('')
    print('------------------------------------')
    # print z
    if t==0:
      print('Bus Voltage: ' + str(z.T))
    elif t==1:
      print('[Bus Voltage, Bus angle]:' + str(z.T))
    elif t==2:
      print('[P, Q]:' + str(z.T))
    else:
      pass
    print('')
    print('------------------------------------')
    # 计算Gamma的特征值
    if self.is_distribute is True:
      Gamma = np.sqrt(self.Precondition_center) * self.Phi * np.sqrt(self.Precondition_center)
      a,b = np.linalg.eig(Gamma)
      print('Gamma的条件数: ' + str(linalg.cond(Gamma)))
      print('Gamma的最大特征值: ' + str(max(a)))
      print('Gamma的最小特征值: ' + str(min(a)))
      print('------------------------------------')
      # np.savetxt('Gamma', Gamma, delimiter = ',')
      exit()
      
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
    Phi = H.T * self.R.I * H
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
    self.Phi_distribute = Phi_distribute
    self.Phi = Phi
    self.x_est_center = Phi.I*H.T*self.R.I*self.z_observed

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
    for m in self.Phi_distribute:
      j=0
      for n in m:
        if (len(n.nonzero()[0]) != 0):# and (i!=j):
          pg[i,j] = 1
        j += 1
      i += 1
    self.Phi_graph = pg

    self.Precondition_distribute = {}	#test

  def __calc_tmp_alpha(self, node, node_neighbor):
    return self.H_distribute[node_neighbor][node].T*self.R_I_distribute_diag[node]*self.z_distribute[node]

  def gen_graph(self):
    return self.nodes_graph

  def gen_H(self):
    return self.H

## 注入虚假数据
#  sparse_amount: 要攻击多少个状态(若measure_tobe_injected非None, 则以它为准)
#  amptitude: 攻击幅度: (eg: amptitude=10 -> 攻击幅度在区间(0,10)内)
#  measure_tobe_injected: 自定义的对测量值攻击向量
#  delete_previous_injected: 若为False, 则参数measure_tobe_injected表示注入这个攻击
#                            若为True, 则在删除measure_tobe_injected这个之前注入的攻击向量后
#                                      以同样的形式(稀疏数)再重新注入另一个攻击.
#                                      若measure_tobe_injected=None, 那么就是随机生成, 不建议这么搞
#                                   注: (现在重新注入攻击还只能随机, 以后补充)
  def inject_falsedata(self, sparse_amount=0, amptitude=0, measure_tobe_injected=None, delete_previous_injected=False):
    if (sparse_amount==0) and (amptitude==0):
      print('未注入攻击!')
      return True
    if (delete_previous_injected is False) or (measure_tobe_injected is None):  # 无历史
      if measure_tobe_injected is None:   # 随机生成sparse_num稀疏的状态攻击向量，再利用Hc生成要攻击的测量值
        state_tobe_injected = np.c_[np.random.random((1,sparse_amount)), np.zeros((1,self.state_size-sparse_amount))][0] * amptitude
        np.random.shuffle(state_tobe_injected)
        measure_tobe_injected = self.H * np.mat(state_tobe_injected).T
      else:   # 自定义攻击向量
        state_tobe_injected = self.Phi.I * self.H.T * self.R.I * (self.z_observed + measure_tobe_injected) - self.x_est_center
        sparse_amount = 0
        for i in state_tobe_injected:
          if i!=0:
            sparse_amount += 1
    else: # 得把历史删了
      self.z_observed -= measure_tobe_injected
      state_tobe_injected = np.c_[np.random.random((1,sparse_amount)), np.zeros((1,self.state_size-sparse_amount))][0] * amptitude
      np.random.shuffle(state_tobe_injected)
      measure_tobe_injected = self.H * np.mat(state_tobe_injected).T

    which_state_tobe_injected_list = []
    cnt = 0
    for i in state_tobe_injected:
      if i != 0:
        which_state_tobe_injected_list.append(cnt)
      cnt+=1

    cnt = 0
    nonzero_cnt = 0
    measure_tobe_injected_list = []
    for i in measure_tobe_injected:
      if i!=0:
        measure_tobe_injected_list.append(cnt)
        nonzero_cnt += 1
      cnt += 1
    self.z_observed += measure_tobe_injected
    z_distribute = []
    row_cnt = 0
    for row in self.node_row_amount:
      z_distribute.append(self.z_observed[row_cnt:row_cnt+row, 0])
      row_cnt += row
    self.z_distribute = z_distribute

    self.x_est_center = self.Phi.I * self.H.T * self.R.I * self.z_observed
    print('对'+str(sparse_amount)+'个状态('+str(which_state_tobe_injected_list)+')注入虚假数据后，更改了'+str(nonzero_cnt)+'个测量值.')
    print('攻击的测量值的位置序列为: '+ str(measure_tobe_injected_list) + '.')
    falsedata_info_dict = {'state_injected':state_tobe_injected, 'measurement_injected':measure_tobe_injected, 'state_injected_amount': sparse_amount, 'measurement_injected_amount':nonzero_cnt, 'amptitude': amptitude, 'measurement_injected_sequence': measure_tobe_injected_list}
    return falsedata_info_dict

  def detect_falsedata(self):
    if self.is_distribute is False:
      detect_res = np.sqrt((self.z_observed - self.H * self.x_est_center).T * (self.z_observed - self.H * self.x_est_center))[0,0]
    else:
      detect_res = np.sqrt((self.z_observed - self.H * self.x_est_distribute).T * (self.z_observed - self.H * self.x_est_distribute))[0,0]
    print('测量误差的二范数为: ' + str(detect_res) + '.')

  def destribute_detect_falsedata(self, is_plot = False):
    est_z_list = []
    for i in range(self.nodes_num):
      tmp_est_z = np.zeros((self.node_row_amount[i],1), dtype=complex)
      tmp_neighbor = self.__get_neighbor(i, 1)
      for j in tmp_neighbor:
        tmp_est_z += self.H_distribute[i][j] * self.x_est_distribute_lists[j]
      est_z_list.append(tmp_est_z)
    
    cnt = 0
    detect_res_list = []
    chi_list = []
    for i in est_z_list:
      detect_res_list.append( np.sqrt((self.z_distribute[cnt] - i).T * (self.z_distribute[cnt] - i))[0,0] )
      chi_list.append( chi2square_val(self.node_row_amount[cnt] - self.node_col_amount[cnt], 0.5) )
      if detect_res_list[cnt] < chi_list[cnt]:
        print('测量残差为: ' + str(detect_res_list[cnt]) + ', 小于置信度为0.5的卡方检验值: ' + str(chi_list[cnt]) + ', 未检测到攻击.')
      else:
        print('测量残差为: ' + str(detect_res_list[cnt]) + ', 大于置信度为0.5的卡方检验值: ' + str(chi_list[cnt]) + ', 检测到攻击.')
      cnt += 1
    if is_plot is True:
      plt.figure('坏值检测')
      plt.title('坏值检测')
      plt.plot(chi_list, 'r--', marker='.')
      plt.plot(detect_res_list, 'b')
      plt.legend(['阈值', '残差'], loc='upper right')
      plt.grid(True)
      plt.show()

  def gen_estimate(self, is_plot=False):
    import threading
    import queue
    self.x_est_distribute_lists = []
    T = MAIN_LOOP_PERIOD
    axis1 = np.mat(range(T))
    sample = np.mat(np.empty([self.state_size, T]))
    for i in range(self.state_size):
      sample[i,:] = axis1
    self.record = []

    self.alpha = []
    self.Precondition = []
    self.est_x = []
    self.pseudo_x = []
    self.sigma_first = []
    self.sigma_second = []
    self.task_lock = queue.Queue(maxsize=self.nodes_num)
    thread_nodes = []

    for i in range(self.nodes_num):
      self.alpha.append( queue.Queue() )
      self.est_x.append( queue.Queue() )
      self.pseudo_x.append( queue.Queue() )
      self.Precondition.append( queue.Queue() )
      self.sigma_first.append( queue.Queue() )
      self.sigma_second.append( queue.Queue() )
      #self.task_lock.append( queue.Queue() )
      lock_con = threading.Condition()

      self.record.append(np.mat(np.empty([self.node_col_amount[i],T])))

      self.x_est_distribute_lists.append(np.mat(np.empty([self.node_col_amount[i],1])))

    for i in range(self.nodes_num):
      thread_nodes.append(threading.Thread(target=self.__set_node, args=(i, lock_con, T)))
    for n in thread_nodes:
      n.setDaemon(True)
      n.start()
    # wait gameover
    for n in thread_nodes:
      n.join()

    # calc x_est
    self.x_est_distribute = np.vstack(self.x_est_distribute_lists)

    # 画出估计结果
    if is_plot is True:
      ### 归一化
      #for i in range(self.nodes_num):
      #  for j in range(0, self.node_col_amount[i]):
      #    self.record[i][j,:] = (self.record[i][j,:] / self.record[i][j,-1])
      
      plt.figure('分布式估计（电压）')
      plt.title(u'状态估计值（电压）')
      plot_cnt = 0
      #plt.plot(sample[0:self.node_col_amount[0],:], self.record[0], 'b.')
      for i in range(self.nodes_num):
        for j in range(0, self.node_col_amount[i], 2):
          plt.plot(self.record[i][j,:].T, 'b', linewidth = 0.5)
      plt.legend([u'电压'], loc='upper right')
      plt.show()

      plt.figure('分布式估计（相角）')
      plt.title(u'状态估计值（电压相角）')
      for i in range(self.nodes_num):
        for j in range(1, self.node_col_amount[i], 2):
          plt.plot(self.record[i][j,:].T, 'r', linewidth = 0.5)
      plt.legend([u'电压相角'], loc='upper right')
      plt.show()

      ### 估计状态误差(\bar{x}-x)
      plt.figure('状态估计误差')
      for i in range(self.nodes_num):
        plt.plot(self.record[i][:,-1] - self.x_real_list[i], 'b.') # 点图
        #plt.bar(np.arange(len(self.x_real_list[i]))+1, (self.record[i][:,-1] - self.x_real_list[i]).T, lw=1)  # 条形图
      plt.legend([u'状态估计误差'], loc='upper right')
      plt.show()

  # num -> which node (start from 0)
  def __set_node(self, num, lock_con, T):
    import threading
    # neighbors whose measurement include mine
    neighbor_in_him = self.__get_neighbor(num, -1)
    # neighbors who is in my measurement
    neighbor_in_me = self.__get_neighbor(num, 1)
    # neighbors
    neighbors = self.__get_neighbor(num, 2)
    
    # Send my alpha(k)_other to neighbor_me
    for i in neighbor_in_me:
      self.alpha[i].put([num, self.H_distribute[num][i].T*self.R_I_distribute_diag[num]*self.z_distribute[num]])

    # Accumulate alpha_me
    alpha_res = 0
    for i in neighbor_in_him:
      comein = self.alpha[num].get()
      if comein[0] not in neighbor_in_him:
        raise Exception('Wrong come in '+str(comein[0]))
      alpha_res += comein[1]
    #print( str(alpha_res) + ' belongs to node '+ str(num) + '.' )

    # Send my Phi(k)_himhim to neighbor_me && Record Phi(k)_himhim
    for i in neighbor_in_me:
      self.Precondition[i].put([num, self.H_distribute[num][i].T*self.R_I_distribute_diag[num]*self.H_distribute[num][i]])

    # Accumulate my Phi_meme and calc Precondition matrix
    my_Precondition = np.mat(np.zeros([self.node_col_amount[num], self.node_col_amount[num]]), dtype=complex)
    for i in neighbor_in_him:
      comein = self.Precondition[num].get()
      if comein[0] not in neighbor_in_him:
        raise Exception('Wrong come in '+str(comein[0]))
      my_Precondition += comein[1]
    my_Precondition = my_Precondition.I
    my_Precondition_sqrt = np.sqrt(my_Precondition)
    self.Precondition_distribute.update({num:my_Precondition_sqrt})

    # initial x_0 set all 0
    x_est = np.mat(np.zeros([self.node_col_amount[num], 1]), dtype=complex)
    ## To estimate ||Gamma||
    # initial b_0 with random ||b_0|| = 1 
    b_bar = np.mat(np.random.rand(self.node_col_amount[num], 1), dtype=complex)
    b_bar = b_bar / np.linalg.norm(b_bar,ord=2)	# normallize
    #
    yita = 1

    # It seems that i in neighbor_in_me is also ok
    v_ij = {}
    for i in neighbors:
      v_ij.update({str(i):{}})
      for j in neighbors:
        v_ij[str(i)].update({str(j):1})

    cnt=0
    ### Calc sigma ###
    ## Calc maximum eigenvalue of Gamma
    for t in range(GAMMA_MAX_EIGEN):
      # Send sigma_first to neighbors
      for i in neighbors:
        self.sigma_first[i].put([num, my_Precondition_sqrt*b_bar, yita])
      # Recieve sigma_first
      comein_dict = {}
      for i in neighbors:
        get = self.sigma_first[num].get()
        comein_dict.update( {str(get[0]) : (get[1],get[2])} )
        if get[0] not in neighbors: # some are not in me
          raise Exception('Wrong come in '+str(get[0]))
      # v_ij
      for i in neighbor_in_me:
        for j in neighbors:
          v_ij[str(i)][str(j)] = comein_dict[str(i)][1] / comein_dict[str(j)][1] * v_ij[str(i)][str(j)]
      # yita_bar_in_me
      yita_bar_in_me = []
      for i in neighbor_in_me:
        yita_bar_in_me.append(v_ij[str(i)][max(v_ij[str(i)], key = v_ij[str(i)].get)])
      # b_hat
      b_hat = {}
      for i in neighbor_in_me:
        b_hat.update({str(i):np.mat(np.zeros([self.node_col_amount[i], 1]), dtype=complex)})
        for j in neighbor_in_me:
          b_hat[str(i)] += v_ij[str(i)][str(j)] * self.H_distribute[num][i].T*self.R_I_distribute_diag[num]*self.H_distribute[num][j] * comein_dict[str(j)][0]
      # Send sigma_second to neighbor_me
      tmp_cnt = 0
      for i in neighbor_in_me:
        self.sigma_second[i].put([num, b_hat[str(i)], yita_bar_in_me[tmp_cnt]])
        tmp_cnt += 1
      # Recieve sigma_second
      comein_dict = {}
      for k in neighbor_in_him:
        get = self.sigma_second[num].get()
        comein_dict.update({ str(get[0]): [get[1], get[2]]})
      # Accumulate b_hat to calc b_tilde
      b_tilde = np.mat(np.zeros([ self.node_col_amount[num], 1 ]), dtype=complex)
      for k in neighbor_in_him:
        b_tilde += comein_dict[str(k)][0]
      b_tilde = my_Precondition_sqrt*b_tilde
      # yita
      yita_candidate = [np.linalg.norm(b_tilde, ord=2)]
      for k in neighbor_in_him:
        yita_candidate.append(comein_dict[str(k)][1])
      yita = 1 / max( yita_candidate )
      # b_bar
      b_bar = yita * b_tilde

      cnt+=1
      # 显示进度
      #print(cnt)

      lock_con.acquire()
      if self.task_lock.qsize() != self.nodes_num-1:
        self.task_lock.put(num)
        lock_con.wait()
      else:
        self.task_lock.queue.clear()
        lock_con.notify_all()
      lock_con.release()
    # Maximum eigenvalue of Gamma
    Gamma_max = 1 / yita
    print(threading.current_thread().name + 'Gamma maximum: ' + str(Gamma_max))
    
    cnt=0
    ## Calc minimum eigenvalue of Gamma
    # initial b_0 with random ||b_0|| = 1 
    b_bar = np.mat(np.random.rand(self.node_col_amount[num], 1), dtype=complex)
    b_bar = b_bar / np.linalg.norm(b_bar,ord=2)	# normallize
    #
    yita = 1

    # It seems that i in neighbor_in_me is also ok
    v_ij = {}
    for i in neighbors:
      v_ij.update({str(i):{}})
      for j in neighbors:
        v_ij[str(i)].update({str(j):1})

    for t in range(GAMMA_MIN_EIGEN):
      # Send sigma_first to neighbors
      for i in neighbors:
        self.sigma_first[i].put([num, my_Precondition_sqrt*b_bar, yita])
      # Recieve sigma_first
      comein_dict = {}
      for i in neighbors:
        get = self.sigma_first[num].get()
        comein_dict.update( {str(get[0]) : (get[1],get[2])} )
        if get[0] not in neighbors: # some are not in me
          raise Exception('Wrong come in '+str(get[0]))
      # v_ij
      for i in neighbor_in_me:
        for j in neighbors:
          v_ij[str(i)][str(j)] = comein_dict[str(i)][1] / comein_dict[str(j)][1] * v_ij[str(i)][str(j)]
      # yita_bar_in_me
      yita_bar_in_me = []
      for i in neighbor_in_me:
        yita_bar_in_me.append(v_ij[str(i)][max(v_ij[str(i)], key = v_ij[str(i)].get)])
      # b_hat
      b_hat = {}
      for i in neighbor_in_me:
        b_hat.update({str(i):np.mat(np.zeros([self.node_col_amount[i], 1]), dtype=complex)})
        for j in neighbor_in_me:
          b_hat[str(i)] += v_ij[str(i)][str(j)] * self.H_distribute[num][i].T*self.R_I_distribute_diag[num]*self.H_distribute[num][j] * comein_dict[str(j)][0]
      # Send sigma_second to neighbor_me
      tmp_cnt = 0
      for i in neighbor_in_me:
        self.sigma_second[i].put([num, b_hat[str(i)], yita_bar_in_me[tmp_cnt]])
        tmp_cnt += 1
      # Recieve sigma_second
      comein_dict = {}
      for k in neighbor_in_him:
        get = self.sigma_second[num].get()
        comein_dict.update({ str(get[0]): [get[1], get[2]]})
      # Accumulate b_hat to calc b_tilde
      b_tilde = np.mat(np.zeros([ self.node_col_amount[num], 1 ]), dtype=complex)
      for k in neighbor_in_him:
        b_tilde += comein_dict[str(k)][0]
      b_tilde = my_Precondition_sqrt*b_tilde
      # Calc b_tilde_min
      b_tilde_min = (Gamma_max+100) * b_bar - b_tilde   # (Gamma_max) * b_bar - b_tilde
      # yita
      yita_candidate = [np.linalg.norm(b_tilde_min, ord=2)]
      for k in neighbor_in_him:
        yita_candidate.append(comein_dict[str(k)][1])
      yita = 1 / max( yita_candidate )

      # b_bar
      b_bar = yita * b_tilde_min

      Gamma_min = (Gamma_max+100) - (1 / yita) # (Gamma_max) - 1 / yita

      cnt+=1
      # 显示进度
      # print(cnt)

      lock_con.acquire()
      if self.task_lock.qsize() != self.nodes_num-1:
        self.task_lock.put(num)
        lock_con.wait()
      else:
        self.task_lock.queue.clear()
        lock_con.notify_all()
      lock_con.release()
		# Minimum eigenvalue of Gamma
    Gamma_min = (Gamma_max+100) - 1 / yita  # (Gamma_max) - 1 / yita
    print(threading.current_thread().name + 'Gamma_min: ' + str(Gamma_min))

    # Calc sigma!
    #Gamma_min = -108.84035369492203-3.836888748846456j
    #Gamma_max = 462.01764897430405+43.01854292973093j
    sigma = 2 / ( Gamma_max + Gamma_min )
    ### End Calc sigma ###

    print(threading.current_thread().name + 'sigma: ' + str(sigma))
    #sigma = 0.0022

    cnt = 0
    for t in range(T):
      # Send my estmate state x to neighbors
      for i in neighbors:
        self.est_x[i].put([num, x_est])

      # Receive est_x
      recv_x_est = []
      for i in neighbors:
        comein = self.est_x[num].get()
        if comein[0] in neighbor_in_me: # some are not in me
          recv_x_est.append(comein)
      self.est_x[num].queue.clear()

      # Send pseudo_x to neighbor_me
      for i in neighbor_in_me:
        pseudo_x = np.mat(np.zeros([self.node_col_amount[i],1]), dtype=complex)
        # Accumulate Phi_ij * x_j
        for j, x_j in recv_x_est:
          pseudo_x += self.H_distribute[num][i].T*self.R_I_distribute_diag[num]*self.H_distribute[num][j]*x_j
        # Send
        self.pseudo_x[i].put([num, pseudo_x])

      # Accumulate my pseudo_x
      pseudo_x = np.mat(np.zeros([self.node_col_amount[num],1]), dtype=complex)
      for k in neighbor_in_him:
        comein = self.pseudo_x[num].get()
        if comein[0] not in neighbor_in_him:
          raise Exception('Wrong come in '+str(comein[0]))
        pseudo_x += comein[1]
      # Calc estimate state
      x_est = x_est - sigma * my_Precondition * (pseudo_x - alpha_res)

      self.record[num][:,t] = x_est#/self.x_observed_list[num] # 归一化 
      # End task

      # 显示进度
      cnt+=1
      # print(cnt)

      lock_con.acquire()
      if self.task_lock.qsize() != self.nodes_num-1:
        self.task_lock.put(num)
        lock_con.wait()
      else:
        self.task_lock.queue.clear()
        lock_con.notify_all()
      lock_con.release()
    self.x_est_distribute_lists[num] = x_est
    # print(threading.current_thread().name, str(x_est.T))

  ## n -> which node (start from 0)
  #  t -> -1: Aji != 0 
  #        1: Aij != 0
  #        2: Aij and Aji !=0
  #				 3: n 的邻居的所有邻居
  #  output: neighbor index (1d-list)(start from 0)
  def __get_neighbor(self, n, t=2):

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

### Calculate the Jacobi matrix (a symbolic list)
# input: M -> a symbolic 1d-list, 
#             every element is a function with unkown values <- X
# output : a symbolic 2d-list, consist a symbolic jacobi matrix.
## Then, use function __symbol_assignment() to assign value 
  def __jacob(self, M, X):
    J = []
    for i in range(len(M)):
      J.append([])
      for j in range(len(X)):
        J[i].append(sb.diff(M[i], X[j]))
    return J

  def delete_measurements(self, amount=1, busTobeDeleted=None):
    if busTobeDeleted is None:   # 随机剔除
      is_full_rank = False
      loop_time = 0
      while is_full_rank is not True:
        sequence_toBeDeleted = np.random.random_integers(self.measure_size ,size=amount)
        sequence_toBeDeleted = np.sort(sequence_toBeDeleted)
        tmp_H = self.H
        tmp_R_I = self.R_I
        tmp_R = self.R
        tmp_z = self.z_observed
        loc = 1
        for measure_location in sequence_toBeDeleted:
          tmp_H = np.delete(tmp_H, measure_location-loc, 0)
          loc += 1
        if np.linalg.matrix_rank(tmp_H) == self.state_size:
          self.measure_size -= amount
          loc=1
          for measure_location in sequence_toBeDeleted:
            tmp_R = np.delete(tmp_R, measure_location-loc, 0)
            tmp_R = np.delete(tmp_R, measure_location-loc, 1)
            tmp_R_I = np.delete(tmp_R_I, measure_location-loc, 0)
            tmp_R_I = np.delete(tmp_R_I, measure_location-loc, 1)
            tmp_z = np.delete(tmp_z, measure_location-loc, 0)
            loc += 1
          self.reduced_R = tmp_R
          self.reduced_R_I = tmp_R_I
          self.reduced_z_real = tmp_z
          self.reduced_H = tmp_H
          is_full_rank = True
          print('经过'+str(loop_time+1)+'轮随机，得到降维后的测量矩阵.')

        if loop_time == 1000:
          print('删掉'+str(amount)+'个测量值随机选了1000次测量矩阵还是没满秩！')
          return False
        loop_time += 1
      measurement_tobe_del_list = list(sequence_toBeDeleted)
    else: # 剔除掉攻击向量
      tmp_H = self.H
      tmp_R_I = self.R_I
      tmp_R = self.R
      tmp_z = self.z_observed

      random_choose_to_del_amount = np.random.randint(len(busTobeDeleted))
      np.random.shuffle(busTobeDeleted)

      del_cnt = 0
      measurement_tobe_del_list = []
      for i in range(random_choose_to_del_amount):
        tmp_H = np.delete(tmp_H, busTobeDeleted[i]-del_cnt, 0)
        tmp_R = np.delete(tmp_R, busTobeDeleted[i]-del_cnt, 0)
        tmp_R = np.delete(tmp_R, busTobeDeleted[i]-del_cnt, 1)
        tmp_R_I = np.delete(tmp_R_I, busTobeDeleted[i]-del_cnt, 0)
        tmp_R_I = np.delete(tmp_R_I, busTobeDeleted[i]-del_cnt, 1)
        tmp_z = np.delete(tmp_z, busTobeDeleted[i]-del_cnt, 0)
        measurement_tobe_del_list.append(busTobeDeleted[i])
        del_cnt += 1
      amount = del_cnt
      if np.linalg.matrix_rank(tmp_H) == self.state_size:
        self.reduced_measure_size = self.measure_size - random_choose_to_del_amount
        self.reduced_R = tmp_R
        self.reduced_R_I = tmp_R_I
        self.reduced_z_real = tmp_z
        self.reduced_H = tmp_H
      else:
        print('删掉这'+ str(del_cnt) +'行测量值后系统测量矩阵不再满秩！('+str(np.linalg.matrix_rank(tmp_H))+'!='+str(np.linalg.matrix_rank(self.H))+').')
        return False

    reduced_x_est = (self.reduced_H.T * self.reduced_R_I * self.reduced_H).I * self.reduced_H.T * self.reduced_R_I * self.reduced_z_real

    print('删除的测量值的位置序列: '+ str(measurement_tobe_del_list) +'.')
    plt.figure('删除 '+str(amount)+'个测量值后预测测量值的变化量')
    plt.plot(np.arange(len(self.reduced_z_real)), (self.reduced_z_real-self.reduced_H*reduced_x_est), '.')

    plt.figure('原残差')
    plt.plot(np.arange(len(self.z_observed)), (self.z_observed-self.H*self.x_est_center), '.')

    plt.figure('前后状态差')
    plt.plot(range(len(reduced_x_est)), self.x_est_center-reduced_x_est, '.')
    plt.show()
    return True

# 置信区间为 confidence，维数为dim 的卡方分布的值
def chi2square_val(dim, confidence):
  mu = 0
  sigma = 1
  x = np.arange(-100,100,0.1)
  pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))

  z_a = np.percentile(pdf, confidence)  # 标准正态分布的confidence分位数
  chi = 1/2*(z_a + np.sqrt(2*dim-1))**2
  return chi

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
### 电网建模
  x_operation = s_bus_state

  model = POWER_GRID_MODEL(118)
  PMU = [3,5,9,12,15,17,21,25,114,28,40,37,34,70,71,53,56,45,49,62,64,68,105,110,76,79,100,92,96,85,86,89]
  model.set_local(s_bus_code, bus_type, PMU)
  model.gen_gbbsh(s_branch_code, resistance, reactance, shunt_conductance, shunt_susceptance)
  model.set_edge(x_operation)
  #model.delete_reference_bus()
########## 集中式建模完毕 ###########

########## 删除某些测量值
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
########## 删除完毕 ###########

######## 分布式建模
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
############# 分布式建模完毕 ###############
  # print(model.gen_graph())
  # model.print_H([1,2])

  # falsedata = model.inject_falsedata(sparse_amount=10, amptitude=100)  # 注入5个幅值0-10的虚假数据
### 分布式估计 ###
  model.gen_estimate(True)
  model.destribute_detect_falsedata(True)

if __name__ == '__main__':
  main()