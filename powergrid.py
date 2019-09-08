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
from sklearn.decomposition import PCA
from decimal import Decimal
from enum import Enum

#************************************************************
#* PowerGrid 类 -- 
#*       电网基类，初始化电网参数
#*       暂时只有线性直流模型的参数
#*
#************************************************************
class PowerGrid(object):
  def __init__(self, size, sim_time=1):
    self.size = size
    self.sim_time = sim_time
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

    self.H = np.mat(np.zeros([0, 2*size]), dtype=complex) # 量测矩阵
    self.R = np.mat(np.zeros([0, 0]), dtype=complex)      # 量测协方差矩阵
    self.z_observed = np.mat(np.zeros([0,0]))             # 量测值
    self.measure_who = []   # Indicate whose z: single i -> bus i
                      #                   [i, i]   -> pmu i
                      #                   [i, j]   -> branch i -> j

    self.x_who = list(range(1,size+1))   # Indicate whose x: per x has 2 members
                      # Defult the order of Bus

    self.is_distribute = False  # Defult is a centralized system
    self.is_reference_deleted = False

  #############################################################
  # 函数 -- 
  #       gen_gbbsh(): 计算建模所需电路参数
  # 输入 -- 
  #       电阻r(resistance),
  #       电抗x(reactance),
  #       分流电导gsh,
  #       分流电纳bsh.
  # 计算 -- 
  #       电导g(conductance),
  #       电纳b(susceptance),
  #       分流导纳ysh(admittance_shunt).
  # 公式 -- Note: 阻抗z(impedance)
  #       z = r + jx              |=> g = r/(r**2 + x**2)
  #       y = g + jb = z^{-1}     |=> b = x/(r**2 + x**2)
  #       branch上的ysh = (相连的两个bus上的ysh)/2
  # 返回 --
  #       ((g), (b), (ysh));
  #############################################################
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

  #############################################################
  # 函数 -- (还不够完善，没用上)
  #       set_variance(): 设置协方差矩阵
  # 输入 -- 
  #       R
  #       |-- None: 单位阵
  #       |-- 列表/元组: 协方差阵的对角线元素
  #       |-- 矩阵: 协方差阵
  # 得到 -- 
  #       协方差矩阵 R
  #       Phi = H.H*inv(R)*H
  # 返回 --
  #       NULL
  #############################################################
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
    self.Phi = self.H.H*self.R_I*self.H

  #############################################################
  # 函数 -- 
  #       summary(): 查看模型各项参数
  # 输入 -- 
  #      * para: 
  #        |-- #     总线编号
  #        |-- [#,#] 两个总线编号
  #        |-- None  待开发
  #      * visualize: 矩阵参数可视化
  #      * record: 是否保存矩阵为txt格式文件
  # 功能 --
  #       看不懂了...
  #       但有一个功能是打印某总线或某两个总线之间的(功率、电压等)信息
  # 返回 --
  #       NULL
  #############################################################
  def summary(self, para=None, record=False, visualize=False):
    # 声明变量
    Gamma = np.empty((self.state_size,self.state_size), dtype=complex)
  
    # 显示bus之间的参数
    if para is not None:
      try:
        stop = self.measure_who.index(para)
      except ValueError:
        print('wrong input!')
        exit()
    
      t = 0 # type: 0->Bus; 1->PMU; 2->Branch
      cnt = 0
      H_cnt = 0

      '''
      ## eliminate Reference bus's state
      bus_ref = self.x_who.index(self.bus_ref)
      H_real = np.delete(self.H, [bus_ref, bus_ref+1], axis=1)
      # and others
      Phi_real = H_real.H*self.R*H_real

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
      #alpha = self.H.H * self.R * self.z_observed
      #print(alpha)
      print('====================================')

      ## WLS Estimate
      x_est = self.Phi.I * self.H.H * self.R.I * self.z_observed
      # z_est = self.H * x_est

      print('====================================')
      for i in self.measure_who:
        if cnt == stop:
          if type(i) is type([]):
            if i[0] == i[1]:
              t=1
              print('Bus(PMU): '+str(i[0]))
              print('')
              print('====================================')
              print('Estimate state[V, angle]: '+str(x_est[[2*self.x_who.index(i[0]), 2*self.x_who.index(i[0])+1], :].T))
              # prepare for z
              z = self.z_observed[(H_cnt, H_cnt+1), :]
            else:
              t=2
              print('Branch: '+str(i[0])+' -> '+str(i[1]))
              print('')
              print('====================================')
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
            print('====================================')
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
      print('====================================')
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
      print('====================================')
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

    # 计算参数矩阵的特性
    print('====================================')
    print('-----------HTH的复数性质-------------')
    a,b = np.linalg.eig(self.H.H*self.H)
    u,v,w = np.linalg.svd(self.H.H*self.H)
    print('复数条件数: ' + str(linalg.cond(self.H)))
    print('特征值: ' + str(a))
    print('奇异值: ' + str(v))
    print('====================================')
    print('-----------Phi的复数性质-------------')
    a,b = np.linalg.eig(self.Phi)
    u,v,w = np.linalg.svd(self.Phi)
    print('复数条件数: ' + str(linalg.cond(self.Phi)))
    print('最大特征值: ' + str(max(a)))
    print('最小特征值: ' + str(min(a)))
    print('最大奇异值: ' + str(max(v)))
    print('最小奇异值: ' + str(min(v)))
    print('-----------Phi的实数性质-------------')
    a,b = np.linalg.eig(self.Phi.astype(float))
    u,v,w = np.linalg.svd(self.Phi.astype(float))
    print('实数数条件数: ' + str(linalg.cond(self.Phi.astype(float))))
    print('最大特征值: ' + str(max(a)))
    print('最小特征值: ' + str(min(a)))
    print('最大奇异值: ' + str(max(v)))
    print('最小奇异值: ' + str(min(v)))
    print('====================================')
    # 计算Gamma的特性
    if self.is_distribute is True:
      print('----------Gamma的复数性质------------')
      Gamma = np.linalg.cholesky(self.Precondition_center).T * self.Phi * np.linalg.cholesky(self.Precondition_center)
      a,b = np.linalg.eig(Gamma)
      u,v,w = np.linalg.svd(Gamma)
      print('复数条件数: ' + str(linalg.cond(Gamma)))
      print('最大特征值: ' + str(max(a)))
      print('最小特征值: ' + str(min(a)))
      print('最大奇异值: ' + str(max(v)))
      print('最小奇异值: ' + str(min(v)))
      #print('对称性:' + str(np.linalg.norm(Gamma.H-Gamma, ord=2)))
      print('----------Gamma的实数性质------------')
      Gamma = Gamma.astype(float)
      a,b = np.linalg.eig(Gamma)
      u,v,w = np.linalg.svd(Gamma)
      print('实数条件数: ' + str(linalg.cond(Gamma)))
      print('最大特征值: ' + str(max(a)))
      print('最小特征值: ' + str(min(a)))
      print('最大奇异值: ' + str(max(v)))
      print('最小奇异值: ' + str(min(v)))
      #print('对称性:' + str(np.linalg.norm(Gamma.H-Gamma, ord=2)))
      print('====================================')

    # 可视化
    if visualize is True:
      plt.matshow(self.Phi.astype(float))
      plt.show()
      if self.is_distribute is True:
        plt.matshow(self.Precondition_center.astype(float))
        plt.show()
        plt.matshow(Gamma.astype(float))
        plt.show()
        plt.matshow(self.nodes_graph)
        plt.show()

    # 保存矩阵到txt文件
    if record is True:
      np.savetxt('./save/H', self.H, delimiter = ',')
      if self.is_distribute is True:
        np.savetxt('./save/Gamma', Gamma, fmt='%.1f', delimiter = ',')
        np.savetxt('./save/Precondition', self.Precondition_center, fmt='%.1f', delimiter = ',')

  def gen_H(self):
    return self.H

  def gen_Phi(self):
    return self.Phi

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

    reduced_x_est = (self.reduced_H.H * self.reduced_R_I * self.reduced_H).I * self.reduced_H.H * self.reduced_R_I * self.reduced_z_real

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
  def chi2square_val(self, dim, confidence):
    mu = 0
    sigma = 1
    x = np.arange(-100,100,0.1)
    pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))

    z_a = np.percentile(pdf, confidence)  # 标准正态分布的confidence分位数
    chi = 1/2*(z_a + np.sqrt(2*dim-1))**2
    return chi
  
    