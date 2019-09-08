from powergrid import PowerGrid
import numpy as np
import random
from scipy.linalg import block_diag
from numpy import linalg
import matplotlib.pyplot as plt
#plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
import matplotlib.pylab as pylab

# Measure variance 
###### 真实参数 ######
# PMU: 电压-0.002%pu; 相角-0.01度
# SCADA: 电压-0.3%pu; 功率-0.3%pu
####### over ########
PMU_VOLTAGE_VARIANCE = .002
PMU_ANGLE_VARIANCE = .01
SCADA_VOLTAGE_VARIANCE = .3
SCADA_POWER_VARIANCE = .3

STATE_VARIENCE = 1e-3

class LinearPowerGrid(PowerGrid):
  def __init__(self, size):
    super().__init__(size=size)
    self.A = np.mat(np.eye(2*size), dtype=complex)        # 状态转移矩阵
    self.G = np.mat(np.zeros((2*size,1), dtype=complex))  # 状态轨迹向量
    self.is_baddata = False
    self.is_FDI = False
    self.is_FDI_PCA = False
    self.time_baddata = 0
    self.time_falsedata = 0

  #############################################################
  # 函数 -- 
  #       set_edge(): 计算H矩阵中由bus决定的参数
  # 输入 -- 
  #       bus_num[]  所有总线的编号
  #       bus_type[] 所有总线的类型: 0->负载总线; 2->发电厂; 3->参考总线
  #       pmu[]      pmu总线的编号  
  # 公式 -- 
  #       SCADA只可以测量(电压)
  #       PMU可以测量(电压,相角)
  # 返回 --
  #       NULL
  #############################################################
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
        self.R = block_diag(self.R, np.eye(1)*(PMU_VOLTAGE_VARIANCE**2))
        self.R = block_diag(self.R, np.eye(1)*(PMU_ANGLE_VARIANCE**2))
      else: # SCADA
        a = np.mat(np.zeros([1, 2*self.size]))
        a[0,2*(bus-1)] = 1
        self.H = np.row_stack((self.H, a))
        self.measure_who.append(bus)
        # Variance matrix
        self.R = block_diag(self.R, np.eye(1)*(SCADA_VOLTAGE_VARIANCE**2))
      if bus_type[bus-1] == 3:  # if reference
        self.bus_ref = bus

  #############################################################
  # 函数 -- 
  #       set_edge(): 计算H矩阵中由branch决定的参数
  # 输入 -- 
  #       x_operation 上一采样时刻的状态估计值(本例中只有一时刻的数据，所以将当前时刻估计值代入)
  # 计算 -- 
  #       求偏导
  #       
  # 得到 --
  #       矩阵 H
  #       协方差矩阵 R (SCADA,PMU测量仪表分别用不同的默认精度)
  #       测量数 measure_size (不该在这个函数里首次定义)
  #       状态数 state_size   (不该在这个函数里首次定义)
  #  (不应该在这个函数里面的)-->
  #       状态估计值 __x_real (真实值)
  #       观测值 __z_real (数据中未给出，使用__x_real*H得到)
  #       观测值 z_observed (__z_real加上噪声后得到)(非真实值)
  #       这一时刻的状态估计值 x_est_center
  #       Phi = H.H*inv(R)*H
  # 返回 --
  #       NULL
  #############################################################
  def set_edge(self, x_operation):
    if len(self.GBBSH) == 0:
      raise Exception('Call function \'gen_gbbsh(branch_num, conductance, susceptance)\' first!')
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
      # print(a[:,2*(branch_out-1):2*(branch_out-1)+2])
      # print(a[:,2*(branch_in-1):2*(branch_in-1)+2])
      tmp_H = np.row_stack((tmp_H, a))
      # print(tmp_H[])
      #
      # Variance matrix
      self.R = block_diag(self.R, np.eye(2)*(SCADA_POWER_VARIANCE**2))
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
    self.__x_real = x
    # 加有噪声的状态
    # self.x_observed = x + np.random.random((self.state_size,1)) * STATE_VARIENCE
    self.__z_real = self.H * self.__x_real
    self.z_observed = self.__z_real + self.R * np.random.random((self.measure_size,1))

    self.Phi = self.H.H*self.R_I*self.H
    

  #############################################################
  # 函数 -- 
  #       delete_reference_bus(): 删除H矩阵中的reference总线
  # 输入 --  
  #       NULL
  # 返回 --
  #       NULL
  #############################################################
  def delete_reference_bus(self):
    self.H = np.delete(self.H, (self.bus_ref-1)*2, 1)
    self.H = np.delete(self.H, (self.bus_ref-1)*2, 1)
    self.__x_real = np.delete(self.__x_real, (self.bus_ref-1)*2, 0)
    self.__x_real = np.delete(self.__x_real, (self.bus_ref-1)*2, 0)
    self.z_observed = self.H * self.__x_real + self.R * np.random.random((self.measure_size,1))
    self.Phi = self.H.H*self.R_I*self.H
    self.state_size -= 2
    self.is_reference_deleted = True

  #############################################################
  # 函数 -- 
  #       estimator(): 最小二乘估计器
  # 输入 -- 
  #       * sim_time: 仿真时间(多少次)
  #       * variance: 状态变化方差大小 (#)
  #       * is_bad_data: 是否会产生坏数据 <False,True>
  #       * falsedata_type: 虚假数据注入攻击的方法
  #             |- 'normal'
  #             |- 'pca'
  # 返回 --
  #       NULL
  #############################################################
  def estimator(self, sim_time=1, variance=STATE_VARIENCE):
    a = self.__x_real
    b = np.zeros((self.size*2,1), dtype=complex)
    state_error_mat = np.mat(np.eye(self.state_size))
    
    if self.is_FDI_PCA is True:
      self.z_observed_history = np.mat(np.empty((self.measure_size,0)))
    res = [[],[],[]]  # 估计、预测、真实
    
    for t in range(sim_time+1):
      if self.is_baddata is True:
        self.__inject_baddata(t)
      if self.is_FDI is True:
        self.__inject_falsedata(t)
      elif self.is_FDI_PCA is True:
        self.z_observed_history = np.column_stack((self.z_observed_history, self.z_observed))
        self.__inject_falsedata_PCA(t)
      self.x_est_center = self.Phi.I*self.H.H*self.R_I*self.z_observed
      res[0].append(complex(self.x_est_center[1]))
      if t != 0:
        print('第%i次估计的残差为: %.3f' % (t, self.detect_baddata()))
      if t is not sim_time:
        a,b,state_error_mat = self.__predict([a,b,state_error_mat])
        self.next(variance=variance)
        res[1].append(complex(self.x_predict[1]))
      res[2].append(complex(self.__x_real[1]))
    plt.figure('时间图')
    plt.plot(range(1,sim_time+1), res[0][1:], 'r*', linewidth = 0.5)
    plt.plot(range(1,sim_time+1), res[1], 'b*', linewidth = 0.5)
    plt.plot(range(1,sim_time+1), res[2][1:], 'y*', linewidth = 0.5)
    plt.legend(['估计','预测','真实'], loc='upper right')
    plt.xlabel("时刻")
    plt.ylabel("幅值")
    plt.show()

  #############################################################
  # 函数 -- 
  #       next()
  # 输入 -- 
  #      * diff: 状态变化量 (array)
  #      * variance: 状态变化方差大小 (#)
  # 功能 --
  #      跳到下一时刻
  # 返回 --
  #      NULL
  #############################################################
  def next(self, diff=None, variance=STATE_VARIENCE):
    if diff is None:
      self.__x_real += np.random.random((self.state_size,1)) * variance
    else:
      self.__x_real += diff
    self.z_observed = self.H * self.__x_real + self.R * np.random.random((self.measure_size,1))

  #############################################################
  # 函数 -- 
  #       __predict()
  # 输入 -- 
  #      * para_before: 上一时刻的参数
  #      * model: 电网参数预测的模型
  #      * para: 
  # 功能 --
  #      预测下一时刻的电网参数，这里以上一时刻估计值作为真实值
  # 返回 --
  #      NULL
  #############################################################
  def __predict(self, para_before=[], model=1, para=None):
    if model == 1:
      alpha = np.zeros((self.size*2,1), dtype=complex)
      beta = np.zeros((self.size*2,1), dtype=complex)
      if para is None:
        for i in range(self.size*2):
          alpha[i] = 0.8
          beta[i] = 0.5
      elif min(para[0]+para[1])>0 and max(para[0]+para[1])<1:
        alpha = para[0]
        beta = para[1]
      a_before = para_before[0]
      b_before = para_before[1]
      state_error_mat_before = para_before[2]
      a = np.zeros((self.size*2,1), dtype=complex)
      b = np.zeros((self.size*2,1), dtype=complex)     
      for i in range(self.size*2):
        a[i] = alpha[i] * self.x_est_center[i] + (1-alpha[i]) * self.__x_real[i]
        b[i] = beta[i] * (a[i] - a_before[i]) + (1-beta[i]) * b_before[i]
      for i in range(self.size*2):
        self.G[i] = (1+beta[i])*(1-alpha[i])*self.x_est_center[i] - beta[i]*a_before[i] + (1-beta[i])*b_before[i]
        self.A[i,i] = alpha[i]*(1+beta[i])
      # 预测下一步
      self.x_predict = self.A * self.x_est_center + self.G
      # 估计状态误差协方差矩阵
      state_error_mat = self.A * state_error_mat_before * self.A.H + np.mat(np.eye(self.state_size))*STATE_VARIENCE

      return a,b,state_error_mat

  #############################################################
  # 函数 -- 
  #       inject_baddata(): 注入坏值
  # 输入 --  
  #       * moment: 在哪个时刻产生了坏数据
  #       * probability: 每个仪表产生坏值的概率: p/10e7
  # 返回 --
  #       baddata_info_dict [type: dic]
  #                     measurement_injected: 注入的测量攻击的值(非攻击向量，)
  #                     measurement_injected_amount: 注入了多少个测量值
  #############################################################
  def inject_baddata(self, moment=1, probability=10):
    self.is_baddata = True
    self.time_baddata = moment
    if probability<100 and probability>0:
      self.baddata_prob = probability
    else:
      raise ValueError('probability 只能设置为 0-100!')  

  def __inject_baddata(self, t):
    if self.time_baddata <= t:
      #np.random.seed(0)
      p = np.array([1-5e-6*self.baddata_prob*self.measure_size, 5e-6*self.baddata_prob*self.measure_size])
      index = np.random.choice([0,1], p=p)
      if index == 1:
        sparse_amount = random.randint(1,10)  # 产生 1-10 个 幅值 0-100 的坏值
        measure_tobe_injected = np.c_[np.ones((1,sparse_amount)), np.zeros((1,self.measure_size-sparse_amount))][0] * 100
        np.random.shuffle(measure_tobe_injected)
        measure_tobe_injected = np.mat(measure_tobe_injected).T
        self.z_observed += measure_tobe_injected
        print('产生了%i个坏值!'%(sparse_amount))
      ## 看往哪些测量注入了坏值，暂时用不上，但是保留
      '''
      cnt = 0
      nonzero_cnt = 0
      measure = []
      measure_tobe_injected_list = []
      for i in np.array(measure_tobe_injected):
        for j in i:
          if round(j,1)!=0.0:
            measure.append(round(j,1))
            measure_tobe_injected_list.append(cnt)
            nonzero_cnt += 1
          cnt += 1
      '''


  #############################################################
  # 函数 -- 
  #       inject_falsedata(): 注入虚假数据
  #           该方法在指定状态后，以尽可能少地篡改测量仪表达成攻击目标（还未实现）
  #           现在仅仅是随意攻击
  # 输入 --  
  #     * dest_state_index: 欲攻击的状态下标 (type:[])
  # 返回 --
  #     * falsedata_info_dict [type:dic]
  #                     攻击向量的特性 - state_injected: 注入的状态攻击向量
  #                                  - measurement_injected: 注入的测量攻击向量
  #                                  - state_injected_amount: 注入了多少个状态值
  #                                  - measurement_injected_amount: 注入了多少个测量值
  #############################################################
  def inject_falsedata(self, moment=1):
    self.is_FDI = True
    self.time_falsedata = moment

  def __inject_falsedata(self, t):
    if self.time_falsedata <= t:
      sparse_amount = random.randint(1,10)  # 产生对 1-10 个状态的幅值 0-100 的虚假数据攻击
      state_tobe_injected = np.c_[np.random.random((1,sparse_amount)), np.zeros((1,self.state_size-sparse_amount))][0] * 100
      np.random.shuffle(state_tobe_injected)
      measure_tobe_injected = self.H * np.mat(state_tobe_injected).T
      self.z_observed += measure_tobe_injected
      # 看哪些状态和测量被攻击了，暂时保留
      '''
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
      '''
      # print('对'+str(sparse_amount)+'个状态注入虚假数据')#后，更改了'+str(nonzero_cnt)+'个测量值.')

  #############################################################
  # 函数 -- 
  #       inject_falsedata_PCA()
  #                     利用量测信息构造虚假数据并注入(!并未成功)
  # 输入 --  
  #     * moment: 什么时刻开始注入攻击
  # 返回 --
  #     * falsedata_info_dict [type:dic]
  #                     攻击向量的特性 - state_injected: 注入的状态攻击向量
  #                                  - measurement_injected: 注入的测量攻击向量
  #                                  - state_injected_amount: 注入了多少个状态值
  #                                  - measurement_injected_amount: 注入了多少个测量值
  #############################################################
  def inject_falsedata_PCA(self, moment=1):
    self.is_FDI_PCA = True
    self.time_falsedata = moment

  def __inject_falsedata_PCA(self, t):
    if self.time_falsedata <= t:
      eigval,eigvec = linalg.eig(self.z_observed_history * self.z_observed_history.T)
      eig_enum = []
      # 给z的奇异值排序
      for i,e in enumerate(eigval):
        eig_enum.append([i,e])
      eig_enum.sort(key=(lambda x:x[1]))
      eig_sorted = []
      for i in eig_enum:
        eig_sorted.append(i[0])
      
      eigvec_sorted = eigvec[:,eig_sorted]
      H_pca = eigvec_sorted[:,:self.state_size]

      #H_pca = Vt[:self.state_size, :].T # shape(m,n), 取前n个特征值对应的特征向量
      sparse_amount = random.randint(1,10)
      state_tobe_injected = np.c_[np.ones((1,sparse_amount)), np.zeros((1,self.state_size-sparse_amount))][0] * 100
      np.random.shuffle(state_tobe_injected)
      measure_tobe_injected = H_pca * np.mat(state_tobe_injected).T
      self.z_observed += measure_tobe_injected
      
      # 想哪些状态和测量值注入了攻击
      '''
      which_state_tobe_injected_list = []
      cnt = 0
      for i in state_tobe_injected:
        if i != 0:
          which_state_tobe_injected_list.append(cnt)
        cnt+=1

      cnt = 0
      nonzero_cnt = 0
      measure = []
      measure_tobe_injected_list = []
      for i in np.array(measure_tobe_injected):
        for j in i:
          if round(j.real,1)>1.0:
            measure.append(round(j.real,1))
            measure_tobe_injected_list.append(cnt)
            nonzero_cnt += 1
          cnt += 1
      '''
      # print('对'+str(sparse_amount)+'个状态注入虚假数据')

  #############################################################
  # 函数 -- 
  #       detect_baddata()
  #                     利用量测信息构造虚假数据并注入(!并未成功，得获得多个时刻的数据)
  # 输入 --  
  #     NULL
  # 返回 --
  #     测量误差的二范数(float)
  #############################################################
  def detect_baddata(self):
    detect_res = np.sqrt((self.z_observed - self.H * self.x_est_center).T * (self.z_observed - self.H * self.x_est_center))[0,0]
    return float(detect_res)
