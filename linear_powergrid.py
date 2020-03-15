#-*- coding: utf-8 -*-
from base import StateEstimationBase
import numpy as np
import random
from scipy.linalg import block_diag
from numpy import linalg
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import time
import os
import json

PMU_VOLTAGE_VARIANCE = .002
PMU_ANGLE_VARIANCE = .01
SCADA_VOLTAGE_VARIANCE = .3
SCADA_POWER_VARIANCE = .3

class LinearPowerGrid(StateEstimationBase):
  def __init__(self, pmu=[], conf_dict={}):
    super().__init__(pmu=pmu, conf_dict=conf_dict)
    # 标记
    self.is_baddata = False
    self.is_FDI = False
    self.is_FDI_PCA = False
    # 变量声明
    self.time_baddata = 0
    self.time_falsedata = 0
    self.x_est_center = np.zeros((2*self.size,1))
    self.z_observed_history = None
    self.x_predict = np.zeros((2*self.size,1))
    # 提前声明
    self.row_reorder_matrix = None
    self.col_reorder_matrix = None
    # 计算参数(R)
    self.set_variance_matrix()
    self.set_linear_model()
    # 创建量测
    self.z_observed = self.create_measurement(self.x_real) + self.R_error * np.mat(np.random.random((self.measure_size,1))) # 真实量测加上噪声

  def set_linear_model(self):
    if self.model_name == 'PowerGrid':
      # 从缓存提取
      if os.path.exists('cache/IEEE_'+str(self.size)+'_linear_info.json') is True:
        with open('cache/IEEE_'+str(self.size)+'_linear_info.json','r',encoding='utf-8') as f:
          saved_conf = json.load(f)
          self.H = np.mat(saved_conf['H_real'],dtype=complex)+np.mat(saved_conf['H_imag'])*(1j)
          self.measure_size = saved_conf['measure_size']
          self.state_size = saved_conf['state_size']
        self.Phi = self.H.H * self.R_I * self.H
      else:
        # 计算H矩阵
        self.H,self.Phi = self.jaccobi_H(self.x_real)
        self.measure_size = self.H.shape[0]  # 量测数量
        self.state_size = self.H.shape[1]    # 状态数量
        # 保存当前配置
        conf_to_save = {
          'H_real': self.H.real.astype(float).tolist(),
          'H_imag': self.H.imag.astype(float).tolist(),
          'measure_size': self.measure_size,
          'state_size': self.state_size,
        }
        with open('cache/IEEE_'+str(self.size)+'_linear_info.json','w',encoding='utf-8') as f:
          f.write(json.dumps(conf_to_save,ensure_ascii=False))
    elif self.model_name == 'WSNs':
      # 从缓存提取
      if os.path.exists('cache/WSNs_'+str(self.size)+'_linear_info.json') is True:
        with open('cache/WSNs_'+str(self.size)+'_linear_info.json','r',encoding='utf-8') as f:
          saved_conf = json.load(f)
          self.H = np.mat(saved_conf['H_real'],dtype=complex)+np.mat(saved_conf['H_imag'])*(1j)
          self.measure_size = saved_conf['measure_size']
          self.state_size = saved_conf['state_size']
        self.Phi = self.H.H * self.R_I * self.H
      else:
        # 计算H矩阵
        self.H,self.Phi = self.jaccobi_H(self.x_real)
        self.measure_size = self.H.shape[0]  # 量测数量
        self.state_size = self.H.shape[1]    # 状态数量
        # 保存当前配置
        conf_to_save = {
          'H_real': self.H.real.astype(float).tolist(),
          'H_imag': self.H.imag.astype(float).tolist(),
          'measure_size': self.measure_size,
          'state_size': self.state_size,
        }
        with open('cache/WSNs_'+str(self.size)+'_linear_info.json','w',encoding='utf-8') as f:
          f.write(json.dumps(conf_to_save,ensure_ascii=False))

  def jaccobi_H(self, x_operation):
    """
    计算H矩阵中由branch与bus决定的参数
      以文件中的out和in为准, 将测量仪表放置在out处.

    输入
    ----
    order: 节点排列的顺序(对分布式系统有用)
    x_operation: 上一采样时刻的状态估计值(第一次将初始时刻真实值代入)

    中间结果
    ----
      测量数: measure_size (不该在这个函数里首次定义)
      状态数: state_size   (不该在这个函数里首次定义)

    公式
    ----
      SCADA只可以测量(电压)
      PMU可以测量(电压,相角)

    返回
    ----
    雅可比矩阵: Jaccobi_H
    """
    # 初始化
    Jaccobi_H = np.mat(np.zeros([0, 2*self.size]), dtype=complex)   # 量测矩阵
    node_list = list(self.bus_info_dict.keys())
    if self.model_name == 'PowerGrid':
      # 配置总线测量
      cnt = 0
      for bus,bus_info_dict in self.bus_info_dict.items():
        if bus_info_dict['attr'] == 'PMU':
          a = np.mat(np.zeros([2, 2*self.size]))
          a[0,cnt] = 1
          a[1,cnt+1] = 1
          Jaccobi_H = np.row_stack((Jaccobi_H, a))
        else:
          a = np.mat(np.zeros([1, 2*self.size]))
          a[0,cnt] = 1
          Jaccobi_H = np.row_stack((Jaccobi_H, a))
        # 配置边测量
        for connect_info_dict in bus_info_dict['connect']:
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
        cnt += 2 # 4~9s
    elif self.model_name == 'WSNs':
      cnt = 0
      for bus,bus_info_dict in self.bus_info_dict.items():
        if bus_info_dict['attr'] == 'PMU':  # ref节点知道自己的位置
          a = np.mat(np.zeros([2, 2*self.size]))
          a[0,cnt] = 1
          a[1,cnt+1] = 1
          Jaccobi_H = np.row_stack((Jaccobi_H, a))
        # 配置边测量
        for connect_info_dict in bus_info_dict['connect']:
          a = np.mat(np.zeros([len(self.h), 2*self.size]), dtype=complex)
          for i,J_i in enumerate(self.jacobi_h_ij):
            for j,J_ij in enumerate(J_i):
              for k,symbol_i in enumerate(self.state_i_symbol):
                J_ij = J_ij.subs(symbol_i, x_operation[cnt+k,0])
              for k,symbol_j in enumerate(self.state_j_symbol):
                J_ij = J_ij.subs(symbol_j, x_operation[2*node_list.index(connect_info_dict['dst'])+k,0])
              a[i, cnt+j] = J_ij
          for j,J_j in enumerate(self.jacobi_h_ji):
            for i,J_ji in enumerate(J_j):
              for k,symbol_i in enumerate(self.state_i_symbol):
                J_ji = J_ji.subs(symbol_i, x_operation[cnt+k,0])
              for k,symbol_j in enumerate(self.state_j_symbol):
                J_ji = J_ji.subs(symbol_j, x_operation[2*node_list.index(connect_info_dict['dst'])+k,0])
              a[j, 2*node_list.index(connect_info_dict['dst'])+i] = J_ji
          Jaccobi_H = np.row_stack((Jaccobi_H, a))  # Augment H
        cnt += 2 # 4~9s
    Phi = Jaccobi_H.H * self.R_I * Jaccobi_H # 0.04s
    return Jaccobi_H, Phi

  def create_measurement(self, x_operation):
    """
    通过模型构造量测(虚假), 并画出非线性模型与线性模型的量测的差值.
    """
    h_operation = np.array((),dtype=complex)
    node_list = list(self.bus_info_dict.keys())
    cnt = 0
    if self.model_name == 'PowerGrid':
      for bus,bus_info_dict in self.bus_info_dict.items():
        # 配置总线测量
        if bus_info_dict['attr'] == 'PMU':
          h_operation = np.append(h_operation, x_operation[cnt])
          h_operation = np.append(h_operation, x_operation[cnt+1])
        else: # SCADA
          h_operation = np.append(h_operation, x_operation[cnt])
        # 配置边测量
        for connect_info_dict in bus_info_dict['connect']:
          for z_tmp in self.h:
            for k,symbol_i in enumerate(self.state_i_symbol):
              z_tmp = z_tmp.subs(symbol_i, x_operation[cnt+k,0])
            for k,symbol_j in enumerate(self.state_j_symbol):
              z_tmp = z_tmp.subs(symbol_j, x_operation[2*node_list.index(connect_info_dict['dst'])+k,0])
            for k,symbol_value in enumerate(self.value_symbol):
              z_tmp = z_tmp.subs(symbol_value, connect_info_dict['para'][k])
            h_operation = np.append(h_operation, z_tmp)
        cnt += 2
    elif self.model_name == 'WSNs':
      for bus,bus_info_dict in self.bus_info_dict.items():
        if bus_info_dict['attr'] == 'PMU':  # ref节点的位置
          h_operation = np.append(h_operation, x_operation[cnt])
          h_operation = np.append(h_operation, x_operation[cnt+1])
        # 配置边测量
        for connect_info_dict in bus_info_dict['connect']:
          for z_tmp in self.h:
            for k,symbol_i in enumerate(self.state_i_symbol):
              z_tmp = z_tmp.subs(symbol_i, x_operation[cnt+k,0])
            for k,symbol_j in enumerate(self.state_j_symbol):
              z_tmp = z_tmp.subs(symbol_j, x_operation[2*node_list.index(connect_info_dict['dst'])+k,0])
            h_operation = np.append(h_operation, z_tmp)
        cnt += 2
    h_operation = np.mat(h_operation, dtype=complex).T
    
    h_operation = self.H * x_operation # 线性模型
    ''' 画出两种量测计算方法的差值
    __z_real = self.H * self.x_real # 通过状态x_real*H得到, 真实情况是非线性的, 因此存在很大的问题.
    plt.figure('两种量测的差值')
    plt.plot(list(range(self.measure_size)), h_operation - __z_real, 'b.')
    plt.show()
    '''
    return h_operation

  def set_variance_matrix(self):
    """
    设置参数
    """
    self.R_real = np.mat(np.zeros([0, 0]), dtype=complex)   # 真实量测协方差矩阵
    self.R = np.mat(np.zeros([0, 0]), dtype=complex)        # 计算量测协方差矩阵
    self.R_error = np.mat(np.zeros([0, 0]), dtype=complex)  # 真实量测误差
    for bus,bus_info_dict in self.bus_info_dict.items():
      if self.model_name == 'PowerGrid':
        # 配置总线测量
        if bus_info_dict['attr'] == 'PMU':
          self.measure_who.append([bus, bus])
          self.R = block_diag(self.R, np.eye(1)*(self.pmu_voltage_variance**2))
          self.R = block_diag(self.R, np.eye(1)*(self.pmu_angle_variance**2))
          self.R_real = block_diag(self.R_real, np.eye(1)*(PMU_VOLTAGE_VARIANCE**2))
          self.R_real = block_diag(self.R_real, np.eye(1)*(PMU_ANGLE_VARIANCE**2))
          self.R_error = block_diag(self.R_error, np.eye(1)*(PMU_VOLTAGE_VARIANCE))
          self.R_error = block_diag(self.R_error, np.eye(1)*(PMU_ANGLE_VARIANCE))
        else: # SCADA
          self.measure_who.append(bus)
          self.R = block_diag(self.R, np.eye(1)*(self.scada_voltage_variance**2))
          self.R_real = block_diag(self.R_real, np.eye(1)*(SCADA_VOLTAGE_VARIANCE**2))
          self.R_error = block_diag(self.R_error, np.eye(1)*(SCADA_VOLTAGE_VARIANCE)*3) # 这个*3是干嘛的?
        if self.bus_type[bus-1] == 3:  # if reference
          self.nodes_ref.append(bus)
        # 配置边测量
        for connect_info_dict in bus_info_dict['connect']:
          self.measure_who.append([bus, connect_info_dict['dst']])
          self.R = block_diag(self.R, np.eye(2)*(self.scada_power_variance**2))
          self.R_real = block_diag(self.R_real, np.eye(2)*(SCADA_POWER_VARIANCE**2))
          self.R_error = block_diag(self.R_error, np.eye(2)*(SCADA_POWER_VARIANCE))
      elif self.model_name == 'WSNs':
        if bus_info_dict['attr'] == 'PMU':
          self.measure_who.append([bus, bus])
          self.R = block_diag(self.R, np.eye(1)*(self.pmu_voltage_variance**2))
          self.R = block_diag(self.R, np.eye(1)*(self.pmu_voltage_variance**2))
          self.R_real = block_diag(self.R_real, np.eye(1)*(PMU_VOLTAGE_VARIANCE**2))
          self.R_real = block_diag(self.R_real, np.eye(1)*(PMU_VOLTAGE_VARIANCE**2))
          self.R_error = block_diag(self.R_error, np.eye(1)*(PMU_VOLTAGE_VARIANCE))
          self.R_error = block_diag(self.R_error, np.eye(1)*(PMU_VOLTAGE_VARIANCE))
        if self.bus_type[bus-1] == 3:  # if reference
          self.nodes_ref.append(bus)
        # 配置边测量
        for connect_info_dict in bus_info_dict['connect']:
          self.measure_who.append([bus, connect_info_dict['dst']])
          self.R = block_diag(self.R, np.eye(1)*(self.scada_power_variance**2))
          self.R = block_diag(self.R, np.eye(1)*(self.pmu_angle_variance**2))
          self.R_real = block_diag(self.R_real, np.eye(1)*(SCADA_POWER_VARIANCE**2))
          self.R_real = block_diag(self.R_real, np.eye(1)*(PMU_ANGLE_VARIANCE**2))
          self.R_error = block_diag(self.R_error, np.eye(1)*(SCADA_POWER_VARIANCE))
          self.R_error = block_diag(self.R_error, np.eye(1)*(PMU_ANGLE_VARIANCE))
    # 总结
    self.R = np.mat(self.R)
    self.R_real = np.mat(self.R_real)
    self.R_error = np.mat(self.R_error)
    self.R_I = self.R.I

  def delete_reference_bus(self):
    """
    删除H矩阵中的reference总线
    
    状态更改
    -------
    * is_reference_deleted: True

    返回
    ----
    NULL
    """
    for ref_node in self.nodes_ref:
      self.H = np.delete(self.H, (ref_node-1)*2, 1)
      self.H = np.delete(self.H, (ref_node-1)*2, 1)
      self.x_real = np.delete(self.x_real, (ref_node-1)*2, 0)
      self.x_real = np.delete(self.x_real, (ref_node-1)*2, 0)
      self.state_size -= 2
    self.z_observed = self.H * self.x_real + self.R * np.random.random((self.measure_size,1))
    self.Phi = self.H.H*self.R_I*self.H
    self.is_reference_deleted = True

  def estimator(self, once=False):
    """
    调用估计器
  
    输入
    ---- 
    self.sim_time: 仿真时间(多少次)，当取0时，一般是被子类调用，只进行一次估计，然后检测，其它什么都别干
    * is_bad_data: 是否会产生坏数据 <False,True>
    * falsedata_type: 虚假数据注入攻击的方法
      |- 'normal'
      |- 'pca'

    返回
    ----
    NULL
    """
    a = np.copy(self.x_real)
    b = np.zeros((self.size*2,1), dtype=complex)
    state_error_mat = np.mat(np.eye(self.state_size))
    is_bad = False
    residual = 0.0
    if once is True:
      self.x_est_center = self.Phi.I*self.H.H*self.R_I*self.z_observed
      is_bad,residual = self.__detect_baddata()
      return is_bad,residual
    else: 
      if self.is_FDI_PCA is True:
        self.z_observed_history = np.mat(np.empty((self.measure_size,0)))
      res = [[],[],[],[]]  # 估计、预测、真实
      for t in range(self.sim_time+1):
        if self.is_baddata is True:
          self.__inject_baddata(t)
        if self.is_FDI is True:
          self.__inject_falsedata(t)
        elif self.is_FDI_PCA is True:
          self.z_observed_history = np.column_stack((self.z_observed_history, self.z_observed))
          self.__inject_falsedata_PCA(t)
        self.x_est_center = self.Phi.I*self.H.H*self.R_I*self.z_observed
        res[0].append(complex(self.x_est_center[0]))
        res[2].append(complex(self.x_real[0]))
        res[3].append(np.array(self.x_est_center-self.x_real)[:,0])
        is_bad,residual = self.__detect_baddata()
        if is_bad is True:
          print('第%i时刻检测到坏值，估计的残差为: %.3f' % (t, residual))
        if t is not self.sim_time:
          a,b,state_error_mat = self.predict(self.x_est_center,[a,b,state_error_mat])
          self.next()
          res[1].append(complex(self.x_predict[0]))
      plt.figure('状态演变')
      plt.subplot(211)
      plt.title('某状态跟随图')
      plt.plot(res[0], 'g*-')
      plt.plot([0]+res[1], 'b*-')
      plt.plot(res[2], 'y*-')
      plt.legend(['估计','预测','真实'], loc='upper right', frameon=False)
      plt.xlabel("时刻")
      plt.ylabel("幅值")
      plt.subplot(212)
      plt.title('状态误差')
      plt.plot(res[3], '.-')
      plt.show()

  def next(self, diff=None):
    """
    将系统更新至下一时刻

    输入
    ---- 
    * diff: 自定义状态变化量 (array)

    返回
    ----
    NULL
    """
    if diff is None:
      self.x_real += np.random.random((self.state_size,1)) * self.state_variance
    else:
      self.x_real += diff + np.random.random((self.state_size,1)) * self.state_variance
    #self.z_observed = self.H * self.x_real + np.random.random((self.measure_size,1))
    self.H,self.Phi = self.jaccobi_H(self.x_est_center)
    self.z_observed = self.create_measurement(self.x_real) + self.R_error * np.mat(np.random.random((self.measure_size,1)))

  def predict(self, est_x, para_before=[], model=1, para=None):
    """
    预测下一时刻的电网参数，这里以上一时刻估计值作为真实值

    输入
    ---- 
    * est_x: 上一时刻的状态估计值
    * para_before: 上一时刻的参数
    * model: 电网参数预测的模型
    * para: None - default [0.8,0.5]
            [alpha,beta]
    
    返回
    ----
    NULL
    """
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
        a[i] = alpha[i] * est_x[i] + (1-alpha[i]) * self.x_real[i]
        b[i] = beta[i] * (a[i] - a_before[i]) + (1-beta[i]) * b_before[i]
      for i in range(self.size*2):
        self.G[i] = (1+beta[i])*(1-alpha[i])*est_x[i] - beta[i]*a_before[i] + (1-beta[i])*b_before[i]
        self.A[i,i] = alpha[i]*(1+beta[i])
      # 预测下一步
      self.x_predict = self.A * est_x + self.G
      # 估计状态误差协方差矩阵
      state_error_mat = self.A * state_error_mat_before * self.A.H + np.mat(np.eye(self.state_size))*self.state_variance

      return a,b,state_error_mat

  def inject_baddata(self, moment=1, probability=10):
    """
    注入坏值

    输入
    ----  
    * moment: 在哪个时刻产生了坏数据
    * probability: 每个仪表产生坏值的概率: p/10e7
    
    返回
    ----
    ---(已删)---
    * baddata_info_dict [type: dic]
    * measurement_injected: 注入的测量攻击的值(非攻击向量，)
    * measurement_injected_amount: 注入了多少个测量值
    """
    self.is_baddata = True
    self.time_baddata = moment
    if probability<100 and probability>0:
      self.baddata_prob = probability
    else:
      raise ValueError('probability 只能设置为 0-100!')  

  def API_inject_baddata(self, t):
    self.__inject_baddata(t)

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
        print('第%i时刻产生了%i个坏值!'%(t,sparse_amount))
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

  def inject_falsedata(self, moment=1, conf_dic=None):
    """
    注入虚假数据
      该方法在指定状态后，以尽可能少地篡改测量仪表达成攻击目标（还未实现）
      现在仅仅是随意攻击
  
    输入
    ----  
    * moment: 攻击开始的时间
    * conf_dic: 攻击的配置 {字典}
      |- which_state
      |- effect
  
    返回
    ----
    * falsedata_info_dict [type:dic]
      攻击向量的特性 - state_injected: 注入的状态攻击向量
                    - measurement_injected: 注入的测量攻击向量
                    - state_injected_amount: 注入了多少个状态值
                    - measurement_injected_amount: 注入了多少个测量值
    """
    self.is_FDI = True
    self.time_falsedata = moment
    self.conf_dic = conf_dic

  def API_inject_falsedata(self, t):
    self.__inject_falsedata(t)

  def __inject_falsedata(self, t):
    if self.time_falsedata <= t:
      state_tobe_injected = np.zeros((1,self.state_size))
      if self.conf_dic is None:
        sparse_amount = random.randint(1,10)  # 产生对 1-10 个状态的幅值 0-100 的虚假数据攻击
        state_tobe_injected = np.c_[np.random.random((1,sparse_amount)), np.zeros((1,self.state_size-sparse_amount))][0] * 100
        np.random.shuffle(state_tobe_injected)
        print('第%i时刻对%i个状态注入了虚假数据'%(t,sparse_amount))#后，更改了'+str(nonzero_cnt)+'个测量值.'%(t,sparse_amount,nonzero_cnt))
      else:
        if self.is_distribute is True:
          for i,j in zip(self.conf_dic['which_state'], self.conf_dic['effect']):
            state_tobe_injected[0,int((np.array(range(self.state_size))*self.col_reorder_matrix)[0,i])] = j
        else:
          for i,j in zip(self.conf_dic['which_state'], self.conf_dic['effect']):
            state_tobe_injected[0,i] = j
      measure_tobe_injected = (self.H + np.multiply(self.H, np.random.rand(self.measure_size,self.state_size)*0.05)) * np.mat(state_tobe_injected).T
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

  def inject_falsedata_PCA(self, moment=1):
    """
    利用量测信息构造虚假数据并注入(!并未成功)
  
    输入
    ----  
    * moment: 什么时刻开始注入攻击
  
    返回 
    ----
    * falsedata_info_dict [type:dic]
    攻击向量的特性 - state_injected: 注入的状态攻击向量
                - measurement_injected: 注入的测量攻击向量
                - state_injected_amount: 注入了多少个状态值
                - measurement_injected_amount: 注入了多少个测量值
    """
    self.is_FDI_PCA = True
    self.time_falsedata = moment

  def API_inject_falsedata_PCA(self, t):
    self.__inject_falsedata_PCA(t)

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
      print('对'+str(sparse_amount)+'个状态注入虚假数据')

  def __detect_baddata(self):
    """
    坏值检测(线性集中系统)
  
    输入 
    ----  
    NULL
  
    返回 
    ----
    * 测量误差的二范数(float)
    """
    is_bad = False
    detect_res = np.sqrt((self.z_observed - self.H * self.x_est_center).T * (self.z_observed - self.H * self.x_est_center))[0,0]
    if detect_res > self.chi2square_val(self.measure_size,0.5):
      is_bad = True
    return is_bad,float(detect_res)
