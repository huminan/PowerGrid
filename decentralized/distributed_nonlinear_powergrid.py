from decentralized.distributed_linear_powergrid import DistributedLinearPowerGrid
from nonlinear_powergrid import NonLinearPowerGrid
from decentralized.estimators import Richardson,Stocastic
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pylab as pylab
import copy
import time

class DistributedNonLinearPowerGrid(DistributedLinearPowerGrid):
  def __init__(self, nodes = [], pmu=[], conf_dict={}):
    super().__init__(nodes=nodes, pmu=pmu, conf_dict=conf_dict)
    self.is_inneriter_plot = conf_dict['is_inneriter_plot']
    self.is_outeriter_plot = conf_dict['is_outeriter_plot']

  def estimator(self,estimator_name='Richardson'):
    """
    进行分布式估计
    """
    self.estimator_conf_dict['is_finite'] = True
    self.estimator_conf_dict['main_period'] = self.nodes_num-2
    child_estimator = Richardson(self.cluster_info_dict, self.neighbors_dict, self.x_real, self.x_est_center, self.estimator_conf_dict)

    res = {'sim_time':self.sim_time, 'state_est':np.empty((self.state_size,0)), 'state_predict':np.empty((self.state_size,1)), 'state_real':np.empty((self.state_size,0)), 'state_error':np.empty((self.state_size,0))}
    a = self.x_real.copy()
    b = np.zeros((self.size*2,1), dtype=complex)
    state_error_mat = np.mat(np.eye(self.state_size))
    # 设置初始值
    self.x_est_distribute_lists = copy.deepcopy(np.array_split(self.x_real, self.node_state_split))
    self.x_est_distribute = self.x_real.copy()
    # 开始仿真
    for t in range(self.sim_time):
      self.next() # 进入下一时刻
      if self.is_baddata is True:
        self.__inject_baddata(t)
      if self.is_FDI is True:
        self.__inject_falsedata(t)
      elif self.is_FDI_PCA is True:
        self.z_observed_history = np.column_stack((self.z_observed_history, self.z_observed))
        self.__inject_falsedata_PCA(t)
      '''
      # 全局状态估计
      is_bad_centralized,residual_centralized = super().estimator(0)
      # 全局估计分布化
      self.x_est_center_list = []
      tmp_cnt = 0
      for i in range(self.nodes_num):
        self.x_est_center_list.append(self.x_est_center[tmp_cnt:tmp_cnt+self.node_col_amount[i],:])
        tmp_cnt += self.node_col_amount[i]
      '''
      # 开始状态估计
      bar=tqdm(total=self.iter_time)
      record = []; record_error = []
      for u in range(self.iter_time):
        # 计算参数并分布化
        J,Phi = self.jaccobi_H(self.x_est_distribute) # 5.4s
        residual = self.z_observed-self.create_measurement(self.x_est_distribute) # 2.47s
        x,residual_distribute,J_distribute,Phi_distribute = self.distribulize(self.x_real,residual,J,Phi) # x没用的, 0.001s
        # 分布式计算delta
        delta_distribute_lists,delta_distribute = child_estimator.algorithm(J_distribute, Phi_distribute, self.R_I_distribute_diag, residual_distribute, is_plot=False) # 1.83s
        # 增量
        for i in range(self.nodes_num):
          self.x_est_distribute_lists[i] += delta_distribute_lists[i]
        self.x_est_distribute += delta_distribute
        # 中间参数
        if self.is_inneriter_plot is True:
          plt.figure('第'+str(t)+'次外部迭代的第'+str(u)+'次内部迭代细节图')
          plt.subplot(311);'''plt.title('测量残差')''';plt.plot(np.arange(1,self.measure_size+1).T, residual);plt.xlabel('测量编号');plt.ylabel('残差')
          plt.subplot(312);'''plt.title('状态增量')''';plt.plot(np.arange(1,self.state_size+1).T, delta_distribute);plt.xlabel('状态编号');plt.ylabel('增量')
          plt.subplot(313);'''plt.title('状态误差')''';plt.plot(np.arange(1,self.state_size+1).T, self.x_est_distribute-self.x_real);plt.xlabel('状态编号');plt.ylabel('误差')
          plt.show()
        record.append(np.array(delta_distribute/self.x_est_distribute)[:,0]);record_error.append(np.array(self.x_est_distribute-self.x_real)[:,0])
        bar.update(1)
      bar.close()
      # 预测
      a,b,state_error_mat = self.predict(self.x_est_distribute,[a,b,state_error_mat])
      # 画出中间过程
      if self.is_outeriter_plot is True:
        x_axis = np.tile(np.arange(1,self.iter_time+1),[self.state_size,1]).T
        plt.figure('第'+str(t)+'次外部迭代状态细节图')
        plt.subplot(211);plt.title('状态相对变化: (变化值)/估计值');plt.xlabel('内迭代次数');plt.ylabel('相对增量');plt.plot(x_axis,record)
        plt.subplot(212);plt.title('状态误差图');plt.xlabel('内迭代次数');plt.ylabel('误差');plt.plot(x_axis,record_error)
        plt.show()
      '''
      is_bad,residual = self.detect_baddata(is_plot=False)
      # 画图
      if is_bad is True:
      print('分布式估计器在第%i时刻检测到坏值，估计的残差为: %s' % (t, str(residual)))
      if is_bad_centralized is True:
        print('集中式估计器在第%i时刻检测到坏值，估计的残差为: %.3f' % (t, residual_centralized))
      '''
      res['state_est'] = np.column_stack((res['state_est'], self.x_est_distribute))
      res['state_real'] = np.column_stack((res['state_real'], self.x_real))
      res['state_error'] = np.column_stack((res['state_error'], np.array(self.x_est_distribute-self.x_real)))
      res['state_predict'] = np.column_stack((res['state_predict'], self.x_predict))
    # 总体报告
    self.plot(res)


