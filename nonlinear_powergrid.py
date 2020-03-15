from linear_powergrid import LinearPowerGrid
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pylab as pylab

class NonLinearPowerGrid(LinearPowerGrid):
  def __init__(self, pmu=[], conf_dict={}):
    super().__init__(pmu=pmu, conf_dict=conf_dict)
    self.x_est_center = np.mat(self.x_real,dtype=complex)

  def estimator(self, once=False):
    """
    调用估计器
  
    输入
    ---- 
    once: 当取True时，一般是被子类调用，只进行一次估计，然后检测，其它什么都别干

    返回
    ----
    NULL
    """
    a = np.copy(self.x_real)
    b = np.zeros((self.size*2,1), dtype=complex)
    state_error_mat = np.mat(np.eye(self.state_size))
    #is_bad = False
    #residual = 0.0
    if once is True:
      for u in range(self.iter_time):
        J,Phi = self.jaccobi_H(self.x_est_center)
        z_model = self.create_measurement(self.x_est_center)
        self.x_est_center += (J.H*self.R_I*J).I* J.H*self.R_I*(self.z_observed - z_model)
      #is_bad,residual = self.__detect_baddata()
      #return is_bad,residual
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
        # 非线性最小二乘估计(迭代)
        bar=tqdm(total=self.iter_time)
        for u in range(self.iter_time):
          J,Phi = self.jaccobi_H(self.x_est_center)
          z_model = self.create_measurement(self.x_est_center)
          self.x_est_center += Phi.I* J.H*self.R_I*(self.z_observed - z_model)
          bar.update(1)
        bar.close()
        res[0].append(complex(self.x_est_center[10]))
        #is_bad,residual = self.__detect_baddata()
        #if is_bad is True:
        #  print('第%i时刻检测到坏值，估计的残差为: %.3f' % (t, residual))
        res[2].append(complex(self.x_real[10]))
        res[3].append(np.array(self.x_est_center-self.x_real)[:,0])
        if t is not self.sim_time:
          a,b,state_error_mat = self.predict(self.x_est_center,[a,b,state_error_mat])
          self.next()
          res[1].append(complex(self.x_predict[10]))
      plt.figure('状态演变')
      plt.subplot(211)
      plt.title('第10状态跟随图')
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
