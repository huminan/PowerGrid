from powergrid import PowerGrid
import numpy as np
from scipy.linalg import block_diag
from numpy import linalg

# Measure variance 
###### 真实参数 ######
# PMU: 电压-0.002%pu; 相角-0.01度
# SCADA: 电压-0.3%pu; 功率-0.3%pu
####### over ########
PMU_VOLTAGE_VARIANCE = .002
PMU_ANGLE_VARIANCE = .01
SCADA_VOLTAGE_VARIANCE = .3
SCADA_POWER_VARIANCE = .3

class LinearPowerGrid(PowerGrid):
  def __init__(self, size):
    super().__init__(size=size)

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
 #       上一时刻的状态估计值 x_real (非真实值)
 #       观测值 z_observed() (数据中未给出，使用x_real加上噪声后*H求得)(非真实值)
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
    self.x_real = x
    # 加有噪声的状态
    self.x_observed = x + np.random.random((self.state_size,1))

    self.z_observed = self.H * self.x_observed
    self.x_est_center = (self.H.H*self.R_I*self.H).I*self.H.H*self.R_I*self.z_observed
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
    self.x_real = np.delete(self.x_real, (self.bus_ref-1)*2, 0)
    self.x_real = np.delete(self.x_real, (self.bus_ref-1)*2, 0)
    self.x_observed = np.delete(self.x_observed, (self.bus_ref-1)*2, 0)
    self.x_observed = np.delete(self.x_observed, (self.bus_ref-1)*2, 0)
    self.z_observed = self.H * self.x_observed
    self.x_est_center = (self.H.H*self.R_I*self.H).I*self.H.H*self.R_I*self.z_observed
    self.Phi = self.H.H*self.R_I*self.H
    self.state_size -= 2
    self.is_reference_deleted = True

#############################################################
 # 函数 -- 
 #       inject_baddata(): 注入坏值
 # 输入 --  
 #       sparse_amount 攻击向量的稀疏性(状态攻击向量)
 #       amptitude     攻击的幅值(后面考虑自定义不同幅值，输入列表)
 #       measure_tobe_injected（还未实现）
 #                     自定义的攻击向量(None - 随机选择要攻击的测量值)
 # 返回 --
 #       baddata_info_dict [type: dic]
 #                     measurement_injected: 注入的测量攻击的值(非攻击向量，)
 #                     measurement_injected_amount: 注入了多少个测量值
#############################################################
  def inject_baddata(self, sparse_amount=0, amptitude=0, measure_tobe_injected=None):
    measure_tobe_injected = np.c_[np.ones((1,sparse_amount)), np.zeros((1,self.measure_size-sparse_amount))][0] * amptitude
    np.random.shuffle(measure_tobe_injected)
    measure_tobe_injected = np.mat(measure_tobe_injected).T
    self.z_observed += measure_tobe_injected

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

    self.x_est_center = self.Phi.I * self.H.H * self.R.I * self.z_observed
    print('更改了'+str(sparse_amount)+'个测量值.')
    baddata_info_dict = {'measurement_injected':measure, 'measurement_injected_amount':sparse_amount}
    return baddata_info_dict

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
    if (sparse_amount==0) and (amptitude==0):
      print('未注入攻击!')
      return True
    if (delete_previous_injected is False) or (measure_tobe_injected is None):  # 无历史
      if measure_tobe_injected is None:   # 随机生成sparse_num稀疏的状态攻击向量，再利用Hc生成要攻击的测量值
        state_tobe_injected = np.c_[np.random.random((1,sparse_amount)), np.zeros((1,self.state_size-sparse_amount))][0] * amptitude
        np.random.shuffle(state_tobe_injected)
        measure_tobe_injected = self.H * np.mat(state_tobe_injected).T
      else:   # 自定义攻击向量
        state_tobe_injected = self.Phi.I * self.H.H * self.R.I * (self.z_observed + measure_tobe_injected) - self.x_est_center
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

    self.x_est_center = self.Phi.I * self.H.H * self.R.I * self.z_observed
    print('对'+str(sparse_amount)+'个状态('+str(which_state_tobe_injected_list)+')注入虚假数据后，更改了'+str(nonzero_cnt)+'个测量值.')
    print('攻击的测量值的位置序列为: '+ str(measure_tobe_injected_list) + '.')
    falsedata_info_dict = {'state_injected':state_tobe_injected, 'measurement_injected':measure_tobe_injected, 'state_injected_amount': sparse_amount, 'measurement_injected_amount':nonzero_cnt, 'amptitude': amptitude, 'measurement_injected_sequence': measure_tobe_injected_list}
    return falsedata_info_dict

#############################################################
 # 函数 -- 
 #       inject_falsedata_PCA()
 #                     利用量测信息构造虚假数据并注入(!并未成功，得获得多个时刻的数据)
 # 输入 --  
 #     * sparse_amount 要攻击多少个状态(若measure_tobe_injected非None, 则以它为准)
 #     * amptitude     攻击的幅值(后面考虑自定义不同幅值，输入列表)
 #     * delete_previous_injected
 #                     ! 还没实现
 # 返回 --
 #     * falsedata_info_dict [type:dic]
 #                     攻击向量的特性 - state_injected: 注入的状态攻击向量
 #                                  - measurement_injected: 注入的测量攻击向量
 #                                  - state_injected_amount: 注入了多少个状态值
 #                                  - measurement_injected_amount: 注入了多少个测量值
#############################################################
  def inject_falsedata_PCA(self, sparse_amount=0, amptitude=0, delete_previous_injected=False):
    eigval,eigvec = linalg.eig(self.z_observed * self.z_observed.T)
    eig_enum = []
    for i,e in enumerate(eigval):
      eig_enum.append([i,e])
    eig_enum.sort(key=(lambda x:x[1]))
    
    eig_sorted = []
    for i in eig_enum:
      eig_sorted.append(i[0])
    
    eigvec_sorted = eigvec[:,eig_sorted]
    H_pca = eigvec_sorted[:,:self.state_size]

    #H_pca = Vt[:self.state_size, :].T # shape(m,n), 取前n个特征值对应的特征向量
    state_tobe_injected = np.c_[np.ones((1,sparse_amount)), np.zeros((1,self.state_size-sparse_amount))][0] * amptitude
    np.random.shuffle(state_tobe_injected)
    measure_tobe_injected = H_pca * np.mat(state_tobe_injected).T
    self.z_observed += measure_tobe_injected
    
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

    self.x_est_center = self.Phi.I * self.H.H * self.R.I * self.z_observed
    print('对'+str(sparse_amount)+'个状态('+str(which_state_tobe_injected_list)+')注入虚假数据后，更改了'+str(nonzero_cnt)+'个测量值.')
    # print('攻击的测量值的位置序列为: '+ str(measure_tobe_injected_list) + '.')
    falsedata_info_dict = {'state_injected':state_tobe_injected, 'measurement_injected':measure, 'state_injected_amount': sparse_amount, 'measurement_injected_amount':nonzero_cnt, 'amptitude': amptitude, 'measurement_injected_sequence': measure_tobe_injected_list}
    return falsedata_info_dict

#############################################################
 # 函数 -- 
 #       detect_baddata()
 #                     利用量测信息构造虚假数据并注入(!并未成功，得获得多个时刻的数据)
 # 输入 --  
 #     NULL
 # 返回 --
 #     NULL
#############################################################
  def detect_baddata(self):
    detect_res = np.sqrt((self.z_observed - self.H * self.x_est_center).T * (self.z_observed - self.H * self.x_est_center))[0,0]
    print('测量误差的二范数为: ' + str(detect_res) + '.')
