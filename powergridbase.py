# coding=utf-8
from base import StateEstimationBase
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
from autograd import grad
import os
import json

import extract_config

###### 真实误差参数 ######
# PMU: 电压-0.002%pu; 相角-0.01度
# SCADA: 电压-0.3%pu; 功率-0.3%pu

def voltage(x, bus):
  return x

def angle(x,bus):
  return x

def active_power(params, v1, v2, a1, a2):
    """
    传输线路上有功功率的计算公式:
    P_ij = G_ij*Vi**2 - G_ij*Vi*Vj*sb.cos(Theta_i - Theta_j) - B_ij*Vi*Vj*sb.sin(Theta_i - Theta_j)

    输入
    ----
    * params: np.array((Gij, Bij, BSHij))
    * v1: 总线电压(out)
    * v2: 总线电压(in)
    * a1: 电压相角(out)
    * a2: 电压相角(in)

    返回
    ----
    * 有功功率
    """
    weights = np.dot(params, np.array([
      [1,0,0],
      [1,0,0],
      [0,1,0]]) )
    states = np.array(v1**2, -v1*v2*np.cos(a1-a2), -v1*v2*np.sin(a1-a2))
    return np.dot(weights, states)
  
def reactive_power(params, v1, v2, a1, a2):
    """
    传输线路上无功功率的计算公式:
    Q_ij = -(B_ij + BSH_ij)*Vi**2 + B_ij*Vi*Vj*sb.cos(Theta_i-Theta_j) - G_ij*Vi*Vj*sb.sin(Theta_i-Theta_j)

    输入
    ----
    * params: np.array((Gij, Bij, BSHij))
    * v1: 总线电压(out)
    * v2: 总线电压(in)
    * a1: 电压相角(out)
    * a2: 电压相角(in)

    返回
    ----
    * 无功功率
    """
    weights = np.dot(params, np.array([
      [0,1,1],
      [0,1,0],
      [1,0,0]]) )
    states = np.array(-v1**2, v1*v2*np.cos(a1-a2), -v1*v2*np.sin(a1-a2))
    return np.dot(weights, states)

class PowerGridBase(StateEstimationBase):
  def __init__(self, conf_path, bus_rule_path, branch_rule_path, pmu=[], conf_dict={}):
    # GUI 配置
    self.model_name = conf_dict['model_name']
    self.size = int(conf_dict['network_size'])
    self.state_variance = float(conf_dict['state_change'])
    self.sim_time = int(conf_dict['sim_time'])
    self.pmu_voltage_variance = float(conf_dict['pmu_voltage_variance'])
    self.pmu_angle_variance = float(conf_dict['pmu_angle_variance'])
    self.scada_voltage_variance = float(conf_dict['scada_voltage_variance'])
    self.scada_power_variance = float(conf_dict['scada_power_variance'])
    # 非线性参数
    self.iter_time = int(conf_dict['nonlinear_iter_time'])
    self.stop_error = int(conf_dict['nonlinear_stop_error'])
    
    self.pmu = []
    for i in range(self.size):
      if i in pmu:
        self.pmu.append('PMU')
      else:
        self.pmu.append('SCADA')
    # 状态轨迹
    self.A = np.mat(np.eye(2*self.size), dtype=complex)        # 状态转移矩阵
    self.G = np.mat(np.zeros((2*self.size,1), dtype=complex))  # 状态轨迹向量
    # 初始化
    self.GBBSH = []
    self.branch = []
    self.H = np.mat(np.zeros([0, 2*self.size]), dtype=complex)   # 量测矩阵
    self.R_real = np.mat(np.zeros([0, 0]), dtype=complex)   # 真实量测协方差矩阵
    self.R = np.mat(np.zeros([0, 0]), dtype=complex)        # 计算量测协方差矩阵
    self.z_observed = np.mat(np.zeros([0,0]))             # 量测值
    self.measure_who = []   # Indicate whose z: single i -> bus i
                      #                   [i, i]   -> pmu i
                      #                   [i, j]   -> branch i -> j
    self.x_who = list(range(1,self.size+1))   # Indicate whose x: per x has 2 members
                      # Defult the order of Bus
    # 标签
    self.is_distribute = False  # Defult is a centralized system
    self.is_reference_deleted = False

    # 构建系统模型
    self.model_function()

    # 从配置文件提取
    if os.path.exists('cache/IEEE_'+str(self.size)+'_info.json') is True:  # 从缓存提取
      with open('cache/IEEE_'+str(self.size)+'_info.json','r',encoding='utf-8') as f:
        saved_conf = json.load(f, object_hook=as_complex)
        self.x_real = np.mat(saved_conf['x_real'])
        self.bus_type = saved_conf['bus_type']
        GBBSH = saved_conf['GBBSH']
        bus = saved_conf['bus']
        branch = saved_conf['branch']
        self.bus_info_dict = self.set_nodes_info(bus,branch,self.pmu,GBBSH,'single')
    else:
      self.bus_cdf = extract_config.tools(conf_path, bus_rule_path, 0)
      self.branch_cdf = extract_config.tools(conf_path, branch_rule_path, 1)
      x_real = self.bus_cdf.get_items([6, 8]) #第一次获得的真实状态值
      x = np.mat(np.zeros([2*self.size,1]))
      for cnt,(i, j) in enumerate(zip(x_real[0], x_real[1])):
        x[2*cnt, 0] = i
        x[2*cnt+1, 0] = j
      self.x_real = x
      self.bus_type = self.bus_cdf.get_items(5)  # Bus type: 0 -> Load
                                            #           2 -> Generation
                                            #           3 -> Reference
      self.__gen_gbbsh(is_ignore=False)
      bus = self.bus_cdf.get_items(0)
      self.bus_info_dict = self.set_nodes_info(bus, zip(self.branch[0],self.branch[1]), self.pmu, zip(self.GBBSH[0], self.GBBSH[1], self.GBBSH[2]), 'single')
      # 保存当前配置
      conf_to_save = {
        'x_real': self.x_real.tolist(),
        'bus_type': self.bus_type,
        'GBBSH': list(zip(self.GBBSH[0], self.GBBSH[1], self.GBBSH[2])),
        'bus': bus,
        'branch': list(zip(self.branch[0],self.branch[1]))
      }
      with open('cache/IEEE_'+str(self.size)+'_info.json','w',encoding='utf-8') as f:
        f.write(json.dumps(conf_to_save,ensure_ascii=False,cls=ComplexEncoder))

  def model_function(self):
    if self.model_name is 'PowerGrid':
      # 为求偏导准备的符号
      G_ij = sb.symbols("G_ij")
      B_ij = sb.symbols("B_ij")
      BSH_ij = sb.symbols("BSH_ij")
      Vi = sb.symbols("V_i")
      Vj = sb.symbols("V_j")
      Theta_i = sb.symbols("Theta_i")
      Theta_j = sb.symbols("Theta_j")
      # alpha_ij = sb.symbols("alpha_ij") 不知道什么用
      # 边量测计算公式
      P_ij = G_ij*Vi**2 - G_ij*Vi*Vj*sb.cos(Theta_i - Theta_j) - B_ij*Vi*Vj*sb.sin(Theta_i - Theta_j)
      Q_ij = -(B_ij + BSH_ij)*Vi**2 + B_ij*Vi*Vj*sb.cos(Theta_i-Theta_j) - G_ij*Vi*Vj*sb.sin(Theta_i-Theta_j)
      self.h = [P_ij, Q_ij]
      self.value_symbol = [G_ij, B_ij, BSH_ij]
      self.state_i_symbol = [Vi, Theta_i]
      self.state_j_symbol = [Vj, Theta_j]
      self.jacobi_h_ij = self.__jacob(self.h, self.state_i_symbol)
      self.jacobi_h_ji = self.__jacob(self.h, self.state_j_symbol)
      #self.h_egde = [active_power, reactive_power]
    elif self.model_name is 'WSNs':
      X = sb.symbols("X")
      Y = sb.symbols("Y")
      Xi = sb.symbols("X_i")
      Xj = sb.symbols("X_j")
      Yi = sb.symbols("Y_i")
      Yj = sb.symbols("Y_j")
      # 本地测量
      M_ij = sb.sqrt((X-Xi)**2 + (Y-Yi)**2) # 已知节点对未知节点的测量
      # 边量测计算公式
      Z_ij = sb.sqrt((Xj-Xi)**2 + (Yj-Yi)**2) # 未知节点之间的测量
      self.h = [Z_ij]
      self.value_symbol = [X, Y]
      self.state_i_symbol = [Xi, Yi]
      self.state_j_symbol = [Xj, Yj]
      self.jacobi_h_ij = self.__jacob(self.h, self.state_i_symbol)
      self.jacobi_h_ji = self.__jacob(self.h, self.state_j_symbol)

  def __gen_gbbsh(self, is_ignore=False):
    """
    计算建模所需电路参数

    输入
    ---- 
    * 电阻r(resistance),
    * 电抗x(reactance),
    * 分流电导gsh,
    * 分流电纳bsh,
    * is_ignore: 是否忽略接地shunt和传输电阻，若忽略则H是实数矩阵 <False,True>
  
    计算
    ---- 
    * 电导g(conductance),
    * 电纳b(susceptance),
    * 分流导纳ysh(admittance_shunt)
  
    公式
    ---- 
    Note: 阻抗z(impedance)
      z = r + jx              |=> g = r/(r**2 + x**2)
      y = g + jb = z^{-1}     |=> b = x/(r**2 + x**2)
      branch上的ysh = (相连的两个bus上的ysh)/2
  
    返回
    ----
    self.GBBSH = ((g), (b), (ysh));
    """
    branch_num = self.branch_cdf.get_items([0,2])   # extract the bus number of out-in side.
    resistance = self.branch_cdf.get_items(6)  # line resistance
    reactance = self.branch_cdf.get_items(7)   # line reactance
    shunt_conductance = self.bus_cdf.get_items(16)  # Shunt conductance
    shunt_susceptance = self.bus_cdf.get_items(17)  # Shunt susceptance

    self.branch = tuple(branch_num)
    if is_ignore is False:
      conductance = tuple([r/(r**2 + x**2) for r,x in zip(resistance, reactance)])
      susceptance = tuple([-x/(r**2 + x**2) for r,x in zip(resistance, reactance)])
    else:
      conductance = tuple([0 for r,x in zip(resistance, reactance)])
      susceptance = tuple([1 for r,x in zip(resistance, reactance)])
    self.GBBSH.append(conductance)
    self.GBBSH.append(susceptance)

    shunt_admittance_bus =[]
    shunt_admittance_branch = []
    if is_ignore is False:
      for c, s in zip(shunt_conductance, shunt_susceptance):
        shunt_admittance_bus.append( complex(c, s) )
      for o, i in zip(branch_num[0], branch_num[1]):
        shunt_admittance_branch.append((shunt_admittance_bus[o-1] + shunt_admittance_bus[i-1])/2)
    else: # 忽略对地shunt，电导为0
      for o, i in zip(branch_num[0], branch_num[1]):
        shunt_admittance_branch.append(0) 
    self.GBBSH.append(shunt_admittance_branch)

    self.GBBSH = tuple(self.GBBSH)
