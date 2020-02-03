# coding=utf-8
import scipy.io as sio
import numpy as np
import random
import threading
from tqdm import tqdm
import queue
import matplotlib.pyplot as plt
#plt.rcParams['font.sans-serif'] = ['SimHei']  # linux正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
import matplotlib.pylab as pylab
 
DEFAULT_DIFF_LIMIT = 20   
RICHARDSON_MAIN_PERIOD = 300
RICHARDSON_GAMMA_PERIOD = 100

STOCASTIC_MAIN_PERIOD = 500

class Richardson:
  """
  利用 Richardson 方法进行分布式估计
  """
  def __init__(self, cluster_info_dict, neighbors_dict, x_real, x_est_center, conf_dict):
    # 定义
    self.Precondition_distribute = {}
    self.cluster_info_dict = cluster_info_dict
    self.neighbors_dict = neighbors_dict
    self.x_real = x_real
    self.x_est_center = x_est_center
    # 配置
    self.is_async = conf_dict['is_async']
    self.diff_limit = conf_dict['diff_limit']
    self.is_finite_time = conf_dict['is_finite']
    self.main_period = conf_dict['main_period']
    self.gamma_period = conf_dict['gamma_period']
    self.attacked_nodes = conf_dict['attacked_nodes']
    # 计算
    self.nodes_num = len(cluster_info_dict)
    self.state_size = self.x_real.shape[0]
    '''
    class Richardson(DistributedLinearPowerGrid):
      """
      利用 Richardson 方法进行分布式估计
      """
      def __init__(self, size, conf_path, bus_rule_path, branch_rule_path, nodes = [], pmu=[]):
        super().__init__(size=size, conf_path=conf_path, bus_rule_path=bus_rule_path, branch_rule_path=branch_rule_path, nodes=nodes, pmu=pmu)
        self.main_period = RICHARDSON_MAIN_PERIOD  # 主程序循环次数
        self.gamma_period = RICHARDSON_GAMMA_PERIOD # 计算Gamma特征值的循环次数
        self.diff_limit = DEFAULT_DIFF_LIMIT    # 异步算法中每个节点能比其它节点最多快多少次计算
        self.Precondition_distribute = {}
        self.is_finite_time = False
    '''
    
  def algorithm(self, H_distribute, Phi_distribute, R_I_distribute_diag, z_distribute, is_plot=False):
    """
    进行分布式估计的估计器，在每个子节点上运行的算法
  
    输入
    ---- 
      is_plot 是否画图: <False,True>
  
    返回
    ----
      状态估计结果
    """
    self.H_distribute = H_distribute
    self.Phi_distribute = Phi_distribute
    self.R_I_distribute_diag = R_I_distribute_diag
    self.z_distribute = z_distribute

    axis1 = np.mat(range(self.main_period))
    sample = np.mat(np.empty([self.state_size, self.main_period]))
    self.x_est_distribute_lists = []
    for i in range(self.state_size):
      sample[i,:] = axis1
    self.record = []

    self.alpha = []
    self.Precondition = []
    self.est_x = []
    self.pseudo_x = []
    self.task_lock = queue.Queue(maxsize=self.nodes_num)
    eig_thread_nodes = []
    thread_nodes = []

    for i in range(self.nodes_num):
      self.alpha.append( queue.Queue() )
      self.est_x.append( queue.Queue() )
      self.pseudo_x.append( queue.Queue() )
      self.Precondition.append( queue.Queue() )
      lock_con = threading.Condition()

      self.record.append(np.mat(np.empty([self.cluster_info_dict[i]['col_amount'],self.main_period]),dtype=complex))

      self.x_est_distribute_lists.append(np.mat(np.empty([self.cluster_info_dict[i]['col_amount'],1])))
    ## 分布式计算最大最小特征值
    if self.is_finite_time is False:
      self.sigma = np.empty(self.nodes_num)
      self.gamma_max_record = []
      self.gamma_min_record = []
      self.sigma_first = []
      self.sigma_second = []
      for i in range(self.nodes_num):
        self.sigma_first.append( queue.Queue() )
        self.sigma_second.append( queue.Queue() )
        self.gamma_max_record.append([])
        self.gamma_min_record.append([])

      for i in range(self.nodes_num):
        eig_thread_nodes.append(threading.Thread(target=self.__maxmin_eigenvalue, args=(i, lock_con, False)))
      for n in eig_thread_nodes:
        n.setDaemon(True)
        n.start()
      # wait gameover
      for n in eig_thread_nodes:
        n.join()
    ## 分布式计算状态
    if self.is_finite_time is True:  # 有限步算法
      for i in range(self.nodes_num):
        thread_nodes.append(threading.Thread(target=self.__finite_time_estimator, args=(i, lock_con)))
    elif self.is_async is False:           # 同步算法
      for i in range(self.nodes_num):
        thread_nodes.append(threading.Thread(target=self.__sync_estimator, args=(i, lock_con)))
    else:                            # 异步算法
      for i in range(self.nodes_num):
        thread_nodes.append(threading.Thread(target=self.__async_estimator, args=(i, lock_con)))
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
      if self.is_finite_time is False:
        plt.figure('Gamma最大最小特征值')
        for i in range(self.nodes_num):
          plt.plot(self.gamma_max_record[i] ,'b--', linewidth = 0.5)
          plt.plot(self.gamma_min_record[i] ,'r', linewidth = 0.5)
        plt.show()
      # 分布式估计过程
      plt.figure('分布式估计过程')
      # 电压
      plt.subplot(211)
      plt.title('电压估计')
      for i in self.attacked_nodes: # 只画受攻击节点的状态估计图像
        for j in range(0, self.cluster_info_dict[i]['col_amount'], 2):
          plt.plot(self.record[i][j,:].T, 'b', linewidth = 0.5)
      plt.legend([u'电压'], loc='upper right')
      plt.xlabel("迭代次数")
      plt.ylabel("幅值")
      # 电压相角
      plt.subplot(212)
      plt.title('电压相角估计')
      for i in self.attacked_nodes:
        for j in range(1, self.cluster_info_dict[i]['col_amount'], 2):
          plt.plot(self.record[i][j,:].T, 'r', linewidth = 0.5)
      plt.legend([u'电压相角'], loc='upper right')
      plt.xlabel("迭代次数")
      plt.ylabel("幅值")
      plt.show()
    #print(self.x_est_distribute_lists)
    ''' 保存 '''
    '''
    mat_save_dict = {}
    for i in range(self.nodes_num):
      for j in range(self.nodes_num):
        mat_save_dict['H'+str(i)+str(j)] = self.H_distribute[i][j]
        mat_save_dict['Phi'+str(i)+str(j)] = self.Phi_distribute[i][j]
        mat_save_dict['R'+str(i)] = self.R_I_distribute_diag[i]
      print(self.sigma[i])
    sio.savemat('./save/sys.mat', mat_save_dict)
    '''
    return self.x_est_distribute_lists,self.x_est_distribute

  def __maxmin_eigenvalue(self, num, lock_con, is_finite_time=False):
    """
    分布式计算最大最小特征值
    
    输入
    ---- 
    * num: 该节点的节点号(从0开始计)
    * lock_con: 锁（保证通信的同步）
    * is_finite_time: 是否使用有限步算法（需在非回环拓扑情况下）
  
    返回
    ----
    * eig_max,eig_min
    """
    # neighbors whose measurement include mine
    neighbor_in_him = self.neighbors_dict[num]['him']
    # neighbors who is in my measurement
    neighbor_in_me = self.neighbors_dict[num]['me']
    # neighbors
    neighbors = self.neighbors_dict[num]['neighbors']

    # Send my Phi(k)_himhim to neighbor_me && Record Phi(k)_himhim
    for i in neighbor_in_me:
      self.Precondition[i].put([num, self.H_distribute[num][i].H*self.R_I_distribute_diag[num]*self.H_distribute[num][i]])

    # Accumulate my Phi_meme and calc Precondition matrix
    my_Precondition = np.mat(np.zeros([self.cluster_info_dict[num]['col_amount'], self.cluster_info_dict[num]['col_amount']]), dtype=complex)
    for i in neighbor_in_him:
      comein = self.Precondition[num].get()
      if comein[0] not in neighbor_in_him:
        raise Exception('Wrong come in '+str(comein[0]))
      my_Precondition += comein[1]
    my_Precondition = my_Precondition.I
    my_Precondition_sqrt = np.linalg.cholesky(my_Precondition)
    self.Precondition_distribute.update({num:my_Precondition_sqrt})

    ## 计算Gamma的最大特征值 ##
    # initial b_0 with random ||b_0|| = 1 
    b_bar = np.mat(np.random.rand(self.cluster_info_dict[num]['col_amount'], 1), dtype=complex) # 随机初始值
    #b_bar = np.mat(np.ones((self.cluster_info_dict[num]['col_amount'], 1)))*0.7
    b_bar = b_bar / np.linalg.norm(b_bar,ord=2)	# normallize
    yita = 1
    # It seems that i in neighbor_in_me is also ok
    v_ij = {}
    for i in neighbors:
      v_ij.update({str(i):{}})
      for j in neighbors:
        v_ij[str(i)].update({str(j):1})
    # 显示进度
    if num == 0:
      self.pbar=tqdm(total=self.gamma_period)
    # 开始迭代
    for t in range(self.gamma_period):
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
        b_hat.update({str(i):np.mat(np.zeros([self.cluster_info_dict[i]['col_amount'], 1]), dtype=complex)})
        for j in neighbor_in_me:
          b_hat[str(i)] += v_ij[str(i)][str(j)] * self.H_distribute[num][i].H*self.R_I_distribute_diag[num]*self.H_distribute[num][j] * comein_dict[str(j)][0]
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
      b_tilde = np.mat(np.zeros([ self.cluster_info_dict[num]['col_amount'], 1 ]), dtype=complex)
      for k in neighbor_in_him:
        b_tilde += comein_dict[str(k)][0]
      b_tilde = my_Precondition_sqrt.H*b_tilde
      # yita
      yita_candidate = [np.linalg.norm(b_tilde, ord=2)]
      for k in neighbor_in_him:
        yita_candidate.append(comein_dict[str(k)][1])
      yita = 1 / max( yita_candidate )
      # b_bar
      b_bar = yita * b_tilde

      # 记录
      self.gamma_max_record[num].append(1/yita)

      lock_con.acquire()
      if self.task_lock.qsize() != self.nodes_num-1:
        self.task_lock.put(num)
        lock_con.wait()
      else:
        self.pbar.update(1)
        self.task_lock.queue.clear()
        lock_con.notify_all()
      lock_con.release()
    # 结束进度条
    if num == 0:
      self.pbar.close()
    # Maximum eigenvalue of Gamma
    Gamma_max = 1 / yita
    # print('%s-Gamma最大特征值: %.3f' % (threading.current_thread().name, Gamma_max))
    
    ## 计算Gamma的最小特征值 ##
    # 初始化
    # initial b_0 with random ||b_0|| = 1 
    b_bar = np.mat(np.random.rand(self.cluster_info_dict[num]['col_amount'], 1), dtype=complex)
    b_bar = b_bar / np.linalg.norm(b_bar,ord=2)	# normallize
    #
    yita = 1
    # It seems that i in neighbor_in_me is also ok
    v_ij = {}
    for i in neighbors:
      v_ij.update({str(i):{}})
      for j in neighbors:
        v_ij[str(i)].update({str(j):1})
    # 显示进度
    if num == 0:
      self.ppbar=tqdm(total=self.gamma_period)
    # 开始迭代
    for t in range(self.gamma_period):
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
        b_hat.update({str(i):np.mat(np.zeros([self.cluster_info_dict[i]['col_amount'], 1]), dtype=complex)})
        for j in neighbor_in_me:
          b_hat[str(i)] += v_ij[str(i)][str(j)] * self.H_distribute[num][i].H*self.R_I_distribute_diag[num]*self.H_distribute[num][j] * comein_dict[str(j)][0]
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
      b_tilde = np.mat(np.zeros([ self.cluster_info_dict[num]['col_amount'], 1 ]), dtype=complex)
      for k in neighbor_in_him:
        b_tilde += comein_dict[str(k)][0]
      b_tilde = my_Precondition_sqrt.H*b_tilde
      # Calc b_tilde_min
      b_tilde_min = (Gamma_max) * b_bar - b_tilde   # (Gamma_max) * b_bar - b_tilde
      # yita
      yita_candidate = [np.linalg.norm(b_tilde_min, ord=2)]
      for k in neighbor_in_him:
        yita_candidate.append(comein_dict[str(k)][1])
      yita = 1 / max( yita_candidate )
      # b_bar
      b_bar = yita * b_tilde_min
      # 记录
      Gamma_min = (Gamma_max) - (1 / yita) # (Gamma_max) - 1 / yita
      self.gamma_min_record[num].append(Gamma_min)

      lock_con.acquire()
      if self.task_lock.qsize() != self.nodes_num-1:
        self.task_lock.put(num)
        lock_con.wait()
      else:
        self.ppbar.update(1)
        self.task_lock.queue.clear()
        lock_con.notify_all()
      lock_con.release()
    # 结束进度条
    if num == 0:
      self.ppbar.close()
		# Minimum eigenvalue of Gamma
    Gamma_min = (Gamma_max) - 1 / yita  # (Gamma_max) - 1 / yita
    #print('%s-Gamma最小特征值: %.3f' % (threading.current_thread().name, Gamma_min))
    ## 计算 sigma
    sigma = 2 / ( Gamma_max + Gamma_min )
    # print(threading.current_thread().name + 'sigma: ' + str(sigma))
    self.sigma[num] = sigma

  def __sync_estimator(self, num, lock_con):
    """
    各分布式节点的同步估计器
  
    输入
    ---- 
    * num 该节点的节点号(从0开始计)
    * lock_con 锁（保证通信的同步）
  
    返回
    ----
    NULL
    """
    # neighbors whose measurement include mine
    neighbor_in_him = self.neighbors_dict[num]['him']
    # neighbors who is in my measurement
    neighbor_in_me = self.neighbors_dict[num]['me']
    # neighbors
    neighbors = self.neighbors_dict[num]['neighbors']
    '''<<初始化>>'''
    # Send my alpha(k)_other to neighbor_me
    for i in neighbor_in_me:
      self.alpha[i].put([num, self.H_distribute[num][i].H*self.R_I_distribute_diag[num]*self.z_distribute[num]])
    # Accumulate alpha_me
    alpha_res = np.zeros((self.cluster_info_dict[num]['col_amount'],1),dtype=complex)
    for i in neighbor_in_him:
      comein = self.alpha[num].get()
      if comein[0] not in neighbor_in_him:
        raise Exception('Wrong come in '+str(comein[0]))
      alpha_res += comein[1]
    #print( str(alpha_res) + ' belongs to node '+ str(num) + '.' )
    # Send my Phi(k)_himhim to neighbor_me && Record Phi(k)_himhim
    for i in neighbor_in_me:
      self.Precondition[i].put([num, self.H_distribute[num][i].H*self.R_I_distribute_diag[num]*self.H_distribute[num][i]])
    # Accumulate my Phi_meme and calc Precondition matrix
    my_Precondition = np.mat(np.zeros([self.cluster_info_dict[num]['col_amount'], self.cluster_info_dict[num]['col_amount']]), dtype=complex)
    for i in neighbor_in_him:
      comein = self.Precondition[num].get()
      if comein[0] not in neighbor_in_him:
        raise Exception('Wrong come in '+str(comein[0]))
      my_Precondition += comein[1]
    my_Precondition = my_Precondition.I
    my_Precondition_sqrt = np.linalg.cholesky(my_Precondition)
    self.Precondition_distribute.update({num:my_Precondition_sqrt})
    # 初始化x_0，全部置0
    x_est = np.mat(np.zeros([self.cluster_info_dict[num]['col_amount'], 1]), dtype=complex)

    '''<<开始计算>>'''
    # 显示进度条
    if num == 0:
      self.pbar=tqdm(total=self.main_period)
    # 开始迭代
    for t in range(self.main_period):
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
        pseudo_x = np.mat(np.zeros([self.cluster_info_dict[i]['col_amount'],1]), dtype=complex)
        # Accumulate Phi_ij * x_j
        for j, x_j in recv_x_est:
          pseudo_x += self.H_distribute[num][i].H*self.R_I_distribute_diag[num]*self.H_distribute[num][j]*x_j
        # Send
        self.pseudo_x[i].put([num, pseudo_x])
      # Accumulate my pseudo_x
      pseudo_x = np.mat(np.zeros([self.cluster_info_dict[num]['col_amount'],1]), dtype=complex)
      for k in neighbor_in_him:
        comein = self.pseudo_x[num].get()
        if comein[0] not in neighbor_in_him:
          raise Exception('Wrong come in '+str(comein[0]))
        pseudo_x = pseudo_x + comein[1]
      # Calc estimate state
      x_est = x_est - self.sigma[num] * my_Precondition * (pseudo_x - alpha_res)

      self.record[num][:,t] = x_est#/self.x_real_list[num] # 归一化 

      lock_con.acquire()
      if self.task_lock.qsize() != self.nodes_num-1:
        self.task_lock.put(num)
        lock_con.wait()
      else:
        self.pbar.update(1)
        self.task_lock.queue.clear()
        lock_con.notify_all()
      lock_con.release()
    if num == 0:
      self.pbar.close()
    self.x_est_distribute_lists[num] = x_est
    # print(threading.current_thread().name, str(x_est.T))

  def __async_estimator(self, num, lock_con):
    """
    各分布式节点的异步估计器
  
    输入
    ---- 
    * num 该节点的节点号(从0开始计)
    * lock_con 锁, 计算Gamma最大最小特征值时依然使用同步算法
  
    返回
    ----
    NULL
    """
    import threading
    # neighbors whose measurement include mine
    neighbor_in_him = self.neighbors_dict[num]['him']
    # neighbors who is in my measurement
    neighbor_in_me = self.neighbors_dict[num]['me']
    # neighbors
    neighbors = self.neighbors_dict[num]['neighbors']
    
    ## 计算初始化参数
    # Send my alpha(k)_other to neighbor_me
    for i in neighbor_in_me:
      self.alpha[i].put([num, self.H_distribute[num][i].H*self.R_I_distribute_diag[num]*self.z_distribute[num]])

    # Accumulate alpha_me
    alpha_res = np.zeros((self.cluster_info_dict[num]['col_amount'],1),dtype=complex)
    for i in neighbor_in_him:
      comein = self.alpha[num].get()
      if comein[0] not in neighbor_in_him:
        raise Exception('Wrong come in '+str(comein[0]))
      alpha_res += comein[1]
    #print( str(alpha_res) + ' belongs to node '+ str(num) + '.' )

    # Send my Phi(k)_himhim to neighbor_me && Record Phi(k)_himhim
    for i in neighbor_in_me:
      self.Precondition[i].put([num, self.H_distribute[num][i].H*self.R_I_distribute_diag[num]*self.H_distribute[num][i]])

    # Accumulate my Phi_meme and calc Precondition matrix
    my_Precondition = np.mat(np.zeros([self.cluster_info_dict[num]['col_amount'], self.cluster_info_dict[num]['col_amount']]), dtype=complex)
    for i in neighbor_in_him:
      comein = self.Precondition[num].get()
      if comein[0] not in neighbor_in_him:
        raise Exception('Wrong come in '+str(comein[0]))
      my_Precondition += comein[1]
    my_Precondition = my_Precondition.I
    my_Precondition_sqrt = np.linalg.cholesky(my_Precondition)
    self.Precondition_distribute.update({num:my_Precondition_sqrt})

    ## 计算状态 ##
    # 初始化参数
    recv_x_est = {}
    recv_pseudo_x = {}
    recv_est_timestap = {}
    recv_pseudo_timestap = {}
    for i in neighbor_in_me:
      recv_x_est[i] = np.zeros((self.cluster_info_dict[i]['col_amount'],1), dtype=complex)
      recv_est_timestap[i] = 0
    for i in neighbor_in_him:
      recv_pseudo_x[i] = np.zeros((self.cluster_info_dict[num]['col_amount'],1), dtype=complex)
      recv_pseudo_timestap[i] = 0
    # 记录这一轮的值
    pseudo_x = {}
    for i in neighbor_in_me:
      pseudo_x[i] = np.mat(np.zeros([self.cluster_info_dict[i]['col_amount'],1]), dtype=complex)
    x_est = np.mat(np.zeros([self.cluster_info_dict[num]['col_amount'], 1]), dtype=complex)
    # 显示进度条
    ppbar=tqdm(total=self.main_period)
    # 开始迭代
    for t in range(self.main_period):
      # Send my estmate state x to neighbors
      for i in neighbors:
        self.est_x[i].put([num, x_est, t])
      # Receive est_x
      wait = True
      while wait:
        if max(recv_est_timestap.values())-t >= self.diff_limit: # 如果有节点比自己快非常多
          if t-min(recv_est_timestap.values()) > 0: # 如果有节点比自己慢，就等待
            pass
          else: # 自己最慢，就继续
            if self.est_x[num].empty():
              wait = False
              break
        comein = self.est_x[num].get()
        if comein[0] in neighbor_in_me: # some are not in me
          if recv_est_timestap[comein[0]] <= comein[2]:
            recv_est_timestap[comein[0]] = comein[2]
            recv_x_est[comein[0]] = comein[1]
          if self.est_x[num].empty():
            if t-min(recv_est_timestap.values()) >= self.diff_limit: # 如果有节点比自己慢很多，就等待
              for i in neighbor_in_me:
                self.pseudo_x[i].put([num, pseudo_x[i], t])
            else: 
              wait = False

      # Send pseudo_x to neighbor_me
      for i in neighbor_in_me:
        pseudo_x[i] = np.mat(np.zeros([self.cluster_info_dict[i]['col_amount'],1]), dtype=complex)
        # Accumulate Phi_ij * x_j
        for j, x_j in recv_x_est.items():
          pseudo_x[i] += self.H_distribute[num][i].H*self.R_I_distribute_diag[num]*self.H_distribute[num][j]*x_j
        # Send
        self.pseudo_x[i].put([num, pseudo_x[i], t])
      # Receive pseudo_x
      wait = True
      while wait:
        if max(recv_pseudo_timestap.values())-t >= self.diff_limit: # 如果有节点比自己快非常多，就别等包了
          if t-min(recv_pseudo_timestap.values()) > 0: # 如果有节点比自己慢，就等待
            pass
          else: # 自己最慢，就继续
            if self.pseudo_x[num].empty():
              wait = False
              break
        comein = self.pseudo_x[num].get()
        if recv_pseudo_timestap[comein[0]] <= comein[2]:
          recv_pseudo_timestap[comein[0]] = comein[2]
          recv_pseudo_x[comein[0]] = comein[1]
        if self.pseudo_x[num].empty():  # 如果空了
          if t-min(recv_pseudo_timestap.values()) >= self.diff_limit: # 如果有节点比自己慢很多，就等待
            for i in neighbors:
              self.est_x[i].put([num, x_est, t])
          else:
            wait = False

      # Accumulate my pseudo_x
      my_pseudo_x = np.mat(np.zeros([self.cluster_info_dict[num]['col_amount'],1]), dtype=complex)
      for k in neighbor_in_him:
        my_pseudo_x += recv_pseudo_x[k]
      # Calc estimate state
      x_est = x_est - self.sigma[num] * my_Precondition * (my_pseudo_x - alpha_res)
      if np.max(x_est) > 5000:
        raise ValueError("发散")
      self.record[num][:,t] = x_est#/self.x_real_list[num] # 归一化 

      ppbar.update(1)
    ppbar.close()
    self.x_est_distribute_lists[num] = x_est
    # print(threading.current_thread().name, str(x_est.T))

  def __finite_time_estimator(self, num, lock_con):
    """
    各分布式节点在有限时间内达到次优估计结果
      该算法只可用于无回环拓扑网络
  
    输入
    ---- 
    * num 该节点的节点号(从0开始计)
    * lock_con 锁（保证通信的同步）
  
    返回
    ----
    NULL
    """
    '''<<初始化变量>>'''
    # neighbors whose measurement include mine
    neighbor_in_him = self.neighbors_dict[num]['him']
    # neighbors who is in my measurement
    neighbor_in_me = self.neighbors_dict[num]['me']
    # neighbors
    neighbors = self.neighbors_dict[num]['nn']
    # Send my alpha(k)_other to neighbor_me
    for i in neighbor_in_me:
      self.alpha[i].put([num, self.H_distribute[num][i].H*self.R_I_distribute_diag[num]*self.z_distribute[num]])
    # Accumulate alpha_me
    alpha_res = np.mat( np.zeros((self.cluster_info_dict[num]['col_amount'],1),dtype=complex) )
    for i in neighbor_in_him:
      comein = self.alpha[num].get()
      if comein[0] not in neighbor_in_him:
        raise Exception('Wrong come in '+str(comein[0]))
      alpha_res = alpha_res + comein[1]
    '''
    # Send my Phi(k)_himhim to neighbor_me && Record Phi(k)_himhim
    Phi_distribute = {}
    for i in neighbors:
      self.Precondition[i].put([num, self.H_distribute[num][i].H*self.R_I_distribute_diag[num]*self.H_distribute[num][i]])
    # Accumulate my Phi_meme and calc Precondition matrix
    my_Precondition = np.mat(np.zeros([self.cluster_info_dict[num]['col_amount'], self.cluster_info_dict[num]['col_amount']]), dtype=complex)
    for i in neighbor_in_him:
      comein = self.Precondition[num].get()
      if comein[0] not in neighbor_in_him:
        raise Exception('Wrong come in '+str(comein[0]))
      my_Precondition += comein[1]
    my_Precondition = my_Precondition.I
    my_Precondition_sqrt = np.linalg.cholesky(my_Precondition)
    self.Precondition_distribute.update({num:my_Precondition_sqrt})
    '''
    x_neighbors = {}
    Sigma_neighbors = {}
    my_Gamma = self.Phi_distribute[num][num].I
    my_x = my_Gamma * alpha_res
    for i in neighbors:
      x_neighbors[i] = my_x
      Sigma_neighbors[i] = my_Gamma
    # 显示进度条
    if num == 0:
      self.pbar=tqdm(total=self.main_period)
    '''<<开始计算>>'''
    for t in range(self.main_period):
      # 计算并发送
      for j in neighbors:
        gamma_neighbors = self.Phi_distribute[j][num] * x_neighbors[j]
        Gamma_neighbors = self.Phi_distribute[j][num] * Sigma_neighbors[j] * self.Phi_distribute[num][j]
        self.est_x[j].put([num, gamma_neighbors, Gamma_neighbors])
      # 接收并计算
      gamma_res = {}
      Gamma_res = {}
      for j in neighbors:
        gamma_res[j] = np.mat( np.zeros([self.cluster_info_dict[num]['col_amount'], 1],dtype=complex) )
        Gamma_res[j] = np.mat( np.zeros([self.cluster_info_dict[num]['col_amount'], self.cluster_info_dict[num]['col_amount']],dtype=complex) )
      gamma_res[num] = np.mat( np.zeros([self.cluster_info_dict[num]['col_amount'], 1],dtype=complex) )
      Gamma_res[num] = np.mat(np.zeros([self.cluster_info_dict[num]['col_amount'], self.cluster_info_dict[num]['col_amount']],dtype=complex))
      for i in neighbors:
        comein = self.est_x[num].get()
        for j in neighbors:
          if comein[0] != j:
            gamma_res[j] += comein[1]
            Gamma_res[j] += comein[2]
        gamma_res[num] += comein[1]
        Gamma_res[num] += comein[2]
      my_Gamma = (self.Phi_distribute[num][num] - Gamma_res[num]).I
      my_x = my_Gamma * (alpha_res - gamma_res[num])
      for j in neighbors:
        Sigma_neighbors[j] = (self.Phi_distribute[num][num] - Gamma_res[j]).I
        x_neighbors[j] = Sigma_neighbors[j] * (alpha_res - gamma_res[j])

      self.record[num][:,t] = my_x#/self.x_real_list[num] # 归一化 

      lock_con.acquire()
      if self.task_lock.qsize() != self.nodes_num-1:
        self.task_lock.put(num)
        lock_con.wait()
      else:
        self.pbar.update(1)
        self.task_lock.queue.clear()
        lock_con.notify_all()
      lock_con.release()
    if num == 0:
      self.pbar.close()
    self.x_est_distribute_lists[num] = np.mat(my_x,dtype=float)
    # print(threading.current_thread().name, str(x_est.T))

#************************************************************
 #* 类 -- 
 #*       利用 随机近似 方法进行分布式估计
 #*
#************************************************************
class Stocastic:
  """
  利用 Stocastic 方法进行分布式估计
  """
  def __init__(self, cluster_info_dict, neighbors_dict, x_real, x_est_center, conf_dict):
    # 定义
    self.Precondition_distribute = {}
    self.cluster_info_dict = cluster_info_dict
    self.neighbors_dict = neighbors_dict
    self.x_real = x_real
    self.x_est_center = x_est_center
    # 配置
    self.is_async = conf_dict['is_async']
    self.diff_limit = conf_dict['diff_limit']
    self.is_finite_time = conf_dict['is_finite']
    self.main_period = conf_dict['main_period']
    self.gamma_period = conf_dict['gamma_period']
    # 计算
    self.nodes_num = len(cluster_info_dict)
    self.state_size = self.x_real.shape[0]

  def algorithm(self, H_distribute, Phi_distribute, R_I_distribute_diag, z_distribute, is_plot=False):
    """
    进行分布式估计的估计器，在每个子节点上运行的算法
 
    输入
    ---- 
      is_plot 是否画图: <False,True>
  
    返回
    ----
      None
    """
    self.H_distribute = H_distribute
    self.Phi_distribute = Phi_distribute
    self.R_I_distribute_diag = R_I_distribute_diag
    self.z_distribute = z_distribute

    self.record = []
    self.x_est_distribute_lists = []
    self.est_x = []
    self.task_lock = queue.Queue(maxsize=self.nodes_num)
    thread_nodes = []

    for i in range(self.nodes_num):
      self.est_x.append( queue.Queue() )
      lock_con = threading.Condition()

      self.record.append(np.mat(np.empty([self.state_size,self.main_period])))
      self.x_est_distribute_lists.append(np.mat(np.empty([self.state_size,1])))

    ## 分布式计算状态
    if self.is_async is False:           # 同步算法
      for i in range(self.nodes_num):
        thread_nodes.append(threading.Thread(target=self.__sync_estimator, args=(i, lock_con)))
    else:                            # 异步算法
      for i in range(self.nodes_num):
        pass

    for n in thread_nodes:
      n.setDaemon(True)
      n.start()
    # wait gameover
    for n in thread_nodes:
      n.join()
    
    self.x_est_distribute = self.x_est_distribute_lists[0] # ....
    # 画出估计结果
    if is_plot is True:
      plt.figure('分布式估计（电压）')
      for j in range(0, self.state_size, 2):
        plt.plot(self.record[1][j,:].T, 'b', linewidth = 0.5) # 仅画出节点0的状态估计误差曲线
      plt.legend([u'电压'], loc='upper right')
      plt.xlabel("迭代次数")
      plt.ylabel("幅值")
      plt.show()

      plt.figure('相角误差的误差曲线')
      for j in range(1, self.state_size-1, 2):
        plt.plot(self.record[1][j,:].T - self.record[1][self.state_size-1,:].T - (self.x_real[j]-self.x_real[self.state_size-1]), 'r', linewidth = 0.5) # 仅画出节点0的状态估计误差曲线
      plt.legend([u'电压相角'], loc='upper right')
      plt.xlabel("迭代次数")
      plt.ylabel("幅值")
      plt.show()

      ### 估计状态误差(\bar{x}-x)
      plt.figure('状态估计误差')
      tmp_x = range(self.state_size)
      plt.plot(tmp_x, self.record[0][:,-1] - self.x_real, 'b.') # 点图
      #plt.bar(np.arange(len(self.x_real_list[i]))+1, (self.record[i][:,-1] - self.x_real_list[i]).T, lw=1)  # 条形图
      plt.xlabel("状态")
      plt.ylabel("误差")
      plt.show()

    return self.x_est_distribute_lists,self.x_est_distribute

  def __sync_estimator(self, num, lock_con):
    """
    各分布式节点的同步估计器
  
    输入
    ---- 
      num 该节点的节点号(从0开始计)
      lock_con 锁（保证通信的同步）
  
    返回
    ----
      None
    """
    # neighbors
    neighbors = self.neighbors_dict[num]['neighbors']
    # 初始化x_0，全部置0
    x_est = np.mat(np.zeros((self.state_size,1)), dtype=complex)
    ## 计算状态 ##
    # 计算最佳步长a
    a = 0.0000005
    b = 1000
    r = np.mat(np.linalg.cholesky(self.R_I_distribute_diag[num]))
    H = np.column_stack(self.H_distribute[num])
    H_ = r * H
    z_ = r * self.z_distribute[num]
    # print(H_.T * z_)
    # 显示进度条
    if num == 0:
      self.pbar=tqdm(total=self.main_period)
    # 开始迭代
    for t in range(self.main_period):
      # Send my estmate state x to neighbors
      for i in neighbors:
        self.est_x[i].put([num, x_est])
      # Receive est_x
      recv_x_est = []
      for i in neighbors:
        comein = self.est_x[num].get()
        if comein[0] in neighbors: # some are not in?
          recv_x_est.append(comein[1])
      self.est_x[num].queue.clear()
      tmp_res = np.mat(np.zeros([self.state_size, 1]), dtype=complex)
      for i in recv_x_est:
        tmp_res = tmp_res + ( x_est - i )
      x_est = x_est - a * (b * tmp_res - H_.H * (z_ - H_ * x_est) )

      self.record[num][:,t] = x_est

      lock_con.acquire()
      if self.task_lock.qsize() != self.nodes_num-1:
        self.task_lock.put(num)
        lock_con.wait()
      else:
        self.pbar.update(1)
        self.task_lock.queue.clear()
        lock_con.notify_all()
      lock_con.release()
    if num == 0:
      self.pbar.close()
    self.x_est_distribute_lists[num] = x_est
    # print(threading.current_thread().name, str(x_est.T))


#************************************************************
 #* 类 -- 
 #*       利用 行投影 方法进行分布式估计
 #*
#************************************************************
class RowProjection:
  pass