# coding=utf-8
from distributed_linear_powergrid import DistributedLinearPowerGrid

import numpy as np
import random
import threading
from tqdm import tqdm
import queue
import matplotlib.pyplot as plt
#plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
import matplotlib.pylab as pylab

GAMMA_MAX_EIGEN_PERIOD = 100       # 计算Gamma特征值的循环次数
MAIN_LOOP_PERIOD = 500    # 主程序循环次数
DEFAULT_DIFF_LIMIT = 40   # 每个节点能比其它节点最多快多少周期

#************************************************************
 #* 类 -- 
 #*       利用 Richardson 方法进行分布式估计
 #*
#************************************************************
class Richardson(DistributedLinearPowerGrid):
  def __init__(self, size):
    super().__init__(size=size)
    pass

  #############################################################
  # 函数 -- 
  #       estimator(): 进行分布式估计的估计器，在每个子节点上运行的算法
  # 输入 -- 
  #       main_period: 迭代周期
  #       gamma_period: 迭代周期
  #       is_async: 是否异步: <False,True>
  #       is_plot 是否画图: <True,False>
  # 返回 --
  #       NULL
  #############################################################
  def estimator(self, main_period=MAIN_LOOP_PERIOD, gamma_period=GAMMA_MAX_EIGEN_PERIOD, is_async=False, is_plot=True):
    self.x_est_distribute_lists = []
    axis1 = np.mat(range(main_period))
    sample = np.mat(np.empty([self.state_size, main_period]))
    for i in range(self.state_size):
      sample[i,:] = axis1
    self.record = []
    self.gamma_max_record = []
    self.gamma_min_record = []

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
      lock_con = threading.Condition()

      self.record.append(np.mat(np.empty([self.node_col_amount[i],main_period])))
      self.gamma_max_record.append([])
      self.gamma_min_record.append([])

      self.x_est_distribute_lists.append(np.mat(np.empty([self.node_col_amount[i],1])))

    if is_async is False:  # 同步算法
      for i in range(self.nodes_num):
        thread_nodes.append(threading.Thread(target=self.__sync_estimator, args=(i, lock_con, main_period, gamma_period)))
    else:             # 异步算法
      for i in range(self.nodes_num):
        thread_nodes.append(threading.Thread(target=self.__async_estimator, args=(i, lock_con, main_period, gamma_period)))

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
      plt.figure('Gamma最大最小特征值')
      for i in range(self.nodes_num):
        plt.plot(self.gamma_max_record[i] ,'b--', linewidth = 0.5)
        plt.plot(self.gamma_min_record[i] ,'r', linewidth = 0.5)
      plt.show()

      plt.figure('分布式估计（电压）')
      #plt.title(u'状态估计值（电压）')
      #plt.plot(sample[0:self.node_col_amount[0],:], self.record[0], 'b.')
      for i in range(self.nodes_num):
        for j in range(0, self.node_col_amount[i], 2):
          plt.plot(self.record[i][j,:].T, 'b', linewidth = 0.5)
      plt.legend([u'电压'], loc='upper right')
      plt.xlabel("迭代次数")
      plt.ylabel("幅值")
      plt.show()

      plt.figure('分布式估计（相角）')
      #plt.title(u'状态估计值（电压相角）')
      for i in range(self.nodes_num):
        for j in range(1, self.node_col_amount[i], 2):
          plt.plot(self.record[i][j,:].T, 'r', linewidth = 0.5)
      plt.legend([u'电压相角'], loc='upper right')
      plt.xlabel("迭代次数")
      plt.ylabel("幅值")
      plt.show()

      ### 估计状态误差(\bar{x}-x)
      tmp_cnt=0
      plt.figure('状态估计误差')
      for i in range(self.nodes_num):
        tmp_x = range(tmp_cnt,tmp_cnt+self.node_col_amount[i])
        plt.plot(tmp_x, self.record[i][:,-1] - self.x_real_list[i], 'b.') # 点图
        #plt.bar(np.arange(len(self.x_real_list[i]))+1, (self.record[i][:,-1] - self.x_real_list[i]).T, lw=1)  # 条形图
        tmp_cnt += self.node_col_amount[i]
      #plt.legend([u'状态估计误差'], loc='upper right')
      plt.xlabel("状态")
      plt.ylabel("误差")
      plt.show()

  #############################################################
  # 函数 -- 
  #       __sync_estimator(): 各分布式节点的同步估计器
  # 输入 -- 
  #       num 该节点的节点号(从0开始计)
  #       lock_con 锁（保证通信的同步）
  #       main_period: 迭代周期
  #       gamma_period: 迭代周期
  # 返回 --
  #       NULL
  #############################################################
  def __sync_estimator(self, num, lock_con, main_period, gamma_period):
    # neighbors whose measurement include mine
    neighbor_in_him = self.get_neighbor(num, -1)
    # neighbors who is in my measurement
    neighbor_in_me = self.get_neighbor(num, 1)
    # neighbors
    neighbors = self.get_neighbor(num, 2)
    
    # Send my alpha(k)_other to neighbor_me
    for i in neighbor_in_me:
      self.alpha[i].put([num, self.H_distribute[num][i].H*self.R_I_distribute_diag[num]*self.z_distribute[num]])

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
      self.Precondition[i].put([num, self.H_distribute[num][i].H*self.R_I_distribute_diag[num]*self.H_distribute[num][i]])

    # Accumulate my Phi_meme and calc Precondition matrix
    my_Precondition = np.mat(np.zeros([self.node_col_amount[num], self.node_col_amount[num]]), dtype=complex)
    for i in neighbor_in_him:
      comein = self.Precondition[num].get()
      if comein[0] not in neighbor_in_him:
        raise Exception('Wrong come in '+str(comein[0]))
      my_Precondition += comein[1]
    my_Precondition = my_Precondition.I
    my_Precondition_sqrt = np.linalg.cholesky(my_Precondition)
    self.Precondition_distribute.update({num:my_Precondition_sqrt})

    # 初始化x_0，全部置0
    x_est = np.mat(np.zeros([self.node_col_amount[num], 1]), dtype=complex)

    ## 计算Gamma的最大特征值 ##
    # initial b_0 with random ||b_0|| = 1 
    b_bar = np.mat(np.random.rand(self.node_col_amount[num], 1), dtype=complex)
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
      self.pbar=tqdm(total=gamma_period)
    # 开始迭代
    for t in range(gamma_period):
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
      b_tilde = np.mat(np.zeros([ self.node_col_amount[num], 1 ]), dtype=complex)
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
    print('%s-Gamma最大特征值: %.3f' % (threading.current_thread().name, Gamma_max))
    
    ## 计算Gamma的最小特征值 ##
    # 初始化
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
    # 显示进度
    if num == 0:
      self.ppbar=tqdm(total=gamma_period)
    # 开始迭代
    for t in range(gamma_period):
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
      b_tilde = np.mat(np.zeros([ self.node_col_amount[num], 1 ]), dtype=complex)
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
    print('%s-Gamma最小特征值: %.3f' % (threading.current_thread().name, Gamma_min))
    # 计算sigma
    sigma = 2 / ( Gamma_max + Gamma_min )
    # print(threading.current_thread().name + 'sigma: ' + str(sigma))

  ## 计算状态 ##
    # 显示进度条
    if num == 0:
      self.pbar=tqdm(total=main_period)
    # 开始迭代
    for t in range(main_period):
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
          pseudo_x += self.H_distribute[num][i].H*self.R_I_distribute_diag[num]*self.H_distribute[num][j]*x_j
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

  #############################################################
  # 函数 -- 
  #       __async_estimator(): 各分布式节点的异步估计器
  # 输入 -- 
  #       num 该节点的节点号(从0开始计)
  #       lock_con 锁, 计算Gamma最大最小特征值时依然使用同步算法
  #       main_period: 迭代周期
  #       gamma_period: 迭代周期
  # 返回 --
  #       NULL
  #############################################################
  def __async_estimator(self, num, lock_con, main_period, gamma_period):
    import threading
    # neighbors whose measurement include mine
    neighbor_in_him = self.get_neighbor(num, -1)
    # neighbors who is in my measurement
    neighbor_in_me = self.get_neighbor(num, 1)
    # neighbors
    neighbors = self.get_neighbor(num, 2)
    
  ## 计算初始化参数
    # Send my alpha(k)_other to neighbor_me
    for i in neighbor_in_me:
      self.alpha[i].put([num, self.H_distribute[num][i].H*self.R_I_distribute_diag[num]*self.z_distribute[num]])

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
      self.Precondition[i].put([num, self.H_distribute[num][i].H*self.R_I_distribute_diag[num]*self.H_distribute[num][i]])

    # Accumulate my Phi_meme and calc Precondition matrix
    my_Precondition = np.mat(np.zeros([self.node_col_amount[num], self.node_col_amount[num]]), dtype=complex)
    for i in neighbor_in_him:
      comein = self.Precondition[num].get()
      if comein[0] not in neighbor_in_him:
        raise Exception('Wrong come in '+str(comein[0]))
      my_Precondition += comein[1]
    my_Precondition = my_Precondition.I
    my_Precondition_sqrt = np.linalg.cholesky(my_Precondition)
    self.Precondition_distribute.update({num:my_Precondition_sqrt})

    # 初始化x_0，全部置0
    x_est = np.mat(np.zeros([self.node_col_amount[num], 1]), dtype=complex)

  ## 计算Gamma的最大特征值 ##
    # initial b_0 with random ||b_0|| = 1 
    b_bar = np.mat(np.random.rand(self.node_col_amount[num], 1), dtype=complex)
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
      self.pbar=tqdm(total=gamma_period)
    # 开始迭代
    for t in range(gamma_period):
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
      b_tilde = np.mat(np.zeros([ self.node_col_amount[num], 1 ]), dtype=complex)
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
    print('%s-Gamma最大特征值: %.3f' % (threading.current_thread().name, Gamma_max))
    
  ## 计算Gamma的最小特征值 ##
    # 初始化
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
    # 显示进度
    if num == 0:
      self.ppbar=tqdm(total=gamma_period)
    # 开始迭代
    for t in range(gamma_period):
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
      b_tilde = np.mat(np.zeros([ self.node_col_amount[num], 1 ]), dtype=complex)
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
    print('%s-Gamma最小特征值: %.3f' % (threading.current_thread().name, Gamma_min))
    # 计算sigma
    sigma = 2 / ( Gamma_max + Gamma_min )
    # print(threading.current_thread().name + 'sigma: ' + str(sigma))

  ## 计算状态 ##
    # 初始化参数
    recv_x_est = {}
    recv_pseudo_x = {}
    recv_est_timestap = {}
    recv_pseudo_timestap = {}
    for i in neighbor_in_me:
      recv_x_est[i] = np.zeros((self.node_col_amount[i],1), dtype=complex)
      recv_est_timestap[i] = 0
    for i in neighbor_in_him:
      recv_pseudo_x[i] = np.zeros((self.node_col_amount[num],1), dtype=complex)
      recv_pseudo_timestap[i] = 0
    # 显示进度条
    ppbar=tqdm(total=main_period)
    # 开始迭代
    for t in range(main_period):
      # Send my estmate state x to neighbors
      for i in neighbors:
        self.est_x[i].put([num, x_est, t])
      # Receive est_x
      wait = True
      while wait:
        comein = self.est_x[num].get()
        if comein[0] in neighbor_in_me: # some are not in me
          recv_x_est[comein[0]] = comein[1]
          if recv_est_timestap[comein[0]] < comein[2]:
            recv_est_timestap[comein[0]] = comein[2]
          if self.est_x[num].empty():
            if t-min(recv_est_timestap.values()) <= DEFAULT_DIFF_LIMIT + random.randint(-int(0.2*DEFAULT_DIFF_LIMIT),int(0.2*DEFAULT_DIFF_LIMIT)):
              wait = False
            else: # 如果有节点比自己慢很多，就等待
              pass
          else: # 没收完，就继续接收
            pass
      # Send pseudo_x to neighbor_me
      for i in neighbor_in_me:
        pseudo_x = np.mat(np.zeros([self.node_col_amount[i],1]), dtype=complex)
        # Accumulate Phi_ij * x_j
        for j, x_j in recv_x_est.items():
          pseudo_x += self.H_distribute[num][i].H*self.R_I_distribute_diag[num]*self.H_distribute[num][j]*x_j
        # Send
        self.pseudo_x[i].put([num, pseudo_x, t])
      # Receive pseudo_x
      wait = True
      while wait:
        comein = self.pseudo_x[num].get()
        recv_pseudo_x[comein[0]] = comein[1]
        if recv_pseudo_timestap[comein[0]] < comein[2]:
          recv_pseudo_timestap[comein[0]] = comein[2]
        if self.pseudo_x[num].empty():  # 如果空了
          if t-min(recv_pseudo_timestap.values()) <= DEFAULT_DIFF_LIMIT + random.randint(-int(0.2*DEFAULT_DIFF_LIMIT),int(0.2*DEFAULT_DIFF_LIMIT)): # 如果有节点比自己慢很多，就等待，加入随机数防卡死
            wait = False
      # Accumulate my pseudo_x
      pseudo_x = np.mat(np.zeros([self.node_col_amount[num],1]), dtype=complex)
      for k in neighbor_in_him:
        pseudo_x += recv_pseudo_x[k]
      # Calc estimate state
      x_est = x_est - sigma * my_Precondition * (pseudo_x - alpha_res)
      if np.max(x_est) > 5000:
        raise ValueError("发散")
      self.record[num][:,t] = x_est#/self.x_observed_list[num] # 归一化 

      ppbar.update(1)
    ppbar.close()
    self.x_est_distribute_lists[num] = x_est
    # print(threading.current_thread().name, str(x_est.T))

#************************************************************
 #* 类 -- 
 #*       利用 随机近似 方法进行分布式估计
 #*
#************************************************************
class Stocastic(DistributedLinearPowerGrid):
  pass

#************************************************************
 #* 类 -- 
 #*       利用 行投影 方法进行分布式估计
 #*
#************************************************************
class RowProjection(DistributedLinearPowerGrid):
  pass