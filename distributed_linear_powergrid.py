from linear_powergrid import LinearPowerGrid
import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
#plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
import matplotlib.pylab as pylab
from tqdm import tqdm

# 循环次数
GAMMA_MAX_EIGEN = 100       # 计算Gamma最大特征值的循环次数 200
GAMMA_MIN_EIGEN = 100      # 计算Gamma最小特征值的循环次数 500
MAIN_LOOP_PERIOD = 2000    # 主程序循环次数

class DistributedLinearPowerGrid(LinearPowerGrid):
  def __init__(self, size):
    super().__init__(size=size)
    self.is_distribute = True

#############################################################
 # 函数 -- 
 #       set_nodes(): 将系统分布化，分割成多个子系统(节点)
 #                    也就是将H矩阵行列重新排列
 # 输入 -- 
 #       nodes[][]  节点包含的所有总线编号
 #       x_operation 上一时刻估计的状态值 
 # 得到 -- 
 #       nodes_num 分割成多少个节点
 #       node_row_amount[] 每个节点有多少行(测量值) (顺序为节点顺序)
 #       node_col_amount[] 每个节点有多少列(状态值) (顺序为节点顺序)
 #       z_distribute[]    每个节点的测量向量 (顺序为节点顺序)
 #       x_real_list[]     每个节点上一时刻的状态向量 (...)
 #       x_observed_list[] 每个节点在上一时刻状态向量加噪声后 (...)(没啥用)
 #       H_distribute[]    每个节点的局部H矩阵 (...)
 #       R_I_distribute_diag[] 每个节点的局部协方差矩阵(...)
 #       nodes_graph(np.mat) 各节点间的连接图矩阵
 #       Phi_graph(np.mat)   各节点间Phi矩阵的连接图矩阵
 #       Precondition_center(np.mat) precondition矩阵
 #       Precondition_distribute{}   空的字典, 在这里先声明, 之后迭代计算时用到 
 # 返回 --
 #       NULL
#############################################################
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
    Phi = H.H * self.R.I * H
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
    self.Phi = Phi
    self.x_est_center = Phi.I*H.H*self.R.I*self.z_observed

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
    for m in Phi_distribute:
      j=0
      for n in m:
        if (len(n.nonzero()[0]) != 0):# and (i!=j):
          pg[i,j] = 1
        j += 1
      i += 1
    self.Phi_graph = pg

    self.Precondition_distribute = {}	#test

#############################################################
 # 函数 -- 
 #       fit(): 进行分布式估计
 # 输入 -- 
 #       is_plot 是否画图: [True,False]
 # 返回 --
 #       NULL
#############################################################
  def fit(self, is_plot=False):
    import threading
    import queue
    self.x_est_distribute_lists = []
    T = MAIN_LOOP_PERIOD
    axis1 = np.mat(range(T))
    sample = np.mat(np.empty([self.state_size, T]))
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
      #self.task_lock.append( queue.Queue() )
      lock_con = threading.Condition()

      self.record.append(np.mat(np.empty([self.node_col_amount[i],T])))
      self.gamma_max_record.append([])
      self.gamma_min_record.append([])

      self.x_est_distribute_lists.append(np.mat(np.empty([self.node_col_amount[i],1])))

    for i in range(self.nodes_num):
      thread_nodes.append(threading.Thread(target=self.__estimator, args=(i, lock_con, T)))
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
 #       __estimator(): 各分布式节点的估计器
 # 输入 -- 
 #       num 该节点的节点号(从0开始计)
 #       lock_con 锁（保证通信的同步）
 #       T 估计的迭代次数
 # 返回 --
 #       NULL
#############################################################
  def __estimator(self, num, lock_con, T):
    import threading
    # neighbors whose measurement include mine
    neighbor_in_him = self.__get_neighbor(num, -1)
    # neighbors who is in my measurement
    neighbor_in_me = self.__get_neighbor(num, 1)
    # neighbors
    neighbors = self.__get_neighbor(num, 2)
    
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

  ## 计算Gamma的最大特征值 ##
    # 初始化
    # initial x_0 set all 0
    x_est = np.mat(np.zeros([self.node_col_amount[num], 1]), dtype=complex)
    ## To estimate ||Gamma||
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
      self.pbar=tqdm(total=GAMMA_MAX_EIGEN)
    # 开始迭代
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
      self.ppbar=tqdm(total=GAMMA_MAX_EIGEN)
    # 开始迭代
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
      self.pbar=tqdm(total=T)
    # 开始迭代
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
 #       __get_neighbor(): 获取节点的邻居
 # 输入 -- 
 #       n 节点号(从0开始计)
 #       t -> -1: Aji != 0 
 #             1: Aij != 0
 #             2: Aij and Aji !=0
 #				     3: n 的邻居的所有邻居
 # 返回 --
 #       邻居节点号 (1d-list)(start from 0)
#############################################################
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
  super().inject_falsedata(sparse_amount=sparse_amount, amptitude=amptitude, measure_tobe_injected=measure_tobe_injected, delete_previous_injected=delete_previous_injected)
  z_distribute = []
  row_cnt = 0
  for row in self.node_row_amount:
    z_distribute.append(self.z_observed[row_cnt:row_cnt+row, 0])
    row_cnt += row
  self.z_distribute = z_distribute

#############################################################
 # 函数 -- 
 #       detect_baddata(): 坏值检测
 # 输入 -- 
 #       n 节点号(从0开始计)
 #       t -> -1: Aji != 0 
 #             1: Aij != 0
 #             2: Aij and Aji !=0
 #				     3: n 的邻居的所有邻居
 # 返回 --
 #       邻居节点号 (1d-list)(start from 0)
#############################################################
  def detect_baddata(self, is_plot = False):
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
      chi_list.append( self.chi2square_val(self.node_row_amount[cnt] - self.node_col_amount[cnt], 0.5) )
      if detect_res_list[cnt] < chi_list[cnt]:
        print('测量残差为: %.3f, 小于置信度为0.5的卡方检验值: %.3f, 未检测到攻击.' % (float(detect_res_list[cnt]), chi_list[cnt]))
      else:
        print('测量残差为: %.3f, 大于置信度为0.5的卡方检验值: %.3f, 检测到攻击.' % (float(detect_res_list[cnt]), chi_list[cnt]))
      cnt += 1
    if is_plot is True:
      plt.figure('坏值检测')
      #plt.title('坏值检测')
      plt.plot(chi_list, 'r--', marker='.')
      plt.plot(detect_res_list, 'b', marker='.')
      plt.legend(['阈值', '残差'], loc='upper right')
      plt.xlabel("子系统号")
      plt.ylabel("幅值")
      plt.grid(True)
      plt.show()

  def gen_graph(self):
    return self.nodes_graph