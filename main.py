#-*- coding: utf-8 -*-
from linear_powergrid import LinearPowerGrid
from nonlinear_powergrid import NonLinearPowerGrid
from decentralized.distributed_linear_powergrid import DistributedLinearPowerGrid
from decentralized.distributed_nonlinear_powergrid import DistributedNonLinearPowerGrid
from decentralized.estimators import Richardson,Stocastic
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import json
import os

CENTRAL = False
LINEAR = False

# 不显示warnings
import warnings
#warnings.filterwarnings("ignore")
#

psap_conf = 'ieee118psp.txt'
psap_rule_line = './rules/rule_ieee118psap_line'
psap_rule_bus = './rules/rule_ieee118psap_bus'

cdf_conf = 'ieee118cdf.txt'
cdf_rule_bus = './rules/rule_ieee118cdf_bus'
cdf_rule_branch = './rules/rule_ieee118cdf_branch'

PMU = [3,5,9,12,15,17,21,25,114,28,40,37,34,70,71,53,56,45,49,62,64,68,105,110,76,79,100,92,96,85,86,89]

node1 = [1,2,3,4,5,6,7,8,9,10,11,12,14,16,117]
node2 = [13,15,17,18,19,20,21,22,23,25,26,27,28,29,30,31,32,33,113,114,115]
node3 = [24,38,70,71,72,73,74]
node4 = [34,35,36,37,39,40,41,42,43]
node5 = [44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,77,80,81,100,116]
node6 = [75,76,78,79,82,95,96,97,98,118]
node7 = [83,84,85,86,87,88,89,90,91,92,93,94]
node8 = [99,101,102,103,104,105,106,107,108,109,110,111,112]
nodes = [node1,node2,node3,node4,node5,node6,node7,node8]
'''
# acycle
node1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,117]
node2 = [23,25,26,27,28,29,31,32,113,114,115]
node3 = [15,16,17,18,19,20,21,22,24,30,33,34,35,36,37,39,40,71,72,73]
node4 = [38,41,42,43,44,45,46,47,48,69,70,74,75,76,77,118]
node5 = [49,50,51,54,65,66,68,78,79,80,81,82,95,96,97,98,99,116]
node6 = [52,53,55,56,57,58,59,60,61,62,63,64,67]
node7 = [83,84,85,86,87,88,89,90,91,92,93,94,100,101,102,103,104,105,106,107,108,109,110,111,112]
nodes = [node1,node2,node3,node4,node5,node6,node7]
'''

class Window(object):
  def __init__(self):
    self.wind_main = tk.Tk()
    self.wind_main.title('状态估计系统仿真')
    # 定义变量（默认值）
    self.networkSizeVal = tk.StringVar()  # 网络大小
    self.isCentralizedVal = tk.BooleanVar()
    self.isLinearVal = tk.BooleanVar()
    self.nonLinearIterVal = tk.StringVar()  # 迭代次数
    self.nonLinearStopVal = tk.StringVar()  # 停止条件(迭代误差阈值)
    self.mainPeriodVal = tk.StringVar()  # 主算法迭代次数
    self.childPeriodVal = tk.StringVar()  # 子算法迭代次数(目前只针对Richardson方法计算特征值)
    self.isAsynchronizeVal = tk.BooleanVar()  # 同步/异步
    self.asynToleranceDiffVal = tk.StringVar()  # 异步算法最大容忍落后差
    self.simTimeVal = tk.StringVar() # 仿真时间
    self.stateChangeVal = tk.StringVar() # 每一时刻的状态变化
    self.pmuVoltageVarianceVal = tk.StringVar()
    self.pmuAngleVarianceVal = tk.StringVar()
    self.scadaVoltageVarianceVal = tk.StringVar()
    self.scadaAngleVarianceVal = tk.StringVar()
    self.isPlotVal = tk.BooleanVar() # 是否每次迭代过程都画(针对分布式线性)
    self.isInnerIterPlotVal = tk.BooleanVar() # 是否画内部迭代(针对分布式非线性)
    self.isStateIterPlotVal = tk.BooleanVar() # 是否画状态更新迭代(针对分布式非线性)
    self.decentralizedMethodVal = tk.StringVar()
    self.modelVal = tk.StringVar()
    self.network_size_list = ('118',) # 下拉栏内容
    self.networkSizeVal.set('118')
    self.isCentralizedVal.set(False)
    self.isLinearVal.set(False)
    self.nonLinearIterVal.set('5')
    self.nonLinearStopVal.set('5')
    self.mainPeriodVal.set('150')
    self.childPeriodVal.set('100')
    self.isAsynchronizeVal.set(False)
    self.asynToleranceDiffVal.set('15')
    self.simTimeVal.set('4')
    self.stateChangeVal.set('0.3')
    self.pmuVoltageVarianceVal.set('0.002')
    self.pmuAngleVarianceVal.set('0.01')
    self.scadaVoltageVarianceVal.set('0.3')
    self.scadaAngleVarianceVal.set('0.3')
    self.isPlotVal.set(True)
    self.decentralizedMethodVal.set('Richardson')
    self.modelVal.set('PowerGrid')
    self.windowSize = [300,450] # 窗口默认大小
    # 读取配置文件
    if os.path.exists('config.json') is True:
      with open('config.json','r',encoding='utf-8') as f:
        self.conf_dict = json.load(f)
        self.isCentralizedVal.set(self.conf_dict['is_centralized'])
        self.isLinearVal.set(self.conf_dict['is_linear'])
        self.nonLinearIterVal.set(self.conf_dict['nonlinear_iter_time'])
        self.nonLinearStopVal.set(self.conf_dict['nonlinear_stop_error'])
        self.mainPeriodVal.set(self.conf_dict['decentralized_main_period'])
        self.childPeriodVal.set(self.conf_dict['decentralized_child_period'])
        self.isAsynchronizeVal.set(self.conf_dict['is_asyn'])
        self.asynToleranceDiffVal.set(self.conf_dict['asyn_tolerance_diff'])
        self.simTimeVal.set(self.conf_dict['sim_time'])
        self.stateChangeVal.set(self.conf_dict['state_change'])
        self.pmuVoltageVarianceVal.set(self.conf_dict['pmu_voltage_variance'])
        self.pmuAngleVarianceVal.set(self.conf_dict['pmu_angle_variance'])
        self.scadaVoltageVarianceVal.set(self.conf_dict['scada_voltage_variance'])
        self.scadaAngleVarianceVal.set(self.conf_dict['scada_power_variance'])
        self.isPlotVal.set(self.conf_dict['is_plot'])
        self.isInnerIterPlotVal.set(self.conf_dict['is_inneriter_plot'])
        self.isStateIterPlotVal.set(self.conf_dict['is_outeriter_plot'])
        self.networkSizeVal.set(self.conf_dict['network_size'])
        self.decentralizedMethodVal.set(self.conf_dict['decentralized_method'])
        self.modelVal.set(self.conf_dict['model_name'])
        # GUI自用变量(以后应该从conf_dict剔除)
        self.network_size_list = self.conf_dict['network_size_list']
        self.windowSize = self.conf_dict['window_size']

    ''' 菜单 '''
    menu_dict = {'Files':['Export']} # 菜单项目
    menu_object_dict = {} # 子菜单对象字典
    menubar = tk.Menu(self.wind_main)
    for key,val in menu_dict.items():
      menu_object_dict[key] = tk.Menu(menubar)  # 创建子菜单对象们
      for item in val:
        menu_object_dict[key].add_command(label=item,command=lambda:self.menu_select_event(obj=item)) # 创建子菜单内的选项
    for menu_son,obj in menu_object_dict.items(): # 主菜单绑定子菜单们
      menubar.add_cascade(label=menu_son, menu=obj)

    self.wind_main['menu'] = menubar
    # 计数器
    line0_cnt = 0
    line1_cnt = 0
    ''' 第一列 '''
    line0_frame = tk.Frame(self.wind_main)
    # 网络大小
    tmp_frame = tk.Frame(line0_frame)
    framework_lable = tk.Label(tmp_frame, text='SIZE')
    framework_lable.grid(row=0,column=0)
    self.network_size = ttk.Combobox(tmp_frame, textvariable=self.networkSizeVal, width=5) # 选择分布式方法
    self.network_size.grid(row=0,column=1)
    self.network_size['values']=self.network_size_list
    self.network_size['state']='readonly'
    self.network_size.current(0)
    tmp_frame.grid(row=line0_cnt,column=0);line0_cnt+=1
    # 选择框架(分布式/集中式)
    framework_lable = tk.Label(line0_frame, text='- select framework -')
    framework_lable.grid(row=line0_cnt,column=0, pady=10, padx=15,sticky=tk.W);line0_cnt+=1
    centralized = tk.Radiobutton(line0_frame, text='centralized', variable=self.isCentralizedVal, value=True,command=lambda:self.framework_select_event(centralized=True))
    decentralized = tk.Radiobutton(line0_frame, text='decentralized', variable=self.isCentralizedVal, value=False,command=lambda:self.framework_select_event(centralized=False))
    centralized.grid(row=line0_cnt,column=0,sticky=tk.W);line0_cnt+=1
    decentralized.grid(row=line0_cnt,column=0,sticky=tk.W);line0_cnt+=1
    # 选择框架(线性/非线性)
    linearlized_lable = tk.Label(line0_frame, text='- linear/nonlinear -')
    linearlized_lable.grid(row=line0_cnt,column=0, pady=10, padx=15,sticky=tk.W);line0_cnt+=1
    linearlized = tk.Radiobutton(line0_frame, text='linear', variable=self.isLinearVal, value=True,command=lambda:self.linearlized_select_event(linearlized=True))
    nonlinearlized = tk.Radiobutton(line0_frame, text='nonlinear', variable=self.isLinearVal, value=False,command=lambda:self.linearlized_select_event(linearlized=False))
    linearlized.grid(row=line0_cnt,column=0,sticky=tk.W);line0_cnt+=1
    nonlinearlized.grid(row=line0_cnt,column=0,sticky=tk.W);line0_cnt+=1
    # 非线性配置栏
    nonlinear_conf_frame = tk.Frame(line0_frame)
    linearlized_lable = tk.Label(nonlinear_conf_frame, text='- nonlinear -')
    linearlized_lable.grid(row=0,column=0,columnspan=2, pady=10, padx=15,sticky=tk.W)
    iter_time_label = tk.Label(nonlinear_conf_frame, text='iter time')
    self.iter_time = tk.Entry(nonlinear_conf_frame, state=tk.NORMAL, textvariable=self.nonLinearIterVal, width=5)
    iter_time_label.grid(row=1,column=0,sticky=tk.E)
    self.iter_time.grid(row=1,column=1)
    stop_label = tk.Label(nonlinear_conf_frame, text='stop error')
    self.stop = tk.Entry(nonlinear_conf_frame, state=tk.NORMAL, textvariable=self.nonLinearStopVal, width=5)
    stop_label.grid(row=2,column=0,sticky=tk.E)
    self.stop.grid(row=2,column=1)
    sym_label = tk.Label(nonlinear_conf_frame, text='%')
    sym_label.grid(row=2,column=2,sticky=tk.W)
    nonlinear_conf_frame.grid(row=line0_cnt,column=0,sticky=tk.E);line0_cnt+=1
    #只读配置
    if self.isLinearVal.get() is True:
      self.iter_time.config(state=tk.DISABLED)
      self.stop.config(state=tk.DISABLED)
    # 分布式配置栏
    decentralized_conf_frame = tk.Frame(line0_frame)
    decentralized_lable = tk.Label(decentralized_conf_frame, text='- decentralized -')
    decentralized_lable.grid(row=0,column=0,columnspan=2, pady=10, padx=15,sticky=tk.W)
    self.decentralized_method = ttk.Combobox(decentralized_conf_frame,textvariable=self.decentralizedMethodVal) # 选择分布式方法
    self.decentralized_method.grid(row=1,column=0,columnspan=2)
    self.decentralized_method['value']=('Richardson','Richardson(finite)','Stocastics')
    self.decentralized_method['state']='readonly'
    self.decentralized_method.bind("<<ComboboxSelected>>",self.decentralized_method_event)
    main_period_label = tk.Label(decentralized_conf_frame, text='main period')
    self.main_period = tk.Entry(decentralized_conf_frame, state=tk.NORMAL, textvariable=self.mainPeriodVal, width=5)
    main_period_label.grid(row=2,column=0,sticky=tk.E)
    self.main_period.grid(row=2,column=1)
    child_period_label = tk.Label(decentralized_conf_frame, text='child period')
    self.child_period = tk.Entry(decentralized_conf_frame, state=tk.NORMAL, textvariable=self.childPeriodVal, width=5)
    child_period_label.grid(row=3,column=0,sticky=tk.E)
    self.child_period.grid(row=3,column=1)
    self.synchronized = tk.Radiobutton(decentralized_conf_frame, text='synchronized', variable=self.isAsynchronizeVal, value=False,command=lambda:self.synchronized_select_event(synchronized=True))
    self.asynchronized = tk.Radiobutton(decentralized_conf_frame, text='asynchronized', variable=self.isAsynchronizeVal, value=True,command=lambda:self.synchronized_select_event(synchronized=False))
    self.synchronized.grid(row=4,column=0,sticky=tk.W)
    self.asynchronized.grid(row=5,column=0,sticky=tk.W)
    tolerance_label = tk.Label(decentralized_conf_frame, text='tolerate diff')
    self.tolerance = tk.Entry(decentralized_conf_frame, state=tk.NORMAL, textvariable=self.asynToleranceDiffVal, width=5)
    tolerance_label.grid(row=6,column=0,sticky=tk.E)
    self.tolerance.grid(row=6,column=1)
    decentralized_conf_frame.grid(row=line0_cnt,column=0,sticky=tk.E);line0_cnt+=1
    line0_frame.grid(row=0,column=0,sticky=tk.N)
    # 只读配置
    if self.isCentralizedVal.get() is True:   # 集中式
      self.main_period.config(state=tk.DISABLED)
      self.child_period.config(state=tk.DISABLED)
      self.synchronized.config(state=tk.DISABLED)
      self.asynchronized.config(state=tk.DISABLED)
      self.tolerance.config(state=tk.DISABLED)
    else:   # 分布式
      if self.isAsynchronizeVal.get() is False:
        self.tolerance.config(state=tk.DISABLED)
      if self.decentralizedMethodVal.get() != 'Richardson':
        self.child_period.config(state=tk.DISABLED)

    '''第二列'''
    line1_frame = tk.Frame(self.wind_main)
    # 选择模型
    model_frame = tk.Frame(line1_frame)
    model_lable = tk.Label(model_frame, text='- select model -')
    model_lable.grid(row=0,column=0,columnspan=2, pady=10, padx=15,sticky=tk.W)
    PowerGrid = tk.Radiobutton(model_frame, text='PowerGrid', variable=self.modelVal, value='PowerGrid',command=lambda:self.model_select_event(model_name='PowerGrid'))
    WSNs = tk.Radiobutton(model_frame, text='WSNs', variable=self.modelVal, value='WSNs',command=lambda:self.model_select_event(model_name='WSNs'))
    PowerGrid.grid(row=1,column=0,sticky=tk.W)
    WSNs.grid(row=2,column=0,sticky=tk.W)
    model_frame.grid(row=line1_cnt,column=0);line1_cnt+=1
    # 基本配置
    configuration_frame = tk.Frame(line1_frame)
    normal_lable = tk.Label(configuration_frame, text='- configuration -')
    normal_lable.grid(row=0,column=0,columnspan=2, pady=10,padx=15,sticky=tk.W)
    simTime_label = tk.Label(configuration_frame, text='simulation time')
    sim_time = tk.Entry(configuration_frame, state=tk.NORMAL, textvariable=self.simTimeVal, width=5)
    simTime_label.grid(row=1,column=0,sticky=tk.W)
    sim_time.grid(row=1,column=1)
    state_change_label = tk.Label(configuration_frame, text='state change')
    state_change = tk.Entry(configuration_frame, state=tk.NORMAL, textvariable=self.stateChangeVal, width=5)
    state_change_label.grid(row=2,column=0,sticky=tk.W)
    state_change.grid(row=2,column=1)
    configuration_frame.grid(row=line1_cnt,column=0);line1_cnt+=1
    # 误差配置(PMU,SCADA)
    variance_frame = tk.Frame(line1_frame)
    variance_lable = tk.Label(variance_frame, text='- variance -')
    variance_lable.grid(row=0,column=0,columnspan=2, pady=10,padx=15,sticky=tk.W)
    self.pmu_voltage_label = tk.Label(variance_frame, text='PMU Volt')
    self.pmu_angle_label = tk.Label(variance_frame, text='PMU Angl')
    self.scada_voltage_label = tk.Label(variance_frame, text='SCA Volt')
    self.scada_angle_label = tk.Label(variance_frame, text='SCA Powr')
    pmu_voltage = tk.Entry(variance_frame, state=tk.NORMAL, textvariable=self.pmuVoltageVarianceVal, width=5)
    pmu_angle = tk.Entry(variance_frame, state=tk.NORMAL, textvariable=self.pmuAngleVarianceVal, width=5)
    scada_voltage = tk.Entry(variance_frame, state=tk.NORMAL, textvariable=self.scadaVoltageVarianceVal, width=5)
    self.scada_angle = tk.Entry(variance_frame, state=tk.NORMAL, textvariable=self.scadaAngleVarianceVal, width=5)
    self.pmu_voltage_label.grid(row=1,column=0,sticky=tk.E)
    pmu_voltage.grid(row=1,column=1)
    self.pmu_angle_label.grid(row=2,column=0,sticky=tk.E)
    pmu_angle.grid(row=2,column=1)
    self.scada_voltage_label.grid(row=3,column=0,sticky=tk.E)
    scada_voltage.grid(row=3,column=1)
    self.scada_angle_label.grid(row=4,column=0,sticky=tk.E)
    self.scada_angle.grid(row=4,column=1)
    variance_frame.grid(row=line1_cnt,column=0);line1_cnt+=1
    line1_frame.grid(row=0,column=1,sticky=tk.N)
    # 只读配置
    if self.modelVal.get() == 'WSNs':
      self.pmu_voltage_label.config(text='Ref x/y')
      self.pmu_angle_label.config(text='Angle  ')
      self.scada_voltage_label.config(text='Distance')
      self.scada_angle_label.config(text=' ')
      self.scada_angle.config(state=tk.DISABLED)

    # 画图配置
    plot_frame = tk.Frame(line1_frame)
    plot_lable = tk.Label(plot_frame, text='- plot -')
    plot_lable.grid(row=0,column=0, pady=10)
    '''
    confirm_button = tk.Button(plot_frame, text='Plot configure', padx=10, pady=5, command=self.plot_button)
    confirm_button.grid(row=1,column=0,sticky=tk.W+tk.E)
    '''
    is_plot = tk.Checkbutton(plot_frame, text='plot every time', variable=self.isPlotVal, onvalue=True, offvalue=False)
    self.is_inneriter_plot = tk.Checkbutton(plot_frame, text='plot inner iter', variable=self.isInnerIterPlotVal, onvalue=True, offvalue=False)
    self.is_outeriter_plot = tk.Checkbutton(plot_frame, text='plot outer iter', variable=self.isStateIterPlotVal, onvalue=True, offvalue=False)
    is_plot.grid(row=1,column=0)
    self.is_inneriter_plot.grid(row=2,column=0)
    self.is_outeriter_plot.grid(row=3,column=0)
    plot_frame.grid(row=line1_cnt,column=0);line1_cnt+=1
    # 只读配置
    if self.isLinearVal.get() is True and self.isCentralizedVal is True:
      self.is_inneriter_plot.config(state=tk.DISABLED)
      self.is_outeriter_plot.config(state=tk.DISABLED)

    # 确认按钮
    confirm_button = tk.Button(self.wind_main, text='Confirm', padx=10, pady=5, command=self.confirm)
    confirm_button.grid(row=11,column=0,columnspan=3,pady=20,sticky=tk.W+tk.E)
    '''窗口大小'''
    width=self.windowSize[0]
    height=self.windowSize[1]
    screenwidth = self.wind_main.winfo_screenwidth()  
    screenheight = self.wind_main.winfo_screenheight()
    alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth-width)/2, (screenheight-height)/2)
    self.wind_main.geometry(alignstr)

  def menu_select_event(self, obj):
    '''
    if obj == 'Export':
      wind_export = tk.Toplevel(self.wind_main)
      wind_export.title('导出')
      self.isPlotVal = tk.BooleanVar() # 是否
      self.isInnerIterPlotVal = tk.BooleanVar() # 是否
      is_plot = tk.Checkbutton(wind_export, text='plot every time', variable=self.isPlotVal, onvalue=True, offvalue=False)
      is_inneriter_plot = tk.Checkbutton(wind_export, text='plot nonlinear inner iter', variable=self.isInnerIterPlotVal, onvalue=True, offvalue=False)
      is_plot.grid(row=0,column=0)
      is_inneriter_plot.grid(row=1,column=0)
      # 确认按钮
      confirm_button = tk.Button(wind_export, text='Confirm', padx=10, pady=5, command=self.plot_confirm)
      confirm_button.grid(row=2,column=0,pady=20,sticky=tk.W+tk.E)
      # 窗口大小
      width=200
      height=150
      screenwidth = wind_export.winfo_screenwidth()  
      screenheight = wind_export.winfo_screenheight()
      alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth-width)/2, (screenheight-height)/2)
      wind_export.geometry(alignstr)
    '''

  def model_select_event(self,model_name):
    '''
    选择模型事件
    '''
    if model_name == 'WSNs':
      self.network_size['values']=('8','4')
      self.network_size_list = ('8','4')
      # variance
      self.pmu_voltage_label.config(text='Ref x/y')
      self.pmu_angle_label.config(text='Angle')
      self.scada_voltage_label.config(text='Distance')
      self.scada_angle_label.config(text=' ')
      self.scada_angle.config(state=tk.DISABLED)
    elif model_name == 'PowerGrid':
      self.network_size['values']=('118',)
      self.network_size_list = ('118',)
      #variance
      self.pmu_voltage_label.config(text='PMU Volt')
      self.pmu_angle_label.config(text='PMU Angl')
      self.scada_voltage_label.config(text='SCA Volt')
      self.scada_angle_label.config(text='SCA Powr')
      self.scada_angle.config(state=tk.NORMAL)
    self.network_size.current(0)

  def framework_select_event(self,centralized):
    '''
    选中框架的事件
    '''
    if centralized is True:
      self.main_period.config(state=tk.DISABLED)
      self.child_period.config(state=tk.DISABLED)
      self.synchronized.config(state=tk.DISABLED)
      self.asynchronized.config(state=tk.DISABLED)
      self.tolerance.config(state=tk.DISABLED)
      # plot button
      self.is_inneriter_plot.config(state=tk.DISABLED)
      self.is_outeriter_plot.config(state=tk.DISABLED)
    else:
      self.main_period.config(state=tk.NORMAL)
      self.child_period.config(state=tk.NORMAL)
      self.synchronized.config(state=tk.NORMAL)
      self.asynchronized.config(state=tk.NORMAL)
      self.tolerance.config(state=tk.NORMAL)
      # plot button
      if self.isLinearVal.get() is False:
        self.is_inneriter_plot.config(state=tk.NORMAL)
        self.is_outeriter_plot.config(state=tk.NORMAL)

  def linearlized_select_event(self,linearlized):
    '''
    选中是否线性模型的事件
    '''
    if linearlized is True:
      self.iter_time.config(state=tk.DISABLED)
      self.stop.config(state=tk.DISABLED)
      # plot button
      self.is_inneriter_plot.config(state=tk.DISABLED)
      self.is_outeriter_plot.config(state=tk.DISABLED)
    else:
      self.iter_time.config(state=tk.NORMAL)
      self.stop.config(state=tk.NORMAL)
      # plot button
      if self.isCentralizedVal.get() is False:
        self.is_inneriter_plot.config(state=tk.NORMAL)
        self.is_outeriter_plot.config(state=tk.NORMAL)

  def decentralized_method_event(self, event):
    if self.decentralizedMethodVal.get() == 'Richardson':
      self.child_period.config(state=tk.NORMAL)
    else:
      self.child_period.config(state=tk.DISABLED)

  def synchronized_select_event(self, synchronized):
    if synchronized is True:
      self.tolerance.config(state=tk.DISABLED)
    else:
      self.tolerance.config(state=tk.NORMAL)

  '''
  def plot_button(self):
  """
  画图配置
  """
    wind_plot = tk.Toplevel(self.wind_main)
    wind_plot.title('画图配置')
    self.isPlotVal = tk.BooleanVar() # 是否每次迭代过程都画(针对分布式线性)
    self.isInnerIterPlotVal = tk.BooleanVar() # 是否画内部迭代(针对分布式非线性)
    is_plot = tk.Checkbutton(wind_plot, text='plot every time', variable=self.isPlotVal, onvalue=True, offvalue=False)
    is_inneriter_plot = tk.Checkbutton(wind_plot, text='plot nonlinear inner iter', variable=self.isInnerIterPlotVal, onvalue=True, offvalue=False)
    is_plot.grid(row=0,column=0)
    is_inneriter_plot.grid(row=1,column=0)
    # 确认按钮
    confirm_button = tk.Button(wind_plot, text='Confirm', padx=10, pady=5, command=self.plot_confirm)
    confirm_button.grid(row=2,column=0,pady=20,sticky=tk.W+tk.E)
    # 窗口大小
    width=200
    height=150
    screenwidth = wind_plot.winfo_screenwidth()  
    screenheight = wind_plot.winfo_screenheight()
    alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth-width)/2, (screenheight-height)/2)
    wind_plot.geometry(alignstr)
  '''

  def confirm(self):
    """
    仿真按钮触发事件
    """
    # 记录当前窗口大小
    self.windowSize[0]=self.wind_main.winfo_width()
    self.windowSize[1]=self.wind_main.winfo_height()
    # 记录当前配置
    conf_dict = {
      'network_size': self.networkSizeVal.get(),
      'is_centralized': self.isCentralizedVal.get(),
      'is_linear': self.isLinearVal.get(),
      'sim_time': self.simTimeVal.get(),
      'state_change': self.stateChangeVal.get(),
      'pmu_voltage_variance': self.pmuVoltageVarianceVal.get(),
      'pmu_angle_variance': self.pmuAngleVarianceVal.get(),
      'scada_voltage_variance': self.scadaVoltageVarianceVal.get(),
      'scada_power_variance': self.scadaAngleVarianceVal.get(),
      'nonlinear_iter_time': self.nonLinearIterVal.get(),
      'nonlinear_stop_error': self.nonLinearStopVal.get(),
      'decentralized_method': self.decentralizedMethodVal.get(),
      'decentralized_main_period': self.mainPeriodVal.get(),
      'decentralized_child_period': self.childPeriodVal.get(),
      'is_asyn': self.isAsynchronizeVal.get(),
      'asyn_tolerance_diff': self.asynToleranceDiffVal.get(),
      'is_plot': self.isPlotVal.get(),
      'is_inneriter_plot': self.isInnerIterPlotVal.get(),
      'is_outeriter_plot': self.isStateIterPlotVal.get(),
      'model_name': self.modelVal.get(),
      # GUI自用变量
      'network_size_list': self.network_size_list,
      'window_size': self.windowSize,
    }
    # 保存当前配置
    with open('config.json','w',encoding='utf-8') as f:
      f.write(json.dumps(conf_dict,ensure_ascii=False))
    # 开始跑仿真
    false_dic = {'which_state':[10,28,50,100,200,17,75,91,125,171],'effect':[5,4.5,4,3.5,3,30,30,30,30,30]} #自定义FDI
    if conf_dict['is_centralized'] is True:
      if conf_dict['is_linear'] is True:
        model = LinearPowerGrid(PMU, conf_dict)
      else:
        model = NonLinearPowerGrid(PMU, conf_dict)
    else:
      if conf_dict['is_linear'] is True:
        model = DistributedLinearPowerGrid(nodes, PMU, conf_dict)
        model.inject_falsedata(moment=0,conf_dic=false_dic)  # 在1时刻注入虚假数据
      else:
        model = DistributedNonLinearPowerGrid(nodes, PMU, conf_dict)
    model.estimator()

def main():
  w = Window()
  w.wind_main.mainloop()
  exit()
  ### 电网建模 ###
  #model.inject_baddata(moment=1, probability=50)  # 在第5个仿真时刻开始每个仪表有10/10e6的概率产生坏数据
  #model.inject_falsedata(moment=5)  # 在第5个仿真时刻开始注入隐匿虚假数据
  #model.inject_falsedata_PCA(moment=500)  # 在第30个仿真时刻开始注入PCA隐匿虚假数据
  #model.summary()

  '''
  model = Richardson(118, cdf_conf, cdf_rule_bus, cdf_rule_branch, nodes, PMU)
  #model.set_variance(pmu=[1000,1000],scada=[1,1])
  #model.summary()
  #model.delete_reference_bus()
  #model.inject_baddata(moment=1, probability=50)  # 坏数据
  false_dic = {'which_state':[10,28,50,100,200,17,75,91,125,171],'effect':[5,4.5,4,3.5,3,30,30,30,30,30]}
  #model.inject_falsedata(moment=0,conf_dic=false_dic)  # 注入虚假数据
  #model.detect_baddata(centralized=True)

  ### 分布式建模 ###
  #model.set_variance(R)  # 方差矩阵 (Default: 单位矩阵)
  #model.summary()
  ### 分布式建模完毕 ###
  ### 分布式估计 ###
  model.estimator_config(state_variance=1)
  model.estimator(sim_time=2, is_async=False, plot=[0,2])
  '''

  '''
  model = Stocastic(118, cdf_conf, cdf_rule_bus, cdf_rule_branch, nodes, PMU)
  model.estimator_config(state_variance=1, main_period=120 ,gamma_period=60, diff_limit=15, is_finite_time=True)
  model.estimator(sim_time=2, is_async=False, plot=[0,2])
  '''
if __name__ == '__main__':
  main()
