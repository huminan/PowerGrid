import numpy as np

BUS_PRE_IGNORE_ROWS = 2   # ignore previous PRE_IGNORE_ROWS rows.
BRANCH_PRE_IGNORE_ROWS = 1

### Location in P_bus_data ###
BUS_NAME_START_COLS = 4   # for character
BUS_DATA_START_COLS = 14

BUS_FLOW_AREA_NUM_COL = 0 # for number
BUS_LOSS_ZONE_NUM_COL = 1
BUS_TYPE_COL = 3

### Location in P_bus_data ###
BRANCH_FLOW_AREA_NUM_COL = 0 # for number
BRANCH_LOSS_ZONE_NUM_COL = 1
BRANCH_TYPE_COL = 3


filename = 'ieee118cdf.txt' # txt文件和当前脚本在同一目录下，所以不用写具体路径
P_bus_name = []
P_bus_data = []
P_branch_data = []
bus_num_cnt = 0
branch_num_cnt = 0

with open(filename, 'r') as file_to_read:
### BUS DATA ###
  for i in range(BUS_PRE_IGNORE_ROWS):
    lines = file_to_read.readline()
  while True:
    lines = file_to_read.readline() # 整行读取数据

    if not lines:
      break
    if lines[0:4] == '-999':
      break

    P_bus_name.append(lines[BUS_NAME_START_COLS:BUS_DATA_START_COLS].strip())

    P_bus_data.append([])
    tmp_cnt = 0
    for t_data in lines[BUS_DATA_START_COLS:].split():   # 将整行数据分割处理
      if tmp_cnt == BUS_FLOW_AREA_NUM_COL:
        P_bus_data[bus_num_cnt].append(int(t_data[1]))
      else:
        P_bus_data[bus_num_cnt].append(float(t_data))
      tmp_cnt += 1
    bus_num_cnt += 1

### BRANCH DATA ###
  for i in range(BRANCH_PRE_IGNORE_ROWS):
    lines = file_to_read.readline()
  while True:
    lines = file_to_read.readline() # 整行读取数据

    if not lines:
      break
    if lines[0:4] == '-999':
      break

    P_branch_data.append([])
    for t_data in lines[BUS_DATA_START_COLS:].split():   # 将整行数据分割处理
      P_branch_data[branch_num_cnt].append(float(t_data))
    branch_num_cnt += 1



