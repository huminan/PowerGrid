# coding=utf-8

# Definitions of rule's file
FILE_FORMAT_NUM = 2   # how many parameters to determine rule's format
SEPARATOR_FORMAT_NUM = 2  # how many parameters in the rule
FIXEDSEAT_FORMAT_NUM = 3

class tools(object):
  def __init__(self, filename, rulepath, section):
    self.filename = filename
    self.rulepath = rulepath
    self.section = section
    self.items_name = []   # list items' names
    self.rules = {}   # dict int/float/char of every items
    self.items = {}   # dict items' values
    self.num_cnt = 0
    self.flg_end = '-999' # Default end flag
    self.__extract_rule()

  def __rules(self, i_rule, i_file_to_read):
    RULES = {
      'SEPARATOR': lambda:self.__rule_separator(i_file_to_read),
      'FIXEDSEAT': lambda:self.__rule_fixedseat(i_file_to_read),
    }
    func = RULES.get(i_rule)
    if func is None:
      return None
    func()
    return True

  def __rule_separator(self, i_file_to_read):
    while True:   # Extract data
      lines = i_file_to_read.readline() # 整行读取数据
      if not lines:
        break
      if lines[0]=='#':
        continue
      if lines.strip()=='':
        continue
      t_line_elements = lines.split()
      if len(t_line_elements) != SEPARATOR_FORMAT_NUM:
        raise Exception('extract_rule: Too many or too less elements in one line!')
        break
      if t_line_elements[1] == '9': # Update end flag
        self.flg_end = t_line_elements[0]
        continue
      self.rules[t_line_elements[0]]=int(t_line_elements[1])
      self.items_name.append(t_line_elements[0])
    for v_names in self.items_name: # Initiallize items
      self.items[v_names]=[]
    self.__separator()

###########################################
# __separator(self, v_separator=None)
# Extracting data through separators, in this way, data must be complete even it is null, otherwise work bad.
# v_separator: default nothing, can be set ',' or ';' and so on;
###########################################
  def __separator(self, v_separator=None):
    with open(self.filename, 'r') as file_to_read:
      t_cnt = 0
      while t_cnt < self.section:
        lines = file_to_read.readline()
        if not lines:
          break
        if lines[0]=='#':
          continue
        if lines[0:len(self.flg_end)] == self.flg_end:
          t_cnt += 1

      while True:
        lines = file_to_read.readline()
        if not lines:
          break
        if lines[0]=='#':
          continue
        if lines[0:len(self.flg_end)] == self.flg_end:
          break

        cnt = 0
        for t_data in lines.split(v_separator):
          if self.rules[self.items_name[cnt]] == 1:
            print(self.items_name[cnt], cnt, self.num_cnt)
            self.items[self.items_name[cnt]].append(int(t_data))
          elif self.rules[self.items_name[cnt]] == 2:
            self.items[self.items_name[cnt]].append(float(t_data))
          elif self.rules[self.items_name[cnt]] == 3:
            self.items[self.items_name[cnt]].append(t_data)
          cnt += 1
        self.num_cnt += 1

  def __rule_fixedseat(self, i_file_to_read):
    while True:   # Extract data
        lines = i_file_to_read.readline() # 整行读取数据
        if not lines:
          break
        if lines[0]=='#':
          continue
        if lines.strip()=='':
          continue
        t_line_elements = lines.split()
        if t_line_elements[1] == '9': # Update end flag
          self.flg_end = t_line_elements[0]
          continue
        if len(t_line_elements) != FIXEDSEAT_FORMAT_NUM:
          raise Exception('extract_rule: Too many elements in one line!')
          break
        t_range_elements = t_line_elements[2].split('-')
        self.rules[t_line_elements[0]]=(int(t_line_elements[1]), int(t_range_elements[0]), int(t_range_elements[1]))
        self.items_name.append(t_line_elements[0])

    for v_names in self.items_name: # Initiallize items
      self.items[v_names]=[]
    self.__fixedseat()

  ''' Extracting data through setting data in a fixed location'''
  def __fixedseat(self):
    with open(self.filename, 'r') as file_to_read:
      t_cnt = 0
      while t_cnt < self.section:
        lines = file_to_read.readline()
        if not lines:
          break
        if lines[0]=='#':
          continue
        if lines[0:len(self.flg_end)] == self.flg_end:
          t_cnt += 1

      while True:
        lines = file_to_read.readline()
        if not lines:
          break
        if lines[0]=='#':
          continue
        if lines[0:len(self.flg_end)] == self.flg_end:
          break

        for t_rule, t_value in self.rules.items():
          tt_value = lines[t_value[1]-1:t_value[2]].strip() 
          if t_value[0] == 1:
            if tt_value is '':
              self.items[t_rule].append(0)
            else:
              self.items[t_rule].append(int(tt_value))
          elif t_value[0] == 2:
            if tt_value is '':
              self.items[t_rule].append(0)
            else:
              self.items[t_rule].append(float(tt_value))
          elif t_value[0] == 3:
            self.items[t_rule].append(tt_value)
        self.num_cnt += 1

  def __extract_rule(self):
    with open(self.rulepath, 'r') as file_to_read:
      while True:   # Determine which format to extract
        lines = file_to_read.readline()
        if not lines:
          raise Exception('extract_rule: No file format!')
          return
        if lines[0]=='#':
          continue
        t_line_elements = lines.split()
        if len(t_line_elements) != FILE_FORMAT_NUM:
          raise Exception('extract_rule: Wrong file format!')
          return
        if self.__rules(t_line_elements[0], file_to_read) is None:
          raise Exception('extract_rule: Unkown format!')
          return
        break
    file_to_read.close()
    return

  def get_items(self, index):
    if type(index) is type([]):
      if len(index) != 2:
        raise Exception('get_items: Wrong list elements!')
      res = []
      for i in range(index[0], index[1]):
        res.append(self.items[self.items_name[i]])
      return res
    else:
      return self.items[self.items_name[index]]

  def get_item(self, index, iindex):
    return self.items[self.items_name[index]][iindex]

  def s_print(self, code):
    print(self.items_name[code])
    print(self.items[self.items_name[code]])
#    print(sum(self.items[self.items_name[code]]))