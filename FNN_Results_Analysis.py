
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')


# ### 1. 导入包，定义全局变量

# In[2]:


import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ### 2. 数据集的导入与导出

# #### P.S. 导入数据集的代码在这里哦

# In[3]:


# 每次录入计算结果时，要先把之前已经录入好的 FNN_Results_df 给导进来
FNN_Results_df = pd.read_pickle('Results_dataframe/2019-4-16')


# In[10]:


FNN_Results_df


# #### P.S. 有新训练得到的模型结果的话，从这里开始导入数据集

# In[11]:


columns=['dianwei','k','n','m','hidden_layers','hidden_dim',
         'activation','learning_rate','batch_size','train_loss','validate_mse']


# In[5]:


import os
path = 'Results\MSEs/2019-4-16'
files = os.listdir(path)


# In[12]:


for file in files:
    if file.endswith('.txt'):
        txt = np.loadtxt(path+'/'+file, dtype=str, delimiter=',')
        rows = txt.shape[0]
        # 将文件名拆分为 [dianwei, k, n ,m] 列表
        filename = file
        filename = filename.strip('.txt').split('-')
        # 将 [dianwei, k, n ,m] 列表复制为列加入 txt array
        filename_array = np.tile(filename,(rows,1))
        # 注意 np.c_[a,b] 方法扩充 array 的使用
        txt_new = np.c_[filename_array,txt]
        df = pd.DataFrame(txt_new, columns=columns)
        FNN_Results_df = FNN_Results_df.append(df)


# #### 导出保存：所有工况的模型需分批算完，因此每次得到的结果用上述代码添加到 FNN_Results_df 这个 dataframe 中，存储并以日期命名

# In[13]:


# 存储的版本未指定 dianwei 为 index 列，以及未指定各列数据类型，这样方便后续继续添加
FNN_Results_df.to_pickle('Results_dataframe/2019-4-16')


# #### P.S. 为了小论文作图，这里我们把中文的点位名称都替换为英文吧

# In[4]:


dianwei_list = FNN_Results_df['dianwei'].unique()
station_list = ['Hefei','Chaohu','Bengbu','Wuzhou','Guilin','Suzhou','Jiyuan','Danjiangkou']
dianwei_list


# In[5]:


for dianwei,station in zip(dianwei_list,station_list):
    FNN_Results_df['dianwei'] = FNN_Results_df['dianwei'].replace(dianwei,station)


# In[6]:


# 将 dianwei 设为 index 列
FNN_Results_df.set_index(['dianwei'], inplace=True)


# In[7]:


# 为各列指定数据类型
FNN_Results_df[['k','n','m','hidden_layers','hidden_dim','batch_size']] = FNN_Results_df[['k','n','m','hidden_layers','hidden_dim','batch_size']].astype('int')
FNN_Results_df[['learning_rate','train_loss','validate_mse']] = FNN_Results_df[['learning_rate','train_loss','validate_mse']].astype('float')


# In[9]:


# 存储的版本为最终进行数据分析的版本
FNN_Results_df.to_pickle('Results_dataframe/sites_Results_dataframe')


# ### 3. 数据分析之概述

# In[3]:


# 读取数据
FNN_Results_df = pd.read_pickle('Results_dataframe/sites_Results_dataframe')


# In[77]:


FNN_Results_df.to_csv('Results/Results.csv')


# In[76]:


FNN_Results_df.describe()


# #### 从 describe 看，train_loss 和 validate_mse 均存在极端异常值，所以考虑一下是否剔除
# -  按照箱线图中默认的异常值检测方法：四分位距IQR法   
# IQR = 75% - 25%   
# Uplimit = 75% + 1.5 * IQR     
# 
# - 经过试验，认为将上下界设置为 [5，95] 的百分位数会比较可信

# ### P.S. 准备好下面我们开始作图！

# In[4]:


# 使正常显示负号
mpl.rcParams['axes.unicode_minus']=False
sns.set_style('whitegrid')
# 直接在 jupyter notebook 的console中生成图形
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('default')
# 箱线图的 color 提前安排一下
color = dict(boxes='DarkGreen',whiskers='DarkOrange',medians='DarkBlue',caps='Gray')


# In[5]:


# 为作图重命名dataframe的列名
FNN_Results_df.columns = ['k','n','m','layers','neurons',
         'activation','lr','batch size','train','validate']


# In[11]:


FNN_Results_df


# #### 筛选train_MSE小于0.01的模型，看超参数分布，再从其中筛选和validate_MSE小于0.01的模型

# In[9]:


train_FNN_Results_df = new_FNN_Results_df[new_FNN_Results_df['train'] <= 0.01]
validate_train_FNN_Results_df = train_FNN_Results_df[train_FNN_Results_df['validate'] <= 0.01]


# In[11]:


validate_train_FNN_Results_df.describe()


# In[12]:


train_FNN_Results_df.describe()


# In[13]:


dianwei_list = FNN_Results_df.index.unique()
k_list = FNN_Results_df['k'].unique()
n_list = FNN_Results_df['n'].unique()
m_list = [1,2,4]
hidden_layers_list = FNN_Results_df['layers'].unique()
hidden_dim_list = FNN_Results_df['neurons'].unique()
activation_list = FNN_Results_df['activation'].unique()
learning_rate_list = FNN_Results_df['lr'].unique()
batch_size_list = FNN_Results_df['batch size'].unique()


# In[45]:


train_FNN_Results_df[train_FNN_Results_df['k'] == 10].shape[0]


# In[14]:


# 统计各个参数值出现的频次
def freq_of_param(parameter,param_list):
    train_list = []
    validate_list = []
    for a in param_list:
        train_list.append(train_FNN_Results_df[train_FNN_Results_df[parameter] == a].shape[0])
        validate_list.append(validate_train_FNN_Results_df[validate_train_FNN_Results_df[parameter] == a].shape[0])
    freq_dataframe = pd.DataFrame({'train':train_list,
                                  'validate':validate_list},
                                 index=param_list)
    return freq_dataframe


# In[15]:


for param,param_list,xlabel in zip(['k','n','m','layers','neurons','activation','lr','batch size'],
                           [k_list,n_list,m_list,hidden_layers_list,hidden_dim_list,activation_list,[0.001,0.003,0.01,0.03],batch_size_list],
                                  ['sample size','n','m','hidden layers','neurons','activation function','learning rate','batch size']):
    freq_dataframe = freq_of_param(param,param_list)
    freq_dataframe.plot.bar(figsize=(4,3),rot=0)
    plt.grid(linestyle='--',linewidth=0.4)
    ax = plt.gca()
    ax.set_xlabel(xlabel,fontsize=12)
    ax.set_ylabel('Frequency',fontsize=12)
    plt.savefig('freq_of_'+str(param)+'.png',bbox_inches = 'tight',dpi=300)


# ##### 先来一个点位分布的图

# ##### 箱线图及频率分布图不受异常值影响，可以反映数据整体情况

# - 设置 showfliers=False 表示不显示离群值
# - 设置 whis=[5,95] 表示离群值的范围为 [5,95] 以外的点，如果不设置的话，默认 whis=1.5，就是1.5倍IQR的意思

# In[24]:


FNN_Results_df[['train','validate']].plot.box(figsize=(6,4),showfliers=False,color=color, whis=[5,95])
plt.grid(linestyle='--',linewidth=0.4,axis='y')
plt.xticks(fontsize=14, fontweight='normal') 

plt.savefig('1.png',bbox_inches = 'tight',dpi=72)


# In[24]:


FNN_Results_df[['train','validate']].plot.hist(figsize=(6,4),bins=500,range=[0,0.5],histtype='step')
plt.grid(linestyle='--',linewidth=0.4)
ax = plt.gca()
ax.set_xlabel('MSE',fontsize=12)
ax.set_ylabel('Frequency',fontsize=12)
plt.savefig('hist.png',bbox_inches = 'tight',dpi=300)


# - 0.2 处峰值产生的原因——学习率 > 0.3
# #### 因此，我们把 lr=0.3的数据去掉以后再分析

# In[6]:


new_FNN_Results_df = FNN_Results_df[FNN_Results_df['lr'] != 0.3]


# In[7]:


# 再试一下去掉 lr=0.1 的
new_FNN_Results_df = new_FNN_Results_df[new_FNN_Results_df['lr'] != 0.1]


# In[8]:


new_FNN_Results_df.describe()


# In[73]:


new_FNN_Results_df.to_excel('Results/new_Results.xlsx')


# In[27]:


new_FNN_Results_df[['train','validate']].plot.hist(figsize=(6,4),bins=500,range=[0,0.5],histtype='step')
plt.grid(linestyle='--',linewidth=0.4)
ax = plt.gca()
ax.set_xlabel('MSE',fontsize=12)
ax.set_ylabel('Frequency',fontsize=12)
plt.savefig('new_hist.png',bbox_inches = 'tight',dpi=300)


# ### 4. 可视化数据分析
# ### 4.1 按 dianwei, k, n, m, hidden_layers, hidden_dim, activation, learning_rate, batch_size 分析

# In[12]:


dianwei_list = FNN_Results_df.index.unique()
k_list = FNN_Results_df['k'].unique()
n_list = FNN_Results_df['n'].unique()
m_list = [1,2,4]
hidden_layers_list = FNN_Results_df['layers'].unique()
hidden_dim_list = FNN_Results_df['neurons'].unique()
activation_list = FNN_Results_df['activation'].unique()
learning_rate_list = FNN_Results_df['lr'].unique()
batch_size_list = FNN_Results_df['batch size'].unique()


# #### 4.1.1 关键函数 slice 的定义，用于对整个 dataframe 进行切片分析

# In[17]:


# 需要注意的是：这个函数的输入参数必须是 list，比如 dianwei=['安徽巢湖裕溪口'],k=[250]
def slice(dianwei=dianwei_list,
         k=k_list,
         n=n_list,
         m=m_list,
         hidden_layers=hidden_layers_list,
         hidden_dim=hidden_dim_list,
         activation=activation_list,
         learning_rate=learning_rate_list,
         batch_size=batch_size_list):
    sub_df = FNN_Results_df[
        (FNN_Results_df.index.isin(dianwei)) &
        (FNN_Results_df['k'].isin(k)) &
        (FNN_Results_df['n'].isin(n)) &
        (FNN_Results_df['m'].isin(m)) &
        (FNN_Results_df['layers'].isin(hidden_layers)) &
        (FNN_Results_df['neurons'].isin(hidden_dim)) &
        (FNN_Results_df['activation'].isin(activation)) &
        (FNN_Results_df['lr'].isin(learning_rate)) &
        (FNN_Results_df['batch size'].isin(batch_size))
    ][['train','validate']]
    return sub_df


# In[30]:


# 需要注意的是：这个函数的输入参数必须是 list，比如 dianwei=['安徽巢湖裕溪口'],k=[250]
def new_slice(dianwei=dianwei_list,
         k=k_list,
         n=n_list,
         m=m_list,
         hidden_layers=hidden_layers_list,
         hidden_dim=hidden_dim_list,
         activation=activation_list,
         learning_rate=[0.001,0.003,0.01,0.03],
         batch_size=batch_size_list):
    sub_df = new_FNN_Results_df[
        (new_FNN_Results_df.index.isin(dianwei)) &
        (new_FNN_Results_df['k'].isin(k)) &
        (new_FNN_Results_df['n'].isin(n)) &
        (new_FNN_Results_df['m'].isin(m)) &
        (new_FNN_Results_df['layers'].isin(hidden_layers)) &
        (new_FNN_Results_df['neurons'].isin(hidden_dim)) &
        (new_FNN_Results_df['activation'].isin(activation)) &
        (new_FNN_Results_df['lr'].isin(learning_rate)) &
        (new_FNN_Results_df['batch size'].isin(batch_size))
    ][['train','validate']]
    return sub_df


# #### 4.1.2 按照 dianwei 作图

# In[58]:


l = len(dianwei_list)
fig,axes = plt.subplots(1,l,figsize=(10,4))
for i,dianwei in enumerate(dianwei_list):
    slice(dianwei=[dianwei]).plot.box(showfliers=False,ylim=[0,0.6],ax=axes[i],color = color,widths=0.4, whis=[5,95])
    axes[i].grid(linestyle='--',linewidth=0.4,axis='y')
    axes[i].set_title(dianwei,fontsize=12)
    axes[i].set_xticklabels(['Tr','Vd'])
    
    # 除了第一个图显示y刻度，其他均不显示
    if i == (l-1): break
    axes[i+1].set_yticklabels([])
    axes[i+1].tick_params(axis='y',width=0)
    
axes[0].set_ylabel('MSE',fontsize=12)
plt.savefig('dianwei_box.png',bbox_inches = 'tight',dpi=300)


# In[57]:


l = len(dianwei_list)
fig,axes = plt.subplots(1,l,figsize=(10,4))
for i,dianwei in enumerate(dianwei_list):
    new_slice(dianwei=[dianwei]).plot.box(showfliers=False,ylim=[0,0.17],ax=axes[i],color = color,widths=0.4, whis=[5,95])
    axes[i].grid(linestyle='--',linewidth=0.4,axis='y')
    axes[i].set_title(dianwei,fontsize=12)
    axes[i].set_xticklabels(['Tr','Vd'])
    
    # 除了第一个图显示y刻度，其他均不显示
    if i == (l-1): break
    axes[i+1].set_yticklabels([])
    axes[i+1].tick_params(axis='y',width=0)
    
axes[0].set_ylabel('MSE',fontsize=12)
plt.savefig('new_dianwei_box.png',bbox_inches = 'tight',dpi=300)


# In[59]:


plt.figure(figsize=(6,4))
for i,dianwei in enumerate(dianwei_list):
    slice(dianwei=[dianwei])['validate'].plot.hist(label=dianwei,density=True,bins=500,range=[0,0.16],histtype='step')
plt.legend(loc='upper right')


# #### 比较 median 和 std   
# 
# - 由于离群值的存在，因此求 mean 好像没啥意义，而 median 可以通过箱线图来反映，因此好像没必要作这组柱状图来比较，所以就先不作了吧

# In[28]:


for dianwei in station_list:
    print(new_slice(dianwei=[dianwei],k=[500],m=[1])['validate'].median())


# In[29]:


for dianwei in station_list:
    print(new_slice(dianwei=[dianwei],k=[500],m=[1])['validate'].min())


# In[64]:


columns=['dianwei','median_train','median_validate','std_train','std_validate']
index=dianwei_list
dianwei_median_std_df = pd.DataFrame(columns=columns)
for dianwei in dianwei_list:
    l = [dianwei]
    l.extend(slice(dianwei=[dianwei]).median().tolist())
    l.extend(slice(dianwei=[dianwei]).std().tolist())
    df = pd.DataFrame([l],columns=columns)
    dianwei_median_std_df = dianwei_median_std_df.append(df)
dianwei_median_std_df.set_index(['dianwei'],inplace=True)

print(dianwei_median_std_df)


# #### 4.1.3 按照 k 值作图

# In[26]:


for k in k_list:
    print(new_slice(k=[k]).median())


# In[37]:


fig,axes = plt.subplots(1,3,figsize=(4,4))

slice(k=[100]).plot.box(showfliers=False,ylim=[0,0.5],ax=axes[0],color = color,widths=0.5, whis=[5,95])
axes[0].grid(linestyle='--',linewidth=0.4,axis='y')
axes[0].set_title('k=100',fontsize=12)
axes[0].set_xticklabels(['Tr','Vd'])

slice(k=[250]).plot.box(showfliers=False,ylim=[0,0.5],ax=axes[1],color = color,widths=0.5, whis=[5,95])
axes[1].grid(linestyle='--',linewidth=0.4,axis='y')
axes[1].set_title('250',fontsize=12)
axes[1].set_yticklabels([])
axes[1].set_xticklabels(['Tr','Vd'])
axes[1].tick_params(axis='y',width=0)

slice(k=[500]).plot.box(showfliers=False,ylim=[0,0.5],ax=axes[2],color = color,widths=0.5, whis=[5,95])
axes[2].grid(linestyle='--',linewidth=0.4,axis='y')
axes[2].set_title('500',fontsize=12)
axes[2].set_yticklabels([])
axes[2].set_xticklabels(['Tr','Vd'])
axes[2].tick_params(axis='y',width=0)

axes[0].set_ylabel('MSE',fontsize=12)
plt.savefig('k_box.png',bbox_inches = 'tight',dpi=300)


# In[62]:


fig,axes = plt.subplots(1,3,figsize=(4,4))

new_slice(k=[100]).plot.box(showfliers=False,ylim=[0,0.16],ax=axes[0],color = color,widths=0.5, whis=[5,95])
axes[0].grid(linestyle='--',linewidth=0.4,axis='y')
axes[0].set_title('k=100',fontsize=12)
axes[0].set_xticklabels(['Tr','Vd'])

new_slice(k=[250]).plot.box(showfliers=False,ylim=[0,0.16],ax=axes[1],color = color,widths=0.5, whis=[5,95])
axes[1].grid(linestyle='--',linewidth=0.4,axis='y')
axes[1].set_title('250',fontsize=12)
axes[1].set_yticklabels([])
axes[1].set_xticklabels(['Tr','Vd'])
axes[1].tick_params(axis='y',width=0)

new_slice(k=[500]).plot.box(showfliers=False,ylim=[0,0.16],ax=axes[2],color = color,widths=0.5, whis=[5,95])
axes[2].grid(linestyle='--',linewidth=0.4,axis='y')
axes[2].set_title('500',fontsize=12)
axes[2].set_yticklabels([])
axes[2].set_xticklabels(['Tr','Vd'])
axes[2].tick_params(axis='y',width=0)

axes[0].set_ylabel('MSE',fontsize=12)
plt.savefig('new_k_box.png',bbox_inches = 'tight',dpi=300)


# In[66]:


plt.figure(figsize=(6,4))
slice(k=[100])['validate'].plot.hist(label=100,density=True,bins=1000,range=[0,0.15],histtype='step')
slice(k=[250])['validate'].plot.hist(label=250,density=True,bins=1000,range=[0,0.15],histtype='step')
slice(k=[500])['validate'].plot.hist(label=500,density=True,bins=1000,range=[0,0.15],histtype='step')
plt.grid(linestyle='--',linewidth=0.4)
plt.legend(loc='upper right')
plt.xlabel('MSE of validation',fontsize=12)
plt.ylabel('Frequency',fontsize=12)
plt.savefig('k_hist.png',bbox_inches = 'tight',dpi=300)


# In[32]:


plt.figure(figsize=(6,4))
new_slice(k=[100])['validate'].plot.hist(label=100,bins=1000,range=[0,0.15],histtype='step')
new_slice(k=[250])['validate'].plot.hist(label=250,bins=1000,range=[0,0.15],histtype='step')
new_slice(k=[500])['validate'].plot.hist(label=500,bins=1000,range=[0,0.15],histtype='step')
plt.grid(linestyle='--',linewidth=0.4)
plt.legend(loc='upper right')
plt.xlabel('Validation MSE',fontsize=12)
plt.ylabel('Frequency',fontsize=12)
plt.savefig('new_k_hist.png',bbox_inches = 'tight',dpi=300)


# #### 4.1.4 按照 n 值作图

# In[207]:


fig,axes = plt.subplots(1,4,figsize=(5,4))
new_slice(n=[1]).plot.box(showfliers=False,ylim=[0,0.14],ax=axes[0],color = color,widths=0.5, whis=[5,95])
axes[0].grid(linestyle='--',linewidth=0.4,axis='y')
axes[0].set_title('n=1',fontsize=12)
axes[0].set_xticklabels(['Tr','Vd'])

new_slice(n=[2]).plot.box(showfliers=False,ylim=[0,0.14],ax=axes[1],color = color,widths=0.5, whis=[5,95])
axes[1].set_yticklabels([])
axes[1].grid(linestyle='--',linewidth=0.4,axis='y')
axes[1].set_title('2',fontsize=12)
axes[1].set_xticklabels(['Tr','Vd'])
axes[1].tick_params(axis='y',width=0)

new_slice(n=[3]).plot.box(showfliers=False,ylim=[0,0.14],ax=axes[2],color = color,widths=0.5, whis=[5,95])
axes[2].set_yticklabels([])
axes[2].grid(linestyle='--',linewidth=0.4,axis='y')
axes[2].set_title('3',fontsize=12)
axes[2].set_xticklabels(['Tr','Vd'])
axes[2].tick_params(axis='y',width=0)

new_slice(n=[4]).plot.box(showfliers=False,ylim=[0,0.14],ax=axes[3],color = color,widths=0.5, whis=[5,95])
axes[3].set_yticklabels([])
axes[3].grid(linestyle='--',linewidth=0.4,axis='y')
axes[3].set_title('4',fontsize=12)
axes[3].set_xticklabels(['Tr','Vd'])
axes[3].tick_params(axis='y',width=0)

axes[0].set_ylabel('MSE',fontsize=12)
plt.savefig('new_n_box.png',bbox_inches = 'tight',dpi=300)


# In[76]:


plt.figure(figsize=(6,4))
for i,n in enumerate(n_list):
    new_slice(n=[n])['validate'].plot.hist(label=n,density=True,bins=800,range=[0,0.15],histtype='step')
plt.grid(linestyle='--',linewidth=0.4)
plt.legend(loc='upper right')
plt.xlabel('Validation MSE',fontsize=12)
plt.ylabel('Frequency',fontsize=12)
plt.savefig('new_n_hist.png',bbox_inches = 'tight',dpi=300)


# In[77]:


columns=['n','mean_train','mean_validate','std_train','std_validate']
index=n_list
n_mean_std_df = pd.DataFrame(columns=columns)
for n in n_list:
    l = [n]
    l.extend(slice(n=[n]).mean().tolist())
    l.extend(slice(n=[n]).std().tolist())
    df = pd.DataFrame([l],columns=columns)
    n_mean_std_df = n_mean_std_df.append(df)
n_mean_std_df.set_index(['n'],inplace=True)

print(n_mean_std_df)
n_mean_std_df.plot.bar()


# #### 4.1.5 按照 m 值作图

# In[208]:


fig,axes = plt.subplots(1,3,figsize=(4,4))
new_slice(m=[1]).plot.box(showfliers=False,ylim=[0,0.14],ax=axes[0],color = color,widths=0.5, whis=[5,95])
axes[0].grid(linestyle='--',linewidth=0.4,axis='y')
axes[0].set_title('m=1',fontsize=12)
axes[0].set_xticklabels(['Tr','Vd'])

axes[1].set_yticklabels([])
new_slice(m=[2]).plot.box(showfliers=False,ylim=[0,0.14],ax=axes[1],color = color,widths=0.5, whis=[5,95])
axes[1].grid(linestyle='--',linewidth=0.4,axis='y')
axes[1].set_title('2',fontsize=12)
axes[1].set_xticklabels(['Tr','Vd'])
axes[1].tick_params(axis='y',width=0)

axes[2].set_yticklabels([])
new_slice(m=[4]).plot.box(showfliers=False,ylim=[0,0.14],ax=axes[2],color = color,widths=0.5, whis=[5,95])
axes[2].grid(linestyle='--',linewidth=0.4,axis='y')
axes[2].set_title('4',fontsize=12)
axes[2].set_xticklabels(['Tr','Vd'])
axes[2].tick_params(axis='y',width=0)

axes[0].set_ylabel('MSE',fontsize=12)
plt.savefig('new_m_box.png',bbox_inches = 'tight',dpi=300)


# In[31]:


plt.figure(figsize=(6,4))
for i,m in enumerate(m_list):
    new_slice(m=[m])['validate'].plot.hist(label=m,bins=1000,range=[0,0.15],histtype='step')
plt.legend(loc='upper right')
plt.grid(linestyle='--',linewidth=0.4)
plt.legend(loc='upper right')
plt.xlabel('Validation MSE',fontsize=12)
plt.ylabel('Frequency',fontsize=12)
plt.savefig('new_m_hist.png',bbox_inches = 'tight',dpi=300)


# In[25]:


for m in m_list:
    print(new_slice(m=[m])['validate'].median())


# In[162]:


columns=['m','mean_train','mean_validate','std_train','std_validate']
index=m_list
m_mean_std_df = pd.DataFrame(columns=columns)
for m in m_list:
    l = [m]
    l.extend(new_slice(m=[m]).median().tolist())
    l.extend(new_slice(m=[m]).std().tolist())
    df = pd.DataFrame([l],columns=columns)
    m_mean_std_df = m_mean_std_df.append(df)
m_mean_std_df.set_index(['m'],inplace=True)

print(m_mean_std_df)
m_mean_std_df.plot.bar()


# #### 4.1.5 按照 hidden_layers 值作图

# In[84]:


l = len(hidden_layers_list)
fig,axes = plt.subplots(1,l,figsize=(10,4))
# 设置子图间距
# plt.subplots_adjust(wspace=0, hspace=0)

for i,hidden_layers in enumerate(hidden_layers_list):
    new_slice(hidden_layers=[hidden_layers]).plot.box(showfliers=False, ylim=[0,0.17],ax=axes[i],color = color,widths=0.4, whis=[5,95])
    axes[i].grid(linestyle='--',linewidth=0.4,axis='y')
    axes[i].set_title(hidden_layers)
    axes[i].set_xticklabels(['Tr','Vd'])
    
    # 除了第一个图显示y刻度，其他均不显示
    if i == (l-1): break
    axes[i+1].set_yticklabels([])
    axes[i+1].tick_params(axis='y',width=0)

axes[0].set_title('layers=1',fontsize=12)
axes[0].set_ylabel('MSE',fontsize=12)
plt.savefig('new_layers_box.png',bbox_inches = 'tight',dpi=300)


# In[85]:


plt.figure(figsize=(6,4))
for i,hidden_layers in enumerate(hidden_layers_list):
    new_slice(hidden_layers=[hidden_layers])['validate'].plot.hist(label=hidden_layers,density=True,bins=1000,range=[0,0.16],histtype='step')
plt.legend(loc='upper right')
plt.grid(linestyle='--',linewidth=0.4)
plt.xlabel('MSE',fontsize=12)
plt.ylabel('Frequency',fontsize=12)
plt.savefig('layers_hist.png',bbox_inches = 'tight',dpi=300)


# In[86]:


columns=['hidden_layers','mean_train','mean_validate','std_train','std_validate']
index=hidden_layers_list
hidden_layers_mean_std_df = pd.DataFrame(columns=columns)
for hidden_layers in hidden_layers_list:
    l = [hidden_layers]
    l.extend(new_slice(hidden_layers=[hidden_layers]).mean().tolist())
    l.extend(new_slice(hidden_layers=[hidden_layers]).std().tolist())
    df = pd.DataFrame([l],columns=columns)
    hidden_layers_mean_std_df = hidden_layers_mean_std_df.append(df)
hidden_layers_mean_std_df.set_index(['hidden_layers'],inplace=True)

print(hidden_layers_mean_std_df)
hidden_layers_mean_std_df.plot.bar()


# #### 4.1.6 按照 hidden_dim 值作图

# In[93]:


l = len(hidden_dim_list)
fig,axes = plt.subplots(1,l,figsize=(10,4))
for i,hidden_dim in enumerate(hidden_dim_list):
    new_slice(hidden_dim=[hidden_dim]).plot.box(showfliers=False, ylim=[0,0.21],ax=axes[i],color = color,widths=0.4, whis=[5,95])
    axes[i].grid(linestyle='--',linewidth=0.4,axis='y')
    axes[i].set_title(hidden_dim)
    axes[i].set_xticklabels(['Tr','Vd'])
    
    if i == (l-1): break
    axes[i+1].set_yticklabels([])
    axes[i+1].tick_params(axis='y',width=0)
    
axes[0].set_title('neurons=4',fontsize=12)
axes[0].set_ylabel('MSE',fontsize=12)
plt.savefig('new_neurons_box.png',bbox_inches = 'tight',dpi=300)


# In[94]:


plt.figure(figsize=(16,6))
for i,hidden_dim in enumerate(hidden_dim_list):
    new_slice(hidden_dim=[hidden_dim])['validate'].plot.hist(label=hidden_dim,density=True,bins=800,range=[0,0.1],histtype='step')
plt.legend(loc='upper right')


# In[95]:


columns=['hidden_dim','mean_train','mean_validate','std_train','std_validate']
index=hidden_dim_list
hidden_dim_mean_std_df = pd.DataFrame(columns=columns)
for hidden_dim in hidden_dim_list:
    l = [hidden_dim]
    l.extend(new_slice(hidden_dim=[hidden_dim]).median().tolist())
    l.extend(new_slice(hidden_dim=[hidden_dim]).std().tolist())
    df = pd.DataFrame([l],columns=columns)
    hidden_dim_mean_std_df = hidden_dim_mean_std_df.append(df)
hidden_dim_mean_std_df.set_index(['hidden_dim'],inplace=True)

print(hidden_dim_mean_std_df)
hidden_dim_mean_std_df.plot.bar()


# #### 4.1.7 按照 activation 值作图

# In[19]:


fig,axes = plt.subplots(1,2,figsize=(3,4))
new_slice(activation=['tanh']).plot.box(showfliers=False, ylim=[0,0.17],ax=axes[0],color = color,widths=0.5, whis=[5,95])
axes[0].grid(linestyle='--',linewidth=0.4,axis='y')
axes[0].set_title('Tanh')
axes[0].set_xticklabels(['Tr','Vd'])

axes[1].set_yticklabels([])
new_slice(activation=['relu']).plot.box(showfliers=False, ylim=[0,0.17],ax=axes[1],color = color,widths=0.5, whis=[5,95])
axes[1].grid(linestyle='--',linewidth=0.4,axis='y')
axes[1].set_title('ReLU')
axes[1].set_xticklabels(['Tr','Vd'])
axes[1].tick_params(axis='y',width=0)

axes[0].set_ylabel('MSE',fontsize=12)
plt.savefig('new_activation_box.png',bbox_inches = 'tight',dpi=300)


# In[20]:


plt.figure(figsize=(6,4))
new_slice(activation=['tanh'])['validate'].plot.hist(label="Tanh",density=True,bins=500,range=[0,0.3],histtype='step')
new_slice(activation=['relu'])['validate'].plot.hist(label="ReLU",density=True,bins=500,range=[0,0.3],histtype='step')
plt.legend(loc='upper right')
plt.grid(linestyle='--',linewidth=0.4)
plt.xlabel('Validation MSE',fontsize=12)
plt.ylabel('Frequency',fontsize=12)
plt.savefig('new_activation_hist.png',bbox_inches = 'tight',dpi=300)


# #### 4.1.8 按照 learning_rate 值作图

# In[116]:


l = len(learning_rate_list)
fig,axes = plt.subplots(1,l,figsize=(8,4))
for i,learning_rate in enumerate(learning_rate_list):
    slice(learning_rate=[learning_rate]).plot.box(showfliers=False, ylim=[0,0.5],ax=axes[i],color = color,widths=0.4, whis=[5,95])
    axes[i].grid(linestyle='--',linewidth=0.4,axis='y')
    axes[i].set_title(learning_rate)
    axes[i].set_xticklabels(['Tr','Vd'])
    
    if i == (l-1): break
    axes[i+1].set_yticklabels([])
    axes[i+1].tick_params(axis='y',width=0)
    
axes[0].set_title('lr=0.001',fontsize=12)
axes[0].set_ylabel('MSE',fontsize=12)
plt.savefig('lr_box.png',bbox_inches = 'tight',dpi=300)


# In[113]:


l = len([0.001,0.003,0.01,0.03])
fig,axes = plt.subplots(1,l,figsize=(5,4))
for i,learning_rate in enumerate([0.001,0.003,0.01,0.03]):
    new_slice(learning_rate=[learning_rate]).plot.box(showfliers=False, ylim=[0,0.15],ax=axes[i],color = color,widths=0.4, whis=[5,95])
    axes[i].grid(linestyle='--',linewidth=0.4,axis='y')
    axes[i].set_title(learning_rate)
    axes[i].set_xticklabels(['Tr','Vd'])
    
    if i == (l-1): break
    axes[i+1].set_yticklabels([])
    axes[i+1].tick_params(axis='y',width=0)
    
axes[0].set_title('lr=0.001',fontsize=12)
axes[0].set_ylabel('MSE',fontsize=12)
plt.savefig('new_lr_box.png',bbox_inches = 'tight',dpi=300)


# In[121]:


plt.figure(figsize=(6,4))
for i,learning_rate in enumerate(learning_rate_list):
    slice(learning_rate=[learning_rate])['validate'].plot.hist(label=learning_rate,density=True,bins=500,range=[0,0.5],histtype='step')
plt.legend(loc='upper right')
plt.grid(linestyle='--',linewidth=0.4)
plt.xlabel('Validation MSE',fontsize=12)
plt.ylabel('Frequency',fontsize=12)
plt.savefig('new_lr_hist.png',bbox_inches = 'tight',dpi=300)


# In[110]:


columns=['learning_rate','mean_train','mean_validate','std_train','std_validate']
index=learning_rate_list
learning_rate_mean_std_df = pd.DataFrame(columns=columns)
for learning_rate in learning_rate_list:
    l = [learning_rate]
    l.extend(slice(learning_rate=[learning_rate]).mean().tolist())
    l.extend(slice(learning_rate=[learning_rate]).std().tolist())
    df = pd.DataFrame([l],columns=columns)
    learning_rate_mean_std_df = learning_rate_mean_std_df.append(df)
learning_rate_mean_std_df.set_index(['learning_rate'],inplace=True)

print(learning_rate_mean_std_df)
learning_rate_mean_std_df.plot.bar()


# #### 4.1.9 按照 batch_size 值作图

# In[32]:


l = len(batch_size_list)
fig,axes = plt.subplots(1,l,figsize=(8,4))
for i,batch_size in enumerate(batch_size_list):
    new_slice(batch_size=[batch_size]).plot.box(showfliers=False, ylim=[0,0.15],ax=axes[i],color = color,widths=0.4, whis=[5,95])
    axes[i].grid(linestyle='--',linewidth=0.4,axis='y')
    axes[i].set_title(batch_size)
    axes[i].set_xticklabels(['Tr','Vd'])
    if i == (l-1): break
    axes[i+1].set_yticklabels([])
    axes[i+1].tick_params(axis='y',width=0)
    
axes[0].set_title('batch size=2',fontsize=12)
axes[0].set_ylabel('MSE',fontsize=12)
plt.savefig('new_batch_box.png',bbox_inches = 'tight',dpi=300)


# In[33]:


for batch_size in batch_size_list:
    print(new_slice(batch_size=[batch_size])['validate'].median())


# ## P.S. 综合上述👆及下述👇分析，最优参数组合应为
# - k = 500
# - n = 1
# - m = 1
# - hidden_layers = 1
# - hidden_dim = 12
# - activation = 'tanh'
# - learning_rate = 0.01
# - batch_size = 32

# In[195]:


df = slice(hidden_layers=[1],hidden_dim=[12],activation=['tanh'],learning_rate=[0.01],batch_size=[32])
df.plot.box(color = color,widths=0.4, whis=[5,95])


# In[198]:


df = slice(hidden_dim=[12],activation=['tanh'],learning_rate=[0.01],batch_size=[32])
df.plot.box(color = color,widths=0.4, whis=[5,95])


# In[200]:


df = slice(hidden_layers=[1],activation=['tanh'],learning_rate=[0.01],batch_size=[32])
df.plot.box(color = color,widths=0.4, whis=[5,95])


# In[201]:


df = slice(hidden_layers=[1],hidden_dim=[12],learning_rate=[0.01],batch_size=[32])
df.plot.box(color = color,widths=0.4, whis=[5,95])


# In[197]:


df = slice(activation=['tanh'],learning_rate=[0.01])
df.plot.box(color = color,widths=0.4, whis=[5,95])


# ### 4.2 二维热力图分析
# - 取的是相应slice的中位值 median

# #### 4.2.1 hidden_layers 相关

# In[125]:


layers_dim_heatmap = []
for hidden_layers in hidden_layers_list:
    l = []
    for hidden_dim in hidden_dim_list:
        median = slice(hidden_layers=[hidden_layers], hidden_dim=[hidden_dim])['validate'].median()
        l.append(median)
    layers_dim_heatmap.append(l)
layers_dim_heatmap = np.array(layers_dim_heatmap)


# 分别对每个点位作layers_dim_heatmap

# In[18]:


for dianwei in station_list:
    layers_dim_heatmap = []
    for hidden_layers in hidden_layers_list:
        l = []
        for hidden_dim in hidden_dim_list:
            median = new_slice(dianwei=[dianwei], hidden_layers=[hidden_layers], hidden_dim=[hidden_dim])['validate'].median()
            l.append(median)
        layers_dim_heatmap.append(l)
    layers_dim_heatmap = np.array(layers_dim_heatmap)
    plt.figure(figsize=(5.4,4))
    sns.heatmap(layers_dim_heatmap,
                xticklabels=hidden_dim_list,
                yticklabels=hidden_layers_list,
                vmin=0.0225, vmax=0.04)
    plt.tick_params(width=0)
    plt.ylabel('hidden layers',fontsize=12)
    plt.xlabel('neurons',fontsize=12)
    plt.title('Validation MSE of '+dianwei)

    plt.savefig('layers_dim_heatmap_'+dianwei+'.png',bbox_inches = 'tight',dpi=300)


# In[19]:


new_layers_dim_heatmap = []
for hidden_layers in hidden_layers_list:
    l = []
    for hidden_dim in hidden_dim_list:
        median = new_slice(hidden_layers=[hidden_layers], hidden_dim=[hidden_dim])['validate'].median()
        l.append(median)
    new_layers_dim_heatmap.append(l)
new_layers_dim_heatmap = np.array(new_layers_dim_heatmap)


# In[31]:


new_layers_dianwei_heatmap = []
for hidden_layers in hidden_layers_list:
    l = []
    for dianwei in dianwei_list:
        median = new_slice(hidden_layers=[hidden_layers], dianwei=[dianwei])['validate'].median()
        l.append(median)
    new_layers_dianwei_heatmap.append(l)
new_layers_dianwei_heatmap = np.array(new_layers_dianwei_heatmap)


# In[122]:


layers_k_heatmap = []
for hidden_layers in hidden_layers_list:
    l = []
    for k in k_list:
        median = slice(hidden_layers=[hidden_layers], k=[k])['validate_mse'].median()
        l.append(median)
    layers_k_heatmap.append(l)
layers_k_heatmap = np.array(layers_k_heatmap)


# In[142]:


new_layers_n_heatmap = []
for hidden_layers in hidden_layers_list:
    l = []
    for n in n_list:
        median = new_slice(hidden_layers=[hidden_layers], n=[n])['validate'].median()
        l.append(median)
    new_layers_n_heatmap.append(l)
new_layers_n_heatmap = np.array(new_layers_n_heatmap)


# In[20]:


plt.figure(figsize=(5.4,4))
sns.heatmap(new_layers_dim_heatmap,
            xticklabels=hidden_dim_list,
            yticklabels=hidden_layers_list,
            vmin=0.0225, vmax=0.04)
plt.tick_params(width=0)
plt.ylabel('hidden layers',fontsize=12)
plt.xlabel('neurons',fontsize=12)
plt.title('Validation MSE')

plt.savefig('layers_dim_heatmap.png',bbox_inches = 'tight',dpi=300)


# In[32]:


plt.figure(figsize=(5.4,4))
sns.heatmap(new_layers_dianwei_heatmap,
            xticklabels=dianwei_list,
            yticklabels=hidden_layers_list,
           vmin=0.0225, vmax=0.04)
plt.tick_params(width=0)
plt.ylabel('hidden layers',fontsize=12)
plt.xlabel('Sites',fontsize=12)
plt.title('Validation MSE')

plt.savefig('layers_dianwei_heatmap.png',bbox_inches = 'tight',dpi=300)


# In[131]:


sns.heatmap(layers_k_heatmap,
            xticklabels=k_list,
            yticklabels=hidden_layers_list)


# In[145]:


plt.figure(figsize=(5,4))
sns.heatmap(new_layers_n_heatmap,
            xticklabels=n_list,
            yticklabels=hidden_layers_list)
plt.tick_params(width=0)
plt.ylabel('hidden layers',fontsize=12)
plt.xlabel('n',fontsize=12)
plt.title('Validation MSE')

plt.savefig('layers_n_heatmap.png',bbox_inches = 'tight',dpi=300)


# #### 4.2.2 hidden_dim 相关

# In[133]:


dim_n_heatmap = []
for hidden_dim in hidden_dim_list:
    l = []
    for n in n_list:
        median = slice(hidden_dim=[hidden_dim], n=[n])['validate_mse'].median()
        l.append(median)
    dim_n_heatmap.append(l)
dim_n_heatmap = np.array(dim_n_heatmap)


# In[33]:


new_dim_dianwei_heatmap = []
for hidden_dim in hidden_dim_list:
    l = []
    for dianwei in dianwei_list:
        median = new_slice(hidden_dim=[hidden_dim], dianwei=[dianwei])['validate'].median()
        l.append(median)
    new_dim_dianwei_heatmap.append(l)
new_dim_dianwei_heatmap = np.array(new_dim_dianwei_heatmap)


# In[34]:


plt.figure(figsize=(5.4,4))
sns.heatmap(new_dim_dianwei_heatmap,
            xticklabels=dianwei_list,
            yticklabels=hidden_dim_list,
           vmin=0.0225, vmax=0.04)
plt.tick_params(width=0)
plt.ylabel('neurons',fontsize=12)
plt.xlabel('Sites',fontsize=12)
plt.title('Validation MSE')

plt.savefig('neurons_dianwei_heatmap.png',bbox_inches = 'tight',dpi=300)


# In[23]:


# 画一个m和n的热力图，看会不会预测4周后的话，就是用n=4的输入会比较好？
m_n_heatmap = []
for m in m_list:
    l = []
    for n in n_list:
        median = new_slice(m=[m], n=[n])['validate'].median()
        l.append(median)
    m_n_heatmap.append(l)
m_n_heatmap = np.array(m_n_heatmap)
plt.figure(figsize=(5,4))
sns.heatmap(m_n_heatmap,
            xticklabels=n_list,
            yticklabels=m_list)
plt.tick_params(width=0)
plt.ylabel('m',fontsize=12)
plt.xlabel('n',fontsize=12)
plt.title('Validation MSE')

plt.savefig('m_n_heatmap.png',bbox_inches = 'tight',dpi=300)


# In[135]:


dim_k_heatmap = []
for hidden_dim in hidden_dim_list:
    l = []
    for k in k_list:
        median = slice(hidden_dim=[hidden_dim], k=[k])['validate_mse'].median()
        l.append(median)
    dim_k_heatmap.append(l)
dim_k_heatmap = np.array(dim_k_heatmap)


# In[134]:


sns.heatmap(dim_n_heatmap,
            xticklabels=n_list,
            yticklabels=hidden_dim_list)


# In[136]:


sns.heatmap(dim_k_heatmap,
            xticklabels=k_list,
            yticklabels=hidden_dim_list)


# #### 4.2.3 learning_rate 相关

# In[21]:


new_lr_bs_heatmap = []
for learning_rate in [0.001,0.003,0.01,0.03]:
    l = []
    for batch_size in batch_size_list:
        median = new_slice(learning_rate=[learning_rate], batch_size=[batch_size])['validate'].median()
        l.append(median)
    new_lr_bs_heatmap.append(l)
new_lr_bs_heatmap = np.array(new_lr_bs_heatmap)


# In[22]:


new_lr_layers_heatmap = []
for learning_rate in [0.001,0.003,0.01,0.03]:
    l = []
    for hidden_layers in hidden_layers_list:
        median = new_slice(learning_rate=[learning_rate], hidden_layers=[hidden_layers])['validate'].median()
        l.append(median)
    new_lr_layers_heatmap.append(l)
new_lr_layers_heatmap = np.array(new_lr_layers_heatmap)


# In[23]:


new_lr_dim_heatmap = []
for learning_rate in [0.001,0.003,0.01,0.03]:
    l = []
    for hidden_dim in hidden_dim_list:
        median = new_slice(learning_rate=[learning_rate], hidden_dim=[hidden_dim])['validate'].median()
        l.append(median)
    new_lr_dim_heatmap.append(l)
new_lr_dim_heatmap = np.array(new_lr_dim_heatmap)


# In[24]:


plt.figure(figsize=(5.4,4))
sns.heatmap(new_lr_bs_heatmap,
            xticklabels=batch_size_list,
            yticklabels=[0.001,0.003,0.01,0.03],
            vmin=0.0225, vmax=0.04)
plt.tick_params(width=0)
plt.ylabel('learning rate',fontsize=12)
plt.xlabel('batch size',fontsize=12)
plt.title('Validation MSE')

plt.savefig('lr_bs_heatmap.png',bbox_inches = 'tight',dpi=300)


# - 大 batch size 宜对应较大 lr，趋势并不明显（有相应依据，可百度之）。但是 lr 的进一步增大将导致模型性能迅速下降

# In[25]:


plt.figure(figsize=(5.4,4))
sns.heatmap(new_lr_layers_heatmap,
            xticklabels=hidden_layers_list,
            yticklabels=[0.001,0.003,0.01,0.03],
            vmin=0.0225, vmax=0.04)
plt.tick_params(width=0)
plt.ylabel('learning rate',fontsize=12)
plt.xlabel('hidden layers',fontsize=12)
plt.title('Validation MSE')

plt.savefig('lr_layers_heatmap.png',bbox_inches = 'tight',dpi=300)


# In[26]:


plt.figure(figsize=(5.4,4))
sns.heatmap(new_lr_dim_heatmap,
            xticklabels=hidden_dim_list,
            yticklabels=[0.001,0.003,0.01,0.03],
            vmin=0.0225, vmax=0.04)
plt.tick_params(width=0)
plt.ylabel('learning rate',fontsize=12)
plt.xlabel('neurons',fontsize=12)
plt.title('Validation MSE')

plt.savefig('lr_dim_heatmap.png',bbox_inches = 'tight',dpi=300)


# - 小的 lr 宜对应大的 hidden nodes 值——趋势不是很明显，但单调性较好

# #### 4.2.4 activation 相关

# In[27]:


act_dim_heatmap = []
for activation in activation_list:
    l = []
    for hidden_dim in hidden_dim_list:
        median = new_slice(activation=[activation], hidden_dim=[hidden_dim])['validate'].median()
        l.append(median)
    act_dim_heatmap.append(l)
act_dim_heatmap = np.array(act_dim_heatmap)


# In[30]:


sns.heatmap(act_dim_heatmap,
            xticklabels=hidden_dim_list,
            yticklabels=activation_list,
           vmin=0.0225, vmax=0.04)


# In[168]:


plt.figure(figsize=(6,4))
sns.heatmap(act_dim_heatmap,
            xticklabels=hidden_dim_list,
            yticklabels=activation_list,
            annot=True)


# - 这个很有意思，hidden_dim 和其他变量作图结果基本都是 12 最好，但这里对于 relu 函数，好像是随 nodes 数单调变化的

# In[156]:


# 保存一下上面计算得到的二维数组，以便后续取用
heatmap_2d_array_dict = {'hidden_layers-hidden_dim':layers_dim_heatmap,
                         'hidden_layers-dianwei':layers_dianwei_heatmap,
                         'hidden_layers-k':layers_k_heatmap,
                         'hidden_layers-n':layers_n_heatmap,
                         'hidden_dim-k':dim_k_heatmap,
                         'hidden_dim-n':dim_n_heatmap,
                         'learning_rate-batch_size':lr_bs_heatmap,
                         'learning_rate-hidden_layers':lr_layers_heatmap,
                         'learning_rate-hidden_dim':lr_dim_heatmap,
                         'activation-hidden_layers':act_layers_heatmap,
                         'activation-hidden_dim':act_dim_heatmap}


# In[157]:


np.save('Dictionary_datasets/heatmap_2d_array_dict.npy',heatmap_2d_array_dict)


# ### 4.3 方差分析
# #### 4.3.1 单因素方差分析

# In[16]:


from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# In[22]:


list_parameters = ['validate ~ C(k)','validate ~ C(m)','validate ~ C(n)',
                   'validate ~ C(layers)','validate ~ C(neurons)','validate ~ C(activation)',
                  'validate ~ C(lr)','validate ~ C(batch_size)']


# In[24]:


for parameter in list_parameters:
    eq = ols(parameter,new_FNN_Results_df).fit()
    anov_eq = anova_lm(eq)
    print(anov_eq)


# - df值为自由度
# - sum_sq 为deviance (within groups, and residual)，总方差和（分别有groups和residual的）
# - mean_sq 为variance (within groups, and residual)，平均方差和（分别有groups和residual的）
# - F值为组间均方与组内均方的比值，即F值越大，说明组间变异大
# - PR值越小，越可以拒绝原无差异假设

# In[25]:


print(pairwise_tukeyhsd(new_FNN_Results_df['validate'],new_FNN_Results_df['layers']))


# In[26]:


print(pairwise_tukeyhsd(new_FNN_Results_df['validate'],new_FNN_Results_df['lr']))


# #### 4.3.2 多因素方差分析

# In[27]:


# 重命名列名，主要是把 batch size 的两个单词连起来，要不然没法写方差分析式
new_FNN_Results_df.columns = ['k','n','m','layers','neurons',
         'activation','lr','batch_size','train','validate']


# In[29]:


formula = 'validate ~ C(k) + C(n) + C(m) + C(layers) + C(neurons) + C(activation) + C(lr) + C(batch_size)'
anova_results = anova_lm(ols(formula,new_FNN_Results_df).fit())
print(anova_results)


# In[45]:


formula = 'validate ~ C(layers)*C(neurons)'
anova_results = anova_lm(ols(formula,new_FNN_Results_df).fit())
print(anova_results)


# In[50]:


formula = 'validate ~ C(n)*C(m)*C(layers)'
anova_results = anova_lm(ols(formula,new_FNN_Results_df).fit())
print(anova_results)


# In[51]:


formula = 'validate ~ C(n)*C(m)*C(neurons)'
anova_results = anova_lm(ols(formula,new_FNN_Results_df).fit())
print(anova_results)


# In[46]:


formula = 'validate ~ C(layers)*C(batch_size)'
anova_results = anova_lm(ols(formula,new_FNN_Results_df).fit())
print(anova_results)


# In[49]:


formula = 'validate ~ C(lr)*C(layers)'
anova_results = anova_lm(ols(formula,new_FNN_Results_df).fit())
print(anova_results)


# ### 5 探讨水质时序预测问题的 benchmark
# - t 时刻预测值采用 t-1 时刻真实值，由此算得的 MSE 作为 benchmark，筛选有效模型
# - 以“安徽巢湖裕溪口-250-1-1”数据集为例，在excel中算得 benchmark 为 0.018774

# In[21]:


df = slice(dianwei=['安徽巢湖裕溪口'],k=[250],n=[1],m=[1])


# In[33]:


sub_df = df[df['validate_mse']<0.018774]


# In[34]:


print(df.describe())
print(sub_df.describe())


# - 喏，有效的模型只有 1171 个，仅占比 22%

# ### 6 原始水质时间序列分析

# ### 6.1 频率分布

# In[4]:


dianwei_list = ['安徽合肥湖滨', '安徽巢湖裕溪口', '安徽蚌埠蚌埠闸', '广西梧州界首', '广西桂林阳朔', '江苏苏州西山', '河南济源小浪底', '湖北丹江口胡家岭']


# In[5]:


Date_Filled = np.load('Dictionary_datasets/Date_Filled.npy').item()
NaN_Filled_mean = np.load('Dictionary_datasets/NaN_Filled_mean.npy').item()


# In[236]:


for dianwei in dianwei_list:
    NaN_Filled_mean[dianwei].to_excel(dianwei+'.xlsx')


# In[6]:


site_list = ['Hefei','Chaohu','Bengbu','Wuzhou','Guilin','Suzhou','Jiyuan','Danjiangkou']


# In[272]:


for zhibiao in ['pH','DO','COD','NH3-N']:
    fig,axes = plt.subplots(3,8,figsize=(8,4))
    for i,k in enumerate(k_list):
        for j,dianwei in enumerate(dianwei_list):
            Date_Filled[dianwei][zhibiao][:k].plot.hist(ax=axes[i][j],density=True,bins=50)
            
            axes[i][j].set_ylabel('')
            axes[i][j].set_yticklabels([])
            axes[i][j].tick_params(axis='y',width=0) 
            axes[i][j].set_xlabel('')
            axes[i][j].set_xticklabels([])
            axes[i][j].tick_params(axis='x',width=0)
            
    for a,site in enumerate(site_list):
        axes[0][a].set_title(site,fontsize=12)
    axes[0][0].set_ylabel('k=100',fontsize=12)
    axes[1][0].set_ylabel('250',fontsize=12)
    axes[2][0].set_ylabel('500',fontsize=12)
    plt.savefig(zhibiao,bbox_inches = 'tight',dpi=300)


# In[7]:


from matplotlib.ticker import MultipleLocator, FormatStrFormatter
ymajorFormatter = FormatStrFormatter('%1.1f')

fig,axes = plt.subplots(8,4,figsize=(9,12))
plt.subplots_adjust(wspace=0.28,hspace=0.26)
i_list=range(0,8)
for i,dianwei,site in zip(i_list,dianwei_list,site_list):
    for j,zhibiao in enumerate(['pH',"DO",'COD','NH3-N']):
        Date_Filled[dianwei][zhibiao][:300].plot.hist(ax=axes[i][j],density=True,bins=30)

        axes[i][j].grid(linestyle='--',linewidth=0.4)
        axes[i][j].set_ylabel('')
        axes[i][j].set_xlabel('')
        axes[i][j].yaxis.set_major_formatter(ymajorFormatter)

        if (i==0)&(j==0):
            axes[i][j].set_title(zhibiao,fontsize=12)
        if (i==0)&(j!=0):
            axes[i][j].set_title(zhibiao+u'(mg/L)',fontsize=12)
        if j==0:
            axes[i][j].set_ylabel(site,fontsize=12)
            
plt.savefig('metadata_hist',bbox_inches = 'tight',dpi=300)


# - 上图是分别对数据集大小为[100，250，500]的4个点位的水质指标作频率直方图，看数据分布情况，结果好像也分析不出来为啥中间两个点位模型效果比两边的两个要好？

# #### 序列图

# In[48]:


Date_Filled['安徽巢湖裕溪口']['2004']


# In[324]:


from matplotlib.ticker import MultipleLocator, FormatStrFormatter
ymajorFormatter = FormatStrFormatter('%1.1f')

fig,axes = plt.subplots(8,4,figsize=(16,8))
plt.subplots_adjust(wspace=0.17,hspace=0.2)
i_list=range(0,8)
for i,dianwei,site in zip(i_list,dianwei_list,site_list):
    for j,zhibiao in enumerate(['pH','DO','COD','NH3-N']):
        Date_Filled[dianwei][zhibiao].plot(ax=axes[i][j],linewidth=0.8)

        axes[i][j].grid(linestyle='--',linewidth=0.4)
        axes[i][j].set_ylabel('')
        axes[i][j].set_xlabel('')
        axes[i][j].yaxis.set_major_formatter(ymajorFormatter)
        
        axes[i][j].set_xticklabels([])
        axes[i][j].tick_params(axis='x',width=0)
        
        if (i==0)&(j==0):
            axes[i][j].set_title(zhibiao,fontsize=12)
        if (i==0)&(j!=0):
            axes[i][j].set_title(zhibiao+u'(mg/L)',fontsize=12)
            
        if j==0:
            axes[i][j].set_ylabel(site,fontsize=12)
            
plt.savefig('Series',bbox_inches = 'tight',dpi=300)


# ### 6.2 纯随机性检验
# - 自相关图：将 n 步自相关系数绘制成一个折线图
# - 注意：统计学意义上的 aotucorrelation 和信号处理中用到的 autocorrelation 是不同的——如果用 matplotlib.pyplot.acorr 来绘制自相关图, 得到的是信号处理中常用的算法。这里使用 pandas.plotting.autocorrelation_plot，来绘制 n 步的皮尔逊相关
# - 尤其注意：用有缺失值的序列是没办法作自相关图的，所以要先把序列补齐

# #### 6.2.1 数据平稳定检验
# - 就是求移动平均数与均方差

# In[49]:


# 一个季度13周，所以以 13 为周期求移动 mean 及 std
rolmean = NaN_Filled_mean['安徽蚌埠蚌埠闸']['DO'].rolling(52).mean()
rolstd = NaN_Filled_mean['安徽蚌埠蚌埠闸']['DO'].rolling(52).std()
plt.figure(figsize=(8,4))
plt.plot(NaN_Filled_mean['安徽蚌埠蚌埠闸']['DO'],label='Time Series')
plt.plot(rolmean,label='Rolling Mean')
plt.plot(rolstd,label='Rolling Std')
plt.legend(loc='best')


# - DF（Dicky-Fuller）检验

# In[235]:


from statsmodels.tsa.stattools import adfuller as ADF
columns = ['site','indicator','test statistic','p-value','lag used','number of observations','critical value(1%)','critical value(5%)','critical value(10%)']
ADF_results = pd.DataFrame(columns=columns)
for dianwei,site in zip(dianwei_list,site_list):
    for zhibiao in ['pH','DO','COD','NH3-N']:
        l=[site,zhibiao]
        l.extend(ADF(NaN_Filled_mean[dianwei][zhibiao])[0:4])
        for value in ADF(NaN_Filled_mean[dianwei][zhibiao])[4].values(): 
            l.append(value)
        df = pd.DataFrame([l],columns=columns)
        ADF_results = ADF_results.append(df)
print(ADF_results)
ADF_results.to_excel('ADF_results.xlsx')


# - 这个结果显示大部分序列都是稳定的？    
# 所以只存在周期性，没有趋势性，也算是稳定序列吗？

# #### 6.2.2 序列自相关图
# - 有两种图供君选择（然鹅为啥两个图看起来并不一样😭）

# In[189]:


from pandas.plotting import autocorrelation_plot


# In[270]:


from matplotlib.ticker import MultipleLocator, FormatStrFormatter
ymajorFormatter = FormatStrFormatter('%1.1f')

fig,axes = plt.subplots(8,4,figsize=(9,12))
plt.subplots_adjust(wspace=0.28,hspace=0.25)
i_list=range(0,8)
for i,dianwei,site in zip(i_list,dianwei_list,site_list):
    for j,zhibiao in enumerate(['pH','DO','COD','NH3-N']):
        autocorrelation_plot(NaN_Filled_mean[dianwei][zhibiao][:250], ax=axes[i][j])
        
        axes[i][j].grid(linestyle='--',linewidth=0.4)
        axes[i][j].set_ylabel('')
        axes[i][j].set_xlabel('')
        axes[i][j].yaxis.set_major_formatter(ymajorFormatter)
        
        if i != 7:
            axes[i][j].set_xticklabels([])
            axes[i][j].tick_params(axis='x',width=0)
        
        if i==0:
            axes[i][j].set_title(zhibiao,fontsize=12)
        if j==0:
            axes[i][j].set_ylabel(site,fontsize=12)
            
plt.savefig('autocorrelation',bbox_inches = 'tight',dpi=300)


# In[256]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
# 参数设置请参考官网- http://www.statsmodels.org/dev/generated/statsmodels.graphics.tsaplots.plot_acf.html#statsmodels.graphics.tsaplots.plot_acf


# In[260]:


fig,axes = plt.subplots(8,4,figsize=(9,12))
plt.subplots_adjust(wspace=0.26,hspace=0.15)
i_list=range(0,8)
for i,dianwei,site in zip(i_list,dianwei_list,site_list):
    for j,zhibiao in enumerate(['pH','DO','COD','NH3-N']):
        plot_acf(NaN_Filled_mean[dianwei][zhibiao][:100], 
                 ax=axes[i][j],
                 marker='.',markersize=1.5,color='steelblue',
                alpha=0.05)
        
        axes[i][j].grid(linestyle='--',linewidth=0.4)
        axes[i][j].set_ylabel('')
        axes[i][j].set_xlabel('')
        axes[i][j].set_title('')
        if i != 7:
            axes[i][j].set_xticklabels([])
            axes[i][j].tick_params(axis='x',width=0)
        
        if i==0:
            axes[i][j].set_title(zhibiao,fontsize=12)
        if j==0:
            axes[i][j].set_ylabel(site,fontsize=12)
            
plt.savefig('acf',bbox_inches = 'tight',dpi=300)


# In[259]:


fig,axes = plt.subplots(8,4,figsize=(9,12))
plt.subplots_adjust(wspace=0.26,hspace=0.15)
i_list=range(0,8)
for i,dianwei,site in zip(i_list,dianwei_list,site_list):
    for j,zhibiao in enumerate(['pH','DO','COD','NH3-N']):
        plot_pacf(NaN_Filled_mean[dianwei][zhibiao][:100], 
                 ax=axes[i][j],
                 marker='.',markersize=1.5,color='steelblue',
                alpha=0.05)
        
        axes[i][j].grid(linestyle='--',linewidth=0.4)
        axes[i][j].set_ylabel('')
        axes[i][j].set_xlabel('')
        axes[i][j].set_title('')
        if i != 7:
            axes[i][j].set_xticklabels([])
            axes[i][j].tick_params(axis='x',width=0)
        
        if i==0:
            axes[i][j].set_title(zhibiao,fontsize=12)
        if j==0:
            axes[i][j].set_ylabel(site,fontsize=12)
            
plt.savefig('pacf',bbox_inches = 'tight',dpi=300)


# ### 7 差分、分解

# In[55]:


from matplotlib.ticker import MultipleLocator, FormatStrFormatter
ymajorFormatter = FormatStrFormatter('%1.1f')

fig,axes = plt.subplots(8,4,figsize=(16,8))
plt.subplots_adjust(wspace=0.17,hspace=0.2)
i_list=range(0,8)
for i,dianwei,site in zip(i_list,dianwei_list,site_list):
    for j,zhibiao in enumerate(['pH','DO','COD','NH3-N']):
        Date_Filled[dianwei][zhibiao].plot(ax=axes[i][j],linewidth=0.8)
        Date_Filled[dianwei][zhibiao].diff(1).plot(ax=axes[i][j],linewidth=0.8)

        axes[i][j].grid(linestyle='--',linewidth=0.4)
        axes[i][j].set_ylabel('')
        axes[i][j].set_xlabel('')
        axes[i][j].yaxis.set_major_formatter(ymajorFormatter)
        
        axes[i][j].set_xticklabels([])
        axes[i][j].tick_params(axis='x',width=0)
        
        if (i==0)&(j==0):
            axes[i][j].set_title(zhibiao,fontsize=12)
        if (i==0)&(j!=0):
            axes[i][j].set_title(zhibiao+u'(mg/L)',fontsize=12)
            
        if j==0:
            axes[i][j].set_ylabel(site,fontsize=12)
            
plt.savefig('First_Diff_Series',bbox_inches = 'tight',dpi=300)


# In[54]:


from matplotlib.ticker import MultipleLocator, FormatStrFormatter
ymajorFormatter = FormatStrFormatter('%1.1f')

fig,axes = plt.subplots(8,4,figsize=(16,8))
plt.subplots_adjust(wspace=0.17,hspace=0.2)
i_list=range(0,8)
for i,dianwei,site in zip(i_list,dianwei_list,site_list):
    for j,zhibiao in enumerate(['pH','DO','COD','NH3-N']):
        Date_Filled[dianwei][zhibiao].plot(ax=axes[i][j],linewidth=0.8)
        Date_Filled[dianwei][zhibiao].diff(52).plot(ax=axes[i][j],linewidth=0.8)

        axes[i][j].grid(linestyle='--',linewidth=0.4)
        axes[i][j].set_ylabel('')
        axes[i][j].set_xlabel('')
        axes[i][j].yaxis.set_major_formatter(ymajorFormatter)
        
        axes[i][j].set_xticklabels([])
        axes[i][j].tick_params(axis='x',width=0)
        
        if (i==0)&(j==0):
            axes[i][j].set_title(zhibiao,fontsize=12)
        if (i==0)&(j!=0):
            axes[i][j].set_title(zhibiao+u'(mg/L)',fontsize=12)
            
        if j==0:
            axes[i][j].set_ylabel(site,fontsize=12)
            
plt.savefig('Seasonal_Diff_Series',bbox_inches = 'tight',dpi=300)


# In[43]:


from statsmodels.tsa.seasonal import seasonal_decompose 

for zhibiao in ['pH','DO','COD','NH3-N']:
    for dianwei in dianwei_list:
        decomposition = seasonal_decompose(NaN_Filled_mean[dianwei][zhibiao], freq=52) 
        decomposition.plot()

