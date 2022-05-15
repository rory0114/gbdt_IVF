
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 10:41:00 2021

@author: rory
"""
import pandas as pd 
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick  
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


    
def dis_test(a,b,way = 't'):
    """
利用t检验或者卡方检验看a，b是否可以融合
返回置信度，越大，越相似
    """

    if way == 't':
        t,p = stats.ttest_ind(a,b)
    elif way == 'chi2':
        #assert set(a) == set(b) and len(set(a)) == 2
        t,p = stats.chisquare(a,b)
    return p


def merge_t(x,y,cut_index,max_bins):
    """
    x,y: 按x排序好的数据和对应的y
    cut_index: 分桶的index值 类别数包括头尾[0,1,2,3,4] 类别数+1
    
    """
    distance=[]
    if len(cut_index)<max_bins:
        return x[cut_index].tolist()
    for i in range(len(cut_index)-2):
        distance.append(dis_test(y[cut_index[i]:cut_index[i+1]],y[cut_index[i+1]:cut_index[i+2]]))
    while len(distance)>=max_bins:
        minidx = np.argmax(distance)
        cut_index.pop(minidx+1)
        distance.pop(minidx)
        if minidx == 0:
            distance[minidx] =dis_test(y[cut_index[minidx]:cut_index[minidx+1]],y[cut_index[minidx+1]:cut_index[minidx+2]])
            continue
        elif minidx == len(distance):
            distance[minidx-1] =dis_test(y[cut_index[minidx-1]:cut_index[minidx]],y[cut_index[minidx]:cut_index[minidx+1]])
            continue
        else:
            distance[minidx-1] = dis_test(y[cut_index[minidx-1]:cut_index[minidx]],y[cut_index[minidx]:cut_index[minidx+1]])
            distance[minidx] =dis_test(y[cut_index[minidx]:cut_index[minidx+1]],y[cut_index[minidx+1]:cut_index[minidx+2]])
        
    return x[cut_index].tolist()
            

        
        
    
    
def bin_label(x,cut_val):
    if pd.isnull(x):
        return np.nan
    for i in range(1,len(cut_val)):
        if float(x)<=float(cut_val[i]):
            return cut_val[i]
        

def bins(dataframe,col,y_col,start_bins = 100,max_bins = 10,min_bins = 5,set_bins = 10):
    """
    col:单维输入变量列名
    y_col：的label变量的列名
    start_bins: 一开始划分多少bins作为merge的初始值
    max_bins：输出的bins的最大可能值，也是开始进行非排序的分桶的开始
    min_bins：非排序的分桶结束阶段
    """
    
    if type(y_col) == str:
        y_col = [y_col]
    df_temp = dataframe[[col]+y_col]
    df_temp =  df_temp.dropna()
    df_temp = df_temp.sort_values(col)
    #return df_temp[col]
    set_len = len({x for x in set(df_temp[col].values.flatten()) if x==x}) 
    if set_len<=7:
        df_temp['cut'] = df_temp[col]
        return(df_temp['cut'],set_len)
    else:
        col_cut = pd.qcut(df_temp[col].values.flatten(),start_bins,duplicates ='drop')
        df_temp['cut'] = col_cut
        
    
    
    #df_temp = df_temp.sort_values(col)
    df_temp['ind'] = df_temp.index
    df_temp['seq'] = range(len(df_temp)) 
    df_temp = df_temp.set_index('seq')
    dff = df_temp[['cut']]
    
    cut_index = dff.ne(dff.shift()).apply(lambda x: x.index[x].tolist()).values.flatten()
    cut_index = list(cut_index.reshape(-1))
    cut_index.append(len(dff)-1)
    #print(len(cut_index))
    cut_num = merge_t(df_temp[col],df_temp[y_col],cut_index,max_bins) 
    print(col,cut_num)
    x_bin_label = [bin_label(i,cut_num) for i in dataframe[col]] 
    #df_temp = data
    return x_bin_label,cut_num
   #  df_temp = df_temp.groupby('cut')[y_col].agg(['mean','count'])
   # # df_temp =
    
   #  df_temp['distance'] = df_temp.eval()
    
    
    
   #  df_temp['cut'] = df_temp.index
    
    
    
   #  df_temp['col_name'] = str(col)
    
    
   #  df_temp = df_temp.set_index(['col_name','cut'])
        
   #  return df_temp



def bins_df(dataframe,x_col = None, y_col = None,draw = False,start_bins = 100,max_bins = 10,min_bins = 5,set_bins = 10):
    rescol = []
    cutdict = {}
    if x_col is None:
        x_col = dataframe.columns
    if y_col is None:
        y_col = dataframe.columns
    for col in x_col:
        try:
            result_col,cutnum = bins(dataframe,col,y_col,start_bins = start_bins,max_bins = max_bins)
            dataframe[col+'_bins'] = result_col
            rescol.append(col+'_bins')
            cutdict[col] = cutnum
            #print(col)
            if draw == True:
                draw_relation(result,col,y_col)
        except:
            pass
    rescol.append(y_col)
    return dataframe.loc[:,rescol],cutdict
        
    



def print_result(result):
    df = pd.DataFrame()
    for idx in result:
        df_col = pd.DataFrame.from_dict(result[idx])
        df = df.append(df_col)
    return df

def cut_mean_col(dataframe,col,y_col):
    """
    col:单维输入变量
    y_col：可以使多维的label变量
    x_bins:划分的分桶数，如果输入的原类别数少于x_bins直接对原来的值进行聚合
    cut_type:{'qcut,cut'} 默认qcut，按照值域平均分，qcut按照使得每个分桶里的个数一样分
    """

    df_temp = dataframe.groupby(col)[y_col].agg(['mean','count'])
    df_temp[col] = df_temp.index
    df_temp['col_name'] = str(col)
    df_temp = df_temp.set_index(['col_name',col])
        
    return df_temp



def cut_mean_df(dataframe,x_col = None, y_col = None):
    result = {}
    for col in x_col:
        
        try:
            result_col = cut_mean_col(dataframe,col,y_col)
            result[col] = result_col
            print(col)
            if draw == True:
                draw_relation(result,col,y_col)
        except:
            pass
    
    return result

def print_result(result):
    df = pd.DataFrame()
    for idx in result:
        df_col = pd.DataFrame.from_dict(result[idx])
        df = df.append(df_col)
    return df

def draw_result(result,col_name,x_col,y_col,graph_width=5,start_col = 0,col_num = 15,y_lim =  None ):
    g = sns.FacetGrid(result, col=col_name,col_wrap=graph_width,sharex = False,ylim = y_lim)
    g.map(sns.pointplot, x_col, y_col)
    return g
    
        
def bin_label_df(df,cols,cutdict):
    recol = []
    for col in cols:
        
        x_bin_label = [bin_label(i,cutdict[col]) for i in df[col]] 
        df[col+'_bins'] = x_bin_label
        recol.append(col+'_bins')
    return df[recol]
    

def findand(idxi,idxj):
    return [(a and b) for a,b in zip(idxi,idxj)]
    

df = pd.read_csv('../data/d622.csv',encoding = 'gbk',index_col = 0)
df_s = df.loc[:,df.dtypes!='O']
afc = []
for i,row in df_s[['左侧AFC_x','右侧AFC_x']].iterrows():
    if pd.isna(row['左侧AFC_x']) or pd.isna(row['右侧AFC_x']):
        afc.append(np.nan)
    else:
        afc.append(row['右侧AFC_x']+row['左侧AFC_x'])
  
df['AFC_x'] = afc
dfn = df[(df['label1']==0)|(df['label1']==1)]

dfa = pd.read_csv('../data/For-binning.csv',encoding = 'gbk')

coleng = pd.read_csv('../data/coleng30.csv',encoding = 'gbk',names=['eng','chn','unit'])
for c in dfa:
    dfn[c] = dfa[c].values

units = coleng.unit.values  
#--------筛选条件-------------------------------
#df_s = df_s[df_s['不明原因性不孕']==1]
#------------------------------------------------------------------
#df_s = df_s[df_s['拟采用精液来源_x']==1]
#df_s = df_s[(df_s['临床妊娠_x'] == 1) | (df_s['临床妊娠_x'] == 3)]
#df_s['label1'] = df_s['临床妊娠_x'].apply(lambda x : 1 if x==1 else 0)
#df_sf = pd.read_csv('../data/sig_factor.csv',encoding = 'gbk')
select_factor =coleng['chn'].values
#select_factor
ecdict = {}
for _,r in coleng.iterrows():
    ecdict[r['chn']]=r['eng']
    
df_s = dfn[np.append(select_factor,'label1')]
ccol = df_s['子宫内膜厚度_x']
ncol = []
for i in ccol:
    if i >3:
        ncol.append(i/10)
    else:
        ncol.append(i)
df_s['子宫内膜厚度_x'] = ncol
a,cutdict = bins_df(df_s,select_factor[:],'label1',start_bins = 50,max_bins = 5)
##df_all_factor_tet = bin_label_df(df_all_factor_te,miccol,cutdict)
# result = cut_mean_df(a,a.columns,a.columns[-1])
# t = print_result(result)
# t['factor'] = [i[0] for i in t.index.to_list()]
# t['factor'] = [i[:-5] for i in t['factor']]
# t['xbin'] = [i[1] for i in t.index.to_list()]
# t['xbin'] = ['%.3g'%i for i in t['xbin']]
t = pd.read_csv('../data/newtt.csv',encoding = 'gbk')
t['factor'] = [ecdict[i] for i in t['factor']]

#------------------draw bin relation
g = draw_result(t,'factor','xbin','mean',graph_width=5,y_lim=[0,0.8])
#df_all_factor_tet = bin_label_df(df_all_factor_te,miccol,cutdict)

sns.set(font_scale = 0.8)
plt.rcParams.update({'font.size':13})
cnt = 0
drawname = coleng['eng'].values
xtick_list = {}
for i in g.axes_dict:
    g_temp = g.axes_dict[i]
    g_temp.set_title(drawname[cnt],fontdict = {'size':12})
    g_temp.set_ylim()
    g_temp.set_xlabel(units[cnt])
      
    cnt+=1
    if cnt%5 == 1:
        g_temp.set_ylabel('Pregnancy accuracy')  

from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor

b = a.fillna(0)
X = b.iloc[:,:-1].values
y = b.iloc[:,-1].values
clf = GradientBoostingRegressor(max_depth=2)
# scores = cross_val_score(clf, X, y, cv=5,scoring='r2')
from sklearn.metrics import r2_score
clf.fit(X,y)
# rrr = clf.predict(X)

# df_model = pd.DataFrame()
# df_model['score'] = rrr
# df_model['label'] = y
# RESULT_BIN = 20
# a_m,cutdict = bins_df(df_model,df_model.columns[[0]],'label',start_bins = 50,max_bins = RESULT_BIN)
# result = cut_mean_df(a_m,a_m.columns,a_m.columns[-1])
# t = print_result(result)
# t['factor'] = [i[0] for i in t.index.to_list()]
# t['factor'] = [i[:-5] for i in t['factor']]
# t['xbin'] = [i[1] for i in t.index.to_list()]
# t['xbin'] = ['%.3g'%i for i in t['xbin']]
# #t = pd.read_csv('m_result.csv',encoding = 'gbk')


# # plt.plot(t['xbin'],t['mean'])
# # plt.bar(t['xbin'],t['count'])
# #--------------------feature importance

# #plt.barh(select_factor,clf.feature_importances_)
# #------------huatu1
# fmt='%.2f%%'
# yticks = mtick.FormatStrFormatter(fmt)  #设置百分比形式的坐标轴
# t = t.iloc[:-2,:]
# a=t['mean']  #数据
# b=t['count']
# l=[i for i in range(RESULT_BIN)]
# lx=t['xbin']

# fig = plt.figure()  
# ax1 = fig.add_subplot(111)  
# ax1.plot(l,a,'or-',label=u'rate');
# ax1.yaxis.set_major_formatter(yticks)
# for i,(_x,_y) in enumerate(zip(l,b)):  
#     plt.text(_x,_y,b[i],color='black',fontsize=10,)  #将数值显示在图形上
# ax1.legend(loc=1)
# ax1.set_ylim([0,1]);
# ax1.set_ylabel('rate');
# plt.legend(prop={'family':'SimHei','size':8})  #设置中文
# ax2 = ax1.twinx() # this is the important function  
# plt.bar(l,b,alpha=0.3,color='blue',label=u'sample')  
# ax2.legend(loc=2)
# ax2.set_ylim([0, 10000])  #设置y轴取值范围
# plt.legend(prop={'family':'SimHei','size':8},loc="upper left") 
# plt.xticks(l,lx)
# plt.show()

# #------------


# from sklearn.metrics import roc_curve, auc 
# fpr,tpr,threshold = roc_curve(y, rrr) ###计算真正率和假正率
# roc_auc = auc(fpr,tpr)

# #plt.figure()
# lw = 2
# plt.figure(figsize=(10,10))
# plt.plot(fpr, tpr, color='darkorange',
#           lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# plt.legend(loc="lower right")







###-----------------------tree

from sklearn import tree
import graphviz
sf = [ecdict[i[:-5]] for i in a.columns[:-1]]
ts = tree.export_graphviz(clf.estimators_[1,0],feature_names=sf)
grraphviz.Source(ts) # doctest: +SKIP
graph.render("tree") # doctest: +SKIPaph = graphviz.Source(ts) # doctest: +SKIP
