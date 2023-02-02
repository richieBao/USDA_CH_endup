# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 23:37:20 2021

@author: Richie Bao-caDesign设计(cadesign.cn)
"""

class DisplayablePath(object):
    '''
    class - 返回指定路径下所有文件夹及其下文件的结构。代码未改动，迁移于'https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python'
    '''
    
    display_filename_prefix_middle = '├──'
    display_filename_prefix_last = '└──'
    display_parent_prefix_middle = '    '
    display_parent_prefix_last = '│   '

    def __init__(self, path, parent_path, is_last):
        from pathlib import Path
        
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    @classmethod
    def make_tree(cls, root, parent=None, is_last=False, criteria=None):
        from pathlib import Path
        
        root = Path(str(root))
        criteria = criteria or cls._default_criteria

        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = sorted(list(path
                               for path in root.iterdir()
                               if criteria(path)),
                          key=lambda s: str(s).lower())
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir():
                yield from cls.make_tree(path,
                                         parent=displayable_root,
                                         is_last=is_last,
                                         criteria=criteria)
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    @classmethod
    def _default_criteria(cls, path):
        return True

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    def displayable(self):
        if self.parent is None:
            return self.displayname

        _filename_prefix = (self.display_filename_prefix_last
                            if self.is_last
                            else self.display_filename_prefix_middle)

        parts = ['{!s} {!s}'.format(_filename_prefix,
                                    self.displayname)]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(self.display_parent_prefix_middle
                         if parent.is_last
                         else self.display_parent_prefix_last)
            parent = parent.parent

        return ''.join(reversed(parts))
    
def filePath_extraction(dirpath,fileType):
    import os
    
    '''
    funciton  -以所在文件夹路径为键，值为包含该文件夹下所有文件名的列表。文件类型可以自行定义 
    
    Paras:
        dirpath - 根目录，存储所有待读取的文件
        fileType - 待读取文件的类型
    '''
    filePath_Info={}
    i=0
    for dirpath,dirNames,fileNames in os.walk(dirpath): #os.walk()遍历目录，使用help(os.walk)查看返回值解释
       i+=1
       if fileNames: #仅当文件夹中有文件时才提取
           tempList=[f for f in fileNames if f.split('.')[-1] in fileType]
           if tempList: #剔除文件名列表为空的情况，即文件夹下存在不为指定文件类型的文件时，上一步列表会返回空列表[]
               filePath_Info.setdefault(dirpath,tempList)
    return filePath_Info    

def start_time():
    import datetime
    '''
    function-计算当前时间
    '''
    start_time=datetime.datetime.now()
    print("start time:",start_time)
    return start_time

def duration(start_time):
    import datetime
    '''
    function-计算持续时间
    
    Paras:
    start_time - 开始时间
    '''
    end_time=datetime.datetime.now()
    print("end time:",end_time)
    duration=(end_time-start_time).seconds/60
    print("Total time spend:%.2f minutes"%duration)

def is_outlier(data,threshold=3.5):
    import numpy as np
    '''
    function-判断异常值
        
    Params:
        data - 待分析的数据，列表或者一维数组
        threshold - 判断是否为异常值的边界条件    
    '''
    MAD=np.median(abs(data-np.median(data)))
    modified_ZScore=0.6745*(data-np.median(data))/MAD
    #print(modified_ZScore)
    is_outlier_bool=abs(modified_ZScore)>threshold    
    return is_outlier_bool,data[~is_outlier_bool]

def print_html(df,row_numbers=5):
    from IPython.display import HTML
    '''
    function - 在Jupyter中打印DataFrame格式数据为HTML
    
    Paras:
        df - 需要打印的DataFrame或GeoDataFrame格式数据
        row_numbers - 打印的行数，如果为正，从开始打印如果为负，从末尾打印
     '''
    if row_numbers>0:
        return HTML(df.head(row_numbers).to_html())
    else:
        return HTML(df.tail(abs(row_numbers)).to_html())
    
flatten_lst=lambda lst: [m for n_lst in lst for m in flatten_lst(n_lst)] if type(lst) is list else [lst]

def kneed_lineGraph(x,y):
    import matplotlib.pyplot as plt
    from data_generator import DataGenerator
    from knee_locator import KneeLocator
    '''
    function - 绘制折线图，及其拐点。需调用kneed库的KneeLocator，及DataGenerator文件

    Paras:
    x - 横坐标，用于横轴标签
    y - 纵坐标，用于计算拐点    
    '''
    #如果调整图表样式，需调整knee_locator文件中的plot_knee（）函数相关参数
    kneedle=KneeLocator(x, y, curve='convex', direction='decreasing')
    print('曲线拐点（凸）：',round(kneedle.knee, 3))
    print('曲线拐点（凹）：',round(kneedle.elbow, 3))
    kneedle.plot_knee(figsize=(8,8))
    
def imgs_layoutShow(imgs_root,imgsFn_lst,columns,scale,figsize=(15,10)):
    import math,os
    import matplotlib.pyplot as plt
    from PIL import Image
    '''
    function - 显示一个文件夹下所有图片，便于查看。
    
    Paras:
        imgs_root - 图像所在根目录
        imgsFn_lst - 图像名列表
        columns - 列数
    '''
    rows=math.ceil(len(imgsFn_lst)/columns)
    fig,axes=plt.subplots(rows,columns,sharex=True,sharey=True,figsize=figsize)   #布局多个子图，每个子图显示一幅图像
    ax=axes.flatten()  #降至1维，便于循环操作子图
    for i in range(len(imgsFn_lst)):
        img_path=os.path.join(imgs_root,imgsFn_lst[i]) #获取图像的路径
        img_array=Image.open(img_path) #读取图像为数组，值为RGB格式0-255        
        img_resize=img_array.resize([int(scale * s) for s in img_array.size] ) #传入图像的数组，调整图片大小
        ax[i].imshow(img_resize)  #显示图像
        ax[i].set_title(i+1)
    fig.tight_layout() #自动调整子图参数，使之填充整个图像区域  
    fig.suptitle("images show",fontsize=14,fontweight='bold',y=1.02)
    plt.show()    
    
def imgs_layoutShow_FPList(imgs_fp_list,columns,scale,figsize=(15,10)):
    import math,os
    import matplotlib.pyplot as plt
    from PIL import Image
    '''
    function - 显示一个文件夹下所有图片，便于查看。

    Paras:
        imgs_root - 图像所在根目录
        imgsFn_lst - 图像名列表
        columns - 列数
    '''
    rows=math.ceil(len(imgs_fp_list)/columns)
    fig,axes=plt.subplots(rows,columns,figsize=figsize,)   #布局多个子图，每个子图显示一幅图像 sharex=True,sharey=True,
    ax=axes.flatten()  #降至1维，便于循环操作子图
    for i in range(len(imgs_fp_list)):
        img_path=imgs_fp_list[i] #获取图像的路径
        img_array=Image.open(img_path) #读取图像为数组，值为RGB格式0-255        
        img_resize=img_array.resize([int(scale * s) for s in img_array.size] ) #传入图像的数组，调整图片大小
        ax[i].imshow(img_resize,)  #显示图像 aspect='auto'
        ax[i].set_title(i+1)
    invisible_num=rows*columns-len(imgs_fp_list)
    if invisible_num>0:
        for i in range(invisible_num):
            ax.flat[-(i+1)].set_visible(False)
    fig.tight_layout() #自动调整子图参数，使之填充整个图像区域  
    fig.suptitle("images show",fontsize=14,fontweight='bold',y=1.02)
    plt.show()    
    
class AttrDict(dict):
    """
    # Code adapted from:
    # https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/collections.py

    Source License
    # Copyright (c) 2017-present, Facebook, Inc.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    ##############################################################################
    #
    # Based on:
    # --------------------------------------------------------
    # Fast R-CNN
    # Copyright (c) 2015 Microsoft
    # Licensed under The MIT License [see LICENSE for details]
    # Written by Ross Girshick
    # --------------------------------------------------------
    """
    IMMUTABLE = '__immutable__'

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__[AttrDict.IMMUTABLE] = False

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if not self.__dict__[AttrDict.IMMUTABLE]:
            if name in self.__dict__:
                self.__dict__[name] = value
            else:
                self[name] = value
        else:
            raise AttributeError(
                'Attempted to set "{}" to "{}", but AttrDict is immutable'.
                format(name, value)
            )

    def immutable(self, is_immutable):
        """Set immutability to is_immutable and recursively apply the setting
        to all nested AttrDicts.
        """
        self.__dict__[AttrDict.IMMUTABLE] = is_immutable
        # Recursively set immutable state
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)
        for v in self.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)

    def is_immutable(self):
        return self.__dict__[AttrDict.IMMUTABLE]
    
def gantt_chart_H(df_,category_field,datatime_filed,interval='3h'):
    '''
    指定分类列，给定时间列，按时间间隔提取时间段，并打印数据按时间的分布

    Parameters
    ----------
    df_ : DataFrame
        待打印的数据.
    category_field : string
        分类列字段名.
    datatime_filed : string
        时间列字段名.
    interval : string, optional
        时间周期. The default is '3h'.

    Returns
    -------
    SFT : DataFrame
        用于gantt图打印的DataFrame格式数据.

    '''    
    import plotly.figure_factory as ff
    from tqdm import tqdm
    import pandas as pd
    import numpy as np
    from random import randint
    import plotly.io as pio
    pio.renderers.default='browser'
    pd.options.mode.chained_assignment=None 
    
    df=df_.copy(deep=True)  
    category_name=df_[category_field].unique()
    print(f"category_field number:{len(category_name)}\ncategory_field:\n{category_name}")
    Start_Finish_Task=[]
    for c_n in tqdm(category_name):        
        category_df=df[df[category_field]==c_n]      
        t_interval=category_df.groupby(pd.Grouper(key='ts',axis=0,freq=interval))['ts'].agg(['first','last','count'])
        t_interval['count']=t_interval['count']*int(interval[:-1])
        t_interval.columns=['Start', 'Finish', 'Duration_Hours']
        t_interval['Task']=c_n
        Start_Finish_Task.append(t_interval)   
        
    SFT=pd.concat(Start_Finish_Task,ignore_index=True,axis=0)#[:1000] 
    colors=["rgb({},{},{})".format(randint(0,255),randint(0,255),randint(0,255) ) for i in range(len(np.unique(SFT.Task)))]
    fig=ff.create_gantt(SFT,group_tasks=True,index_col='Task',colors=colors)
    fig.update_layout(autosize=False,width=2100,height=1200,)
    fig.show()        
    
    return SFT    

def df_linking_geometry(df,gdf,key_df,key_gdf,how="inner"):
    '''
    包含有属性的DataFrame格式数据合并包含有地理空间信息的GeoDataFrame格式数据，仅合并“geometry”字段和用于链接的关键字段列

    Parameters
    ----------
    df : DataFrame
        属性数据，含有用于链接的关键字段.
    gdf : GeoDataFrame
        还有地理空间信息geometry字段和用于链接的关键字段.
    key_df : string
        属性数据用于链接的关键字段名.
    key_gdf : string
        含有地理空间信息数据用于链接的关键字段名.
    how : string, optional
        {‘left’, ‘right’, ‘outer’, ‘inner’, ‘cross’}，具体解释参考pandas.DataFrame.merg. The default is "inner".

    Returns
    -------
    gdf_linked : GeoDataFrame
        属性数据合并地理空间信息后的数据.

    '''    
    gdf_linked=gdf[[key_gdf,"geometry"]].merge(df,left_on=key_gdf,right_on=key_df,how=how,)
    
    return gdf_linked

def gdf_plot_annotate(gdf_,value_column,annotate_column,**setting):
    '''
    打印GeoDataFrame格式地理空间信息数据

    Parameters
    ----------
    gdf_ : GeoDataFrame
        待打印的数据.
    value_column : string
        数值显示字段名.
    annotate_column : string
        标注显示字段名.
    **setting : key args
        用于配置图表的参数，键和默认值如下
        setting_dict=dict(annotate_fontsize=8,
                          figsize=(10,10),    
                          legend_position="right",
                          legend_size="5%",
                          legend_pad=0.1,
                          legend_bbox_to_anchor=(1, 1),
                          cmap='OrRd',
                          labelsize=8,
                          scheme=None, # 等值分类图，例如 ‘BoxPlot’, ‘EqualInterval’, ‘FisherJenks’,‘FisherJenksSampled’, ‘HeadTailBreaks’, ‘JenksCaspall’, 
                                                         ‘JenksCaspallForced’, ‘JenksCaspallSampled’, ‘MaxP’, ‘MaximumBreaks’, ‘NaturalBreaks’, ‘Quantiles’, 
                                                         ‘Percentiles’, ‘StdMean’, ‘UserDefined’等
                          k=5, # 分类数量， 对应scheme参数，如果scheme参数为None，则k参数忽略
                          categorical=False # 为True时为分类数据，为False时为数值数据
                         ).

    Returns
    -------
    ax : TYPE
        DESCRIPTION.

    '''     
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    gdf=gdf_.copy(deep=True)
    setting_dict=dict(annotate_fontsize=8,
                      figsize=(10,10),    
                      legend_position="right",
                      legend_size="5%",
                      legend_pad=0.1,
                      legend_bbox_to_anchor=(1, 1),
                      cmap='OrRd',
                      labelsize=8,
                      scheme=None,
                      k=5,
                      categorical=False
                     )
    setting_dict.update(setting)
    gdf["index"]=gdf.index
    
    fig, ax=plt.subplots(figsize=setting_dict["figsize"])
    divider=make_axes_locatable(ax) 
    if setting_dict["scheme"]:
        gdf.plot(column=value_column,scheme=setting_dict["scheme"], k= setting_dict["k"],ax=ax,legend=True,cmap=setting_dict["cmap"],legend_kwds={'bbox_to_anchor':setting_dict["legend_bbox_to_anchor"]}) 
    elif setting_dict["categorical"]:
        gdf.plot(column=value_column,categorical=True,ax=ax,legend=True,cmap=setting_dict["cmap"],edgecolor='white',legend_kwds={'bbox_to_anchor':setting_dict["legend_bbox_to_anchor"]}) 
    else:   
        cax=divider.append_axes(setting_dict["legend_position"], size=setting_dict["legend_size"], pad=setting_dict["legend_pad"]) # 配置图例参数
        gdf.plot(column=value_column,scheme=setting_dict["scheme"], k= setting_dict["k"],ax=ax,cax=cax,legend=True,cmap=setting_dict["cmap"]) 
    gdf.apply(lambda x: ax.annotate(text=x[annotate_column], xy=x.geometry.centroid.coords[0], ha='center',fontsize=setting_dict["annotate_fontsize"]),axis=1) # 增加标注
    ax.tick_params(axis='both', labelsize=setting_dict["labelsize"])

    plt.show()
    
def df_group_resample(df,val_column,time_column,rules,methods=["mean","min","max","sum"],group_column=None,geometry_column=None):
    '''
    时空数据（面板数据），按照给定的分组，时间长度，数值计算方法重采样数值列。

    Parameters
    ----------
    df : DataFrame
        时空数据.
    val_column : string
        用于重采样的数据值.
    time_column : string
        时间列.
    rules : string
        偏移量（时间长度），例'H'，'D'，`W`，`M`，`Y`，'30S'，`3T`，`Q`，`17min`等.
    methods : list(string), optional
        数值采样方法，包括均值、最小和最大值，及和. The default is ["mean","min","max","sum"].
    group_column : string, optional
        分组列名. The default is None.
    geometry_column : string, optional
        几何列. The default is None.

    Returns
    -------
    GeoDataFrame
        重采样后时空数据.

    '''    
    from pandas.api.types import is_datetime64_any_dtype as is_datetime
    import pandas as pd
    import geopandas as gpd
    from tqdm import tqdm
    from shapely import wkt
    
    df_notna=df[df[val_column].notna()]
    if is_datetime(df_notna[time_column]):
        df_notna.rename(columns={time_column:"ts"},inplace=True)
    else:
        df_notn['ts']=pd.to_datetime(df_notna[time_column])
    df_notna.sort_values(by=["ts"],inplace=True)
    df_notna[val_column]=df_notna[val_column].astype("float")
    if geometry_column:
        crs=df_notna.crs
        print(f"CRS={crs}")
    
    if group_column:
        df_group=df_notna.groupby(group_column)    
        nodeID_geometry_mapping={}
        g_v_resample_lst=[]
        for g_n,g_v in tqdm(df_group):
            g_v.set_index('ts',inplace=True)       
            if geometry_column:
                nodeID_geometry_mapping[g_n]=g_v.iloc[[0]].geometry.values[0].wkt
            g_v_resample_r_dict={}
            for r in rules:               
                g_v_resample=g_v[val_column].resample(r)
                g_v_resample_methods_dict={}
                for m in methods:
                    if m=="mean":
                        g_v_resample_mean=g_v_resample.mean()
                        g_v_resample_methods_dict["mean"]=g_v_resample_mean
                    elif m=="min":
                        g_v_resample_min=g_v_resample.min()
                        g_v_resample_methods_dict["min"]=g_v_resample_min=g_v_resample_min
                    elif m=="max":
                        g_v_resample_max=g_v_resample.max()
                        g_v_resample_methods_dict["max"]=g_v_resample_min=g_v_resample_max
                    elif m=="sum":
                        g_v_resample_max=g_v_resample.sum()
                        g_v_resample_methods_dict["sum"]=g_v_resample_min=g_v_resample_max                        
                    else:
                        pass
                g_v_resample_r_dict[r]=pd.concat(g_v_resample_methods_dict,axis=1)  
                
            g_v_resample_r_df=pd.concat(g_v_resample_r_dict,axis=1)     
            g_v_resample_r_df.columns=g_v_resample_r_df.columns.map("_".join).str.strip("_")
            g_v_resample_r_df[group_column]=g_n
            g_v_resample_lst.append(g_v_resample_r_df)
            
        g_v_resample_df=pd.concat(g_v_resample_lst)
        if geometry_column:
            g_v_resample_df[geometry_column]=g_v_resample_df[group_column].map(nodeID_geometry_mapping)
            g_v_resample_df[geometry_column]=g_v_resample_df[geometry_column].apply(wkt.loads)
            g_v_resample_df=gpd.GeoDataFrame(g_v_resample_df,geometry=geometry_column,crs=crs)  
            g_v_resample_df.reset_index(inplace=True)
        return g_v_resample_df
    
    else:
        df_notna.set_index('ts',inplace=True) 
        df_notna_resample_r_dict={}
        for r in rules:                  
            df_notna_resample=df_notna[val_column].resample(r)
            df_notna_resample_methods_dict={}
            for m in methods:
                if m=="mean":
                    df_notna_resample_mean=df_notna_resample.mean()
                    df_notna_resample_methods_dict["mean"]=df_notna_resample_mean
                elif m=="min":
                    df_notna_resample_min=df_notna_resample.min()
                    df_notna_resample_methods_dict["min"]=df_notna_resample_min
                elif m=="max":
                    df_notna_resample_max=df_notna_resample.max()
                    df_notna_resample_methods_dict["max"]=df_notna_resample_max
                elif m=="sum":
                    df_notna_resample_max=df_notna_resample.sum()
                    df_notna_resample_methods_dict["sum"]=df_notna_resample_max                
                else:
                        pass        
            df_notna_resample_r_dict[r]=pd.concat(df_notna_resample_methods_dict,axis=1)    
        df_notna_resample_r_df=pd.concat(df_notna_resample_r_dict,axis=1)   
        df_notna_resample_r_df.reset_index(inplace=True)
        return df_notna_resample_r_df    
    
def rec_quadrats_gdf(leftBottom_coordi,rightTop_coordi,h_distance,v_distance,crs=4326,to_crs=None):
    '''
    构建网格式样方

    Parameters
    ----------
    leftBottom_coordi : list(float)
        定位左下角坐标.
    rightTop_coordi : list(float)
        定位右上角坐标.
    h_distance : float
        单个样方宽度.
    v_distance : float
        单个样方长度.
    crs : int, optional
        投影编号. The default is 4326.
    to_crs : int, optional
        转换投影编号. The default is None.

    Returns
    -------
    grids_gdf : GeoDataFrame
        Polygon地理几何形式的GeoDataFrame格式样方数据.

    '''    
    import numpy as np
    from shapely.geometry import MultiLineString
    from shapely.ops import polygonize
    import geopandas as gpd
    
    x=np.arange(leftBottom_coordi[0], rightTop_coordi[0], h_distance)
    y=np.arange(leftBottom_coordi[1], rightTop_coordi[1], v_distance)
    hlines=[((x1, yi), (x2, yi)) for x1, x2 in zip(x[:-1], x[1:]) for yi in y]
    vlines=[((xi, y1), (xi, y2)) for y1, y2 in zip(y[:-1], y[1:]) for xi in x]
    grids=list(polygonize(MultiLineString(hlines + vlines)))
    
    grids_gdf=gpd.GeoDataFrame({'geometry':grids},crs=crs)
    if to_crs:
        grids_gdf.to_crs(to_crs,inplace=True)
        
    return grids_gdf    
    
def zonal_stats_raster(raster_fn,sampling_zone,band=1,stats=['majority'],add_stats=['frequency'],nodata=-9999):#
    '''
    区域统计，包括['count', 'min', 'max', 'mean', 'sum', 'std', 'median', 'majority', 'minority', 'unique', 'range', 'nodata', 'nan']，以及自定义的'frequency'，即频数统计

    Parameters
    ----------
    raster_fn : String
        待区域统计的栅格数据路径名.
    sampling_zone : GeoDataFrame
        用于栅格区域统计的polygon几何对象.
    band : int, optional
        数据波段. The default is 1.
    stats : List(String), optional
        默认统计的统计量名称. The default is ['majority'].
    add_stats :List(String) , optional
        自定义统计量名. The default is ['frequency'].

    Returns
    -------
    GeoDataFrame
        返回统计量值.

    '''
    import rasterio as rio
    import rasterstats as rst
    import pandas as pd
     
    sampling_zone_copy=sampling_zone.copy(deep=True)
    
    def frequency(x):
        data=x.data[~x.mask]
        return pd.value_counts(data)
    
    add_stats_dict={'frequency':frequency}
    with rio.open(raster_fn,'r') as src:
        band=src.read(band)
        sampling_zone_copy=sampling_zone_copy.to_crs(src.crs)
        zs_result=rst.zonal_stats(sampling_zone_copy,band,nodata=nodata,affine=src.transform,stats=stats,add_stats={i:add_stats_dict[i] for i in add_stats})
    
    for stat in stats:
        sampling_zone_copy[stat]=[dic[stat] for dic in zs_result]
    for stat in add_stats:
        if stat=='frequency':
            fre=pd.concat([dic[stat].to_frame().T for dic in zs_result])
            fre.rename(columns={col:"{}_{}".format(stat[:3],col) for col in fre.columns},inplace=True)
            fre.reset_index(inplace=True)  
    try:        
        zonal_stats_gdf=pd.concat([sampling_zone_copy,fre],axis=1)   
        
    except:
        zonal_stats_gdf=sampling_zone_copy
    return zonal_stats_gdf

def weights_plot(gdf,weights,annotate_column=None,**setting):  
    '''
    打印显示空间权重

    Parameters
    ----------
    gdf : GeoDataFrame
        地理空间数据.
    weights : libpysal.weights
        有PySAL库计算的空间权重.
    annotate_column : string, optional
        用于标注的列名. The default is None.
    **setting : key args
        打印样式参数配置，包括：
                        setting_dict=dict(figsize=(10,10),
                                  annotate_fontsize=8,
                                  ax=None,
                                  ).

    Returns
    -------
    ax : AxesSubplot
        子图.

    '''    
    setting_dict=dict(figsize=(10,10),
                      annotate_fontsize=8,
                      ax=None,
                      )
    setting_dict.update(setting)

    if setting_dict["ax"]:
        ax=setting_dict["ax"]
        gdf.plot(edgecolor="grey",facecolor="w",figsize=setting_dict["figsize"],ax=ax)
    else:
        ax=gdf.plot(edgecolor="grey",facecolor="w",figsize=setting_dict["figsize"])
    f,ax=weights.plot(gdf,
                      ax=ax,
                      edge_kws=dict(color='r', linestyle=':', linewidth=1),
                      node_kws=dict(marker='')
                      )
    if annotate_column:
        gdf["index"]=gdf.index
        gdf.apply(lambda x: ax.annotate(text=x[annotate_column], xy=x.geometry.centroid.coords[0], ha='center',fontsize=setting_dict["annotate_fontsize"]),axis=1) # 增加标注
        
def G_drawing(G,edge_labels=None,node_labels=None,routes=[],nodes=[],**kwargs):
    '''
    绘制复杂网络

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        复杂网络（图）.
    edge_labels : string, optional
        显示边属性. The default is None.
    node_labels : string, optional
        显示节点属性. The default is None.
    routes : list(G vertex), optional
        构成图路径的顶点. The default is None.  
    nodes : list(G vertex), optional
        顶点的嵌套列表，用于不同顶点集的不同显示（颜色和大小等）. The default is None.        
    **kwargs : kwargs
        图表样式参数，包括options和sytle，默认值为：
            options={
                    "font_size": 20,
                    "font_color":"black",
                    "node_size": 150,
                    "node_color": "olive",
                    "edgecolors": "olive",
                    "linewidths": 7,
                    "width": 1,
                    "with_labels":True,    
                    }
             style={
                    "figsize":(3,3),   
                    "tight_layout":True,
                    "pos_func":nx.spring_layout,
                    "edge_label_font_size":10,
                    "pos":None
                    }.

    Returns
    -------
    None.

    '''    
    import matplotlib.pyplot as plt
    import networkx as nx
    import matplotlib.colors as mcolors
    import random
    from pylab import mpl
    plt.rc('axes', unicode_minus=False) # 解决图表负号不正确显示问题
    mpl.rcParams['font.sans-serif']=['SimHei'] # 解决中文字符乱码问题

    def generate_color():
        color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: random.randint(0, 255), range(3)))
        return color
    
    options={
    "font_size": 20,
    "font_color":"black",
    "node_size": 150,
    "node_color": "olive",
    "edgecolors": "olive",
    "linewidths": 7,
    "width": 1,
    "with_labels":True,    
    "cmap":None,
    }
    options.update((k, kwargs[k]) for k in set(kwargs).intersection(options))
    
    style={
    "figsize":(3,3),   
    "tight_layout":True,
    "pos_func":nx.spring_layout,
    "edge_label_font_size":10,
    "pos":None,
    "edge_colors":list(mcolors.TABLEAU_COLORS.values()),
    "edge_widths":[3]*len(routes),
    "title":None,
    "nodes_size":[200]*len(nodes),
    "nodes_color":[generate_color() for i in range(len(nodes))]#list(mcolors.TABLEAU_COLORS.values()),
    }
    
    style.update((k, kwargs[k]) for k in set(kwargs).intersection(style))        
    fig,ax=plt.subplots(figsize=style['figsize'],tight_layout=style["tight_layout"]) 
    
    if style['pos']:
        pos=style['pos']
    else:
        pos=list(map(style["pos_func"],[G]))[0]    
        
    if routes:
        route_edges=[[(r[n],r[n+1]) for n in range(len(r)-1)] for r in routes]
        [nx.draw_networkx_edges(G,pos=pos,edgelist=edgelist,edge_color=style['edge_colors'][idx],width=style['edge_widths'][idx],) for idx,edgelist in enumerate(route_edges)]        

    
    if node_labels:
        options["with_labels"]=False
        nx.draw(G, pos=pos,ax=ax,**options)
        node_labels=nx.get_node_attributes(G,node_labels)
        nx.draw_networkx_labels(G, pos, labels=node_labels,ax=ax)
    else:
        nx.draw(G, pos=pos,ax=ax,**options)        
    
    if edge_labels:
        edge_labels=nx.get_edge_attributes(G,edge_labels)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,ax=ax,font_size=style["edge_label_font_size"])  
        
    if nodes:
        [nx.draw_networkx_nodes(G,pos=pos,nodelist=sub_nodes,node_size=style['nodes_size'][idx],node_color=style['nodes_color'][idx]) for idx,sub_nodes in enumerate(nodes)]    
        
    plt.title(style['title'])
    plt.show()  
        