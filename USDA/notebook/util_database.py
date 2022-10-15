# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 22:05:41 2021

@author: Richie Bao-caDesign设计(cadesign.cn)
"""

def df2SQLite(db_fp,df,table,method='fail'):
    from sqlalchemy import create_engine
    
    '''
    function - pandas方法，把DataFrame格式数据写入数据库（同时创建表）
    
    Paras:
        db_fp - 数据库文件路径
        df - 待写入数据库的DataFrame格式数据
        table - 表名称
        method - 写入方法，'fail'，'replace'或'append'
    '''
    engine=create_engine('sqlite:///'+'\\\\'.join(db_fp.split('\\')),echo=True) 
    try:    
        df.to_sql(table,con=engine,if_exists="%s"%method)
        if method=='replace':            
            print("_"*10,'the %s table has been overwritten...'%table)                  
        elif method=='append':
            print("_"*10,'the %s table has been appended...'%table)
        else:
            print("_"*10,'the %s table has been written......'%table)
    except:
        print("_"*10,'the %s table has been existed......'%table)
        
def SQLite2df(db_fp,table):
    import pandas as pd
    
    '''
    function - pandas方法，从SQLite数据库中读取表数据
    
    Paras:
        db_fp - 数据库文件路径
        table - 所要读取的表        
    '''    
    
    return pd.read_sql_table(table, 'sqlite:///'+'\\\\'.join(db_fp.split('\\'))) #pd.read_sql_table从数据库中读取指定的表

#CREATE EXTENSION postgis;    
def gpd2postSQL(gdf,table_name,**kwargs):
    from sqlalchemy import create_engine
    
    '''
    function - 将GeoDataFrame格式数据写入PostgreSQL数据库
    
    Paras:
        gdf - GeoDataFrame格式数据，含geometry字段（几何对象，点、线和面，数据值对应定义的坐标系统）
        table_name - 写入数据库中的表名
        **kwargs - 连接数据库相关信息，包括myusername（数据库的用户名），mypassword（用户密钥），mydatabase（数据库名）
    '''     
    engine=create_engine("postgres://{myusername}:{mypassword}@localhost:5432/{mydatabase}".format(myusername=kwargs['myusername'],mypassword=kwargs['mypassword'],mydatabase=kwargs['mydatabase']))  
    gdf.to_postgis(table_name, con=engine, if_exists='replace', index=False,)  
    print("_"*50)
    print('The GeoDataFrame has been written to the PostgreSQL database.The table name is {}.'.format(table_name))

def postSQL2gpd(table_name,geom_col='geometry',**kwargs):
    from sqlalchemy import create_engine
    import geopandas as gpd
    
    '''
    function - 读取PostgreSQL数据库中的表为GeoDataFrame格式数据
    
    Paras:
        table_name - 待读取数据库中的表名
        geom_col='geometry' - 几何对象，常规默认字段为'geometry'
        **kwargs - 连接数据库相关信息，包括myusername（数据库的用户名），mypassword（用户密钥），mydatabase（数据库名）
    '''
    engine=create_engine("postgres://{myusername}:{mypassword}@localhost:5432/{mydatabase}".format(myusername=kwargs['myusername'],mypassword=kwargs['mypassword'],mydatabase=kwargs['mydatabase']))  
    gdf=gpd.read_postgis(table_name, con=engine,geom_col=geom_col)
    print("_"*50)
    print('The data has been read from PostgreSQL database. The table name is {}.'.format(table_name))    
    return gdf          