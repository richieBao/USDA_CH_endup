# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 20:10:26 2022

@author: Richie Bao-caDesign设计(cadesign.cn)
"""
def gpd2postSQL(gdf,table_name,**kwargs):  
    '''
    function - 将GeoDataFrame格式数据写入PostgreSQL数据库
    
    Paras:
        gdf - GeoDataFrame格式数据，含geometry字段（几何对象，点、线和面，数据值对应定义的坐标系统）
        table_name - 写入数据库中的表名
        **kwargs - 连接数据库相关信息，包括myusername（数据库的用户名），mypassword（用户密钥），mydatabase（数据库名）
    '''    
    from sqlalchemy import create_engine
    #The URI should start with postgresql:// instead of postgres://. SQLAlchemy used to accept both, but has removed support for the postgres name.
    engine=create_engine("postgresql://{myusername}:{mypassword}@localhost:5432/{mydatabase}".format(myusername=kwargs['myusername'],mypassword=kwargs['mypassword'],mydatabase=kwargs['mydatabase']))  
    gdf.to_postgis(table_name, con=engine, if_exists='replace', index=False,)  
    print("_"*50)
    print('The GeoDataFrame has been written to the PostgreSQL database.The table name is {}.'.format(table_name))
    
def df2postSQL(df,table_name,if_exists='replace',**kwargs):
    '''
    function - 将DataFrame格式数据写入PostgreSQL数据库
    
    Paras:
        df - DataFrame格式数据
        table_name - 写入数据库中的表名
        **kwargs - 连接数据库相关信息，包括myusername（数据库的用户名），mypassword（用户密钥），mydatabase（数据库名）
    '''     
    from sqlalchemy import create_engine
    #The URI should start with postgresql:// instead of postgres://. SQLAlchemy used to accept both, but has removed support for the postgres name.
    engine=create_engine("postgresql://{myusername}:{mypassword}@localhost:5432/{mydatabase}".format(myusername=kwargs['myusername'],mypassword=kwargs['mypassword'],mydatabase=kwargs['mydatabase']))  
    conn=engine.connect()
    df.to_sql(table_name, con=conn, if_exists=if_exists,index=False)    
    # gdf.to_postgis(table_name, con=engine, if_exists='replace', index=False,)  
    print("_"*50)    
    print('The GeoDataFrame has been written to the PostgreSQL database.The table name is {}.'.format(table_name))     

def postSQL2gpd(table_name,geom_col='geometry',**kwargs):    
    '''
    function - 读取PostgreSQL数据库中的表为GeoDataFrame格式数据
    
    Paras:
        table_name - 待读取数据库中的表名
        geom_col='geometry' - 几何对象，常规默认字段为'geometry'
        **kwargs - 连接数据库相关信息，包括myusername（数据库的用户名），mypassword（用户密钥），mydatabase（数据库名）
    '''
    from sqlalchemy import create_engine
    import geopandas as gpd   
    
    engine=create_engine("postgresql://{myusername}:{mypassword}@localhost:5432/{mydatabase}".format(myusername=kwargs['myusername'],mypassword=kwargs['mypassword'],mydatabase=kwargs['mydatabase']))  
    gdf=gpd.read_postgis(table_name, con=engine,geom_col=geom_col)
    print("_"*50)
    print('The data has been read from PostgreSQL database. The table name is {}.'.format(table_name))    
    return gdf 

def postSQL2df(table_name,**kwargs):    
    '''
    function - 读取PostgreSQL数据库中的表为DataFrame格式数据
    
    Paras:
        table_name - 待读取数据库中的表名
        **kwargs - 连接数据库相关信息，包括myusername（数据库的用户名），mypassword（用户密钥），mydatabase（数据库名）
    '''
    from sqlalchemy import create_engine
    import pandas as pd   
    
    engine=create_engine("postgresql://{myusername}:{mypassword}@localhost:5432/{mydatabase}".format(myusername=kwargs['myusername'],mypassword=kwargs['mypassword'],mydatabase=kwargs['mydatabase']))  
    conn=engine.connect()
    df=pd.read_sql('SELECT * FROM {}'.format(table_name), conn)

    print("_"*50)
    print('The data has been read from PostgreSQL database. The table name is {}.'.format(table_name))    
    return df 

def cfg_load_yaml(ymlf_fp):
    '''
    读取 yaml 格式的配置文件

    Parameters
    ----------
    ymlf_fp : string
        配置文件路径.

    Returns
    -------
    cfg : yaml-dict
        读取到python中的配置信息.
    '''
    import yaml
    with open (ymlf_fp,'r') as ymlfile:
        cfg=yaml.safe_load(ymlfile)   
    return cfg

def json2gdf(json_fn,numeric_columns=None,epsg=None):
    '''
    读取.geojson(json)文件为GeoDataFrame格式文件，选择配置投影

    Parameters
    ----------
    json_fn : string
        文件路径.
    epsg : int, optional
        坐标投影系统，epsg编号. The default is None.

    Returns
    -------
    gdf : GeoDataFrmae
        转换后的GeoDataFrame格式文件.

    '''
    import geopandas as gpd
    
    gdf=gpd.read_file(json_fn)
    if epsg:
        gdf.to_crs(epsg,inplace=True)   
    print("fields_{}".format(gdf.columns))    
    if numeric_columns:
        gdf=gdf.astype(numeric_columns)

    return gdf

def shp2gdf(fn,epsg=None,boundary=None,encoding='utf-8'):    
    '''
    function - 转换.shp地理信息数据为GeoDataFrame(geopandas)数据格式，可以配置投影
    
    Paras:
        fn - .shp文件路径
        epsg - 配置投影，默认为None
        boundary - 配置裁切边界，默认为None
        encoding - 配置编码，默认为'utf-8'
    '''
    import geopandas as gpd
    
    shp_gdf=gpd.read_file(fn,encoding=encoding)
    print('original data info:{}'.format(shp_gdf.shape))
    shp_gdf.dropna(how='all',axis=1,inplace=True)
    print('dropna-how=all,result:{}'.format(shp_gdf.shape))
    shp_gdf.dropna(inplace=True)
    print('dropna-several rows,result:{}'.format(shp_gdf.shape))
    if epsg is not None:
        shp_gdf_proj=shp_gdf.to_crs(epsg=epsg)
        print(shp_gdf_proj.crs)
    if boundary:
        shp_gdf_proj['mask']=shp_gdf_proj.geometry.apply(lambda row:row.within(boundary))
        shp_gdf_proj.query('mask',inplace=True)        
    
    return shp_gdf_proj

def shp2gdf_updated(fn,boundary=None,encoding='utf-8'):    
    '''
    function - 转换.shp地理信息数据为GeoDataFrame(geopandas)数据格式，配置投影为给定边界的crs，并应用gpd.clip()方法裁切
    
    Paras:
        fn - .shp文件路径
        boundary - 配置裁切边界，默认为None
        encoding - 配置编码，默认为'utf-8'
    '''
    import geopandas as gpd
    from tqdm import tqdm
    tqdm.pandas()  
    
    shp_gdf=gpd.read_file(fn,encoding=encoding)    
    if boundary is not None:        
        shp_gdf['mask']=shp_gdf.geometry.progress_apply(lambda row:row.is_valid)
        shp_gdf=shp_gdf[shp_gdf['mask']==True]
        shp_clip_gdf=gpd.clip(shp_gdf.to_crs(boundary.crs),boundary)    
        return shp_clip_gdf
    else:
        return shp_gdf
    
def raster_reprojection(raster_fp,save_path,dst_crs):
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    import rasterio as rio
    '''
    function - 转换栅格投影
    
    Paras:
        raster_fp - 待转换投影的栅格
        dst_crs - 目标投影
        save_path - 保存路径
    '''
    with rio.open(raster_fp) as src:
        transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            "compress":'lzw',
        })
        with rio.open(save_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)      

def raster_reprojection_batch(srcs,dsts,epsg):
    '''
    批量转换栅格投影

    Parameters
    ----------
    srcs : string
        待转换投影的栅格路径名.
    dsts : string
        转换后存储栅格路径名.
    epsg : int
        坐标投影系统，epsg编号.

    Returns
    -------
    None.

    '''
    from tqdm import tqdm
    for i in tqdm(range(len(srcs))):
        raster_reprojection(srcs[i],dsts[i],epsg)    

if __name__=="__main__":
    import geopandas as gpd
    
    cfg=cfg_load_yaml('config.yml')  
    UN=cfg["postgreSQL"]["myusername"]
    PW=cfg["postgreSQL"]["mypassword"]
    DB=cfg["postgreSQL"]["mydatabase"] 
    GC='geometry'      
    
    Chicago_epsg=cfg['Chicago_epsg']
    
    #A.建筑轮廓（含层高）写入数据库
    # Building_Footprints_fn=cfg['raw_data']['Building_Footprints_fn']    
    # Building_Footprints=json2gdf(Building_Footprints_fn,numeric_columns={'no_stories':'int','stories':'int'},epsg=Chicago_epsg)
    
    # Building_Footprints_TN=cfg['table_name']['Building_Footprints_TN']    
    # BF_columns_selection=['no_stories','stories','geometry']
    # gpd2postSQL(Building_Footprints[BF_columns_selection].dropna(),table_name=Building_Footprints_TN,myusername=UN,mypassword=PW,mydatabase=DB) 
    # Building_Footprints=postSQL2gpd(table_name=Building_Footprints_TN,geom_col=GC,myusername=UN,mypassword=PW,mydatabase=DB)
    
    #B.道路中心线 
    # street_center_lines_fn=cfg['raw_data']['Street_Center_Lines_fn']
    # street_center_lines=json2gdf(street_center_lines_fn,epsg=Chicago_epsg)
    # street_center_lines_TN=cfg['table_name']['street_center_lines_TN']
    # gpd2postSQL(street_center_lines,table_name=street_center_lines_TN,myusername=UN,mypassword=PW,mydatabase=DB) 
    # street_center_lines=postSQL2gpd(table_name=street_center_lines_TN,geom_col=GC,myusername=UN,mypassword=PW,mydatabase=DB)
    
    #C.土地利用
    #研究边界
    # Chicago_boundary_fp=cfg['raw_data']['Chicago_boundary']
    # Chicago_boundary_gdf=shp2gdf(Chicago_boundary_fp,epsg=Chicago_epsg)
    # Chicago_boundary_TN=cfg['table_name']['Chicago_boundary_TN']
    # gpd2postSQL(Chicago_boundary_gdf,table_name=Chicago_boundary_TN,myusername=UN,mypassword=PW,mydatabase=DB)    
    # Chicago_boundary=postSQL2gpd(table_name=Chicago_boundary_TN,geom_col=GC,myusername=UN,mypassword=PW,mydatabase=DB)
    

    
    
    #读取土地利用数据并裁切到研究区域
    landuse_fn=cfg['raw_data']['landuse_fn']
    # landuse_gdf=shp2gdf_updated(landuse_fn,boundary=Chicago_boundary)
    # landuse_mapping=cfg['landuse']['landuse_mapping']
    # landuse_gdf['landuse_name']=landuse_gdf['LANDUSE'].map(landuse_mapping)
    
    # landuse_TN=cfg['table_name']['landuse_TN']
    # gpd2postSQL(landuse_gdf,table_name=landuse_TN,myusername=UN,mypassword=PW,mydatabase=DB)
    # landuse_gdf=postSQL2gpd(table_name=landuse_TN,geom_col=GC,myusername=UN,mypassword=PW,mydatabase=DB)
    
    #4paper
    boudary4paper_fp=r'./data/raw_data/boundary4paper/boudary4paper.shp'
    boudary4paper=gpd.read_file(boudary4paper_fp)
    lu4paper_gdf=shp2gdf_updated(landuse_fn,boundary=boudary4paper)    
    gpd2postSQL(lu4paper_gdf,'lu4paper',myusername=UN,mypassword=PW,mydatabase=DB) 
    
    #D. 树木（高，中，低）栅格数据重投影
    # HighVegetation_fn=cfg['raw_data']['HighVegetation_fn']
    # MediumVegetation_fn=cfg['raw_data']['MediumVegetation_fn']
    # LowVegetation_fn=cfg['raw_data']['LowVegetation_fn']
    # HighVegetation_reprojection_fn=cfg['processed_data']['HighVegetation_reprojection_fn']
    # MediumVegetation_reprojection_fn=cfg['processed_data']['MediumVegetation_reprojection_fn']
    # LowVegetation_reprojection_fn=cfg['processed_data']['LowVegetation_reprojection_fn']
    # raster_reprojection_batch(srcs=[HighVegetation_fn,MediumVegetation_fn,LowVegetation_fn],dsts=[HighVegetation_reprojection_fn,MediumVegetation_reprojection_fn,LowVegetation_reprojection_fn],epsg=Chicago_epsg)    
