> Created on Thu Jan  5 22:29:54 2023 @author: Richie Bao-caDesign设计(cadesign.cn)

## 2.6.7 NAIP航拍影像与分割模型库及Colaboratory和Planetary Computer Hub

### 2.6.7.1 NAIP航拍影像和构建图像数据集

（美）国家农业图像项目（National Agriculture Imagery Program，NAIP），由美国农业部农业服务局（the U.S. Department of Agriculture's Farm Service Agency，FSA）通过位于犹他州（Utah）盐湖城（Salt Lack City）航空摄影驻地办公室（Aerial Photography Field Office，APFO）管理。NAIP在农业种植季或植被生长季（leaf on），采集地面采样距离为1m分辨率的航空影像（结合具有地理参照的图像特征进行正射校正）。每个单独的图像瓦片（tile）都为3.75-分钟经度和3.75-分钟纬度的四分四边形，加上每边300m的缓冲距离。NAIP图像包含红、绿和蓝波段，且通常含有近红外波段。美国地质调查局（U.S. Geological Survey，USGS）地球资源观测与科学（ Earth Resources Observation and Science，EROS）中心发布格式为GeoTIFF和JPEG2000（内嵌地理坐标信息的压缩文件）的NAIP产品。10：1的有损压缩以减小图像大小，但图像质量也会有所下降。APFO要求图像符合制图标准，所有交付的图像均会用自动化视觉检验图像的质量，确保其准确性且符合规范<sup>①</sup>。数据至少每3年更新一次。

NAIP航拍影像获取图途径可以从USGS的[EarthExplorer](https://earthexplorer.usgs.gov/)<sup>②</sup>中检索下载；或从微软（Microsoft）的[行星计算机（Planetary Computer）](https://planetarycomputer.microsoft.com/)<sup>③</sup>上检索下载。Planetary Computer借助云的力量支持全球环境可持续发展决策，集合具有应用接口（APIs）千万亿字节的全球环境数据目录，弹性的科学环境允许用户回答有关数据的全球问题，并将其提交到相关保护利益者手中。例如官网应用中例举了全球规模的环境监测案例，使数据驱动决策成为可能，这包括全球土地利用和土地覆盖分类（Land use and Land cover，LULC），森林砍伐风险分析，生态系统监测（森林变化、栖息地连通性、土地使用等），保护规划，森林碳（排放）风险评估，基于AI的土地覆盖物分类等。

#### 1）NAIP航拍影像数据下载

此次实验所用数据为特拉华州（Delaware）的全部NAIP数据，约147个单独影像文件，而EarthExplorer不支持批量下载，因此使用[TorchGeo](https://torchgeo.readthedocs.io/en/stable/user/installation.html)<sup>④</sup>库提供的`download_url`方法（基本类似[torchvision](https://pytorch.org/vision/stable/index.html)<sup>⑤</sup>库提供的`download_url`方法）从Planetary Computer云平台下载。首先在EarthExplorer地图中绘制检索区域（Polygon方式），选择NAIP数据集（Data Sets标签），检索结果的文件名（Entity ID）和区域边界可以导出为Shapefile、CSV、Comma(,)Delimited和IKMZ等文件格式，然后Python（`GeoPandas`库）读取，根据影像获取年份，移除重叠影像，提取获得最终要下载的文件名。

参数管理使用`AttrDict()`方法（具体查看“Cityscapes数据集——参数管理”一节）。data子属性存储数据文件路径。文件名Shapefile数据存储于`naip_de_entityID`变量路径名文件夹下，NAIP影像数据存储于`Chesapeake_root`变量路径名文件夹下，对应的土地覆盖数据存储于`Chesapeake_LC`变量路径名文件夹下。

> 植被生长期（leaf-on），当树木或者灌木等生长有枝叶时采集航拍影像可以获得植被物种特有的光谱反射率（spectral reflection），可用于区分植被类型；也可以检测农作物的长势和健康状况等。植被落叶期（leaf-off），当树木或者灌木等落叶，枝头很少或者没有叶子时采集航拍影像可以更清楚的看到地面特征，有助于绘制建筑物和道路等对象。


```python
from util_misc import AttrDict
import os
__C=AttrDict() 
args=__C

__C.data=AttrDict()
__C.data.Chesapeake_root=r'E:\data\Delaware'
__C.data.Chesapeake_LC=os.path.join(args.data.Chesapeake_root,'LC')
__C.data.Chesapeake_imagery=os.path.join(args.data.Chesapeake_root,'imagery')
__C.data.naip_de_entityID='./data/naip_Delaware/naip_63b945b372d740b7.shp'
```

读取查看从EarthExplorer检索获取的NAIP数据文件名字段，包括有`NAIP Entit(y)`，和几何对象`geometry`字段。


```python
import geopandas as gpd
naip_de_entityID_gdf=gpd.read_file(args.data.naip_de_entityID)
naip_de_entityID_gdf.sort_values(by='Acquisitio',ascending=False,inplace=True)
naip_de_entityID_gdf.head(1).squeeze()
```




    NAIP Entit                         M_3907562_SW_18_060_20211019
    State                                                        DE
    Agency                                                     USDA
    Vendor                                            USDA-FSA-APFO
    Map Projec                                                  UTM
    Projection                                                  18N
    Datum                                                     NAD83
    Resolution                                    0.600000000000000
    Units                                                     METER
    Number of                                                     4
    Sensor Typ                                                 CNIR
    Project Na                202112_DELAWARE_NAIP_0X6000M_UTM_CNIR
    Acquisitio                                           2021-10-19
    Center Lat                                    39&deg;01'52.48"N
    Center Lon                                    75&deg;20'37.51"W
    NW Corner                                     39&deg;03'51.42"N
    NW Corne_1                                    75&deg;22'39.55"W
    NE Corner                                     39&deg;03'52.13"N
    NE Corne_1                                    75&deg;18'36.62"W
    SE Corner                                     38&deg;59'53.50"N
    SE Corne_1                                    75&deg;18'35.58"W
    SW Corner                                     38&deg;59'52.79"N
    SW Corne_1                                    75&deg;22'38.28"W
    Center L_1                                           39.0312443
    Center L_2                                          -75.3437527
    NW Corne_2                                           39.0642833
    NW Corne_3                                          -75.3776527
    NE Corne_2                                           39.0644805
    NE Corne_3                                          -75.3101722
    SE Corne_2                                           38.9981944
    SE Corne_3                                          -75.3098833
    SW Corne_2                                           38.9979971
    SW Corne_3                                          -75.3772999
    geometry      POLYGON ((-75.3772999 38.9979971, -75.3776527 ...
    Name: 1343, dtype: object



打印NAIP影像瓦片分布可以发现，很多瓦片重叠，一方面是不同年份的影像，另一方面是同一年份不同时间段可能也有重叠，因此需要移除重叠的瓦片。定义`IoU_2Polygons()`函数计算交并比（Intersection over Union，IoU）；定义`drop_overlapping_polygons()`函数，根据根据交并比移除过度重叠的瓦片。移除的方式是保留第一个出现的瓦片，而移除后续与之重叠的瓦片。


```python
naip_de_entityID_gdf.boundary.plot(figsize=(10,10))
```




    <AxesSubplot: >




<img src="./imgs/2_6_7/output_6_1.png" height='auto' width='auto' title="caDesign">    



```python
def IoU_2Polygons(polygon1,polygon2):
    '''
    计算两个Poygon（Shapely）对象的交并比

    Parameters
    ----------
    polygon1 : POLYGON（Shapely）
        多边形对象1.
    polygon2 : POLYGON（Shapely）
        多边形对象2.

    Returns
    -------
    iou : float
        交并比 Intersection over Union，IoU.

    '''    
    from shapely.geometry import Polygon
    
    intersect_area=polygon1.intersection(polygon2).area
    union_area=polygon1.union(polygon2).area
    iou=intersect_area/union_area
    
    return iou

def drop_overlapping_polygons(gdf_,iou=0.5):
    from tqdm import tqdm 
    '''
    移除GeoDataFrame格式文件Polygon对象重叠的行。保留第一个出现的对象，而移除后面与之重叠的对象

    Parameters
    ----------
    gdf_ : GeoDataFrame
        为Polygon对象的地理信息数据.
    iou : float, optional
        交并比（Intersection over Union，IoU）. The default is 0.5.

    Returns
    -------
    gdf_non_overlapping : GeoDataFrame
        移除重叠的Polygon后的GeoDataFrame格式数据.

    '''    
    gdf=gdf_.copy(deep=True)
    polygons_dict=gdf['geometry'].to_dict()    
    tabu_idx=[]
    for idx,row in tqdm(gdf.iterrows(),total=gdf.shape[0]):  
        tabu_idx.append(idx)
        polygons_except4one_dict={key:value for key, value in polygons_dict.items() if key not in tabu_idx}
        pg_gdf=row.geometry
        for k,pg_dict in polygons_except4one_dict.items():  
            iou_2pgs=IoU_2Polygons(pg_dict,pg_gdf)    
            if iou_2pgs>iou:
                polygons_dict.pop(k)
                
    gdf_non_overlapping=gdf.loc[list(polygons_dict.keys())]         
    return gdf_non_overlapping
```

提取2018年的数据行，并执行移除重叠多边形的操作。


```python
naip_de_entityIDd_2018=[i for i in naip_de_entityID_gdf['Acquisitio'].unique() if i.split('-')[0]=='2018']
naip_de_entityID_2018_gdf=naip_de_entityID_gdf.loc[naip_de_entityID_gdf['Acquisitio'].isin(naip_de_entityIDd_2018)]
naip_de_entityID_2018_nonOverlapping_gdf=drop_overlapping_polygons(naip_de_entityID_2018_gdf)
```

      0%|          | 0/162 [00:00<?, ?it/s]C:\Users\richi\anaconda3\envs\torchgeo3\lib\site-packages\shapely\set_operations.py:133: RuntimeWarning: invalid value encountered in intersection
      return lib.intersection(a, b, **kwargs)
    100%|██████████| 162/162 [00:00<00:00, 199.94it/s]
    

打印处理后的NAIP影像瓦片分布，检测瓦片的重叠情况，确定已经移除了重叠的瓦片，并且基本能够覆盖整个特拉华州。


```python
naip_de_entityID_2018_nonOverlapping_gdf.boundary.plot(figsize=(10,10))
```




    <AxesSubplot: >



<img src="./imgs/2_6_7/output_11_1.png" height='auto' width='auto' title="caDesign">
    



搜索位于Planetary Computer上的NAIP数据集，[NAIP: National Agriculture Imagery Program](https://planetarycomputer.microsoft.com/dataset/naip)<sup>⑥</sup>条目提供了NAIP的信息，并在`Example Notebook`标签下给出了获取数据的Python代码。这里参考`TorchGeo`库提供的方法，通过文件名（Entity ID）构建文件下载地址，因为2018年的NAIP数据位于38075和39075两个目录下，因此分别构建下载地址下载对应的影像文件，核心使用`download_url`方法，从而无需自行构建检索下载工具。


```python
from torchgeo.datasets.utils import download_url # from torchvision.datasets.utils import download_url
naip_38075_url=("https://naipeuwest.blob.core.windows.net/naip/v002/de/2018/de_060cm_2018/38075/")
naip_39075_url=("https://naipeuwest.blob.core.windows.net/naip/v002/de/2018/de_060cm_2018/39075/")
tiles=[i.lower()+'.tif' for i in naip_de_entityID_2018_nonOverlapping_gdf['NAIP Entit'].tolist()]
tiles_38075=[i for i in tiles if '38075' in i.split('_')[1]]
tiles_39075=[i for i in tiles if '39075' in i.split('_')[1]]

naip_download_rul=lambda url,tile,root:download_url(url+tile,root)

downloaded_tiles=[]
failed_tiles=[]
for tile in tiles_38075:
    try:
        naip_download_rul(naip_38075_url,tile,args.data.Chesapeake_imagery)
        downloaded_tiles.append(tile)
    except:
        failed_tiles.append(tile)
        
for tile in tiles_39075:
    try:
        naip_download_rul(naip_39075_url,tile,args.data.Chesapeake_imagery)
        downloaded_tiles.append(tile)
    except:
        failed_tiles.append(tile)        
```
Downloading https://naipeuwest.blob.core.windows.net/naip/v002/de/2018/de_060cm_2018/38075/m_3807536_ne_18_060_20181104.tif to E:\data\Delaware\imagery\m_3807536_ne_18_060_20181104.tif
100%|██████████| 508699496/508699496 [14:35<00:00, 580974.44it/s]
Downloading https://naipeuwest.blob.core.windows.net/naip/v002/de/2018/de_060cm_2018/38075/m_3807535_ne_18_060_20181104.tif to E:\data\Delaware\imagery\m_3807535_ne_18_060_20181104.tif
100%|██████████| 515336170/515336170 [14:03<00:00, 610631.50it/s]
Downloading https://naipeuwest.blob.core.windows.net/naip/v002/de/2018/de_060cm_2018/38075/m_3807527_se_18_060_20181104.tif to E:\data\Delaware\imagery\m_3807527_se_18_060_20181104.tif
100%|██████████| 514118760/514118760 [14:06<00:00, 607084.33it/s]
Downloading https://naipeuwest.blob.core.windows.net/naip/v002/de/2018/de_060cm_2018/38075/m_3807527_ne_18_060_20181104.tif to E:\data\Delaware\imagery\m_3807527_ne_18_060_20181104.tif
100%|██████████| 530246098/530246098 [14:10<00:00, 623712.25it/s]
#### 2）构建数据集并查看样本

几十年来，地球观测卫星、飞机及无人机平台一直在搜集地球表面图像，使得图像具有时空连续性，从而利于人们用遥感影像解决当今人类面临的诸多挑战，例如适应气候变化、自然灾害监测、水资源管理和全球人口日益增长下的粮食安全问题等。从计算机视觉领域则包括土地覆盖分类（语义分割）、森林砍伐和洪水监测（变化检测）、冰川流（像素跟踪）、飓风跟踪和强度估计（回归）及建筑和道路检测（物体检测、实例分割）等应用。`PyTorch Ecosystem`中的`TorchGeo`库试图同时处理深度学习模型和地理空间数据这两个截然不同领域的专业知识，为处理地理空间数据的PyTorch深度学习库，使得用深度学习处理地理空间数据工作变的简单<sup>[1]</sup>。

`TorchGeo`库提供了常用数据的下载工具，目前集成的[数据集](https://torchgeo.readthedocs.io/en/stable/api/datasets.html)<sup>⑦</sup>摘录如下：

<table class="colwidths-given docutils align-center">
<colgroup>
<col style="width: 30%" />
<col style="width: 15%" />
<col style="width: 20%" />
<col style="width: 20%" />
<col style="width: 15%" />
</colgroup>
<thead>
<tr class="row-odd"><th class="head"><p>Dataset</p></th>
<th class="head"><p>Type</p></th>
<th class="head"><p>Source</p></th>
<th class="head"><p>Size (px)</p></th>
<th class="head"><p>Resolution (m)</p></th>
</tr>
</thead>
<tbody>
<tr class="row-even"><td><p><a class="reference internal" href="#aboveground-woody-biomass">Aboveground Woody Biomass</a></p></td>
<td><p>Masks</p></td>
<td><p>Landsat, LiDAR</p></td>
<td><p>40,000x40,000</p></td>
<td><p>30</p></td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="#aster-global-dem">Aster Global DEM</a></p></td>
<td><p>Masks</p></td>
<td><p>Aster</p></td>
<td><p>3,601x3,601</p></td>
<td><p>30</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="#canadian-building-footprints">Canadian Building Footprints</a></p></td>
<td><p>Geometries</p></td>
<td><p>Bing Imagery</p></td>
<td><ul class="simple">
<li></li>
</ul>
</td>
<td><ul class="simple">
<li></li>
</ul>
</td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="#chesapeake-land-cover">Chesapeake Land Cover</a></p></td>
<td><p>Imagery, Masks</p></td>
<td><p>NAIP</p></td>
<td><ul class="simple">
<li></li>
</ul>
</td>
<td><p>1</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="#global-mangrove-distribution">Global Mangrove Distribution</a></p></td>
<td><p>Masks</p></td>
<td><p>Remote Sensing, In Situ Measurements</p></td>
<td><ul class="simple">
<li></li>
</ul>
</td>
<td><p>3</p></td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="#cropland-data-layer">Cropland Data Layer</a></p></td>
<td><p>Masks</p></td>
<td><p>Aerial</p></td>
<td><ul class="simple">
<li></li>
</ul>
</td>
<td><p>30</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="#eddmaps">EDDMapS</a></p></td>
<td><p>Points</p></td>
<td><p>Citizen Scientists</p></td>
<td><ul class="simple">
<li></li>
</ul>
</td>
<td><ul class="simple">
<li></li>
</ul>
</td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="#enviroatlas">EnviroAtlas</a></p></td>
<td><p>Imagery, Masks</p></td>
<td><p>NAIP, NLCD, OpenStreetMap</p></td>
<td><ul class="simple">
<li></li>
</ul>
</td>
<td><p>1</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="#esri2020">Esri2020</a></p></td>
<td><p>Masks</p></td>
<td><p>Sentinel-2</p></td>
<td><ul class="simple">
<li></li>
</ul>
</td>
<td><p>10</p></td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="#eu-dem">EU-DEM</a></p></td>
<td><p>Masks</p></td>
<td><p>Aster, SRTM, Russian Topomaps</p></td>
<td><ul class="simple">
<li></li>
</ul>
</td>
<td><p>25</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="#gbif">GBIF</a></p></td>
<td><p>Points</p></td>
<td><p>Citizen Scientists</p></td>
<td><ul class="simple">
<li></li>
</ul>
</td>
<td><ul class="simple">
<li></li>
</ul>
</td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="#globbiomass">GlobBiomass</a></p></td>
<td><p>Masks</p></td>
<td><p>Landsat</p></td>
<td><p>45,000x45,000</p></td>
<td><p>100</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference external" href="https://www.inaturalist.org/">iNaturalist</a></p></td>
<td><p>Points</p></td>
<td><p>Citizen Scientists</p></td>
<td><ul class="simple">
<li></li>
</ul>
</td>
<td><ul class="simple">
<li></li>
</ul>
</td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="#landsat">Landsat</a></p></td>
<td><p>Imagery</p></td>
<td><p>Landsat</p></td>
<td><p>8,900x8,900</p></td>
<td><p>30</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="#naip">NAIP</a></p></td>
<td><p>Imagery</p></td>
<td><p>Aerial</p></td>
<td><p>6,100x7,600</p></td>
<td><p>1</p></td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="#open-buildings">Open Buildings</a></p></td>
<td><p>Geometries</p></td>
<td><p>Maxar, CNES/Airbus</p></td>
<td><ul class="simple">
<li></li>
</ul>
</td>
<td><ul class="simple">
<li></li>
</ul>
</td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="#sentinel">Sentinel</a></p></td>
<td><p>Imagery</p></td>
<td><p>Sentinel</p></td>
<td><p>10,000x10,000</p></td>
<td><p>10</p></td>
</tr>
</tbody>
</table>

`TorchGeo`库参考`PyTorch Ecosystem`下的` torchvision`、` PyTorch Lightning`，同时提供了用于深度学习训练的数据集（Datasets）构建方法、样本采样（Sampler）、数据加载（DataLoader）及模型训练（Training）方式，大幅度简化了直接使用`PyTorch`构建深度学习模型从数据集（训练、测试、验证等）建立、模型构建（激活函数、模型和损失函数等）到训练的整个流程。`NAIP`方法可以直接构建NAIP航拍影像的数据集，`ChesapeakeDE`方法则可以下载来自于[切萨皮克保护协会（Chesapeake Conservancy Center，CIC）](https://www.chesapeakeconservancy.org/conservation-innovation-center/high-resolution-data/)<sup>⑧</sup>提供的对应NAIP数据土地覆盖类型数据集。CIC成立于2013年，旨在利用尖端技术赋予保护和修复环境以数据驱动力。通过`dataset=naip & chesapeake`方式可以直接对位叠加NAIP的航拍影像和对应的土地覆盖类型数据为单个的数据集，且可以用`RandomGeoSampler`等方法采样数据，返回给定大小的影像`image`及对应的分类`mask`。该土地覆盖类型包括13类<sup>[2]</sup>：

1. Water（水体）：所有开放水域，包括池塘、河流、湖泊（含不依附码头的船只）等，及小型人为的农场池塘和雨水滞留设施等。MMU=25$m^{2} $；
2. Emergent Wetlands（湿地）：位于海洋或河口的低植被区域，视觉上可以观察到包含植被的饱和地面，且位于主要水道（河流、海洋等）。对于弗吉尼亚州（Virginia）的潮汐带，该类湿地通常具有低矮的植被、木本植被和土地贫瘠（裸地）的特征。MMU=225$m^{2} $；
3. Tree Canopy（树冠/林冠）：自然演替或人工种植的落叶和常绿木本植被，高度通常超过3米。单株，离散的林木团和紧密连接的单株都计算在内。MMU=9$m^{2} $；
4. Scrub/Shrub（灌木丛）：落叶和/或常绿木本植被的异质区域，贯穿覆盖有低矮植被和草地，高低变化斑驳分布的灌木丛和幼树为特征，包括离散的灌木丛和紧密联系的单株植被。和由于环境条件发育不良矮小的灌木、幼树等，它们混长于植被较低的异质景观中。MMU=225$m^{2} $；
5. Low Vegetation（低矮植被）：高度小于约3米的植被，包括草坪、耕地、有或无防水布覆盖的苗圃种植区，近期砍伐的森林管理区，和自然地表覆盖物等。MMU=9$m^{2} $；
6. Barren（裸地）：由天然土质构成的无植被区域，包括海滩、泥滩和建筑工地的裸露地面等。MMU=25$m^{2} $；
7. Impervious Structures（不透水结构）：由不透水材料制成高度大于约2米的人造构筑，包括房屋、商城和电力塔等。MMU=9$m^{2} $；
8. Other Impervious（其它不透水类）：水不能渗透、小于约2米的人造构筑。MMU=9$m^{2} $；
9. Impervious Road（不透水路面）：为运输使用和需要维护的不透水表面。MMU=9$m^{2} $；
10. Tree Canopy over Impervious Structure（覆盖于不透水结构上的树冠）：与不透水表面重叠的森林或树木覆盖物，使得不透水结构部分部分或完全不可见。注：不透水表面和树冠层是独立绘制的，通过重叠区域识别。MMU=9$m^{2} $；
11. Tree Canopy over Other Impervious（覆盖于其它不透水结构上的树冠）：与不透水表面重叠的森林或树木覆盖物，使得不透水结构部分部分或完全不可见。MMU=9$m^{2} $；
12. Tree Canopy over Impervious Roads（覆盖于不透水路面上的树冠）：与不透水路面重叠的森林或树木覆盖物，使得不透水结构部分部分或完全不可见。MMU=9$m^{2} $；

254.Aberdeen Proving Ground（阿伯丁实验场）：该类区域无源图像或辅助数据可用，该类存在于马里兰郡的哈福德（Harford,County Maryland）。

> `PyTorch`的`Ecosystem`由来自学术界、工业界、应用程序开发人员和ML（Machine Learning）工程师开发的项目、工具和库组成，其目的是支持、加速和帮助对PyTorch的探索<sup></sup>。


```python
from torchgeo.datasets import NAIP,ChesapeakeDE,stack_samples 

naip=NAIP(args.data.Chesapeake_imagery)
chesapeake=ChesapeakeDE(args.data.Chesapeake_LC, crs=naip.crs, res=naip.res, download=False)
dataset=naip & chesapeake
```


```python
from torchgeo.samplers import RandomGeoSampler
from torch.utils.data import DataLoader

sampler=RandomGeoSampler(dataset, size=256, length=10)
dataloader=DataLoader(dataset, sampler=sampler, collate_fn=stack_samples)
for batch in dataloader:
    image=batch["image"]
    target=batch["mask"]
    break
print(image,'\n',target)
```

    tensor([[[[102,  94,  91,  ...,  87,  93,  91],
              [ 96,  89,  92,  ...,  98,  92,  99],
              [ 96,  85,  88,  ...,  96,  96,  99],
              ...,
              [110, 112, 144,  ..., 117,  63,  51],
              [102, 104, 119,  ..., 137,  90,  83],
              [113, 103, 104,  ..., 129,  81,  74]],
    
             [[121, 113, 115,  ..., 103, 112, 109],
              [114, 106, 107,  ..., 121,  98, 117],
              [117, 104, 108,  ..., 112, 102, 123],
              ...,
              [134, 127, 171,  ..., 111,  55,  63],
              [129, 122, 144,  ..., 118,  77,  92],
              [135, 109, 113,  ..., 115,  76,  73]],
    
             [[ 81,  78,  78,  ...,  78,  79,  80],
              [ 78,  74,  75,  ...,  85,  77,  81],
              [ 79,  78,  76,  ...,  87,  80,  81],
              ...,
              [ 94,  89, 129,  ..., 110,  63,  72],
              [ 94,  86,  97,  ..., 111,  72,  85],
              [ 95,  83,  86,  ..., 103,  75,  81]],
    
             [[153, 149, 150,  ..., 156, 156, 155],
              [147, 142, 148,  ..., 158, 154, 155],
              [146, 149, 144,  ..., 159, 156, 150],
              ...,
              [163, 163, 155,  ...,  60,  30,  30],
              [158, 158, 156,  ...,  66,  39,  39],
              [159, 152, 152,  ...,  57,  41,  35]]]], dtype=torch.uint8) 
     tensor([[[[5, 5, 5,  ..., 5, 5, 5],
              [5, 5, 5,  ..., 5, 5, 5],
              [5, 5, 5,  ..., 5, 5, 5],
              ...,
              [3, 3, 3,  ..., 8, 8, 8],
              [3, 3, 3,  ..., 8, 8, 8],
              [3, 3, 3,  ..., 8, 8, 8]]]], dtype=torch.uint8)
    

`unbind_samples`方法与`stack_sample`方法相逆，将小批量（mini-batch）样本转换成样本列表。


```python
from torchgeo.datasets import unbind_samples

sample=unbind_samples(batch)[0]
sample
```




    {'image': tensor([[[107, 118, 149,  ..., 103, 132, 138],
              [174, 159, 187,  ..., 130, 134, 136],
              [190, 202, 207,  ..., 103, 100,  97],
              ...,
              [ 57,  55,  64,  ..., 166, 162, 163],
              [ 57,  62,  59,  ..., 133, 146, 143],
              [ 58,  56,  60,  ..., 116, 125, 124]],
     
             [[109, 127, 129,  ..., 143, 175, 166],
              [131, 148, 160,  ..., 148, 159, 164],
              [140, 176, 190,  ..., 137, 117, 121],
              ...,
              [ 52,  45,  68,  ..., 136, 143, 144],
              [ 53,  53,  48,  ..., 117, 126, 117],
              [ 51,  54,  53,  ..., 121, 125, 114]],
     
             [[ 80,  86,  85,  ...,  80,  93,  92],
              [ 96, 110, 120,  ...,  89,  81, 102],
              [105, 140, 161,  ...,  82,  83,  73],
              ...,
              [ 68,  56, 107,  ..., 136, 125, 133],
              [ 65,  69,  61,  ...,  96, 116, 107],
              [ 63,  69,  66,  ...,  88,  92,  85]],
     
             [[193, 197, 179,  ..., 188, 206, 212],
              [168, 177, 175,  ..., 197, 212, 203],
              [157, 163, 163,  ..., 192, 180, 152],
              ...,
              [ 17,  16,  18,  ...,  52,  52,  57],
              [ 17,  18,  17,  ..., 131, 115,  97],
              [ 17,  16,  15,  ..., 191, 180, 174]]], dtype=torch.uint8),
     'crs': CRS.from_epsg(26918),
     'bbox': BoundingBox(minx=494628.6, maxx=494884.6, miny=4264922.399999999, maxy=4265178.399999999, mint=1534262399.999999, maxt=1534348799.999999),
     'mask': tensor([[[ 3,  3,  3,  ..., 10, 10, 10],
              [ 3,  3,  3,  ..., 10, 10, 10],
              [11, 11,  3,  ..., 10, 10, 10],
              ...,
              [ 8,  8,  8,  ..., 10, 10, 10],
              [ 8,  8,  8,  ..., 10, 10, 10],
              [ 8,  8,  8,  ..., 10, 10, 10]]], dtype=torch.uint8)}



分别打印样本的影像和对应的分类（标签）。


```python
import matplotlib.pyplot as plt
import torchvision.transforms as T

print(sample['image'].shape)
plt.imshow(T.ToPILImage()(sample['image'])); # plt.imshow(sample['image'][:3].T)
```

    torch.Size([4, 427, 427])
    

<img src="./imgs/2_6_7/output_21_1.png" height='auto' width='auto' title="caDesign">
    


配置土地覆盖类型的颜色字典`LC_color_dict`，并转换为`matplotlib`库的`cmap`颜色图实例，用于地图打印分类颜色。


```python
import matplotlib
import numpy as np

LC_color_dict={
    0: (0, 0, 0, 0),
    1: (0, 197, 255, 255),
    2: (0, 168, 132, 255),
    3: (38, 115, 0, 255),
    4: (76, 230, 0, 255),
    5: (163, 255, 115, 255),
    6: (255, 170, 0, 255),
    7: (255, 0, 0, 255),
    8: (156, 156, 156, 255),
    9: (0, 0, 0, 255),
    10: (115, 115, 0, 255),
    11: (230, 230, 0, 255),
    12: (255, 255, 115, 255),
    13: (197, 0, 255, 255),
    }

cmap_LC, norm=matplotlib.colors.from_levels_and_colors(list(LC_color_dict.keys()),[[v/255 for v in i] for i in LC_color_dict.values()],extend='max')
print(target.shape)
plt.imshow(np.squeeze(target),cmap=cmap_LC);
```

    torch.Size([1, 1, 427, 427])
    


<img src="./imgs/2_6_7/output_23_1.png" height='auto' width='auto' title="caDesign">    

    


可以用`naip.plot()`方法直接打印`unbind_samples`后的小批量样本。


```python
sampler_naip=RandomGeoSampler(naip, size=4096, length=3)
dataloader_naip=DataLoader(naip, sampler=sampler_naip, collate_fn=stack_samples)

i=0
for batch in dataloader_naip:
    sample=unbind_samples(batch)[0]
    naip.plot(sample['image'])
    if i==3:break
    i+=1
```


<img src="./imgs/2_6_7/output_25_0.png" height='auto' width='auto' title="caDesign">    

    

<img src="./imgs/2_6_7/output_25_1.png" height='auto' width='auto' title="caDesign">


    

<img src="./imgs/2_6_7/output_25_2.png" height='auto' width='auto' title="caDesign">
    


#### 3）图像数据增强变换（ data augmentation transforms）

[TorchVison](https://pytorch.org/vision/stable/transforms.html)<sup>⑤</sup>和[Kornia](https://github.com/kornia/kornia)<sup>⑨</sup>均为计算机视觉库，且都提供了图像增强变换的诸多方法。`TorchGeo`的`transforms`模块则针对地理空间数据多个栅格波段追加了如下指数：

`NBR`（Normalized Burn Ratio ）：$\text{NBR} = \frac{\text{NIR} - \text{SWIR}}{\text{NIR} + \text{SWIR}}$；

`NDBI`（Normalized Difference Built-up Index）：$\text{NDBI} = \frac{\text{SWIR} - \text{NIR}}{\text{SWIR} + \text{NIR}}$；

`NDSI`（Normalized Difference Snow Index）：$\text{NDSI} = \frac{\text{G} - \text{SWIR}}{\text{G} + \text{SWIR}}$；

`NDVI`（Normalized Difference Vegetation Index）：$\text{NDVI} = \frac{\text{NIR} - \text{R}}{\text{NIR} + \text{R}}$；

`NDWI`（Normalized Difference Water Index）：$\text{NDWI} = \frac{\text{G} - \text{NIR}}{\text{G} + \text{NIR}}$；

`SWI`（Standardized Water-Level Index）：$\text{SWI} = \frac{\text{VRE1} - \text{SWIR2}}{\text{VRE1} + \text{SWIR2}}$；

`GNDVI`（Green Normalized Difference Vegetation Index）：$ \text{GNDVI} = \frac{\text{NIR} - \text{G}}{\text{NIR} + \text{G}}$；

`BNDVI`（Blue Normalized Difference Vegetation Index ）：$\text{BNDVI} = \frac{\text{NIR} - \text{B}}{\text{NIR} + \text{B}}$；

`NDRE`（Normalized Difference Red Edge Vegetation Index）：$text{NDRE} = \frac{\text{NIR} - \text{VRE1}}{\text{NIR} + \text{VRE1}}$；

`GRNDVI`（Green-Red Normalized Difference Vegetation Index）：$ \text{GRNDVI} =\frac{\text{NIR} - (\text{G} + \text{R})}{\text{NIR} + (\text{G} + \text{R})}$；

`GBNDVI`（Green-Blue Normalized Difference Vegetation Index）：$\text{GBNDVI} =\frac{\text{NIR} - (\text{G} + \text{B})}{\text{NIR} + (\text{G} + \text{B})}$；

`RBNDVI`（Red-Blue Normalized Difference Vegetation Index）：$\text{RBNDVI} =\frac{\text{NIR} - (\text{R} + \text{B})}{\text{NIR} + (\text{R} + \text{B})}$。

下述代码调用了`kornia`库提供的图像增强变换方法，并追加了`NDVI`和`NDWI`两个指数变换图像。


```python
dataloader=DataLoader(dataset, sampler=sampler, collate_fn=stack_samples)
dataloader=iter(dataloader)
batch=next(dataloader)
x, y=batch["image"], batch["mask"]
```


```python
from torchgeo.transforms import AugmentationSequential, indices
import kornia.augmentation as K
import torch.nn as nn

augmentations=AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    # K.RandomAffine(degrees=(0, 90), p=0.25),
    # K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.25),
    # K.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0), p=0.25),
    data_keys=["image",'mask'],
)
transforms=nn.Sequential(
    indices.AppendNDVI(index_nir=3, index_red=0),
    indices.AppendNDWI(index_green=1, index_nir=3),
    augmentations,
)

batch=next(dataloader)
print(batch["image"].shape)
batch=transforms(batch)
print(batch["image"].shape)
```

    torch.Size([1, 4, 427, 427])
    torch.Size([1, 6, 427, 427])
    

`nvidia-smi`（NVSMI）为NVIDIA显卡（例如Tesla, Quadro, GRID和GeForce等）设备提供监控和管理功能。下述打印的信息显示了显卡驱动（driver）版本，CUDA（Compute Unified Device Architecture)）版本等相关信息。


```python
!nvidia-smi
```

    Wed Jan 11 17:54:04 2023       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 512.78       Driver Version: 512.78       CUDA Version: 11.6     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  NVIDIA GeForce ... WDDM  | 00000000:01:00.0  On |                  N/A |
    | N/A   64C    P0    37W /  N/A |   3320MiB /  8192MiB |      1%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |    0   N/A  N/A      1144    C+G   ...ekyb3d8bbwe\HxOutlook.exe    N/A      |
    |    0   N/A  N/A      5804    C+G   ...n1h2txyewy\SearchHost.exe    N/A      |
    |    0   N/A  N/A      6956    C+G   ...d\runtime\WeChatAppEx.exe    N/A      |
    |    0   N/A  N/A      8908    C+G   ...zilla Firefox\firefox.exe    N/A      |
    |    0   N/A  N/A      9428    C+G   ...zilla Firefox\firefox.exe    N/A      |
    |    0   N/A  N/A     12076    C+G   C:\Windows\explorer.exe         N/A      |
    |    0   N/A  N/A     12320    C+G   ...462.76\msedgewebview2.exe    N/A      |
    |    0   N/A  N/A     13148    C+G   ...artMenuExperienceHost.exe    N/A      |
    |    0   N/A  N/A     13856    C+G   ...lPanel\SystemSettings.exe    N/A      |
    |    0   N/A  N/A     14928    C+G   ...e\Current\LogiOverlay.exe    N/A      |
    |    0   N/A  N/A     15388    C+G   ...ystemEventUtilityHost.exe    N/A      |
    |    0   N/A  N/A     15912    C+G   ...2txyewy\TextInputHost.exe    N/A      |
    |    0   N/A  N/A     16884    C+G   ...e\PhoneExperienceHost.exe    N/A      |
    |    0   N/A  N/A     17076    C+G   ...y\ShellExperienceHost.exe    N/A      |
    |    0   N/A  N/A     17812    C+G   ...ge\Application\msedge.exe    N/A      |
    |    0   N/A  N/A     18776    C+G   ...ram Files\LGHUB\lghub.exe    N/A      |
    |    0   N/A  N/A     19372    C+G   ...ray\lghub_system_tray.exe    N/A      |
    |    0   N/A  N/A     20280    C+G   ...arkupHero\Markup Hero.exe    N/A      |
    |    0   N/A  N/A     21124    C+G   ...cw5n1h2txyewy\LockApp.exe    N/A      |
    |    0   N/A  N/A     22632    C+G   ...me\Application\chrome.exe    N/A      |
    |    0   N/A  N/A     27516    C+G   ...tracted\WechatBrowser.exe    N/A      |
    |    0   N/A  N/A     27820    C+G   ...wekyb3d8bbwe\Video.UI.exe    N/A      |
    |    0   N/A  N/A     31488    C+G   ...root\Office16\WINWORD.EXE    N/A      |
    |    0   N/A  N/A     34900    C+G   ...icrosoft VS Code\Code.exe    N/A      |
    +-----------------------------------------------------------------------------+
    

将图像增强变换`nn.Sequential`对象（按照构造函数中传递的对象顺序添加到模块容器中）通过`sequential.to(device[cuda])`转换到GPU中进行计算。打印原始图像和增强变换后的图像。


```python
import torch
from copy import deepcopy

device="cuda" if torch.cuda.is_available() else "cpu"
print(device)
transforms_gpu=deepcopy(transforms).to(device)
```

    cuda
    


```python
import copy
sampler_naip=RandomGeoSampler(naip, size=4096, length=1)
dataloader=DataLoader(dataset, sampler=sampler, collate_fn=stack_samples)

fig, axs=plt.subplots(2,figsize=(10,10))
for n in dataloader:
    batch=copy.deepcopy(n)

    print(batch['image'].shape)
    axs[0].imshow(T.ToPILImage()(batch['image'][0][:3]));
    batch=transforms_gpu(batch)
    print(batch['image'].shape)
    axs[1].imshow(T.ToPILImage()(batch['image'][0][2:6]));    
    break
```

    torch.Size([1, 4, 427, 427])
    torch.Size([1, 6, 427, 427])
    

<img src="./imgs/2_6_7/output_33_1.png" height='auto' width='auto' title="caDesign">


打印真实分类。


```python
plt.imshow(T.ToPILImage()(batch['mask'][0]));
```

<img src="./imgs/2_6_7/output_35_0.png" height='auto' width='auto' title="caDesign">



### 2.6.7.2 分割模型库及Colaboratory和Planetary Computer Hub

#### 1）Segmentation Models

深度学习的不断发展更新，新的网络不断涌现，大量被证实可用的深度学习网络广泛传播，因此集成已有成熟的网络（通常包含预训练参数权重）库为网络的实际应用带来便利，例如前文涉及到的[ONNX Model Zoo](https://github.com/onnx/models)<sup>⑩</sup>、[torchvision.models](https://pytorch.org/vision/stable/models.html)<sup>⑪</sup>等，及[PyTorch Image Models（TIMM）](https://github.com/rwightman/pytorch-image-models)<sup>⑫</sup>、[Transformers](https://github.com/huggingface/transformers)<sup>⑬</sup>和[Segmentation Models](https://smp.readthedocs.io/en/stable/index.html)<sup>⑭</sup>等，且不仅上述所限。以图像语义分割为主的`Segmentation Models`为例，可以仅用几行代码调用预训练的模型，涉及的分割网络包括Unet、Unet++、MAnet、Linknet、FPN、PSPNet、PAN、DeepLabV3和DeepLabV3+等；且包括高达124个可用的编码器（encoders）,所有编码器都有预训练的参数权重，以实现更快更好的模型训练收敛，例如`ResNet（imagenet / ssl / swsl）`、`ResNeXt（imagenet / instagram / ssl / swsl）`、`ResNeSt（imagenet）`、`Res2Ne(X)t（imagenet）`、`RegNet(x/y)（imagenet）`、`GERNet（imagenet）`、`SE-Net（imagenet）`、`SK-ResNe(X)t（imagenet）`、`DenseNet（imagenet）`、`Inception（imagenet / imagenet+background）`、`EfficientNet（imagenet / advprop / noisy-student）`、`MobileNet（imagenet）`、`DPN（imagenet+5k）`、`VGG（imagenet）`、`Mix Vision Transformer（imagenet）`和`MobileOne（imagenet）`等。其中预训练模型最常用的图像数据库为[ImageNet](https://www.image-net.org/index.php)<sup>⑮</sup>（仅包含名词），为按照[WordNet](https://en.wikipedia.org/wiki/WordNet)<sup>⑯</sup>层次结构组织的图像数据库，其中层次结构的每个节点由成百上千的图像描述。该项目在推进计算机视觉和深度学习研究方面发挥了重要作用。（`ImageNet`图像数据库开源用于非商业用途的研究）

下述示例调用网络构建模型和预测（假值）的代码，并打印了该`Unet`网络模型，其中编码器使用了`resnet34`，预训练参数权重来自于图像数据库`imagenet`。预测的结果为`mask`掩码。


```python
import segmentation_models_pytorch as smp
import torch

model=smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=13,                     # model output channels (number of classes in your dataset)
)
mask=model(torch.ones([1, 3,64, 64]))
print(mask[:,0,:])
print('-'*50)
print(model)
```

    tensor([[[-0.0060, -0.1177, -0.1687,  ...,  0.3846,  0.0027,  0.0376],
             [ 0.4949, -0.3818,  0.0144,  ...,  0.3487,  0.0475, -0.0637],
             [ 0.1829, -0.9314, -0.4307,  ...,  1.1400,  0.3028, -0.0593],
             ...,
             [ 0.0888, -0.0988, -0.2369,  ..., -0.4098, -0.2458, -0.5457],
             [-0.6411, -0.4427, -0.9519,  ..., -0.9773, -0.4209, -0.2852],
             [-0.1010, -0.4232, -0.8313,  ..., -0.2045,  0.1289,  0.0031]]],
           grad_fn=<SliceBackward0>)
    --------------------------------------------------
    Unet(
      (encoder): ResNetEncoder(
        (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (layer1): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicBlock(
            (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (2): BasicBlock(
            (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (layer2): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (downsample): Sequential(
              (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): BasicBlock(
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (2): BasicBlock(
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (3): BasicBlock(
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (layer3): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (downsample): Sequential(
              (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): BasicBlock(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (2): BasicBlock(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (3): BasicBlock(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (4): BasicBlock(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (5): BasicBlock(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (layer4): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (downsample): Sequential(
              (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): BasicBlock(
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (2): BasicBlock(
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (decoder): UnetDecoder(
        (center): Identity()
        (blocks): ModuleList(
          (0): DecoderBlock(
            (conv1): Conv2dReLU(
              (0): Conv2d(768, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (attention1): Attention(
              (attention): Identity()
            )
            (conv2): Conv2dReLU(
              (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (attention2): Attention(
              (attention): Identity()
            )
          )
          (1): DecoderBlock(
            (conv1): Conv2dReLU(
              (0): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (attention1): Attention(
              (attention): Identity()
            )
            (conv2): Conv2dReLU(
              (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (attention2): Attention(
              (attention): Identity()
            )
          )
          (2): DecoderBlock(
            (conv1): Conv2dReLU(
              (0): Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (attention1): Attention(
              (attention): Identity()
            )
            (conv2): Conv2dReLU(
              (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (attention2): Attention(
              (attention): Identity()
            )
          )
          (3): DecoderBlock(
            (conv1): Conv2dReLU(
              (0): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (attention1): Attention(
              (attention): Identity()
            )
            (conv2): Conv2dReLU(
              (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (attention2): Attention(
              (attention): Identity()
            )
          )
          (4): DecoderBlock(
            (conv1): Conv2dReLU(
              (0): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (attention1): Attention(
              (attention): Identity()
            )
            (conv2): Conv2dReLU(
              (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (attention2): Attention(
              (attention): Identity()
            )
          )
        )
      )
      (segmentation_head): SegmentationHead(
        (0): Conv2d(16, 13, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): Identity()
        (2): Activation(
          (activation): Identity()
        )
      )
    )
    

所有编码器都有预训练的权重。以与权重训练相同的方式准备数据可能使得训练更快的收敛且获得更好的预测准确度（各类测量指标，例如precision（查准率）、recall（召回率）、F1-Score（（F1-分数）和Intersection over Uion（IOU）等）。但是如果训练整个模型，而不仅是解码器则没有必要。


```python
from segmentation_models_pytorch.encoders import get_preprocessing_fn
preprocess_input=get_preprocessing_fn('resnet18', pretrained='imagenet')
print(preprocess_input)
```

    functools.partial(<function preprocess_input at 0x000001B11D4C4AF0>, input_space='RGB', input_range=[0, 1], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    

输入通道`in_channels`数量根据训练图像数据确定，如果预训练权重来自于`ImageNet`，第一个卷积层的权重将被重复使用。对于单通道（`in_channels=1`）的情况，为第一个卷积层的权重之和，否则通道权重类似为`new_weight[:, i] = pretrained_weight[:, i % 3]`，然后用`new_weight * 3 / new_in_channels`缩放。


```python
model=smp.FPN('resnet34', in_channels=1)
mask=model(torch.ones([1, 1, 64, 64]))
print(mask,mask.shape)
```

    tensor([[[[0.6180, 0.5185, 0.4191,  ..., 0.6451, 0.4130, 0.1810],
              [0.6129, 0.5560, 0.4990,  ..., 1.0183, 0.7343, 0.4504],
              [0.6078, 0.5934, 0.5790,  ..., 1.3915, 1.0556, 0.7197],
              ...,
              [1.6890, 1.6174, 1.5457,  ..., 2.0490, 1.9673, 1.8856],
              [1.8117, 1.7673, 1.7229,  ..., 2.1160, 2.0245, 1.9329],
              [1.9345, 1.9173, 1.9000,  ..., 2.1830, 2.0816, 1.9802]]]],
           grad_fn=<UpsampleBilinear2DBackward1>) torch.Size([1, 1, 64, 64])
    

所有模型都支持`aux_params`辅助分类输出参数，默认值为None，不创建分类辅助输出；如果配置了该参数，模型不仅产生掩码（`mask`），且产生分类标签（联合预测概率），分类头（Classification head）由`GlobalPooling->Dropout(optional)->Linear->Activation(optional)`等层组成。


```python
aux_params=dict(
    pooling='avg',             # one of 'avg', 'max'
    dropout=0.5,               # dropout ratio, default is None
    activation='sigmoid',      # activation function, default is None
    classes=3,                 # define number of output labels
)
model=smp.Unet('resnet34', classes=3, aux_params=aux_params)
mask,label=model(torch.ones([1, 3, 64, 64]))
print(mask.shape)
print(label)
```

    torch.Size([1, 3, 64, 64])
    tensor([[0.0278, 0.3745, 0.2524]], grad_fn=<SigmoidBackward0>)
    

`encoder_depth`深度参数可以配置下采样大小，较小的值可以减轻模型的计算量。


```python
model=smp.DeepLabV3Plus('resnet34', encoder_depth=3)
```

#### 2）Colaboratory和Planetary Computer Hub

深度学习神经网络的参数量通常在百万级之上（例如包括卷积层参数、全连接层参数、输出层参数及训练和测试的参数等），因此对硬件通常有必要的需求，而神经网络是高度并行的，那么适合于并行计算的GPU（Graphics processing unit）图形处理器成为深度学习的重要依赖，在CPU上需要花费几个小时完成的训练任务，在GPU上只需要10几分钟；那么对于GPU需要几天甚至几周完成的训练任务，很难想象用CPU计算完成。目前个人电脑根据需要通常配置有GPU，但是其显存大小、位宽、带宽和计算能力影响到GPU的效能，高的配置可以减小深度学习网络训练的时间，尤其无法避免反复调参的情况下，高效率的GPU配置将大幅度减少等待时间。高配置的GPU往往价格偏高，对于个人使用可能会受到限制；如果科研团队配置有高配的GPU，但是数量可能有限，如果有经常训练模型的需求，团队公用的GPU也限制了使用的自由，因此[Colaborator，CoLab](https://colab.research.google.com/)<sup>⑰</sup>（由Google推出）和[Planetary Computer Hub，PcHub](https://pccompute.westeurope.cloudapp.azure.com/compute/hub)<sup>⑱</sup>（由Microsoft推出）等云端提供的算力则解决了这一问题，同时以社区方式的平台建立，也大大增加了深度学习相关研究的推进和开源分享。CoLab和PcHub均集成了Jupyter笔记服务，且预先安装了常用的Python库，这包括PyTorch和TenserFlow等深度学习库。

CoLab通常配合Google Drive使用，可以实现数据集和训练模型的无缝连接，避免数据丢失（重新下载或上传），尤其训练好的模型丢失造成的损失。CoLab的另一优势在于可以访问计算机系统层，这对于需要修改Pytho库代码的用户而言是必要的。相比CoLab，Hub无法访问系统层，但Hub集成了很多全球环境数据目录，例如naip、sentinel、modis、landsat、us-census、noaa、eclipse等几十种，可以方便快捷的调用数据用于相关分析。

以NAIP航拍影像语义分割为例，使用CoLab书写代码、训练模型，配合Google Drive存储数据和训练结果。分割模型调用`Segmentation Models`库提供的Unet网络，使用`TorchGeo`和`Pytorcch Lighting`库处理。因为`TorchGeo`库目前处于开发上升期，因此有些不方便应用的方法则根据本次实验进行调整。Pyton库无以计数，不同的库处于不同的状态，例如`Pytorch`、`Pandas`等稳定型，有大量使用者和大量开发者维护；有些库则处于开发上升期，代码变更可能会比较频繁，有些方法在前一版本存在，在后一版本可能就会取消，又或者变更到其它模块，或者修改了名称，及调整了参数和返回值等；还有些库已经停止了维护及更新，但有些功能在特定的分析中却非常有用。因此，面对不同库的状态，进行数据分析时，往往也会对调用的库进行修改，修改的方式主要包括两种，一种是，直接在安装环境下修改库代码；另一种是迁移库代码，结合自身分析调整代码等。

> 下述代码为CoLab中执行代码

* 库的安装

因为CoLab没有预先安装`TorchGeo`库，因此需要安装；当前版本的`Matplotlib`库的`plt.imshow`方法提示错误，因此更新到3.6.2版本。


```python
%pip install torchgeo
%pip install matplotlib==3.6.2
```

需要修改`TorchGeo`库的部分代码，满足分析需要，可以通过`!pip show torchgeo`获取该库位置。


```python
!pip show torchgeo
```

    Name: torchgeo
    Version: 0.3.1
    Summary: TorchGeo: datasets, samplers, transforms, and pre-trained models for geospatial data
    Home-page: https://github.com/microsoft/torchgeo
    Author: Adam J. Stewart
    Author-email: ajstewart426@gmail.com
    License: 
    Location: /usr/local/lib/python3.8/dist-packages
    Requires: einops, fiona, kornia, matplotlib, numpy, omegaconf, packaging, pillow, pyproj, pytorch-lightning, rasterio, rtree, scikit-learn, segmentation-models-pytorch, shapely, timm, torch, torchmetrics, torchvision
    Required-by: 
    

* 连接Google Drive

为避免数据丢失，方便数据调用，连接Google Drive到CoLab，文件夹会加载到CoLab本地显示，用法同本地文件。


```python
from google.colab import drive
drive.mount('/content/gdrive')
```

    Mounted at /content/gdrive
    

* 配置数据存储路径

数据文件、训练过程和结果文件均存储于Google Drive中，文件夹可以在Googel Drive中建立，也可以直接在CoLab中建立。文件路径可以在对应文件上右键`Copy path`直接获取。


```python
import os
imagery_data=os.path.join('/content/gdrive/MyDrive/data/delaware', "imagery") # 存储NAIP航拍影像数据
LC_data=os.path.join('/content/gdrive/MyDrive/data/delaware', "LC") # 存储NAIP土地覆盖分类数据
data_dir=os.path.join('/content/gdrive/MyDrive/data/delaware', "training") # 存储模型训练过程和结果文件
```

* NAIP图像数据下载

训练时未使用全部Delaware区域影像，仅选取了4个瓦片。将瓦片编号录入文本文件`naipEntityID_selection.txt`下，传至Google Drive用于读取并下载对应图像数据至Google Drive。下载时直接调用`torchvision`库的`download_url`方法。

<img src="./imgs/2_6_7/2_6_7_01.png" height='auto' width='auto' title="caDesign">


```python
naipEntityID_selection_fn=r'/content/gdrive/MyDrive/data/delaware/naipEntityID_selection.txt'
with open(naipEntityID_selection_fn,'r') as f:
    naipEntityID_selection=f.readlines()
naipEntityID_selection=[line.rstrip() for line in naipEntityID_selection]   
naipEntityID_selection
```




    ['m_3807505_se_18_060_20180827.tif',
     'm_3807505_sw_18_060_20180815.tif',
     'm_3807504_se_18_060_20180815.tif',
     'm_3807504_sw_18_060_20180815.tif']




```python
from torchvision.datasets.utils import download_url
naip_38075_url=("https://naipeuwest.blob.core.windows.net/naip/v002/de/2018/de_060cm_2018/38075/")

naip_download_rul=lambda url,tile,root:download_url(url+tile,root)
for tile in naipEntityID_selection:
    try:
        naip_download_rul(naip_38075_url,tile,imagery_data)
    except:
        print(f'Can not access to:{tile}')
```

    Using downloaded and verified file: /content/gdrive/MyDrive/data/delaware/imagery/m_3807505_se_18_060_20180827.tif
    Using downloaded and verified file: /content/gdrive/MyDrive/data/delaware/imagery/m_3807505_sw_18_060_20180815.tif
    Downloading https://naipeuwest.blob.core.windows.net/naip/v002/de/2018/de_060cm_2018/38075/m_3807504_se_18_060_20180815.tif to /content/gdrive/MyDrive/data/delaware/imagery/m_3807504_se_18_060_20180815.tif
    


      0%|          | 0/485891786 [00:00<?, ?it/s]


    Downloading https://naipeuwest.blob.core.windows.net/naip/v002/de/2018/de_060cm_2018/38075/m_3807504_sw_18_060_20180815.tif to /content/gdrive/MyDrive/data/delaware/imagery/m_3807504_sw_18_060_20180815.tif
    


      0%|          | 0/506248627 [00:00<?, ?it/s]


* 对应NAIP的土地覆盖类型数据下载

`TorchGeo`库下载土地覆盖类型数据时，直接将其转换为训练数据集，这里单独下载为初始的ZIP压缩文件，解压后再单独建立训练数据集。


```python
chesapeakebay_landcover_url="https://cicwebresources.blob.core.windows.net/chesapeakebaylandcover"
base_folder="DE"
zipfile="_DE_STATEWIDE.zip"
chesapeakebay_landcover_url += f"/{base_folder}/{zipfile}"
```


```python
download_url(chesapeakebay_landcover_url, LC_data, filename=zipfile) 
```

    Downloading https://cicwebresources.blob.core.windows.net/chesapeakebaylandcover/DE/_DE_STATEWIDE.zip to /content/gdrive/MyDrive/data/delaware/LC/_DE_STATEWIDE.zip
    


      0%|          | 0/287350495 [00:00<?, ?it/s]


* 迁移解压缩代码

因为下载的土地覆盖类型数据为ZIP格式压缩文件，因此迁移`TorchGeo`库已有解压缩代码，位于`torchgeo.datasets.utils`模块。


```python
import tarfile
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)

class _rarfile:
    class RarFile:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

        def __enter__(self) -> Any:
            try:
                import rarfile
            except ImportError:
                raise ImportError(
                    "rarfile is not installed and is required to extract this dataset"
                )

            # TODO: catch exception for when rarfile is installed but not
            # unrar/unar/bsdtar
            return rarfile.RarFile(*self.args, **self.kwargs)

        def __exit__(self, exc_type: None, exc_value: None, traceback: None) -> None:
            pass


class _zipfile:
    class ZipFile:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

        def __enter__(self) -> Any:
            try:
                # Supports normal zip files, proprietary deflate64 compression algorithm
                import zipfile_deflate64 as zipfile
            except ImportError:
                # Only supports normal zip files
                # https://github.com/python/mypy/issues/1153
                import zipfile  # type: ignore[no-redef]

            return zipfile.ZipFile(*self.args, **self.kwargs)

        def __exit__(self, exc_type: None, exc_value: None, traceback: None) -> None:
            pass


def extract_archive(src: str, dst: Optional[str] = None) -> None:
    """Extract an archive.

    Args:
        src: file to be extracted
        dst: directory to extract to (defaults to dirname of ``src``)

    Raises:
        RuntimeError: if src file has unknown archival/compression scheme
    """
    if dst is None:
        dst = os.path.dirname(src)

    suffix_and_extractor: List[Tuple[Union[str, Tuple[str, ...]], Any]] = [
        (".rar", _rarfile.RarFile),
        (
            (".tar", ".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tbz2", ".tbz", ".txz"),
            tarfile.open,
        ),
        (".zip", _zipfile.ZipFile),
    ]

    for suffix, extractor in suffix_and_extractor:
        if src.endswith(suffix):
            with extractor(src, "r") as f:
                f.extractall(dst)
            return

    suffix_and_decompressor: List[Tuple[str, Any]] = [
        (".bz2", bz2.open),
        (".gz", gzip.open),
        (".xz", lzma.open),
    ]

    for suffix, decompressor in suffix_and_decompressor:
        if src.endswith(suffix):
            dst = os.path.join(dst, os.path.basename(src).replace(suffix, ""))
            with decompressor(src, "rb") as sf, open(dst, "wb") as df:
                df.write(sf.read())
            return

    raise RuntimeError("src file has unknown archival/compression scheme")
```


```python
extract_archive(os.path.join(LC_data, zipfile))
```

* 构建包含图像增强变换的训练数据集

`TorchGeo`库设计时对图像增强变换的操作有两个途径，一个是构建训练数据集时，且融入到了具体的数据类型下，例如`torchgeo.datasets.NAIP(root, crs=None, res=None, transforms=None, cache=True)`集成了NAIP图像下载、图像增强变换和构建训练数据集等多个功能；另一个是用`DataLoader`加载小批量训练样本时，直接对小批量样本执行图像增强变换操作。在后续训练时调用`pytorch_lightning`库提供的`Trainer`类，其`fit(model, train_dataloaders=None, val_dataloaders=None, datamodule=None, ckpt_path=None)`方法提供`datamodule`，为`LightningDataModule`类的一个实例对象。`TorchGeop`构建的`NAIPChesapeakeDataModule(pl.LightningDataModule)`类继承了`LightningDataModule`类，`NAIPChesapeakeDataModule`构建时融合了土地覆盖类型数据的下载，并构建训练数据集，和NAIP训练数据集的构建，及一些图像增强变换操作，这些融合的方式虽然大幅度增加了数据处理的效率，但混杂也带来了很多数据处理分步的不清晰，和不便利，因此下述迁移更新了`NAIPChesapeakeDataModule`类，定义为为`geo_datamodule(pl.LightningDataModule)`。新定义的`geo_datamodule`类，拆离了训练数据集构建，从而把图像增强变换仅作用于训练数据集构建过程中，而不再混杂于`datamodule`的构建过程。这样也方便观察图像增强变换后的数据结果，及用于图像预测时输入数据的变换。

土地覆盖类型中，树冠类有很多子类很难通过航拍影像识别，如果仍然保留进行训练将会影响训练收敛和预测精度，因此定义`labes_merge()`函数作为图像增强变换输入条件，合并部分分类。下述定义的变量`label_merge_mapping`合并了树冠类的各个子类，及灌丛和低矮植被，和不透水人造对象。同时，调整了标签数值，排序为1至8，缩减为8个土地覆盖类。分类数量由13类缩减为8类，也会提高训练收敛的速度及预测的精度。


```python
# 迁移于TorchGeo库：torchgeo.datamodules.naip模块

def chesapeake_transform(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Transform a single sample from the Chesapeake Dataset.

    Args:
        sample: Chesapeake mask dictionary

    Returns:
        preprocessed Chesapeake data
    """
    sample["mask"] = sample["mask"].long()[0]

    return sample

def remove_bbox(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Removes the bounding box property from a sample.

    Args:
        sample: dictionary with geographic metadata

    Returns
        sample without the bbox property
    """
    del sample["bbox"]
    return sample

def labes_merge(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Merge land cover classifications.

    Args:
        sample: dictionary with geographic metadata

    Returns
        sample without the bbox property
    """    

    label_merge_mapping={1:1, # Water ->Water
                        2:2, # Emergent Wetlands -> Emergent Wetlands
                        3:3, # Tree Canopy -> Tree Canopy
                        4:4, # Scrub/Shrub -> Scrub/Shrub
                        5:4, # Low Vegetation -> Scrub/Shrub
                        6:5, # Barren -> Barren(6->5)
                        7:6, # Impervious Structures -> Impervious Structures(7->6)
                        8:6, # Other Impervious -> Impervious Structures(7->6)
                        9:7, # Impervious Road -> Impervious Road(9->7)
                        10:3, # Tree Canopy over Impervious Structure -> Tree Canopy
                        11:3, # Tree Canopy over Other Impervious -> Tree Canopy
                        12:3, # Tree Canopy over Impervious Roads -> Tree Canopy
                        254:8, # Aberdeen Proving Ground -> Aberdeen Proving Ground(254->8)
                        }
    old=sample["mask"]
    indexer=np.array([label_merge_mapping.get(i, -1) for i in range(old.min(), old.max() + 1)])
    new=torch.from_numpy(indexer[(old - old.min())])
    sample["mask"]=new
    return sample
```

`TorchGeo`定义`RasterDataset`类时，输入的参数部分为全局变量（位于类下而于函数外），因此定义类的方式对变量进行赋值，例如对土地覆盖类型构建`TorchGeo`的`RasterDataset`栅格数据集，建立类`delaware_lc_rd(RasterDataset)`，继承`RasterDataset`类，并对父类的`filename_glob`变量进行更新。同时增加迁移的上述两个变换`chesapeake_transform`和`remove_bbox`。


```python
from torchvision.transforms import Compose
from torchgeo.datasets import RasterDataset

chesapeak_transforms=Compose([chesapeake_transform,remove_bbox])

class delaware_lc_rd(RasterDataset):
    filename_glob="DE_STATEWIDE.tif"
    is_image=False

chesapeake=delaware_lc_rd(LC_data,transforms=chesapeak_transforms) 
```


```python
print(f"crs:{chesapeake.crs}\nres:{chesapeake.res}")
```

    crs:ESRI:102039
    res:1.0
    

同样的方式操作NAIP图像，在图像增强变换时，增加了NDVI和NDWI两个指数层。

> 图像增强变换的内容和组合结构需要根据训练效果不断调整实验，使得训练收敛且提高预测精度。


```python
from torchgeo.transforms import indices
import kornia.augmentation as K

def naip_preprocess(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Transform a single sample from the NAIP Dataset.

    Args:
        sample: NAIP image dictionary

    Returns:
        preprocessed NAIP data
    """
    sample["image"] = sample["image"].float()
    sample["image"] /= 255.0

    return sample

class naip_rd(RasterDataset):
    filename_glob = "m_*.*"
    filename_regex = r"""
        ^m
        _(?P<quadrangle>\d+)
        _(?P<quarter_quad>[a-z]+)
        _(?P<utm_zone>\d+)
        _(?P<resolution>\d+)
        _(?P<date>\d+)
        (?:_(?P<processing_date>\d+))?
        \..*$
    """
    is_image=True

naip_transforms=Compose([naip_preprocess,remove_bbox,indices.AppendNDVI(index_nir=3, index_red=0),indices.AppendNDWI(index_green=1, index_nir=3)])
naip=naip_rd(imagery_data,chesapeake.crs,chesapeake.res,naip_transforms)
```


```python
print(f"crs:{naip.crs}\nres:{naip.res}")
```

    crs:ESRI:102039
    res:1.0
    

构建包含图像和类标的训练数据集。`TorchGeo`库重写了`__and__`、`__or__`等方法，可以直接对地理空间数据集执行交集和并集的操作。


```python
dataset=chesapeake & naip
```

* 查看图像增强变换后的数据集样本

因为增加了NDVI和NDWI两个指数层，因此数组的纬度为`[1, 6, 256, 256]`，即包括R、G、B、Nir及NDVI和NDWI等6个波段。


```python
from torchgeo.samplers import RandomGeoSampler
from torchgeo.datasets import stack_samples 
from torch.utils.data import DataLoader

sampler=RandomGeoSampler(dataset,size=256, length=10)
dataloader=DataLoader(dataset, sampler=sampler, collate_fn=stack_samples)

dataloader=iter(dataloader)
batch=next(dataloader)
print(batch['image'].shape)
```

    torch.Size([1, 6, 256, 256])
    


```python
import matplotlib.pyplot as plt
import torchvision.transforms as T

batch=next(dataloader)
print(batch['image'].shape)
plt.imshow(T.ToPILImage()(batch['image'][0][4:]));
```

    torch.Size([1, 6, 256, 256])
    torch.Size([1, 6, 256, 256])
    

<img src="./imgs/2_6_7/output_76_1.png" height='auto' width='auto' title="caDesign">
    



```python
import matplotlib
import numpy as np

LC_color_dict={
    0: (0, 0, 0, 0),
    1: (0, 197, 255, 255),
    2: (0, 168, 132, 255),
    3: (38, 115, 0, 255),
    4: (76, 230, 0, 255),
    5: (163, 255, 115, 255),
    6: (255, 170, 0, 255),
    7: (255, 0, 0, 255),
    8: (156, 156, 156, 255),
    #9: (0, 0, 0, 255),
    #10: (115, 115, 0, 255),
    #11: (230, 230, 0, 255),
    #12: (255, 255, 115, 255),
    #13: (197, 0, 255, 255),
    }

cmap_LC, norm=matplotlib.colors.from_levels_and_colors(list(LC_color_dict.keys()),[[v/255 for v in i] for i in LC_color_dict.values()],extend='max')
plt.imshow(np.squeeze(batch['mask']),cmap=cmap_LC);
```

<img src="./imgs/2_6_7/output_77_0.png" height='auto' width='auto' title="caDesign">
    


```python
batch['mask']
```




    tensor([[[4, 4, 4,  ..., 4, 4, 4],
             [4, 4, 4,  ..., 4, 4, 4],
             [4, 4, 4,  ..., 4, 4, 4],
             ...,
             [4, 4, 4,  ..., 3, 4, 4],
             [4, 4, 4,  ..., 4, 4, 4],
             [4, 4, 4,  ..., 4, 4, 4]]])



* 更新`datamodule`类

新定义的`geo_datamodule`类，移除了原`NAIPChesapeakeDataModule`类中的数据集前期处理部分内容（包括数据下载、图像增强变换等），而集中于训练数据切分为训练数据集（`train_dataloader`）、测试数据集（`test_dataloader`）和验证数据集（`val_dataloader`）的`Dataloder`对象的构建上。


```python
# 迁移并更新于TorchGeo库：torchgeo.datamodules.naip模块

import pytorch_lightning as pl
import torchgeo
from torch.utils.data import DataLoader
import torch
from torchgeo.datasets import BoundingBox,stack_samples
from torchgeo.samplers.batch import RandomBatchGeoSampler
from torchgeo.samplers.single import GridGeoSampler

class geo_datamodule(pl.LightningDataModule):
    """LightningDataModule implementation for the TorchGeo datasets.

    Uses the train/val/test splits from the dataset.
    """  
    # TODO: tune these hyperparams
    length=1000
    stride=128

    def __init__(
        self,
        ds_image: torchgeo.datasets,
        ds_label: torchgeo.datasets,
        batch_size: int=64,
        num_workers: int=0,
        patch_size: int=256,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for TorchGeo dataset based DataLoaders.

        Args:
            ds_image: Image dataset in TorchGeo RasterDataset format
            ds_label: Target(label) dataset in TorchGeo RasterDataset format
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            patch_size: size of patches to sample
        """
        super().__init__()
        self.ds_image=ds_image
        self.ds_label=ds_label
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.patch_size=patch_size    

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: state to set up
        """
        # TODO: these transforms will be applied independently, this won't work if we
        # add things like random horizontal flip

        # TODO: figure out better train/val/test split
        self.dataset = self.ds_label & self.ds_image
        roi=self.dataset.bounds
        midx=roi.minx + (roi.maxx - roi.minx) / 2
        midy=roi.miny + (roi.maxy - roi.miny) / 2
        train_roi=BoundingBox(roi.minx, midx, roi.miny, roi.maxy, roi.mint, roi.maxt)
        val_roi=BoundingBox(midx, roi.maxx, roi.miny, midy, roi.mint, roi.maxt)
        test_roi=BoundingBox(roi.minx, roi.maxx, midy, roi.maxy, roi.mint, roi.maxt)

        self.train_sampler=RandomBatchGeoSampler(
            self.ds_image, self.patch_size, self.batch_size, self.length, train_roi
        )
        self.val_sampler = GridGeoSampler(self.ds_image, self.patch_size, self.stride, val_roi)
        self.test_sampler = GridGeoSampler(self.ds_image, self.patch_size, self.stride, test_roi)

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training.

        Returns:
            training data loader
        """   

        return DataLoader(
            self.dataset,
            batch_sampler=self.train_sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation.

        Returns:
            validation data loader
        """

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing.

        Returns:
            testing data loader
        """

        return DataLoader(            
            self.dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
        ) 
```


```python
datamodule=geo_datamodule(
    ds_image=naip,
    ds_label=chesapeake,
    batch_size=64,
    patch_size=256 
    )
```

* 构建深度学习网络模型

`TorchGeo`的深度学习网络模型构建`SemanticSegmentationTask`类继承了`pytorch_lightning`的`LightningModule`类，并调用了`segmentation_models_pytorch`库的`unet`模型。配置了`aux_params`参数，返回掩码和分类标签。因为增加了NDVI和NDWI两个指数层，因此总共6个通道，配置`in_channels=6`。调整后的土地覆盖类型共计8类，因此配置`num_classes=8`。


```python
from torchgeo.trainers import SemanticSegmentationTask

aux_params=dict(
    pooling='avg',             # one of 'avg', 'max'
    dropout=0.5,               # dropout ratio, default is None
    activation='sigmoid',      # activation function, default is None
    classes=8,                 # define number of output labels
)

task=SemanticSegmentationTask(
    segmentation_model='unet', 
    encoder_name='resnet34',
    encoder_weights='imagenet',
    pretrained=True,
    in_channels=6,
    num_classes=8,
    ignore_index=7,
    loss='ce', # 'jaccard'
    learning_rate=0.1,
    learning_rate_schedule_patience=5,
    ignore_zeros=True,
    aux_params=aux_params,
    )
```

* 配置训练参数

`PyTorch Lightning`库提供的`Trainer`方法大幅度简化了训练参数的配置，如果不希望部分参数自动化处理，也可以自行配置。下述配置了`ModelCheckpoint`，通过配置`moniter`参数监测给定参数的值，定期保存模型。在` LightningModule`中使用` log() `或`log_dict()`记录的每个指标都可以为监控的对象；配置`EarlyStopping`，监控给定的指标，当该指标停止改进时则停止训练，这非常有利于了解到模型训练的程度，避免无效的训练，利于反复调参实验；配置`CSVLogger`，以YAML或CSV格式存储测量指标，用于后续读取分析；同时配置了训练文件、过程网络模型等保存的本地位置（本次存储到Google Drive中）`default_root_dir`；用`min_epochs`和`max_epochs`配置最小和最大的迭代次数；`fast_dev_run`为一种单元测试；配置`accelerator="gpu"`，支持传递不同的加速器类型，例如“cpu”, “gpu”, “tpu”, “ipu”, “hpu”, “mps, “auto”，及自定义的加速器实例等。`Trainer`方法提供的参数高达50多个，可以从官网查看应用。


```python
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

checkpoint_callback=ModelCheckpoint(monitor="val_loss", dirpath=data_dir, save_top_k=1, save_last=True) #,save_on_train_epoch_end=True
early_stopping_callback=EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10)
csv_logger=CSVLogger(save_dir=data_dir, name="segmentation_unet")
in_tests="PYTEST_CURRENT_TEST" in os.environ

trainer=pl.Trainer(
    callbacks=[checkpoint_callback,early_stopping_callback], 
    logger=[csv_logger],
    default_root_dir=data_dir,
    min_epochs=1,
    max_epochs=300,
    fast_dev_run=in_tests,
    accelerator="gpu",   
    #limit_val_batches=500,
    )
```

    INFO:pytorch_lightning.utilities.rank_zero:GPU available: True (cuda), used: True
    INFO:pytorch_lightning.utilities.rank_zero:TPU available: False, using: 0 TPU cores
    INFO:pytorch_lightning.utilities.rank_zero:IPU available: False, using: 0 IPUs
    INFO:pytorch_lightning.utilities.rank_zero:HPU available: False, using: 0 HPUs
    

* 网络模型训练

`fit`方法传入的参数`ckpt_path`，只对已经训练且保存的模型权重值有效，因此需要检查该文件是否已经存在。利用CoLab提供的GPU算力，模型达到给定指标不再有效提升时所需要的时间约为1个小时42分钟（CoLab Pro版）。


```python
from os import path

ckpt_path=os.path.join(data_dir,'last.ckpt')
if path.exists(ckpt_path):
    print('ckpt file exists.')
    trainer.fit(model=task,datamodule=datamodule,ckpt_path=ckpt_path)  
else:
    trainer.fit(model=task,datamodule=datamodule)
```

    /usr/local/lib/python3.8/dist-packages/pytorch_lightning/callbacks/model_checkpoint.py:604: UserWarning: Checkpoint directory /content/gdrive/MyDrive/data/delaware/training exists and is not empty.
      rank_zero_warn(f"Checkpoint directory {dirpath} exists and is not empty.")
    INFO:pytorch_lightning.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO:pytorch_lightning.callbacks.model_summary:
      | Name          | Type             | Params
    ---------------------------------------------------
    0 | model         | Unet             | 24.4 M
    1 | loss          | CrossEntropyLoss | 0     
    2 | train_metrics | MetricCollection | 0     
    3 | val_metrics   | MetricCollection | 0     
    4 | test_metrics  | MetricCollection | 0     
    ---------------------------------------------------
    24.4 M    Trainable params
    0         Non-trainable params
    24.4 M    Total params
    97.787    Total estimated model params size (MB)
    


    Sanity Checking: 0it [00:00, ?it/s]


    /usr/local/lib/python3.8/dist-packages/pytorch_lightning/utilities/data.py:85: UserWarning: Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is 64. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`.
      warning_cache.warn(
    /usr/local/lib/python3.8/dist-packages/pytorch_lightning/trainer/trainer.py:1595: PossibleUserWarning: The number of training batches (15) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.
      rank_zero_warn(
    


    Training: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]


    /usr/local/lib/python3.8/dist-packages/pytorch_lightning/utilities/data.py:85: UserWarning: Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is 8. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`.
      warning_cache.warn(
    


    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



* 从日志（Log）中提取训练测量指标

测量指标已经存入Google Drive文件夹下，读取`metrics.csv`文件数据，并打印`Train loss`和`Validation Accuracy`指标，可以初步判断深度学习网络模型有效收敛，且验证数据集上的预测精度能够达到约0.8以上。


```python
import csv

if not in_tests:
    train_steps = []
    train_rmse = []

    val_steps = []
    val_rmse = []
    with open(
        os.path.join(data_dir, "segmentation_unet", "version_0", "metrics.csv"), "r"  
    ) as f:
        csv_reader = csv.DictReader(f, delimiter=",")
        for i, row in enumerate(csv_reader):
            try:
                train_rmse.append(float(row["train_loss"]))
                train_steps.append(i)
            except ValueError:  # Ignore rows where train RMSE is empty
                pass

            try:
                val_rmse.append(float(row["val_Accuracy"]))
                val_steps.append(i)
            except ValueError:  # Ignore rows where val RMSE is empty
                pass
```


```python
import matplotlib.pyplot as plt

if not in_tests:
    plt.figure()
    plt.plot(train_steps, train_rmse, label="Train loss")
    plt.plot(val_steps, val_rmse, label="Validation Accuracy")
    plt.legend(fontsize=15)
    plt.xlabel("Batches", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.show()
    plt.close()
```

<img src="./imgs/2_6_7/output_90_0.png" height='auto' width='auto' title="caDesign">
    


* 测试数据集测试

用测试数据集测试，测试结果显示测试精度`test_Accuracy`为0.820。


```python
trainer.test(model=task, datamodule=datamodule)
```

    INFO:pytorch_lightning.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    


    Testing: 0it [00:00, ?it/s]


    /usr/local/lib/python3.8/dist-packages/pytorch_lightning/utilities/data.py:85: UserWarning: Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is 12. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`.
      warning_cache.warn(
    

    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
           Test metric             DataLoader 0
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
          test_Accuracy         0.8202915191650391
        test_JaccardIndex       0.2725767493247986
            test_loss           0.5870120525360107
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    




    [{'test_loss': 0.5870120525360107,
      'test_Accuracy': 0.8202915191650391,
      'test_JaccardIndex': 0.2725767493247986}]



* 用训练的模型预测

这里并未使用用于训练下载的4个NAIP图像瓦片，而是下载了新的16个图像瓦片，新建立数据集，随机采样用已经训练好的模型进行预测。从预测结果来看，所提取的样本包括3、4、6三个分类，对应林冠、灌木丛和不透水区域，目视分类图结果，能够较好的对原始的NAIP图像进行语义分割（解译）。


```python
from torchgeo.samplers import GridGeoSampler,RandomGeoSampler
from torchgeo.datasets import NAIP,stack_samples
from torch.utils.data import DataLoader
import torch

X_pre=NAIP(r'/content/gdrive/MyDrive/data/delaware_imagery_temp',transforms=naip_transforms)
X_sample=RandomGeoSampler(X_pre, size=1024,length=100) 
X_dataloader=DataLoader(X_pre, sampler=X_sample, collate_fn=stack_samples)
X_dataloader_=iter(X_dataloader)
```


```python
X_batch=next(X_dataloader_)
X=X_batch["image"]#.float()
```


```python
import matplotlib.pyplot as plt
import torchvision.transforms as T

plt.imshow(T.ToPILImage()(X[0][:3]));
```


<img src="./imgs/2_6_7/output_96_0.png" height='auto' width='auto' title="caDesign">    


```python
unet_model=task.load_from_checkpoint(os.path.join(data_dir,'last.ckpt'))
unet_model.freeze()
```


```python
y_probs=unet_model(X)
print(y_probs.shape)
```

    torch.Size([1, 8, 1024, 1024])
    

将联合预测概率转换为对应的土地覆盖类型标签（即概率最大对应的列）。


```python
import numpy as np
y_pred=np.argmax(y_probs,axis=1)#.reshape(-1,1)
print(y_pred.shape,'\n',y_pred,'\n',y_pred.unique())
```

    torch.Size([1, 1024, 1024]) 
     tensor([[[4, 4, 4,  ..., 4, 4, 4],
             [4, 4, 4,  ..., 4, 4, 4],
             [4, 4, 4,  ..., 6, 4, 4],
             ...,
             [6, 6, 6,  ..., 4, 4, 4],
             [4, 4, 6,  ..., 4, 4, 4],
             [4, 4, 6,  ..., 4, 4, 4]]]) 
     tensor([3, 4, 6])
    


```python
import matplotlib
import numpy as np

plt.imshow(np.squeeze(y_pred),cmap=cmap_LC);
```

<img src="./imgs/2_6_7/output_101_0.png" height='auto' width='auto' title="caDesign">
    
---

注释（Notes）：

① USGS EROS Archive - Aerial Photography - National Agriculture Imagery Program (NAIP)，（<https://www.usgs.gov/centers/eros/science/usgs-eros-archive-aerial-photography-national-agriculture-imagery-program-naip>）。

② EarthExplorer，（<https://earthexplorer.usgs.gov/>）。

③ 行星计算机（Planetary Computer），（<https://planetarycomputer.microsoft.com/>）。

④ TorchGeo，（<https://torchgeo.readthedocs.io/en/stable/user/installation.html>）。

⑤ torchvision，（<https://pytorch.org/vision/stable/index.htm>）。

⑥ NAIP: National Agriculture Imagery Program，（<https://planetarycomputer.microsoft.com/dataset/naip>）。

⑦ Geospatial Datasets（TorchGeo），（<https://torchgeo.readthedocs.io/en/stable/api/datasets.html>）。

⑧ Chesapeake Conservancy，（<https://www.chesapeakeconservancy.org/conservation-innovation-center/high-resolution-data/>）。

⑨ Kornia，（<https://github.com/kornia/kornia>）。

⑩ ONNX Model Zoo，（<https://github.com/onnx/models>）。

⑪ torchvision.models，（<https://pytorch.org/vision/stable/models.html>）。

⑫ PyTorch Image Models，（<https://github.com/rwightman/pytorch-image-models>）。

⑬ Transformers，（<https://github.com/huggingface/transformers>）。

⑭ Segmentation Models，（<https://smp.readthedocs.io/en/stable/index.html>）。

⑮ ImageNet，（<https://www.image-net.org/index.php>）。

⑯ WordNet（Wikipedia），为200多种语言中单词之间语义关系的词汇数据库。（<https://en.wikipedia.org/wiki/WordNet>）。

⑰ Colaborator，CoLab，（<https://colab.research.google.com/>）。

⑱ Planetary Computer Hub，PcHub，（<https://pccompute.westeurope.cloudapp.azure.com/compute/hub>）。

参考文献（References）:

[1]  Geospatial deep learning with TorchGeo，<https://pytorch.org/blog/geospatial-deep-learning-with-torchgeo/>。

[2] Conservation Innovation Center (CIC). LC Class Descriptions, <https://www.chesapeakeconservancy.org/conservation-innovation-center/high-resolution-data/lulc-data-project-2022/>.
