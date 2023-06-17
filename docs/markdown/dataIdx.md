> Created on Sat Jun 17 07:14:27 2023 @author: Richie Bao-caDesign设计(cadesign.cn)

# 数据集检索

为了方便检索、查看数据集和观察各个模型使用了哪些数据，及模型间共同调用同类数据的情况，用[Microsoft Access（MS Access）](https://www.microsoft.com/en-us/microsoft-365/access)<sup>①</sup>数据库管理系统（database management system，DBMS）录入数据建立数据库。然后用 Python 从建立的数据库中读入数据，并用[Dash（plotly）](https://dash.plotly.com/installation)<sup>②</sup>建立网页交互式应用程序，用于数据库检索查看。数据集按两种情况划分，一种是本书章节知识点对应使用的数据集；另一种是增加了诸如[InVEST](https://naturalcapitalproject.stanford.edu/software/invest)<sup>③</sup>等研究模型或相关论文中模型所使用的数据。MS Access界面如下，

<img src="./imgs/dara_retrieval/dr_04.jpg" height='auto' width='auto' title="caDesign">

因为可视化数据需要数据集或数据库等数据文件，数据文件一般文件较大，因此并没有将交互式可视化数据应用布局到 [USDA](https://richiebao.github.io/USDA_PyPI/#/) <sup>④</sup>库中，而是单独建立了[usda-dashboard](https://pypi.org/project/usda-dashboard/) <sup>⑤</sup>PyPI包，可以通过`pip install usda-dashboard`方法安装。通过执行`usda_dashboard.dash_usda_datasetidx()`可以直接打开一个交互式网页应用。

数据集检索应用布局内容包括两个部分，一部分是检索数据库中的表内容，可以在下拉菜单中选取表查看，也可以通过 SQL 语句查询。其中`研究模型`为本书章节和 InVEST等扩展模型对应的数据集的表；`数据集`为所用数据集的信息；`样本集`为模型所用的基本参数表。另一部分是用 Sankey 图表述本书各个章节和InVEST等扩展模型间所用数据集的关系，并对模型所用数据有一个宏观上的概览。


```python
import usda_dashboard
usda_dashboard.dash_usda_datasetidx()
```

    Dash is running on http://127.0.0.1:8050/
    
     * Serving Flask app 'usda_dashboard._dash_USDA_dataIndex'
     * Debug mode: off
    

    WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
     * Running on http://127.0.0.1:8050
    Press CTRL+C to quit
    127.0.0.1 - - [17/Jun/2023 09:20:10] "GET /dash-nba/ HTTP/1.1" 200 -
    127.0.0.1 - - [17/Jun/2023 09:20:10] "GET /_dash-layout HTTP/1.1" 200 -
    

<img src="./imgs/dara_retrieval/dr_01.png" height='auto' width='auto' title="caDesign">

直接调用应用，其结果如上图。可以通过检索的不同方式查看数据，如下图。

<img src="./imgs/dara_retrieval/dr_02.png" height='auto' width='auto' title="caDesign">

源代码托管于GitHub 上的[USDA_dashboard](https://github.com/richieBao/USDA_dashboard) <sup>⑥</sup>代码库。附数据集检索部分的代码如下，


```python
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns', None)

import pyodbc
from sqlalchemy import create_engine
import json
import itertools
import numpy as np

from dash import Dash, dcc, html, Input, Output, dash_table, State,no_update
import plotly.express as px
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from sqlalchemy import create_engine

flatten_lst=lambda lst: [m for n_lst in lst for m in flatten_lst(n_lst)] if type(lst) is list else [lst]   
def update_df_foreign_keys(df,dn_foreign_keys_mapping):
    for col in df.columns:
        if col in dn_foreign_keys_mapping.keys():
            related_df=pd.read_sql('select * from %s'%(dn_foreign_keys_mapping[col][0]),conn)
            df[col]=df[col].apply(lambda idx:related_df[related_df.ID==idx][dn_foreign_keys_mapping[col][1]].item())     
   
    return df

def sql_query2list(statement,conn):
    cursor=conn.cursor()
    cursor.execute(statement) 
    fetched_data=[dict((cursor.description[i][0], value)  for i, value in enumerate(row)) for row in cursor.fetchall()]
    
    return  fetched_data

# datasets query
import pkg_resources
db_fn=pkg_resources.resource_filename('usda_dashboard', 'data/USDA_dataIndex.accdb') # db_fn='.data/USDA_dataIndex.accdb'
conn=pyodbc.connect(r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=%s;'%db_fn)

dn_foreign_keys_mapping={'dn_1':['category','d_category'],
                         'dn_2':['sampleidx','sample_idx'],
                         'dn_3':['category','d_category'],
                         'dn_4':['category','d_category'],
                         'dn_5':['sampleidx','sample_idx'],
                         'dn_6':['sampleidx','sample_idx'],
                         'dn_7':['category','d_category'],
                         'dn_8':['category','d_category'],
                         'dn_9':['category','d_category'],
                         'dn_10':['category','d_category'],
                         'dn_11':['category','d_category'],
                         'dn_12':['category','d_category'],
                         'dn_13':['sampleidx','sample_idx'],
                         'd_category':['category','d_category'],
                         'usda_dn':['dataset_usda','d_name'],
                         'usda_category':['category_usda','usda_category']}

# dashboard
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP]) #
app.config.suppress_callback_exceptions = True

sample_data_list_=sql_query2list('select a.sample_idx from sampleidx as a',conn)
sample_data_list=[i['sample_idx'] for i in sample_data_list_ if i['sample_idx']!='Null']

modelstudy_ms_category=np.unique([i['ms_category'] for i in sql_query2list('select a.ms_category from modelstudy as a',conn)])
usda_points=[i['point'] for i in sql_query2list('select a.point from usda as a',conn)]

table_dict={'related_research':['usda','modelstudy'],
            'available_data':['category','dataset','sampleidx','category_usda','dataset_usda'],
            'sample_data':sample_data_list}

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    },
    'h7':{
        'color': 'k',
        'font-size': '16px',
        'text-transform': 'uppercase'    
        }
}

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [html.H2("数据集检索", style={"text-align": "center"}),html.P('USDA',style={"text-align": "center"})]
                ),                
            ]
        ),
        dbc.Row([
            dbc.Col([
                html.H5('研究模型'),
                dcc.Dropdown(id='d_related_research',options=table_dict['related_research']),
                html.Div(id='t_related_research'),
                # dash_table.DataTable(id='dt_related_research'),                
                dbc.Alert(id='alert_model_datasets'),
                ]),
            dbc.Col([
                html.H5('数据集'),
                dcc.Dropdown(id='d_available_data',options=table_dict['available_data']),
                html.Div(id='t_available_data'),
                ]),   
            dbc.Col([
                html.H5('样本集'),
                dcc.Dropdown(id='d_sample_data',options=table_dict['sample_data']),
                html.Div(id='t_sample_data'),
                ]),               
            ]),
        dbc.Row([html.H5('SQL 查询'),]),
        dbc.Row([   
            dbc.Col([                    
                dcc.Input(id='query',value='',placeholder="e.g.: select a.ref_citation from modelstudy as a",type='text',style={'width':'100%'}),
                ],width=4),     
            dbc.Col([        
                dbc.Button('submit',id='button_query_submit',color="dark",n_clicks=0, ),
                ],width='auto'),              
            dbc.Col([
                # dbc.Alert(id='alert_query'),
                html.Pre(id='pre_query', style=styles['pre']),
                ],),
            ]),        
        dbc.Row([ 
            html.Hr(style={'marginTop': '1em'}),
            html.H5('研究模型与数据集关系'),
            html.Hr(style={'borderColor':'red','marginTop': '1em', "width": "50%"}),
            html.H6('+USDA 章节数据索引'),
            dcc.RadioItems(id='r_usda_points',options=['off','on'],inline=True,value='off'),
            dcc.Graph(id='g_usda_sankey'),
            html.Hr(style={'borderColor':'red','marginTop': '1em', "width": "50%"}),
            html.H6('+相关研究模型'),
            dcc.Checklist(id='cl_modelstudy',options=modelstudy_ms_category,inline=True),
            dcc.Graph(id='g_sankey'),
            ]),   

        ]
    )

@app.callback(
    Output('t_related_research','children'),
    Input('d_related_research',"value"),    
    )
def related_research_table(table_name):
    if table_name is not None:
        df=pd.read_sql('select * from %s'%(table_name),conn)  
        df=update_df_foreign_keys(df,dn_foreign_keys_mapping)
        df=df.reset_index().rename(columns={"index": "id"})
        
        return  dash_table.DataTable(df.to_dict('records'),
                                      [{"name": i, "id": i} for i in df.columns if i != 'id'],
                                      style_table={'overflow':'auto'},
                                      page_size=10,
                                      # page_action='native',
                                      id='dt_related_research')
        
@app.callback(
    Output('t_available_data','children'),
    Input('d_available_data',"value"),    
    )
def available_data_table(table_name):
    if table_name is not None:
        df=pd.read_sql('select * from %s'%(table_name),conn)   
        return dash_table.DataTable(df.to_dict('records'),
                                    [{"name": i, "id": i} for i in df.columns],
                                    style_table={'overflow':'auto'},
                                    page_size=10,)
  
@app.callback(
    Output('t_sample_data','children'),
    Input('d_sample_data',"value"),    
    )
def sample_data_table(table_name):
    if table_name is not None:
        df=pd.read_sql('select * from %s'%(table_name),conn)   
        return dash_table.DataTable(df.to_dict('records'),
                                    [{"name": i, "id": i} for i in df.columns],
                                    style_table={'overflow':'auto'},
                                    page_size=10,)

@app.callback(
    Output('alert_model_datasets','children'),
    Input('dt_related_research','active_cell'),
    Input('dt_related_research','data'),
    Input('dt_related_research','columns'),      
    )
def model_datasets(active_cell,rows,columns):    
    df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    if active_cell:        
        row_values=df.iloc[active_cell['row_id']]
        row_values_dn=row_values[[col for col in df.columns if col in dn_foreign_keys_mapping.keys()]]
        row_values_dn_values=[i for i in row_values_dn.values if i!='Null']
        try:
            mm=row_values['model_method']
        except:
            mm=row_values['point']
        return f'{mm} 数据需求: {row_values_dn_values}.' 
    
@app.callback(
    Output('pre_query','children'),
    Input("button_query_submit", "n_clicks"),
    State('query','value'),
    )
def sql_query(n_clicks,statement):    
    if n_clicks:          
        cursor=conn.cursor()
        try:
            cursor.execute(statement) # select * from Biophysical_UHI_fake category; 
            fetched_data=[dict((cursor.description[i][0], value)  for i, value in enumerate(row)) for row in cursor.fetchall()]
            return json.dumps(fetched_data)
        except:
            return json.dumps('SQL statement is wrong!')
            
    return json.dumps('Null')

@app.callback(
    Output('g_sankey','figure'),
    Input('cl_modelstudy',"value"),    
    )
def display_sankey_modelstudy(related_reserch):
    modelstudy=pd.read_sql('select * from %s'%('modelstudy'),conn) 
    modelstudy=update_df_foreign_keys(modelstudy,dn_foreign_keys_mapping)
    
    dataset=pd.read_sql('select * from %s'%('dataset'),conn) 
    dataset=update_df_foreign_keys(dataset,dn_foreign_keys_mapping) 
    
    source_target_links=[]
    if related_reserch is not None:
        if len(related_reserch)>0:
            modelstudy_selection=modelstudy[modelstudy.ms_category.isin(related_reserch)]
            for dn in dn_foreign_keys_mapping.keys():            
                if dn in modelstudy.columns:
                    source_target_links.append(list(zip(modelstudy['model_method'],modelstudy[dn])))  
    
            category_filter=[i[1] for i in flatten_lst(source_target_links)]        
            for dn in dn_foreign_keys_mapping.keys():
                if dn in dataset.columns:   
                    dataset_selection=dataset[dataset[dn].isin(set(dataset[dn]).intersection(set(category_filter)))]
                    source_target_links.append(list(zip(dataset_selection[dn],dataset_selection['d_name'])))   
            
            source_target_links=flatten_lst(source_target_links)    
            labels=list(set(list(itertools.chain(*source_target_links))))
            source_target_links_df=pd.DataFrame(source_target_links,columns=['source','target'])
            source_target_links_df=source_target_links_df[(source_target_links_df['source']!='Null') & (source_target_links_df['target']!='Null')]
            source_target_links_df['source_id']=source_target_links_df.source.apply(lambda x:labels.index(x))
            source_target_links_df['target_id']=source_target_links_df.target.apply(lambda x:labels.index(x))
            
            style_sankey=dict(
                        pad = 15,
                        thickness = 15,
                        line = dict(color = "black", width = 0.5),
                        color = "lightgray")    
            
            node=dict(label = labels)
            node.update(style_sankey)
            
            link= dict(
              source =source_target_links_df['source_id'], 
              target =source_target_links_df['target_id'],
              value = [1]*len(source_target_links)
              )    
            
            fig = go.Figure(data=[go.Sankey(
                node = node,
                link =link,
                )])
            
            fig.update_layout(title_text=f"Sankey Diagram: {related_reserch}", font_size=13,autosize=False,height=1200)        
            return fig    
        else:
            return {}
    else:
        return {}
    
@app.callback(
    Output('g_usda_sankey','figure'),
    Input('r_usda_points',"value"),    
    )
def display_sankey_usda(selection):
    if selection=='on':
        usda=pd.read_sql('select * from %s'%('usda'),conn) 
        usda=update_df_foreign_keys(usda,dn_foreign_keys_mapping)
        
        dataset_usda=pd.read_sql('select * from %s'%('dataset_usda'),conn) 
        dataset_usda=update_df_foreign_keys(dataset_usda,dn_foreign_keys_mapping)
        
        source_target_links=[]
        source_target_links.append(list(zip(usda['chapter'],usda['point'])))    
        source_target_links.append(list(zip(usda['point'],usda['usda_dn'])))  
        source_target_links.append(list(zip(dataset_usda['d_name'],dataset_usda['usda_category'])))  
        
        source_target_links=flatten_lst(source_target_links)    
        labels=list(set(list(itertools.chain(*source_target_links))))
        source_target_links_df=pd.DataFrame(source_target_links,columns=['source','target'])
        source_target_links_df=source_target_links_df[(source_target_links_df['source']!='Null') & (source_target_links_df['target']!='Null')]
        source_target_links_df['source_id']=source_target_links_df.source.apply(lambda x:labels.index(x))
        source_target_links_df['target_id']=source_target_links_df.target.apply(lambda x:labels.index(x))
        
        style_sankey=dict(
                    pad = 15,
                    thickness = 15,
                    line = dict(color = "black", width = 0.5),
                    color = "lightgray")    
        
        node=dict(label = labels)
        node.update(style_sankey)
        
        link= dict(
          source =source_target_links_df['source_id'], 
          target =source_target_links_df['target_id'],
          value = [1]*len(source_target_links)
          )    
        
        fig = go.Figure(data=[go.Sankey(
            node = node,
            link =link,
            )])
        
        fig.update_layout(title_text=f"Sankey Diagram: USDA", font_size=13,autosize=False,height=900)     
        return fig      
    else:
        return {}
 
    
if __name__ == '__main__':
    app.run_server(debug=True)

```

---

注释（Notes）：

① Microsoft Access（MS Access）数据库管理系统，（<https://www.microsoft.com/en-us/microsoft-365/access>）。

② Dash（plotly）网页交互式应用程序，（<https://dash.plotly.com/installation>）。

③ InVEST（Integrated Valuation of Ecosystem Services and Tradeoffs），（<https://naturalcapitalproject.stanford.edu/software/invest>）。

④ USDA 本书配套Python包PyPI库，（<https://richiebao.github.io/USDA_PyPI>）。

⑤ usda-dashboard，本书配套Python包PyPI库，（<https://pypi.org/project/usda-dashboard/>）。

⑥ USDA_dashboard 本书配套用，托管于GitHub的代码库，（<https://github.com/richieBao/USDA_dashboard>）。
