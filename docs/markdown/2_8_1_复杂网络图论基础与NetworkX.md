> Created on Sun Jan  1 21:31:49 2023 @author: Richie Bao-caDesign设计(cadesign.cn)

## 2.8.1  复杂网络（图论）基础与NetworkX

复杂网络（图，graph）的研究已经成为涉及物理学、数学、生物学、社会科学、信息学以及其它理论和应用科学的多学科研究的重要领域。这一领域的重要性在于存在一种统一的语言来描述在现代社会中具有重大相关性的不同的现实世界系统，从互联网或电网到代谢或蛋白质相互作用的网络<sup>[1]</sup>。复杂网络也成为城市空间数据分析中解决城市问题的重要工具，例如城市街道（交通）系统，斑廊基景观生态，业态分布网络等。图的研究自从18世纪上半叶诞生以来，到19世纪下半叶已经发展成为数学一个系统的分支，至20世纪开始迅速发展，开发了大量算法、属性识别并定义数学模型以更好的理解复杂网络。为了避免对大量繁复算法的直接计算，[NetworkX](https://networkx.org/documentation/stable/index.html#)<sup>①</sup>库定义了大量函数方法再现了许多图的算法，可以直接调用用于实际相关的研究问题中。对于图的解释也以`NetworkX`库为工具来说明，不仅可以图示复杂网络，也可以再现复杂算法。

对于复杂网络的解释会涉及到很多名词定义（术语）、数学符号表达，这里以Reihhard Diestel的《图论》<sup>[2]</sup>，结合`NetworkX`库作为复杂网络解释的参考标准，并适当的引入其它论文专著的表述。

### 2.8.1.1 图<sup>[2]</sup>

用${\mathbb{N}}$表示包括零在内的自然数的集合。模$n$整数集${\mathbb{Z}} / n{\mathbb{Z}} $记作${\mathbb{Z}}_{n} $，其元素记作$ \overline{i} :=i+n{\mathbb{Z}} $；当把${\mathbb{Z}} _{2}=\{ \overline{0}, \overline{1}  \} $看作一个域时，也把它记作${\mathbb{F}} _{2}=\{ 0,1 \} $。对实数$x$，用$\lfloor x\rfloor$表示小于或等于$x$的最大整数，用$\lceil x\rceil$表示大于或等于$x$的最小整数。以2为底的对数函数记作$log$，而自然对数函数记作$ln$。给定集合$A$，及它的互不相交的子集$A_{1} ,  \ldots ,A_{k}$，如果对于每个$i$均有$A_{i} \neq  Ø（空集）$，并且$\bigcup_{i=1}^k A_{i}=A  $，则称集合$A={A_{1} ,  \ldots ,A_{k}}$为$A$的一个**划分（partition）**。给定$A$的另一个划分$\{  A' _{1}, A' _{2} , \ldots , A' _{ \ell }  \}$，如果每个$A' _{i}$都包含于某个$A_{j}$中，则称$A'$为$A$的**细化（refine）**。用$[A]^{k} $表示集合$A$的所有**k-元子集**。另外，有$k$个元素的集合称为**k-集（k-set）**，而$k$个元素的子集称为**k-子集（k-subset）**。

**图（graph）**$G=(V,E)$是一个二元组$(V,E)$使得$E \subseteq [V]^{2} $，所以$E$的元素是$V$的2-元子集。为了避免符号上的混淆，默认$V \cap E=Ø $。集合$V$中的元素称为图$G$的**顶点（vertex）**（或**节点（node）**，**点（point）**），而集合$E$的元素称为**边（edge）**(或**线（line）**)。通常，描绘一个图的方法是把顶点画成一个小圆圈，如果相应的顶点之间有一条边，就用一条线连接这两个小圆圈。如何绘制这些小圆圈和连线时无关紧要的，重要的是要正确体现哪些顶点对之间有边，哪些顶点对之间没有边。

为了方便绘制复杂网络，定义`G_drawing()`函数，可以配置是否显示顶点属性或边属性，及配置图的样式，可以调整顶点的大小和颜色，字体的大小和颜色，及顶点位置、边宽度、图大小等样式。


```python
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
```

用`NetworkX`库的`nx.Graph()`方法定义空的图G，这里通过`add_edges_from`增加边（顶点对）的方式和`add_node`增加点的方式定义图G。其顶点集为$V=\{1,2, \ldots ,8\}$，边集为$E=\{\{1,2\},\{1,5\},\{2,5\},\{3,4\},\{5,7\},\{7,8\}\}$的图。


```python
import networkx as nx
G=nx.Graph()
G.add_edges_from([(1,2),(1,5),(2,5),(3,4),(5,7),(7,8)])
G.add_node(6)
G_drawing(G,node_size=50,font_size=10,nodes=[[3,4],[1,2,5,7,8]],nodes_size=[300,300],figsize=(5,5),routes=[[1,5,7,8]],edge_widths=[9])
```

<img src="./imgs/2_8_1/output_5_0.png" height='auto' width='auto' title="caDesign">
    


具有顶点集$V$的图亦称为**V上的图**（**a graph on V**）。图G的顶点集记为$V(G)$，边集记为$E(G)$。这些约定俗成与这两个集合的记法是独立的：图$H=(W,F)$的顶点集$W$仍记为$V(H)$，而不是$W(H)$。通常并不把一个图和它的顶点集或边集严格的区分开来，例如称一个顶点$v \in G$（而不是$v \in V(G)$），一条边$e \in G$等。

一个图的顶点个数称为图的**阶（order）**，记为$\mid  G\mid $，边数记为$\parallel G \parallel $。根据图的阶，把图分为**有限的（finite）**、**无限的（infinite）**或**可数的（countable）**等。一般假定图均为有限的。

如下代码，`NetworkX`库提供了获取图顶点`G.nodes`、顶点数`G.number_of_nodes()`和边`G.edges`、边数`G.number_of_edges()`的方法。


```python
print(G.nodes)
print(G.number_of_nodes())
print(G.edges)
print(G.number_of_edges())
```

    [1, 2, 5, 3, 4, 7, 6]
    7
    [(1, 2), (1, 5), (2, 5), (5, 7), (3, 4)]
    5
    

对于**空图（empty graph）**，记作$Ø $。阶为0或1的图称为**平凡的（trivial）**。有时平凡图会很有用，例如使用归纳法时；而在其它一些情景中，平凡图则无意义，因此避免非平凡性条件的假设，省略对平凡图和空图的讨论。

给定顶点$v$和边$e$，如果$v \in e$，则称$v$与$e$**关联（incident）**，从而$e$是**在（at）**$v$的边。关联同一条边的两个顶点称为这条边的**端点（endvertex）**或**顶端（end）**，而这条边**连接（joint）**它的两个端点。边$\{x,y\}$通常记为$xy$（或$yx$）。如果$x \in X$且$y \in Y$，则称边$xy$为**一条X-Y边（X-Y edge）**；集合$E$中所有$X-Y$边的集合，记为$E(X,Y)$；而$E(\{x\},Y)$和$E(X,\{y\})$会记为$E(x,Y)$和$E(X,y)$。$E$中所有和顶点$v$关联的边记为$E(v)$。

如果$\{x,y\}$是$G$的一条边，则称两个顶点$x$和$y$是**相邻的（adjacent）**或**邻点（neighbor）**；如果两条边$e \neq f$有一个公共端点，则称$e$和$f$是**相邻的**。若$G$的所有顶点都是两列相邻的，则称$G$是**完全的（complete）**。$n$个顶点的完全图记为$K^{n} $；$K^{3} $称为**三角形（triangle）**。

用`G.neighbors`或`nx.all_neighbors`获取给定顶点的邻点；用`G.edges(node)`可以提取给定公共端点的所有边。


```python
print(list(G.neighbors(5)))
print(list(nx.all_neighbors(G,5)))
print(G.edges(5))
```

    [1, 2, 7]
    [1, 2, 7]
    [(5, 1), (5, 2), (5, 7)]
    

用`nx.complete_graph`方法可以生成给定阶的完全图。


```python
G=nx.complete_graph(5)
G_drawing(G)
```

<img src="./imgs/2_8_1/output_11_0.png" height='auto' width='auto' title="caDesign">
    
  


互不相邻的顶点或边称为**独立顶点**或**独立边（independent vertex/edge）**，更正式地，若一个顶点集或边集中没有两个元素是相邻的，则该集合称为**独立集（independent set）**；独立的顶点集也称为**稳定集（stable set）**。

`maximal_independent_set`(G, nodes=None, seed=None)：返回图的随机最大独立集。可以通过`nodes`参数指定必须包含的顶点列表。最大独立集是一个独立集，因此不可能添加一个新的节点，仍然可以得到一个独立集。

`maximum_independent_set`(G)<sup>[3]</sup>：返回一个近似的最大独立集。


```python
G=nx.lollipop_graph(4, 3)
G.add_edges_from([(6,7),(5,7),(7,8)])
G.add_node("a")
G_drawing(G,node_size=20,font_size=10)
print(nx.maximal_independent_set(G))
print(nx.maximal_independent_set(G, [1]))

from networkx.algorithms.approximation.clique import maximum_independent_set
print(maximum_independent_set(G))
```

<img src="./imgs/2_8_1/output_13_0.png" height='auto' width='auto' title="caDesign">
    

    


    [1, 5, 'a', 8]
    [1, 8, 6, 'a', 4]
    {0, 4, 6, 8, 'a'}
    

设$G=(V,E)$和$G'=( V', E'  ) $是两个图，如果从$G$到$G'$的映射$\varphi : V   \longrightarrow  V' $保留顶点的关联性，即只要$\{x,y\} \in E$就有$\{ \varphi (x), \varphi (y)\} \in E' $，则称$\varphi$是一个**同态（(homomorphis）**。特定地，对于$\varphi$像中的每个顶点$x' $，它的逆映像$ \varphi ^{-1}  (x') $是$G$的一个独立集。如果$\varphi $是一个双射，同时它的$\varphi ^{-1} $也是一个同态（即对于任何$x,y \in V$，有$xy  \in  E \Leftrightarrow  \varphi (x) \varphi (y)  \in  E' $），称$\varphi $是一个**同构（isomorphism）**，或者称$G$和$G'$是**同构的**，记作$G \simeq G' $。从$G$到自身的同构是一个自同构（automorphism）。通常情况下并不区分同构的图，所以同构的图常记为$G=G'$，而不是$G \simeq G' $。如果强调只对给定图的同构类感兴趣，会非正式的称它为**抽象图（abstract graph）**。


在同构意义下封闭的图族叫做**图性质（graph property）**。例如“包含三角形”就是一个图性质：如果$G$包含三个两两相邻的顶点，则每个同构于$G$的图亦有此性质。对于图上的一个映射，如果对于每个同构图它均取相同的值，则这样的映射称为一个图不变量（graph invvariant）。一个图的顶点数和边数就是两个简单的图不变量；图中两两相邻的最大顶点数也是**图不变量**。

`is_isomorphic`(G1, G2, node_match=None, edge_match=None)<sup>[4]</sup>：如果图G1和G2同构，则返回true，否则返回false。如果需要考虑顶点和边属性则可以配置`node_match`和`edge_match`参数。

`numerical_edge_match`(attr, default, rtol=1e-05, atol=1e-08)：返回数值边缘属性的比较函数。属性值必须为数值或可排序的对象，如果G1和G2排序后的值列表相同或者在一个允许的范围内则返回True。

`numerical_node_match`(attr, default, rtol=1e-05, atol=1e-08)：返回数值顶点属性的比较函数。

`could_be_isomorphic`(G1, G2)，`fast_could_be_isomorphic`(G1, G2)和`faster_could_be_isomorphic`(G1, G2)：如果图绝对不是同构的，则返回false；不能保证True为同构。


```python
import networkx.algorithms.isomorphism as iso

G1=nx.DiGraph()
G2=nx.DiGraph()
nx.add_path(G1, [1, 2, 3, 4], weight=1)
nx.add_path(G2, [10, 20, 30, 40], weight=2)
G_drawing(G1,edge_labels='weight')
G_drawing(G2,edge_labels='weight')
em=iso.numerical_edge_match("weight", 1)
print(em)
print(nx.is_isomorphic(G1, G2))  # no weights considered
print(nx.is_isomorphic(G1, G2, edge_match=em))  # match weights
```

<img src="./imgs/2_8_1/output_15_0.png" height='auto' width='auto' title="caDesign">
    
    

<img src="./imgs/2_8_1/output_15_1.png" height='auto' width='auto' title="caDesign">

    
    


    <function numerical_node_match.<locals>.match at 0x00000251FA0B9430>
    True
    False
    

记$G  \cup  G' :=(V \cup  V' ,E \cup  E' )$及$G   \cap   G' :=(V  \cap   V' ,E  \cap   E' )$，若$G   \cap   G'=Ø $，则称$G$和$G'$是**不交的（disjoint）**；如果$V' \subseteq V $且$E' \subseteq E $，则称$G'$是$G$的**子图（subgraph）**（并称$G$是$G'$的**母图（supergraph）**），记作$G' \subseteq G $。非正式的称$G$**包含（contain）**$G' $。若$G' \subseteq G $且$G'\neq  G $，则称$G' $是$G$的**真子图（proper subgraph）**。

若$G' \subseteq G $且$G' $包含了$E$中所有满足$x,y \in  V' $的边$xy$，则称$G' $是$G$的**导出子图（induced subgraph）**，或称$V' $是在图$G$中导出**（induce）**或**支撑（span）**$G' $，并记为$ G':=G[ V' ] $。因此对任意顶点集$U \subseteq V$，$G[U]$表示定义在$U$上的图，它的边恰好是$G$中那些两个端点均在$U$中的边。如果$H$是$G$的子图（不必是导出子图）,简记$G[V[H]]$为$G[H]$。最后，如果$V'$支撑$G$的所有顶点，即$V'=V $，则称$G' $是$G$的一个**支撑子图（spanning subgraph）**。

如下代码定义图$G$和$H$，调用`compose`方法实现图的并；`R.remove_nodes_from(n for n in G if n in H)`方法实现图的差（对于提供的`difference`(G, H)方法会提示错误`Node sets of graphs not equal
`）；用`intersection`实现图的交。顶点2,3,4在$G \cup  G' $中导出（或支撑）一个三角形，但在$G$中则不导出三角形。


```python
G=nx.Graph()
G.add_edges_from([(1,2),(2,3),(2,4),(3,5),(4,5)])
G_drawing(G)
H=nx.Graph()
H.add_edges_from([(3,4),(3,5),(4,6),(5,6)])
G_drawing(H)
```

<img src="./imgs/2_8_1/output_17_0.png" height='auto' width='auto' title="caDesign">
    

<img src="./imgs/2_8_1/output_17_1.png" height='auto' width='auto' title="caDesign">

    



```python
U=nx.compose(G, H)
G_drawing(U)
```

<img src="./imgs/2_8_1/output_18_0.png" height='auto' width='auto' title="caDesign">
    


```python
R=G.copy()
R.remove_nodes_from(n for n in G if n in H)
G_drawing(R)
```

<img src="./imgs/2_8_1/output_19_0.png" height='auto' width='auto' title="caDesign">
    


```python
I=nx.intersection(G, H)
G_drawing(I)
```

<img src="./imgs/2_8_1/output_20_0.png" height='auto' width='auto' title="caDesign">
    
    


 设$U$是$G$的任意一个顶点集合，把$G[V \backslash U]$简记为$G-U$，即$G-U$是从$G$中删除$U \cap V$中所有顶点及相关联的边得到的图。如果$U=\{v\}$是个单点集，把$U-\{v\}$简记为$G-v$；而把$G-V( G' )$简单记作$G- G' $。对于$[V]^{2} $的一个子集$F$，记$G-F:=(V,E \backslash F)$和$G+F:=(V,E \cup F)$。同上，$G-\{e\}$和$G+\{e\}$分别简记为$G-e$和$G+e$。对于一个给定的图性质，若$G$本身具有此性质，而它的任意真子图$(V,F)$（即$F \supsetneq E$）却不具有此性质，则称$G$关于此性质是**边极大的（edge-maxmal）**。
 
更一般的，当称一个图对于某性质是**极大的（maximal）**或**极小的（minimal）**，但没有强调具体的序关系时，均指子图关系。 当提到顶点集或边集的极大性或极小性时，均指集合的包含关系。

如果$G$和$G'$是不交的，那么$G* G'$表示在$G \cup  G' $中连接$G$的所有顶点到$G'$的所有顶点而得到的图，例如$ K^{2}*  K^{3}= K^{5}$。图$G$的**补图（complement）**$\overline{G} $是$V$上边集为$[V]^{2} \backslash E $的图，即$\overline{G} $与$G$有相同顶点，其顶点之间的边当且仅当在$G$里它们没有边相连。图$G$的**线图（line graph）**$L(G)$是$E$上的图，它的顶点集是$E$且使得$x,y \in E$是相邻的当且仅当它们作为边在$G$中是相邻的。

下述代码定义了一个图$G$，使用`complement`方法计算补图，从结果可知补图的边不在图$G$中存在。


```python
G=nx.Graph([(1, 2), (2, 3), (3, 4), (4, 5),(1,5)])
G_drawing(G)
G_complement=nx.complement(G)
G_drawing(G_complement)
G_complement.edges() # This shows the edges of the complemented graph
```

<img src="./imgs/2_8_1/output_22_0.png" height='auto' width='auto' title="caDesign">
    
    

<img src="./imgs/2_8_1/output_22_1.png" height='auto' width='auto' title="caDesign">

    
    





    EdgeView([(1, 3), (1, 4), (2, 4), (2, 5), (3, 5)])



下述代码定义了一个图𝐺 ，使用`line_graph`方法计算线图，$L(G)$将$G$中的每条边抽象成一个顶点，如若原图中两条边相邻，那么就给线图中对应顶点之间连接一条边。因为线图将原图的边化作了顶点，所以也可以将其视作原图的一种对偶。线图的顶点保留了原图边的信息，例如顶点$(0,2)$表示原图$G$中边$\{0，2\}$.


```python
G=nx.star_graph(3)
G_drawing(G)
L=nx.line_graph(G)
G_drawing(L)
```

<img src="./imgs/2_8_1/output_24_0.png" height='auto' width='auto' title="caDesign">
    
    

<img src="./imgs/2_8_1/output_24_1.png" height='auto' width='auto' title="caDesign">

    


### 2.8.1.2 顶点度

设$G=(V,E)$是一个非空图，$G$中的顶点$v$的邻点集记为$N_{G}(v) $，或简记为$N(v)$。更一般地，对于$U \subseteq V$，$U$在$V \backslash U$中的邻点被称作$U$的**邻点（neighbor）((，这个顶点集记为$N(U)$。

顶点$v$的**度（degree）**（或**价（valenccy）**）$d_{G}(v)=d(v) $是指关联$v$的边数$| E(v) | $。由图的定义，它等于$v$的邻点的个数。度为0的顶点叫做**孤立顶点（isolated vertex）**，$G$的**最小度（minimum degree）**记为$\delta (G):=min\{d(v)  |  v \in V\}$，而**最大度（maximum degree）**记为$ \triangle (G):=max\{d(v)  |  v \in V\}$。如果$G$的所有顶点都有相同的顶点度$k$，则称$G$是**k-正则的(k-regular)**，或简称**正则的（regular）**。3-正则图亦称**立方图（cubic graph）**。

图$G$的**平均度（average degree）**定义为： $d(G):= \frac{1}{  |  V| } \sum_{v \in V} d(v)  $，显然$\delta (G) \leq d(G) \leq  \triangle (G)$。

顶点度是连接每个顶点的边数，它是一个局部参数，而平均度则是一个整体性的度量，有时，可以方便的把这个比率记为$\varepsilon (G):= |  E| /  |  V|  $。

当然，$d$和$\varepsilon$这两个量是密切相关的，如果对$G$中所有顶点度求和，那么每条边恰被计算两次，即每个端点计算一次，所以$| E |= \frac{1}{2} \sum_{v \in V} d(v)=   \frac{1}{2} d(G) .  |  V| $，从而有$ \varepsilon (G)=   \frac{1}{2} d(G) $。

`G.degree`方法可以计算返回所有顶点的度。


```python
from statistics import mean
G=nx.star_graph(3)
G_drawing(G)
print( G.degree)
max_degree=max(deg for n, deg in G.degree)
print(max_degree)
average_degree=sum(deg for n, deg in G.degree)/G.number_of_nodes()
print(average_degree)
```

<img src="./imgs/2_8_1/output_26_0.png" height='auto' width='auto' title="caDesign">
    


    [(0, 3), (1, 1), (2, 1), (3, 1)]
    3
    1.5
    

`average_neighbor_degree`(G, source='out', target='out', nodes=None, weight=None)<sup>[5]</sup>：返回每一个顶点邻点的平均度。在无向图中，顶点$i$的邻域$N(i)$包括通过边连接到$i$的顶点。对于有向图，$N(i)$根据`source`参数确定：

1. 如果`source`参数为`in`，则$N(i)$由节点$i$的前驱顶点组成；
2. 如果`source`参数为`out`，则$N(i)$由节点$i$的后继顶点组成；
3.  如果`source`参数为`in+out`，则$N(i)$由节点$i$的前驱和后继顶点共同组成。

顶点$i$的平均邻点度为：$k_{nn,i} = \frac{1}{|N(i)|} \sum_{j \in N(i)} k_j$，式中，$N(i)$为顶点$i$的邻点集；$k_j$是属于邻点集$N(i)$顶点$j$的度。对于有向图，定义一个类似的度量，公式为： $k_{nn,i}^{w} = \frac{1}{s_i} \sum_{j \in N(i)} w_{ij} k_j$，式中，$s_i$是顶点$i$的加权度；$w_{ij}$是连接$i$和$j$的边权重；$N(i)$为顶点$i$的邻点集。


```python
G=nx.path_graph(4)
G.edges[0, 1]["weight"]=5
G.edges[2, 3]["weight"]=3
G_drawing(G,edge_labels='weight')
print(nx.average_neighbor_degree(G))
print(nx.average_neighbor_degree(G, weight="weight"))
```

<img src="./imgs/2_8_1/output_28_0.png" height='auto' width='auto' title="caDesign">
    
    


    {0: 2.0, 1: 1.5, 2: 1.5, 3: 2.0}
    {0: 2.0, 1: 1.1666666666666667, 2: 1.25, 3: 2.0}
    


```python
G=nx.DiGraph()
nx.add_path(G, [0, 1, 2, 3])
G_drawing(G)
nx.average_neighbor_degree(G, source="in", target="in")
```

<img src="./imgs/2_8_1/output_29_0.png" height='auto' width='auto' title="caDesign">
    
   





    {0: 0.0, 1: 1.0, 2: 1.0, 3: 0.0}




```python
nx.average_neighbor_degree(G, source="out", target="out")
```




    {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0}



`is_regular`(G)：确定图G是否为正则图；

`is_k_regular`(G, k)：确定图G是否为k-正则的。


```python
G=nx.cubical_graph()
G_drawing(G)
print(nx.is_regular(G))
print(nx.is_k_regular(G, 3))
```

<img src="./imgs/2_8_1/output_32_0.png" height='auto' width='auto' title="caDesign">
    

    


    True
    True
    

`k_factor`(G, k, matching_weight='weight')<sup>[6]</sup>：计算图G的k-factor（k-因子）。图的k-因子是生成k-正则子图。G的生成k-正则子图是包含G的每个顶点和G的边的子集的子图，使得每个顶点都有度k。


```python
G=nx.cubical_graph()
G.add_edges_from([(3,7),(3,1),(4,6)])
G_drawing(G)
G1=nx.k_factor(G, 3)
G_drawing(G1)
```

<img src="./imgs/2_8_1/output_34_0.png" height='auto' width='auto' title="caDesign">
    


<img src="./imgs/2_8_1/output_34_1.png" height='auto' width='auto' title="caDesign">

    
    


### 2.8.1.3 路和圈

**路（path）**$P=(V,E)$是一个非空图，其顶点集和边集分别为$V= \{x_{0}, x_{1}, \ldots , x_{k}  \}$，$E= \{x_{0} x_{1}, x_{1}x_{2}, \ldots , x_{k-1} x_{k}   \}$，这里的$x_{i} $均互不相同，顶点$x_{0}$和$x_{k}$由**路P连接（link）**，并称它们为路的**端点（endvertex）**或**顶点（end）**；而$ x_{1}, \ldots , x_{k-1}  $称为**P的内部（inner）**顶点。一条路上的边数称为**路的长度（length）**，长度为$k$的路记为$P^{k} $。允许$k$为零，所以$P^{0}=  K^{1} $。

经常用顶点的自然顺序排列表示路，记为$P= x_{0} x_{1}   \ldots  x_{k} $，并称$P$是一个**从（from）**$x_{0}$**到（to）**$ x_{k}$的路（或$x_{0}$和$ x_{k}$**之间（between）**的路）。

对$0 \leq i \leq j \leq k$，记$P$的各种子路如下：

$P x_{i}:= x_{0}   \ldots  x_{i} $

$x_{i}P := x_{i}   \ldots  x_{k} $

$x_{i}P x_{j}:= x_{i}   \ldots  x_{j} $

以及：

$\dot{P}:= x_{1} \ldots  x_{k-1}   $

$P\dot{ x_{i} }:= x_{0} \ldots  x_{i-1}   $

$\dot{ x_{i} }P:= x_{i+1} \ldots  x_{k}   $

$\dot{ x_{i} }P\dot{ x_{j} }:= x_{i+1} \ldots  x_{j-1}   $

可以用类似直观的方法表示路的串联，例如，如果三条路的并$Px \cup xQy \cup yR$还是一条路，则可以简记为$PxQyR$。

下述代码绘制了两条路，$\{3,1,4,2,5\}$和$\{1,2,6,5\}$。其中，`is_path`(G, path)：判读指定路径是否存在。`path_weight`(G, path, weight)：返回指定路径给定边属性（权重）的总成本。


```python
G=nx.Graph()
G.add_edges_from([(1,2),(1,3),(2,4),(1,4),(2,5),(2,6),(5,6),(1,6)])
nx.set_edge_attributes(G,{(3,1):{'weight':2.0},(1,4):{'weight':1.0},(4,2):{'weight':5.0},(2,5):{'weight':2.0}})
route=[3,1,4,2,5]
G_drawing(G,routes=[route,[1,2,6,5]],edge_widths=[5,2],edge_labels='weight')

print(nx.is_path(G,route))
print(nx.path_weight(G,route,weight='weight'))
```

<img src="./imgs/2_8_1/output_36_0.png" height='auto' width='auto' title="caDesign">
    

    


    True
    10.0
    

给定顶点集$A$和$B$及路$P= x_{0} x_{1}   \ldots  x_{k} $，如果$V(P) \cap A=\{ x_{0} \}$且$V(P) \cap B=\{ x_{k} \}$，则称$P$为一条**A-B路（A-B path）**；同前，把$\{a\}-B$记作$a-B$路等。对于两条或以上的路，如果其中任意一条路不包含另一条路的内部顶点，则称它们是**独立路（independent path）**，例如，两条$a-b$路是独立的当且仅当$a$和$b$是其唯一的公共顶点。

给定图$H$，如果路$P$是非平凡的且只与$H$在端点接触，则称$P$是一条**H-路（H-path）**。特别的，任何长度为1的H-路的边不可能是$H$的边。

`add_path`(G_to_add_to, nodes_for_path, **attr)：将路径添加到图G中。

`has_path`(G, source, target)：判断图G中是否有从源`source`到汇`target`顶点的路径。


```python
G=nx.Graph()
route_1=[0, 1, 2, 3]
route_2=[10, 11, 12]
nx.add_path(G, route_1)
nx.add_path(G,route_2, weight=7)
G_drawing(G,routes=[route_1,route_2],edge_widths=[5,2],edge_labels='weight',node_size=20,font_size=10)
print(nx.has_path(G,0,3))
```

<img src="./imgs/2_8_1/output_38_0.png" height='auto' width='auto' title="caDesign">
    

    


    True
    

`all_simple_paths`(G, source, target, cutoff=None)：生成图G中给定源汇顶点的所有路。使用`map(nx.utils.pairwise, paths)`方法将路的顶点列表转换为对应的边列表。

`shortest_simple_paths`(G, source, target, weight=None)<sup>[7]</sup>：生成图G中给定源汇顶点的所有路。


```python
G=nx.complete_graph(4)
paths=list(nx.all_simple_paths(G, source=0, target=3))
print(paths)
print(print(list(nx.shortest_simple_paths(G, 0, 3))))
print([list(path) for path in  map(nx.utils.pairwise, paths)])
G_drawing(G,routes=paths,edge_widths=[10,7,5,4,2])
```

    [[0, 1, 2, 3], [0, 1, 3], [0, 2, 1, 3], [0, 2, 3], [0, 3]]
    [[0, 3], [0, 1, 3], [0, 2, 3], [0, 1, 2, 3], [0, 2, 1, 3]]
    None
    [[(0, 1), (1, 2), (2, 3)], [(0, 1), (1, 3)], [(0, 2), (2, 1), (1, 3)], [(0, 2), (2, 3)], [(0, 3)]]
    

<img src="./imgs/2_8_1/output_40_1.png" height='auto' width='auto' title="caDesign">
    
    


配置`cutoff`参数，仅返回小于等于给定路长度的路。


```python
paths=list(nx.all_simple_paths(G, source=0, target=3, cutoff=2))
print(paths)
G_drawing(G,routes=paths,edge_widths=[10,7,5])
```

    [[0, 1, 3], [0, 2, 3], [0, 3]]
    

<img src="./imgs/2_8_1/output_42_1.png" height='auto' width='auto' title="caDesign">
    



配置`target`参数为一个列表时，返回以任意多个节点结尾的所有路径。


```python
paths=list(nx.all_simple_paths(G, source=0, target=[3,2]))
print(paths)
```

    [[0, 1, 2], [0, 1, 2, 3], [0, 1, 3], [0, 1, 3, 2], [0, 2], [0, 2, 1, 3], [0, 2, 3], [0, 3], [0, 3, 1, 2], [0, 3, 2]]
    

使用函数编程方法（functional programming approach）返回有向非循环图中（ directed acyclic graph）迭代从根节点（root nodes）到叶节点（leaf nodes）的每个路径。


```python
from itertools import chain
from itertools import product
from itertools import starmap
from functools import partial

chaini=chain.from_iterable
G=nx.DiGraph([(0, 1), (1, 2), (0, 3), (3, 2)])
G_drawing(G)
roots=(v for v, d in G.in_degree() if d == 0)
leaves=(v for v, d in G.out_degree() if d == 0)
all_paths=partial(nx.all_simple_paths, G)
list(chaini(starmap(all_paths, product(roots, leaves))))
```

<img src="./imgs/2_8_1/output_46_0.png" height='auto' width='auto' title="caDesign">
    






    [[0, 1, 2], [0, 3, 2]]



使用迭代方法（iterative approach）返回有向非循环图中（ directed acyclic graph）迭代从根节点（root nodes）到叶节点（leaf nodes）的每个路径。


```python
roots=(v for v, d in G.in_degree() if d == 0)
leaves=(v for v, d in G.out_degree() if d == 0)
print([list(nx.all_simple_paths(G, root, leaf)) for leaf in leaves for root in roots])
```

    [[[0, 1, 2], [0, 3, 2]]]
    

在有向非循环图中，迭代从根节点到叶节点的每个路径，将所有叶传递到一起，以避免不必要的计算。


```python
G=nx.DiGraph([(0, 1), (2, 1), (1, 3), (1, 4)])
G_drawing(G)
roots=(v for v, d in G.in_degree() if d == 0)
leaves=[v for v, d in G.out_degree() if d == 0]
print([list(nx.all_simple_paths(G, root, leaves)) for root in roots])
```

<img src="./imgs/2_8_1/output_50_0.png" height='auto' width='auto' title="caDesign">
    

    


    [[[0, 1, 3], [0, 1, 4]], [[2, 1, 3], [2, 1, 4]]]
    

`all_simple_edge_paths`(G, source, target, cutoff=None)<sup>[8]</sup>：返回图G中所有源汇顶点路的边列表。


```python
G=nx.Graph([(1, 2), (2, 4), (1, 3), (3, 4)])
G_drawing(G)
sorted(nx.all_simple_edge_paths(G, 1, 4))
```

<img src="./imgs/2_8_1/output_52_0.png" height='auto' width='auto' title="caDesign">
    
    





    [[(1, 2), (2, 4)], [(1, 3), (3, 4)]]



对于多重图（MultiGraph），返回路的边时，也包含其关联的键。


```python
MG=nx.MultiGraph()
MG.add_edge(1, 2, key="k0")
MG.add_edge(1, 2, key="k1")
MG.add_edge(2, 3, key="k0")
G_drawing(MG)
sorted(nx.all_simple_edge_paths(MG, 1, 3))
```

<img src="./imgs/2_8_1/output_54_0.png" height='auto' width='auto' title="caDesign">
    





    [[(1, 2, 'k0'), (2, 3, 'k0')], [(1, 2, 'k1'), (2, 3, 'k0')]]



`is_simple_path`(G, nodes)：如果顶点来自于简单路（simple path）则返回为True。一个简单路是一个非空顶点列表，其中没有顶点在序列中出现一次以上，且序列中的每对邻点在图中相邻。


```python
G=nx.cycle_graph(4)
G_drawing(G)
print(nx.is_simple_path(G, [2, 3, 0]))
print(nx.is_simple_path(G, [0, 2]))
```

<img src="./imgs/2_8_1/output_56_0.png" height='auto' width='auto' title="caDesign">
    
    


    True
    False
    

若$P= x_{0} \ldots  x_{k}  $是一条路且$k \geq 3$，则称图$C:=P+ x_{k-1}  x_{0} $为**圈（cycle）**。与路一样，经常用顶点的（循环）序列来表示一个圈；上面提到的圈$C$可以表示为$x_{0} x_{1} \ldots  x_{k-1} x_{0}    $。圈中的边数（或顶点数）称为**长度（length）**，长度为$k$的圈亦称为**k-圈（k-cycle）**并记为$C^{k} $。

图G中最短圈的长度叫做**围长（girth）**，记作$g(G)$，而$G$中最长圈的长度称为**周长（circumference）**。（若$G$中不含圈，则围长设为$\infty $而周长为零）。图G中不在圈上但连接圈中两个顶点的边称为这个圈的**弦（chord）**，所以$G$的**导出圈（induced cycle）**是不含弦的圈（即G的导出子图是个圈）。

`add_cycle`(G_to_add_to, nodes_for_cycle, **attr)：将圈添加到图G中。


```python
G=nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
cycle_1= [0, 1, 2, 3,4,5]
cycle_2=[2,3,6,7]
G.add_edges_from([(5,9),(5,10),(7,11)])
nx.add_cycle(G,cycle_1)
nx.add_cycle(G, cycle_2, weight=7)
G_drawing(G,routes=[cycle_1+[cycle_1[0]],cycle_2+[cycle_2[0]]])
```

<img src="./imgs/2_8_1/output_58_0.png" height='auto' width='auto' title="caDesign">
    
    


`find_cycle`(G, source=None, orientation=None)：通过深度优先（depth-first traversal）遍历找到圈，返回圈的有向边列表。

`cycle_basis`(G, root=None)<sup>[9]</sup>：返回基础圈（basis for cycles）的一个列表，且为圈的最小集合，即图G中任何一个圈都为基础圈对象的和。


```python
print(nx.find_cycle(G))
print(nx.cycle_basis(G))
print([sorted(c) for c in nx.minimum_cycle_basis(G)])
```

    [(5, 4), (4, 3), (3, 2), (2, 1), (1, 0), (0, 5)]
    [[2, 7, 6, 3], [1, 2, 3, 4, 5, 0]]
    [[0, 1, 2, 3, 4, 5], [2, 3, 6, 7]]
    

`simple_cycles`(G)<sup>[10][11][12]</sup>：返回有向图的简单圈（simple cyccle）。


```python
edges=[(0, 0), (0, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]
G=nx.DiGraph(edges)
G_drawing(G)
print(sorted(nx.simple_cycles(G)))
```

<img src="./imgs/2_8_1/output_62_0.png" height='auto' width='auto' title="caDesign">
    



    [[0], [0, 1, 2], [0, 2], [1, 2], [2]]
    

`recursive_simple_cycles`(G)<sup>[10]</sup>：返回有向图的简单圈（simple cyccle）。


```python
print(nx.recursive_simple_cycles(G))
```

    [[0, 1, 2], [0, 2], [1, 2]]
    


```python
H=G.copy()
H.remove_edges_from(nx.selfloop_edges(G))
c
sorted(nx.simple_cycles(H))
```

<img src="./imgs/2_8_1/output_65_0.png" height='auto' width='auto' title="caDesign">
    
    





    [[0, 1, 2], [0, 2], [1, 2]]



图G中的两个顶点$x,y$之间的**距离（distance）**$d_{G} (x,y)$定义为$G$中最短$x-y$路的长度；如果这样的路不存在，则令$d(x,y):= \infty $。图$G$中所有顶点对之间的距离最大值称为$G$的**直径（diameter）**，记为$diam G$。当然直径和周长是密切相关的。

`diameter`(G, e=None, usebounds=False)：返回图G的直径。直径是最大偏心距（maximum eccentricity）。

`shortest_path`(G, source=None, target=None, weight=None, method='dijkstra')：给定源汇顶点计算最短路径。如果配置参数`weight`，则按权重值计算最短路径。


```python
G = nx.Graph()
G.add_edge("a", "b", weight=0.6)
G.add_edge("a", "c", weight=0.2)
G.add_edge("c", "d", weight=0.1)
G.add_edge("c", "e", weight=0.7)
G.add_edge("c", "f", weight=0.9)
G.add_edge("a", "d", weight=0.05)
G_drawing(G,edge_labels='weight')
print(nx.diameter(G))
print(nx.shortest_path(G, source="b", target="e"))
print(nx.shortest_path(G, source="b", target="e",weight='weight'))
```

<img src="./imgs/2_8_1/output_67_0.png" height='auto' width='auto' title="caDesign">

    
    


    3
    ['b', 'a', 'c', 'e']
    ['b', 'a', 'd', 'c', 'e']
    

`all_shortest_paths`(G, source, target, weight=None, method='dijkstra')：给定源汇顶点返回所有最短路径。


```python
G=nx.Graph()
nx.add_path(G, [0, 1, 2])
nx.add_path(G, [0, 10, 2])
G_drawing(G)
print([p for p in nx.all_shortest_paths(G, source=0, target=2)])
```

<img src="./imgs/2_8_1/output_69_0.png" height='auto' width='auto' title="caDesign">
    

    


    [[0, 1, 2], [0, 10, 2]]
    

`shortest_path_length`(G, source=None, target=None, weight=None, method='dijkstra')：计算源汇之间最短路长度。如果指定源汇则返回最短路长度；如果只指定源则返回源到各个顶点的最短路长度；如果只指定汇则返回各个顶点到汇的最短路长度；如果不指定源汇则返回所有顶点对的最短路长度。

`average_shortest_path_length`(G, weight=None, method=None：返回平均最短路长度，其公式为：$a =\sum_{s,t \in V} \frac{d(s, t)}{n(n-1)}$，式中，$v$是$G$的顶点集，$d(s, t)$是从顶点$s$到$t$的最短路长度，$n$是图$G$顶点数（阶）。


```python
G=nx.path_graph(5)
G_drawing(G)
print(nx.shortest_path_length(G, source=0, target=4))
print(nx.shortest_path_length(G, source=0))  # target not specified
print(nx.shortest_path_length(G, target=4))  # source not specified
print(dict(nx.shortest_path_length(G)))  # source,target not specified
print(nx.average_shortest_path_length(G))
```

<img src="./imgs/2_8_1/output_71_0.png" height='auto' width='auto' title="caDesign">
    

    


    4
    {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    {4: 0, 3: 1, 2: 2, 1: 3, 0: 4}
    {0: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}, 1: {1: 0, 0: 1, 2: 1, 3: 2, 4: 3}, 2: {2: 0, 1: 1, 3: 1, 0: 2, 4: 2}, 3: {3: 0, 2: 1, 4: 1, 1: 2, 0: 3}, 4: {4: 0, 3: 1, 2: 2, 1: 3, 0: 4}}
    2.0
    

图$G$的**中心点（central vertex）**是指能使得它到任何其它顶点的距离尽可能小的顶点（中心点可以不只一个），这个最短距离称作**半径（radius）**并记为$rad G$。严格的说$radG=min \thinspace max \thinspace  d_{G}(x,y),  \thinspace   x \in V(G) \thinspace y \in V(G) $，为各顶点到其它各顶点的最大距离的最小值。

`center`(G, e=None, usebounds=False)：返回图G的中心点。

`radius`(G, e=None, usebounds=False)：返回图G的半径。


```python
G=nx.Graph([(1, 2), (1, 3), (1, 4), (3, 4), (3, 5), (4, 5),(5,6),(6,7)])
G_drawing(G)
print(list(nx.center(G)))
print(nx.radius(G))
```

<img src="./imgs/2_8_1/output_73_0.png" height='auto' width='auto' title="caDesign">
    

    


    [3, 4, 5]
    3
    

图$G$中长度为$k$的**途径（walk）**是一个非空的顶点和边的交错序列$v_{0} e_{0}  v_{1} e_{1} \ldots   e_{k-1} v_{k}  $，使得对于所有$i<k$均有$e_{i}=\{ v_{i}, v_{i+1}  \} $，若$v_{0}=v_{k} $，则称此途径是**闭的（closed）**。如果途径中的顶点互不相同，会得到$G$中的一条路。

### 2.8.1.4  连通性

如果非空图$G$中的任意两个顶点之间均有一条路相邻，称$G$是**连通的（connected）**（若一个图中有$n$个顶点，并且边数小于$n-1$，则此图一定是非连通的）。若$U \subseteq V(G)$且$G[U]$是联通的，则称（在$G$中）$U$本身是连通的。

`is_connected`(G)：如图是连通的，则返回True，否则返回False。


```python
G=nx.path_graph(4)
G_drawing(G)
print(nx.is_connected(G))
G.add_edges_from([(1,3)])
G.add_node(4)
G_drawing(G)
print(nx.is_connected(G))
```

<img src="./imgs/2_8_1/output_76_0.png" height='auto' width='auto' title="caDesign">
    



    True
    
<img src="./imgs/2_8_1/output_76_2.png" height='auto' width='auto' title="caDesign">

    

    


    False
    

设$G=(V,E)$是一个图，则它的极大连通子图称为**分支（component）**。显然，分支都是导出子图且它们的顶点集划分$V$。因为连通图是非空的，所以空图没有分支。

`number_connected_components`(G)：返回连通分支（connected components）的数量。


```python
G=nx.Graph([(0, 1), (1, 2), (5, 6), (3, 4)])
c
print(nx.number_connected_components(G))
```

<img src="./imgs/2_8_1/output_78_0.png" height='auto' width='auto' title="caDesign">
    

    


    3
    

`connected_components`(G)：返回所有连通分支。

`subgraph`(G, nbunch)：返回顶点列表中的导出子图。


```python
G=nx.path_graph(4)
nx.add_path(G, [10, 11, 12])
G.add_edges_from([(2,4),(1,4),(4,5)])
G_drawing(G)
print(list(nx.connected_components(G)))
print([len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)])
largest_cc=max(nx.connected_components(G), key=len) # 返回最大阶的连通分支
print(largest_cc)
S=[G.subgraph(c).copy() for c in nx.connected_components(G)] # 创建每个连通分支的导出子图
[G_drawing(s) for s in S];
```

<img src="./imgs/2_8_1/output_80_0.png" height='auto' width='auto' title="caDesign">
    

    


    [{0, 1, 2, 3, 4, 5}, {10, 11, 12}]
    [6, 3]
    {0, 1, 2, 3, 4, 5}
    

<img src="./imgs/2_8_1/output_80_2.png" height='auto' width='auto' title="caDesign">
    

    

<img src="./imgs/2_8_1/output_80_3.png" height='auto' width='auto' title="caDesign">

    

    


`node_connected_component`(G, n)：返回包含顶点$n$图$G$的连通分支。


```python
print(nx.node_connected_component(G, 2))  # nodes of component that contains node 0
print(nx.node_connected_component(G, 11)) 
```

    {0, 1, 2, 3, 4, 5}
    {10, 11, 12}
    

给定$A,B \subseteq V$和$X \subseteq V \cup E$，如果$G$的每条$A-B$路均包含$X$中的一个顶点或一条边，则称在$G$中**X分离（separate）**集合$A$和$B$。注意到，这蕴含着$A \cap B \subseteq X$。若两个顶点$a,b \notin X$且$X$分离顶点$\{a\}$，$\{b\}$，则称**X分离**顶点$a,b$；如果$X$分离$G$中的两个顶点，就称**X分离G**。顶点所形成的分离集合亦称为**分隔（separator）**。边的分离集合没有通用的名称，但某些这样的集合会有专用名称，例如**割（cut）**和**键（bond）**。如果一个顶点分离同一个分支中的两个顶点，则称它为**割点（cutvertex）**，而**桥（bridge）**则为分离其两个端点的边，所以图中的桥恰为那些不在任何圈中的边。

如果$A \cup B=V$且$G$没有$A \backslash B$和$B \backslash A$之间的边，则无序对$\{A,B\}$称为$G$的**分离（separation）**。显然，第二个条件等于说，$A \cap B$分离$A$和$B$。若$A \backslash B$和$B \backslash A$均非空，则称这个分离式**真的（proper）**，而$ | A \cap B | $叫做$\{A,B\}$的阶（order），集合$A,B$叫做分离的**侧面（side）**。

>  割点：如果去掉一个点及与它连接的边，该点原来所在的图被分成两部分（不连通），则称该点为割点；割边：如果去掉一条边，该边原来所在的图被分成两部分（不连通），则称该点为割边（桥）。

`d_separated`(G, x, y, z)<sup>[13][14][15][16]</sup>：返回是否顶点集$z$分离顶点集$x$和$y$。


```python
g = nx.DiGraph()
g.add_edges_from(
    [
        ("S1", "S2"),
        ("S2", "S3"),
        ("S3", "S4"),
        ("S4", "S5"),
        ("S1", "O1"),
        ("S2", "O2"),
        ("S3", "O3"),
        ("S4", "O4"),
        ("S5", "O5"),
    ]
)
G_drawing(g)
print(nx.d_separated(g, {"S1", "S2", "O1", "O2"}, {"S4", "S5", "O4", "O5"}, {"S3","S4"}))
print(nx.d_separated(g, {"S1", "S2", "O1", "O2"}, {"S3", "S4", "O4", "O3"}, {"S2"}))
print(nx.d_separated(g, {"S1", "S2", "O1", "O2"}, {"S4", "S5", "O4", "O5"}, {"S1"}))
```

<img src="./imgs/2_8_1/output_84_0.png" height='auto' width='auto' title="caDesign">
    

    


    True
    True
    False
    

若$|G  | >k \in \mathbb{N}$，且对任意满足$|X| <k $的子集$X \subseteq V$均有$G-X$是连通的，则称$G$是**k-连通的（k-connected）**。换言之，$G$中任意两个顶点都不能被少于$k$个其它顶点所分离。所有（非空）图都是0-连通的；而那些1-连通图恰为那些非平凡连通图。使得$G$是k-连通的最大整数$k$称为$G$的**连通度（connectivity）**并记为$\kappa (G)$。所以，$\kappa (G)=0$当且仅当$G$是不连通的或是$K^{1} $。对于任意$n \geq 1$均有$\kappa ( K^{n} )=n-1$。

> 一个网络的健壮程度是指它不容易因为顶点的移除使得源汇顶点连接断开，因此顶点的连通度指，要断开源汇顶点的连通所要移除的最小顶点数。可以证明，连通度也等于顶点之间独立于节顶点的路径数，因此可以通过计算独立于顶点的路径数来量化网络的健壮性<sup>[17]</sup>。

`all_pairs_node_connectivity`(G, nbunch=None, cutoff=None)：计算所有顶点对之间的连通度（node connectivity）。


```python
import pprint

G=nx.cycle_graph(3)
G.add_node("a")
G.add_edge(2, 3)
G_drawing(G) 
pprint.pprint(nx.all_pairs_node_connectivity(G))    
```

<img src="./imgs/2_8_1/output_86_0.png" height='auto' width='auto' title="caDesign">
    

    


    {0: {1: 2, 2: 2, 3: 1, 'a': 0},
     1: {0: 2, 2: 2, 3: 1, 'a': 0},
     2: {0: 2, 1: 2, 3: 1, 'a': 0},
     3: {0: 1, 1: 1, 2: 1, 'a': 0},
     'a': {0: 0, 1: 0, 2: 0, 3: 0}}
    

`local_node_connectivity`(G, source, target, cutoff=None)： 计算源汇顶点之间的连通度。


```python
from networkx.algorithms import approximation as approx
G=nx.octahedral_graph()
G_drawing(G) 
approx.local_node_connectivity(G, 1,3)
```

<img src="./imgs/2_8_1/output_88_0.png" height='auto' width='auto' title="caDesign">
    

    





    4



`node_connectivity`(G, s=None, t=None)：计算有向图或无向图近似的顶点连通度。为使得图$G$为不连通所要删除的最小顶点数。


```python
from networkx.algorithms import approximation as approx
G=nx.octahedral_graph()
G.remove_edges_from([(0,4),(1,2),(0,1)])
G_drawing(G)
print(approx.node_connectivity(G))
```

<img src="./imgs/2_8_1/output_90_0.png" height='auto' width='auto' title="caDesign">
    

    


    2
    

若$| G | >1$且对任意少于$\ell $条边的集合$F \subseteq E$，$G-F$均是连通的，则称$G$是$\ell $边连通的（$\ell $-edge-connected）。使得$G$为$\ell $边连通的最大整数$\ell $叫做$G$的**边连通度（edge-connectivity）**并记为$\lambda (G)$。特别的，若$G$是不连通的，则$\lambda (G)=0$。

`edge_connectivity`(G, s=None, t=None, flow_func=None, cutoff=None)<sup>[18]</sup>：返回有向图或无向图G的边连通度。如果给定一对顶点，则可以返回本地边连通度（local edge connectivity）。

`local_edge_connectivity`(G, s, t, flow_func=None, auxiliary=None, residual=None, cutoff=None)<sup>[18]</sup>： 给定一对顶点，返回本地边连通度（local edge connectivity）。


```python
# Platonic icosahedral graph is 5-edge-connected
G=nx.icosahedral_graph()
G_drawing(G)
print(nx.edge_connectivity(G))
print(nx.edge_connectivity(G, 1, 7))

from networkx.algorithms.connectivity.connectivity import local_edge_connectivity
print(local_edge_connectivity(G, 1, 7))
```

<img src="./imgs/2_8_1/output_92_0.png" height='auto' width='auto' title="caDesign">
    

    


    5
    5
    5
    

### 2.8.1.5 树和森林

一个**无圈（acyclic）**图，即不含任何圈的图，亦称为**森林（forest）**，而连通的森林则称为**树（tree）**。（所以森林里是分支为树的图）。树中度为1的顶点为**叶子（leaf）**，而其它顶点称为内部顶点。每个非平凡的树都有叶子，例如，最长路的端点。这一简单的结论有时会很有用，尤其是对树使用归纳法时：去掉树的一个叶子，剩下的图还是一个树。对于树$T$，当$T$是图$G$的一颗支撑树时，$E(G) \backslash E(T)$中的边是$T$在$G$中的**弦（chord）**。

`is_tree`(G)：返回图$G$是否为树。


```python
G=nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (2, 4), (2, 5)])
G_drawing(G)
print(nx.is_tree(G)) # n-1 edges

G.add_edge(3, 4)
G_drawing(G)
print(nx.is_tree(G)) # n edges
```

<img src="./imgs/2_8_1/output_94_0.png" height='auto' width='auto' title="caDesign">
    

    


    True
    

<img src="./imgs/2_8_1/output_94_2.png" height='auto' width='auto' title="caDesign">
    

    


    False
    

`is_forest`(G)：返回图$G$是否为森林。


```python
G=nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (2, 4), (2, 5)])
G_drawing(G)
print(nx.is_forest(G))

G.add_edge(4, 1)
G_drawing(G)
print(nx.is_forest(G))
```

<img src="./imgs/2_8_1/output_96_0.png" height='auto' width='auto' title="caDesign">
    

    


    True
    

<img src="./imgs/2_8_1/output_96_2.png" height='auto' width='auto' title="caDesign">
    

    


    False
    

有时，把树中的一个顶点做特别处理会方便问题的解决，这个顶点称为树的**根（root）**，而具有固定根$r$的树叫做**有根树（rooted tree）**。把$x \in r T_{y} $记作$x \leq y$，从而定义了V(T)上的一个偏序关系，称它是与$T$和$r$关联的**树序（tree-order）**。这种序可看作是用“高度”来刻画的：若$x<y$，则称$x$在$T$中位于$y$**之下（below）**，而$\lceil y\rceil:=\{x | x \leq y\}$和$\lfloor x\rfloor:=\{y | y \geq x\}$分别称为$y$的**下闭包（down-closure）**和$x$的**上闭包（up-closure）**。如果集合$X \subseteq V(T)$等于它自身的上闭合包，即满足$X=\lfloor X\rfloor:=  \cup _{x \in X} \lfloor x\rfloor$，则称它在$T$中是**上闭的（up-closed）**或是一个**上集合（up-set）**；类似的，可以定义**下闭的（down-closed）**或**下集合（down-set）**。

注意到，$T$的根是这个偏序关系中最小的元素，叶子是极大元素，而$T$的任意边的两个端点是可比的，任意顶点的下闭包是一条**链（chain）**，即两两可比较的元素的集合。到根的距离为$k$的顶点具有**高度（height）**$k$，并组成$k$的第$k$**层（level）**。

对于包含于$G$中的有根树$T$，如果$G$中的任意T-路的两个端点在$T$的树序中是可比的，称有根树$T$在$G$中是**正规的（normal）**。若$T$支撑$G$，这等于是要求：只要两个顶点在$G$中是相邻的，它们在$T$中一定是可比的。正规支撑树也叫做**深度优先搜索树（depth-first search tree，DFS tree）**，这是因为它和图的计算机搜索相关。

`join`(rooted_trees, label_attribute=None)：给定多个有根树，返回一个新的有根树。新的有根树的根连接到给定多个有根树的根。

> 为了更清晰的表达树和森林，安装`pydot`（`conda install pydot`）库，通过`graphviz_layout`方法传递参数`prog`，包括"twopi", "dot", "circo"等获得顶点位置值，绘制图。


```python
from networkx.drawing.nx_pydot import graphviz_layout
import pydot

h=4
left=nx.balanced_tree(2, h)
right=nx.balanced_tree(2, h)

pos=graphviz_layout(left, prog="circo") 
G_drawing(left,pos=pos,figsize=(5,5))

pos=graphviz_layout(right, prog="dot")
G_drawing(right,pos=pos,figsize=(8,5))

joined_tree=nx.join([(left, 0), (right, 0)])
pos=graphviz_layout(joined_tree, prog="twopi")
G_drawing(joined_tree,pos=pos,figsize=(5,5))
```

<img src="./imgs/2_8_1/output_98_0.png" height='auto' width='auto' title="caDesign">
    

    
<img src="./imgs/2_8_1/output_98_1.png" height='auto' width='auto' title="caDesign">


    

    
<img src="./imgs/2_8_1/output_98_2.png" height='auto' width='auto' title="caDesign">


    
    


`dfs_edges`(G, source=None, depth_limit=None)：在深度优先搜索（ depth-first-search (DFS)）下沿给定源顶点，迭代返回边。可以配置参数`depth_limit`限制搜索的深度。


```python
print(list(nx.dfs_edges(right, source=0)))
print('-'*50)
print(list(nx.dfs_edges(right, source=0, depth_limit=2)))
```

    [(0, 1), (1, 3), (3, 7), (7, 15), (7, 16), (3, 8), (8, 17), (8, 18), (1, 4), (4, 9), (9, 19), (9, 20), (4, 10), (10, 21), (10, 22), (0, 2), (2, 5), (5, 11), (11, 23), (11, 24), (5, 12), (12, 25), (12, 26), (2, 6), (6, 13), (13, 27), (13, 28), (6, 14), (14, 29), (14, 30)]
    --------------------------------------------------
    [(0, 1), (1, 3), (1, 4), (0, 2), (2, 5), (2, 6)]
    

`dfs_tree`(G, source=None, depth_limit=None)：在深度优先搜索（ depth-first-search (DFS)）下给定源顶点，返回指定深度的定向树（oriented tree）。


```python
T=nx.dfs_tree(right, source=1, depth_limit=2)
pos=graphviz_layout(T, prog="dot")
G_drawing(T,pos=pos)
```

<img src="./imgs/2_8_1/output_102_0.png" height='auto' width='auto' title="caDesign">
    

    


`dfs_predecessors`(G, source=None, depth_limit=None)：在深度优先搜索（ depth-first-search (DFS)）下给定源顶点，返回指定深度，顶点为键，值为该顶点的祖先（predecessors ）。


```python
print(nx.dfs_predecessors(right, source=0,depth_limit=2))
```

    {1: 0, 3: 1, 4: 1, 2: 0, 5: 2, 6: 2}
    

`dfs_successors`(G, source=None, depth_limit=None)：在深度优先搜索（ depth-first-search (DFS)）下给定源顶点，返回指定深度，顶点为键，值为该顶点的后代（successors）。


```python
print(nx.dfs_successors(right, source=0, depth_limit=2))
```

    {0: [1, 2], 1: [3, 4], 2: [5, 6]}
    


```python

```

### 2.8.1.6 二部图

设$r \geq  2$是一个整数，对于图$G=(V,E)$，如果$V$可以划分为$r$个类使得任意一条边的端点都属于不同的类中（即同一类中的顶点不相邻），则称$G$为**r-部图（r-partite graph）**。通常把2-部图称为**二部图（bipartite graph）**。

若$r$-部图中，不同类中任意两个顶点均相邻，则称它为**完全r-部图（complete r-partite graph）**。所有的$r$-部图被统称为**完全多部图（complete multipartite graph）**。把完全$r$部图$\overline{  k^{ n_{1}} } * \ldots *\overline{ k^{ n_{r} } } $记为$ x_{ x_{1}, \ldots , n_{r}  } $；当$n_{1} = \ldots = n_{r} =:s$时，可简记为$K_s^r $。所以$K_s^r $是一个每个类都恰好有$s$个顶点的完全$r$-部图。图类$K_{1,n} $也叫做**星（star）**，其中$K_{1,n} $中度数为$n$的顶点称为星的**中心（center）**。任意二部图不能包含长度为奇数的圈（**奇圈（odd cycle）**）。


下述使用了两种方法定义二部图，第1种通过参数`bipartite`指定两个顶点集，并自定义边实现；第2种使用`networkx.bipartite`模块提供的生成方式构建。为了清晰显示二部图的关系，用`networkx.bipartite.sets`方法提取顶点集，并配合使用`networkx.bipartite_layout`定义顶点位置，传入到自定义`G_drawing()`函数中绘制二部图。


```python
B=nx.Graph()
# Add nodes with the node attribute "bipartite"
B.add_nodes_from([1, 2, 3, 4], bipartite=0)
B.add_nodes_from(["a", "b", "c"], bipartite=1)
# Add edges only between nodes of opposite node sets
B.add_edges_from([(1, "a"), (1, "b"), (2, "b"), (2, "c"), (3, "c"), (4, "a")])
print(nx.bipartite.sets(B))
pos=nx.bipartite_layout(B, nx.bipartite.sets(B)[0])
print(pos)
G_drawing(B,pos=pos)
```

    ({1, 2, 3, 4}, {'c', 'b', 'a'})
    {1: array([-0.75   , -0.65625]), 2: array([-0.75   , -0.21875]), 3: array([-0.75   ,  0.21875]), 4: array([-0.75   ,  0.65625]), 'c': array([ 1.     , -0.65625]), 'b': array([1.00000000e+00, 2.08166817e-17]), 'a': array([1.     , 0.65625])}
    

<img src="./imgs/2_8_1/output_109_1.png" height='auto' width='auto' title="caDesign">
    

    



```python
G=nx.bipartite.gnmk_random_graph(3, 5, 10, seed=123)
top=nx.bipartite.sets(G)[0]
pos=nx.bipartite_layout(G, top)
G_drawing(G,pos=pos)
```

<img src="./imgs/2_8_1/output_110_0.png" height='auto' width='auto' title="caDesign">
    

    


除了用`networkx.bipartite.sets`方法获取顶点集，也可以直接使用顶点属性的方法提取。


```python
top_nodes={n for n, d in B.nodes(data=True) if d["bipartite"] == 0}
bottom_nodes=set(B)-top_nodes
print(top_nodes,bottom_nodes)
```

    {1, 2, 3, 4} {'c', 'b', 'a'}
    

* Basic functions（基本函数）

`is_bipartite`(G)：判断是否为二部图。

`is_bipartite_node_set`(G, nodes)：判断顶点集是否为二分部顶点集。

`sets`(G[, top_nodes])：返回顶点集。

`color`(G)：返回二部图顶点集染色（ two-coloring）。

`density`(B, nodes)：返回二部图的密度。

`degrees`(B, nodes[, weight])：返回二部图顶点集顶点的度值。


```python
from networkx.algorithms import bipartite
print(bipartite.is_bipartite(B)) 
print(bipartite.is_bipartite_node_set(B, top_nodes))
print(bipartite.sets(B))
print(bipartite.color(B))
print(bipartite.density(B,top_nodes))
print(bipartite.density(B, bottom_nodes))
print(bipartite.degrees(B, top_nodes))
```

    True
    True
    ({1, 2, 3, 4}, {'c', 'b', 'a'})
    {1: 1, 'a': 0, 'b': 0, 2: 1, 'c': 0, 3: 1, 4: 1}
    0.5
    0.5
    (DegreeView({'c': 2, 'b': 2, 'a': 2}), DegreeView({1: 2, 2: 2, 3: 1, 4: 1}))
    

* Edgelist （边集）

`generate_edgelist`(G[, delimiter, data])：
以列表形式返回二部图的边，可以指定返回边的属性值。


```python
from networkx.algorithms import bipartite
G=nx.path_graph(4)
G.add_nodes_from([0, 2], bipartite=0)
G.add_nodes_from([1, 3], bipartite=1)
G[1][2]["weight"]=3
G[2][3]["capacity"]=12
pos=nx.bipartite_layout(G, nx.bipartite.sets(G)[0])
G_drawing(G,pos=pos)
print(list(bipartite.generate_edgelist(G, data=False)))
print(list(bipartite.generate_edgelist(G, data=True)))
print(list(bipartite.generate_edgelist(G, data=['weight'])))
```

<img src="./imgs/2_8_1/output_116_0.png" height='auto' width='auto' title="caDesign">
    

    


    ['0 1', '2 1', '2 3']
    ['0 1 {}', "2 1 {'weight': 3}", "2 3 {'capacity': 12}"]
    ['0 1', '2 1 3', '2 3']
    

`write_edgelist`(G, path[, comments, ...])：将二部图边集合写入至文件。

`read_edgelist`(path[, comments, delimiter, ...])：配合`write_edgelist`方法读取边集合文件为二部图。


```python
nx.write_edgelist(G, "./data/test.edgelist")
```


```python
fh=open("./data/test.edgelist", "r")
print(fh.readlines())
G=bipartite.read_edgelist("./data/test.edgelist")
pos=nx.bipartite_layout(G, nx.bipartite.sets(G)[0])
G_drawing(G,pos=pos,edge_labels='weight')
```

    ['0 1 {}\n', "1 2 {'weight': 3}\n", "2 3 {'capacity': 12}\n"]
    

<img src="./imgs/2_8_1/output_119_1.png" height='auto' width='auto' title="caDesign">
    

    


`parse_edgelist`(lines[, comments, delimiter, ...])：例如以["1 2", "2 3", "3 4","3 5"]或["1 2 {'weight':3}", "2 3 {'weight':27}", "3 4 {'weight':3.0}"]，及["1 2 3", "2 3 27", "3 4 3.0","3 5"]等方式传入，自动解析构建二部图。


```python
from networkx.algorithms import bipartite
lines=["1 2", "2 3", "3 4","3 5"]
G=bipartite.parse_edgelist(lines, nodetype=int)
pos=nx.bipartite_layout(G, nx.bipartite.sets(G)[0])
G_drawing(G,pos=pos)
```

<img src="./imgs/2_8_1/output_121_0.png" height='auto' width='auto' title="caDesign">
    

    



```python
lines=["1 2 {'weight':3}", "2 3 {'weight':27}", "3 4 {'weight':3.0}"]
G=bipartite.parse_edgelist(lines, nodetype=int)
G=nx.complete_bipartite_graph(2, 3)
```


```python
lines=["1 2 3", "2 3 27", "3 4 3.0","3 5"]
G=bipartite.parse_edgelist(lines, nodetype=int, data=(("weight", float),))
pos=nx.bipartite_layout(G, nx.bipartite.sets(G)[0])
G_drawing(G,pos=pos,edge_labels='weight')
```

<img src="./imgs/2_8_1/output_123_0.png" height='auto' width='auto' title="caDesign">
    



### 2.8.1.7 收缩运算和子式

非正式的，对$X$中的某些或全部边进行“细分”，即在这些边上插入若干新的顶点，这样所得到的图叫做$X$的**细分（subdivision）**。换句话说，把$X$中的某些边替换成具有相同端点的路，使得这些路的内点即不在$V(X)$中，也不在其它任何新的路上。当$G$是$X$的细分时，亦称$G$是一个$TX^{ \tau } $。$X$中原始顶点是$TX$的**分支顶点（branch vertices）**，新的顶点叫做**细分顶点（subdividing vertices）**。注意到，细分顶点的度总是2，而分支顶点保持了它在$X$中的顶点度。

如果$Y$包含$TX$作为子图，那么$X$是$Y$的**拓扑子式（topological minor）**。下图<sup>[2]18</sup>中图$G$是一个$TX$，即$X$的细分，因为$G \subseteq Y$，故$X$是$Y$的拓扑子式。

<img src="./imgs/2_8_1/2_8_1_07.png" height='auto' width='auto' title="caDesign">

类似的，把$X$的每个顶点$x$由不相交的连通子图$G_{x} $代替，$X$的边$xy$由$G_{x}-G_{y}$边的非空集合代替，所得到的图叫做$I  X^{8} $。更严格的讲，如果图$G$的顶点集可以划分成连通子集$V_{x} $的集合$\{ V_{x} | x \in V(X) \}$使得不同的顶点$x,y \in X$在$X$中是邻接的，当且仅当$G$包含一条$V_{x}- V_{y}  $，则称$G$是一个$IX$；集合$V_{x}$是$IX$的**分支集（branch set）**。反过来，则说$X$是由$G$收缩子图$G_{x}$而得到的，称其为$G$的**收缩子式（contraction minor）**。

若图$Y$包含$IX$子图，则称$X$是$Y$的**子式（minor）**。而$IX$是$Y$中$X$的**模型（model）**，记作$X \preceq Y$（下图<sup>[2]18</sup>）。因此$X$是$Y$的子式，当且仅当存在从$V(Y)$的子集到$V(X)$上的映射$ \varphi $使得对每个顶点$x \in X$，它的原象$ \varphi ^{-1}(x) $在$Y$中是连通的，同时对每条边$x x'  \in X$，存在$Y$中的一条边连接分支集合$ \varphi ^{-1}(x) $和$\varphi ^{-1}(x' ) $。如果$ \varphi $的定义域是整个$V(Y)$，且只要$x \neq  x' $就有$x x'  \in X$且$Y$包含$ \varphi ^{-1}(x) $和$\varphi ^{-1}(x' ) $之间的边（因此$Y$是一个$IX$），则称$ \varphi $是$Y$到$X$的收缩（contraction）。

因为分支集可能是单点集，所以图的每个子图也是它的一个子式。在无限图中，分支集也允许式无限的。

<img src="./imgs/2_8_1/2_8_1_08.png" height='auto' width='auto' title="caDesign">

`contracted_nodes`(G, u, v, self_loops=True, copy=True：收缩两个顶点为一个，收缩后的顶点与原两个顶点上的任何边相关联。


```python
G=nx.cycle_graph(4)
G_drawing(G)
M=nx.contracted_nodes(G, 1, 3)
G_drawing(M)
P3=nx.path_graph(3)
G_drawing(P3)
nx.is_isomorphic(M, P3)
```

<img src="./imgs/2_8_1/output_126_0.png" height='auto' width='auto' title="caDesign">
    

    
<img src="./imgs/2_8_1/output_126_1.png" height='auto' width='auto' title="caDesign">


    

<img src="./imgs/2_8_1/output_126_2.png" height='auto' width='auto' title="caDesign">


    





    True



`contracted_edge`(G, edge, self_loops=True, copy=True)：返回收缩指定边的结果图。边收缩是将边的两个端点收缩为一个顶点，该顶点与收缩边端点所关联的边相关联。由边收缩产生的图为原始图的子式（minor）。

下述代码给定的参数`edge`为边$\{0,1\}$，可描述为图$M$是图$G$**收缩边**$\{0,1\}$而得到的结果。


```python
C5=nx.cycle_graph(5)
G_drawing(C5)
C4=nx.cycle_graph(4)
G_drawing(C4)
M=nx.contracted_edge(C5, (0, 1), self_loops=False)
G_drawing(M)
nx.is_isomorphic(M, C4)
```

<img src="./imgs/2_8_1/output_128_0.png" height='auto' width='auto' title="caDesign">
    

    

<img src="./imgs/2_8_1/output_128_1.png" height='auto' width='auto' title="caDesign">

    
    

<img src="./imgs/2_8_1/output_128_2.png" height='auto' width='auto' title="caDesign">

    
    





    True



`quotient_graph`(G, partition, edge_relation=None, node_data=None, edge_data=None, relabel=False, create_using=None)：指定顶点的等价关系（equivalence relation）返回商图（quotient graph）。参数`partition`可以为自定义函数或字典和嵌套列表。如果是函数，则必须表征图$G$顶点等价关系。其包含两个参数`u`和`v`，如果`u`和`v`在自定义同一等价关系类中则返回True。返回的图由等价类构成顶点集；如果是字典列表等，则键可以为任何有意义的块标签（block labels），值必须为形成图形节点的有效分区（valid partition），即每个顶点必须正好位于分区的一个块中。


```python
G=nx.complete_bipartite_graph(2, 3)
pos=nx.bipartite_layout(G, nx.bipartite.sets(G)[0])
G_drawing(G,pos=pos)

same_neighbors = lambda u, v: ( u not in G[v] and v not in G[u] and G[u] == G[v])
Q=nx.quotient_graph(G, same_neighbors)
print(list(Q.nodes))
G_drawing(Q,figsize=(20,3))

K2=nx.complete_graph(2)
G_drawing(K2)
print(nx.is_isomorphic(Q, K2))
```

<img src="./imgs/2_8_1/output_130_0.png" height='auto' width='auto' title="caDesign">
    

    

<img src="./imgs/2_8_1/output_130_1.png" height='auto' width='auto' title="caDesign">

    
    
<img src="./imgs/2_8_1/output_130_2.png" height='auto' width='auto' title="caDesign">


    

    





    True




```python
G=nx.path_graph(6)
G_drawing(G)
partition=[{0, 1}, {2, 3}, {4, 5}] # partition = {0: {0, 1}, 2: {2, 3}, 4: {4, 5}}
M=nx.quotient_graph(G, partition, relabel=True)
G_drawing(M)
```

<img src="./imgs/2_8_1/output_131_0.png" height='auto' width='auto' title="caDesign">
    

    

<img src="./imgs/2_8_1/output_131_1.png" height='auto' width='auto' title="caDesign">

    

    


图$G$在$H$中的**嵌入（embdedding）**是一个单射$ \varphi :V(G) \longrightarrow V(H)$，保持了某种感兴趣的图结构。所以，$ \varphi$把$G$”作为子图“嵌入到$H$中意味着$ \varphi$保持了顶点的邻接性。”导出子图“嵌入则意味着$ \varphi$既保持了顶点的邻接性，也保持了非邻接性。如果$ \varphi$是定义在$E(G)$和$V(G)$上的，并且把$G$的边$xy$映射到$H$中连接$ \varphi (x)$和$ \varphi (y)$的独立路上，则称$G$是嵌入到$H$中的”拓扑子式“。类似地，$ \varphi$把$G$作为“子式”嵌入到$H$中是指，$ \varphi$是从$V(G)$到$H$中不相交连通顶点子集（而不是单个顶点）的一个映射，使得对于$G$的任意边$xy$，$H$都有一条连接集合$ \varphi (x)$和$ \varphi (y)$的边。根据研究的对象，可以引进不同的嵌入，例如，可以类似的定义“作为支撑子图”，“作为导出子式”等嵌入。

### 2.8.1.8 欧拉环游

称一个通过图的每条边恰好一次的闭途径为**欧拉环游（Euler tour/circuit）**。如果一个图包含一个欧拉环游，称它是**欧拉的（Eulerain）**。

`is_eulerian`(G)：当且仅当图$G$是欧拉的返回True。


```python
E1=nx.DiGraph({0: [3], 1: [2], 2: [3], 3: [0, 1]})
G_drawing(E1)
print(nx.is_eulerian(E1))
E2=nx.complete_graph(5)
G_drawing(E2)
print(nx.is_eulerian(E2))
E3=nx.petersen_graph()
G_drawing(E3)
print(nx.is_eulerian(E3))
```

<img src="./imgs/2_8_1/output_134_0.png" height='auto' width='auto' title="caDesign">
    

    


    True
    
<img src="./imgs/2_8_1/output_134_2.png" height='auto' width='auto' title="caDesign">

    

    


    True
    
<img src="./imgs/2_8_1/output_134_4.png" height='auto' width='auto' title="caDesign">

    

    


    False
    

`eulerian_circuit`(G, source=None, keys=False)<sup>[19]</sup>：返回图$G$中顺序迭代一个欧拉环游所有边。


```python
G=nx.complete_graph(3)
G_drawing(G)
print(list(nx.eulerian_circuit(G)))
print(list(nx.eulerian_circuit(G, source=1)))
```

<img src="./imgs/2_8_1/output_136_0.png" height='auto' width='auto' title="caDesign">
    

    


    [(0, 2), (2, 1), (1, 0)]
    [(1, 2), (2, 0), (0, 1)]
    

`eulerize`(G)<sup>[19]</sup>：将图G转换为欧拉图（Eulerian graph）。


```python
G=nx.complete_graph(10)
G_drawing(G)
print(nx.is_eulerian(G))
print(G.number_of_nodes(),G.number_of_edges())
H=nx.eulerize(G)
G_drawing(H)
print(nx.is_eulerian(H))
print(H.number_of_nodes(),H.number_of_edges())
```

<img src="./imgs/2_8_1/output_138_0.png" height='auto' width='auto' title="caDesign">
    

    


    False
    10 45
    

<img src="./imgs/2_8_1/output_138_2.png" height='auto' width='auto' title="caDesign">
    

    


    True
    10 50
    

`has_eulerian_path`(G, source=None)：如果图$G$中有**欧拉路（径）（Eulerian path）**则返回True。欧拉路（径）为通过图的每条边恰好一次的途径，不一定闭合。

`eulerian_path`(G, source=None, keys=False)：返回图$𝐺$中顺序迭代一个欧拉路径所有边。

`is_semieulerian`(G)：返回图$G$是否为**半-欧拉（semi-Eulerian）**。半-欧拉是指存在欧拉路径但是没有欧拉环游（Eulerian circuit）。


```python
G=nx.Graph([(0, 1), (1, 2), (0, 2)])
G.add_node(3)
G.add_edge(0,4)
G_drawing(G)
print(nx.has_eulerian_path(G))

G.remove_nodes_from(list(nx.isolates(G)))
G_drawing(G)
print(nx.has_eulerian_path(G))
print(nx.has_eulerian_path(G,source=0))
print(nx.has_eulerian_path(G,source=2))
print(nx.is_semieulerian(G))
```

<img src="./imgs/2_8_1/output_140_0.png" height='auto' width='auto' title="caDesign">
    

    


    False
    

<img src="./imgs/2_8_1/output_140_2.png" height='auto' width='auto' title="caDesign">
    
)
    


    True
    True
    False
    True
    


```python
list(nx.eulerian_path(G))
```




    [(0, 1), (1, 2), (2, 0), (0, 4)]



### 2.8.1.9 若干线性代数知识

设$G=(V,E)$是一个具有$n$个顶点和$m$条边的图，这里$V=\{ v_{1}, \ldots , v_{n}  \}$和$E=\{ e_{1}, \ldots , e_{m}  \}$。图$G$的**顶点空间（vertex space）**$ {\mathcal {V}}(G) $是由所有从$V$到$\mathbb{F}_{2} $上的函数组成的向量空间（这里$\mathbb{F}_{2} =\{0,1\}$是一个二元域）。$ {\mathcal {V}} (G) $的每个元素自然地对应于$V$的一个子集，即具有函数值1的那些顶点所组成的集合，而$V$的每个子集都由它在$ {\mathcal {V}} (G) $中的指示函数唯一的表示。所以，可以把${\mathcal {V}} (G) $看作由$V$的幂集构成的向量空间：两个顶点集$U, U' \subseteq V $的和$U+U' $定义为它们的对称差，且对于任意$U \subseteq V$，有$U=-U$。空（顶点）集$Ø$可以看作$ {\mathcal {V}} (G) $中的零元。因为$\{\{ v_{1} \}, \ldots ,\{ v_{n} \} \}$构成$ {\mathcal {V}} (G) $的一组基（称作**标准基（standard basis）**），故$dim \hspace{0.2em} {\mathcal {V}} (G)=n$。

同样地，从$E$到$\mathbb{F}_{2}$的所有函数构成了$G$的**边空间（edge space）**${\mathcal {E}}(G)$：$E$的子集对应于它的元素，向量加法采用对称差运算，$Ø  \in E$是零元，且对任意$F \subseteq E$有$F=-F$。同理，$\{\{e_{1} \}, \ldots ,\{e_{n} \} \}$是${\mathcal {E}}(G)$的**标准基（standard basis）**，故$dim \hspace{0.2em} {\mathcal {E}} (G)=m$。给定边空间中的两个元素$F$和$F' $，把它们看作从$E$到$\mathbb{F}_{2}$的函数，记：$\langle F, F'  \rangle:= \sum_{e \in E} F(e) F'(e) \in  \mathbb{F}  _{2} $。

上式等于零当且仅当$F$和$F'$有偶数条公共边；特别地，若$F \neq Ø $，则$\langle F, F'  \rangle=0$。给定${\mathcal {E}}(G)$的子空间$\mathcal{F}$，记$\mathcal {F}^{\bot} :=\{D \in \mathcal {E}(G) | 对所有F \in \mathcal {F}满足\langle F, D \rangle=0\}$。

**圈空间（cycle space）**$\mathcal{C}=\mathcal{C}(G)$是由$G$中所有圈（更准确的说，是所有圈的边）支撑的${\mathcal {E}}(G)$的子空间。$\mathcal{C}(G)$的维数有时称为$G$的**圈数（cyclomatic number）**。

如果存在$V$的一个划分$\{ V_{1}, V_{2} \}$，使得$F=E( V_{1},V_{2} )$，那么称边集$F$是$G$的一个**割（cut）**；$F$中的边**横穿（cross）**这个划分；集合$\{ V_{1}$和$ V_{2} \}$是这个割的**侧面（side）**。当$V_{1}=\{v\} $时，这个割记为$E(v)$。$G$中的极小非空割是一个**键**。

最大割问题（Maximum Cut）是求一种分割方法，将图所有顶点分割成两群，同时使得被切断的边数量最大。当每条边都有权重的时候，则需要保证被切断的边权重和最大。下述计算结果如图示：

<img src="./imgs/2_8_1/2_8_1_09.jpg" height='auto' width='auto' title="caDesign">

`randomized_partitioning`(G[, seed, p, weight])：计算图顶点的随机分割及分割值。返回值`cut_size`为最小割值（cut_size，value of the minimum cut）；`partition`为定义最小割的顶点划分（partition）。

`one_exchange`(G[, initial_cut, seed, weight])：计算图顶点的分割及分割值。使用贪心交换策略（greedy one exchange strategy）找到局部最大割（locally maximal cut）及其值，添加到当前割并重复此过程，直到无法改进为止。


```python
from networkx.algorithms.approximation.maxcut import randomized_partitioning
G=nx.random_geometric_graph(6, radius=0.6, seed=3)
G_drawing(G)
print(randomized_partitioning(G))
from networkx.algorithms.approximation.maxcut import one_exchange
maximal_cut=one_exchange(G)
G_drawing(G,nodes=maximal_cut[-1])
print(maximal_cut)
```

<img src="./imgs/2_8_1/output_143_0.png" height='auto' width='auto' title="caDesign">
    

    


    (1, ({5}, {0, 1, 2, 3, 4}))
    

<img src="./imgs/2_8_1/output_143_2.png" height='auto' width='auto' title="caDesign">
    

    


    (6, ({1, 4, 5}, {0, 2, 3}))
    

`cut_size`(G, S, T=None, weight=None)：返回划分两个顶点集割的大小。


```python
G=nx.barbell_graph(3, 0)
G.add_edge(2,5)
S={0, 1, 2}
T={3, 4, 5}
G_drawing(G,nodes=[S,T])
nx.cut_size(G, S, T)
```

<img src="./imgs/2_8_1/output_145_0.png" height='auto' width='auto' title="caDesign">
    

    





    2



`minimum_edge_cut`(G, s=None, t=None, flow_func=None)<sup>[18]</sup>：返回断开图$G$最小基数（cardinality）的边集。


```python
cuts=nx.minimum_edge_cut(G)
print(cuts)
G_drawing(G,routes=cuts,edge_colors=['r']*len(cuts))
G.remove_edges_from(cuts)
G_drawing(G)
```

    {(2, 3), (2, 5)}
    

<img src="./imgs/2_8_1/output_147_1.png" height='auto' width='auto' title="caDesign">
    

    


<img src="./imgs/2_8_1/output_147_2.png" height='auto' width='auto' title="caDesign">
    

    


minimum_edge_cut(G, s=None, t=None, flow_func=None)<sup>[18]</sup>：返回断开图 𝐺 最小基数（cardinality）的顶点集。


```python
G=nx.icosahedral_graph()
node_cut=nx.minimum_node_cut(G)
G_drawing(G,nodes=[node_cut])
print(node_cut)
G.remove_nodes_from(node_cut)
G_drawing(G)
```

<img src="./imgs/2_8_1/output_149_0.png" height='auto' width='auto' title="caDesign">
    

    


    {0, 4, 5, 7, 10}
    
<img src="./imgs/2_8_1/output_149_2.png" height='auto' width='auto' title="caDesign">
    

    


`minimum_st_edge_cut`(G, s, t, flow_func=None, auxiliary=None, residual=None)：返回分割给定源汇顶点的最小基数（cardinality）的边集。


```python
from networkx.algorithms.connectivity import minimum_st_edge_cut
G=nx.icosahedral_graph()
cuts=minimum_st_edge_cut(G, 0, 6)
print(cuts)
G_drawing(G,routes=cuts,edge_colors=['r']*len(cuts))
```

    {(4, 6), (2, 6), (5, 6), (3, 6), (1, 6)}
    

<img src="./imgs/2_8_1/output_151_1.png" height='auto' width='auto' title="caDesign">
    

    


`minimum_st_node_cut`(G, s, t, flow_func=None, auxiliary=None, residual=None)minimum_st_node_cut(G, s, t, flow_func=None, auxiliary=None, residual=None)<sup>[18]</sup>：返回分割给定源汇顶点的最小基数（cardinality）的顶点集。


```python
from networkx.algorithms.connectivity import minimum_st_node_cut
G=nx.icosahedral_graph()
node_cut=minimum_st_node_cut(G, 0, 6)
G_drawing(G,nodes=[node_cut])
```

<img src="./imgs/2_8_1/output_153_0.png" height='auto' width='auto' title="caDesign">
    

    


考虑一个连通图$G=(V,E)$以及它的支撑树$T \subseteq G$。（下图<sup>[2]24</sup>）对于任意弦$e \in E \backslash E(T)$，$T+e$中存在唯一圈$C_{e} $。这些圈$C_{e} $是$G$的关于$T$的**基本圈（fundamental cycle）**。另一方面，给定一条边$f \in T$，森林$T-f$恰好有两个分支，这两个分支之间的边集合$D_{f} \in E $形成了$G$的一个键，这键是关于$T$的$f$的**基本割（fundamental cut）**。注意到，对于所有的边$e \notin T$和$f \in T  $，$f \in  C_{e}  $当且仅当$e \in  D_{f} $。意味着存在某种深刻的对偶关系。

<img src="./imgs/2_8_1/2_8_1_10.png" height='auto' width='auto' title="caDesign">

设图$G=(V,E)$的顶点集是$V=\{ v_{1}, \ldots , v_{n}  \}$，而边集是$E=\{ e_{1}, \ldots , e_{m}  \}$，则它在$\mathbb { F_{2} }$上的**关联矩阵（incidence matrix）**$B= ( b_{ij} )_{n \times m} $定义为：$b_{ij} :=\begin{cases}1 &  v_{i} \in e_{j},  \\0 & 否则\end{cases} $。依据惯例，用$B^{t} $表示$B$的转置，则$B$和$B^{t} $分别定义了关于标准基的线性映射$B:\mathcal{E}(G) \longrightarrow \mathcal{V}(G)$和$B:\mathcal{V}(G) \longrightarrow \mathcal{E}(G)$。不难验证，$B$把边集$F \subset E$映射到关联$F$中奇数条边的顶点集上，而$B^{t} $把集合$U \subseteq V$映射到恰好有一个端点在$U$中的边集合上。

图$G$的**邻接矩阵（adjacency matrix）**$A= ( a_{ij} )_{n \times n} $定义为：$a_{ij} :=\begin{cases}1 &  v_{i}  v_{j} \in E, \\0 & 否则\end{cases} $。作为一个从$\mathcal{V}$到$\mathcal{V}$的映射，邻接矩阵把一个给定集合$U \subseteq V$映射到在$U$中有奇数个邻点的顶点集上。

`incidence_matrix`(G, nodelist=None, edgelist=None, oriented=False, weight=None)：返回图$G$的关联矩阵。

`adjacency_matrix`(G, nodelist=None, dtype=None, weight='weight')：返回图$G$的邻接矩阵。


```python
G=nx.Graph([(1, 1)])
G_drawing(G)
A=nx.adjacency_matrix(G)
print(A.todense())
I=nx.incidence_matrix(G)
print(I.todense())
```

<img src="./imgs/2_8_1/output_156_0.png" height='auto' width='auto' title="caDesign">
    

    


    [[1]]
    [[0.]]
    

    C:\Users\richi\AppData\Local\Temp\ipykernel_14704\1922076179.py:3: FutureWarning: adjacency_matrix will return a scipy.sparse array instead of a matrix in Networkx 3.0.
      A=nx.adjacency_matrix(G)
    C:\Users\richi\AppData\Local\Temp\ipykernel_14704\1922076179.py:5: FutureWarning: incidence_matrix will return a scipy.sparse array instead of a matrix in Networkx 3.0.
      I=nx.incidence_matrix(G)
    


```python
G=nx.path_graph(3)
G_drawing(G)
A=nx.adjacency_matrix(G)
print(A.todense())
I=nx.incidence_matrix(G)
print(I.todense())
```

<img src="./imgs/2_8_1/output_157_0.png" height='auto' width='auto' title="caDesign">
    

    


    [[0 1 0]
     [1 0 1]
     [0 1 0]]
    [[1. 0.]
     [1. 1.]
     [0. 1.]]
    

    C:\Users\richi\AppData\Local\Temp\ipykernel_14704\1125983268.py:3: FutureWarning: adjacency_matrix will return a scipy.sparse array instead of a matrix in Networkx 3.0.
      A=nx.adjacency_matrix(G)
    C:\Users\richi\AppData\Local\Temp\ipykernel_14704\1125983268.py:5: FutureWarning: incidence_matrix will return a scipy.sparse array instead of a matrix in Networkx 3.0.
      I=nx.incidence_matrix(G)
    

`NetworkX`库提供了`to_pandas_adjacency`和`to_numpy_matrix`方法，可以将图$G$的邻接矩阵直接转换为DataFrame和array（NumPy）的数据格式。


```python
nx.to_pandas_adjacency(G)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
nx.to_numpy_matrix(G)
```




    matrix([[0., 1., 0.],
            [1., 0., 1.],
            [0., 1., 0.]])



* Mixing（混合）

`attribute_mixing_matrix`(G, attribute[, ...])：
返回顶点属性连通的混合矩阵，如果`normalized`为False，返回的为属性对出现次数，例如$male \rightarrow male$出现次数为0，$male \rightarrow female$出现次数为1，$female \rightarrow male$出现次数为1，$female \rightarrow female$出现次数为2；为True则返回的为属性对出现的联合概率（ joint probability）。

> `numeric_mixing_matrix`(G, attribute, nodes=None, normalized=True, mapping=None)方法将被移除，直接用`attribute_mixing_matrix`方法，传入属性参数。


```python
G=nx.path_graph(3)
gender={0: 'male', 1: 'female', 2: 'female'}
nx.set_node_attributes(G, gender, 'gender')
G_drawing(G,node_labels='gender',figsize=(5,5))
mapping={'male': 0, 'female': 1}
mix_mat=nx.attribute_mixing_matrix(G, 'gender', mapping=mapping,normalized=False)
# mixing from male nodes to female nodes
print(mix_mat)
print(mix_mat[mapping['male'], mapping['female']])
print(nx.attribute_mixing_matrix(G, 'gender', mapping=mapping))
```

<img src="./imgs/2_8_1/output_162_0.png" height='auto' width='auto' title="caDesign">
    

    


    [[0. 1.]
     [1. 2.]]
    1.0
    [[0.   0.25]
     [0.25 0.5 ]]
    

`degree_mixing_matrix`(G[, x, y, weight, ...])：
返回顶点度连通的混合矩阵，例如下图顶点度只有1和3两种情况，度为1的顶点连通到度为3的顶点次数为3，度为3的顶点连通到度为1的顶点次数为3，无其它情况，因此矩阵其它位置为0。


```python
G=nx.star_graph(3)
mix_mat=nx.degree_mixing_matrix(G,normalized=False)
G_drawing(G)
print(mix_mat)
mix_mat[0, 1]  # mixing from node degree 1 to node degree 3
```

<img src="./imgs/2_8_1/output_164_0.png" height='auto' width='auto' title="caDesign">
    

    


    [[0. 3.]
     [3. 0.]]
    




    3.0



可以使用`mapping`参数，显示所有的度，即使度值不存在于图中，例如度为0、1、2、3的混合矩阵。


```python
max_degree=max(deg for n, deg in G.degree)
mapping={x: x for x in range(max_degree + 1)} # identity mapping
mix_mat=nx.degree_mixing_matrix(G, mapping=mapping,normalized=False)
print(mix_mat)
mix_mat[3, 1]  # mixing from node degree 3 to node degree 1
```

    [[0. 0. 0. 0.]
     [0. 0. 0. 3.]
     [0. 0. 0. 0.]
     [0. 3. 0. 0.]]
    




    3.0



`degree_mixing_dict`(G, x='out', y='in', weight=None, nodes=None, normalized=False)：返回以字典形式存储的顶点度连通次数或者联合概率。


```python
nx.degree_mixing_dict(G)
```




    {3: {1: 3}, 1: {3: 3}}



`attribute_mixing_dict`(G, attribute[, nodes, ...])：返回字典表示的顶点属性混合矩阵。


```python
G=nx.Graph()
G.add_nodes_from([0, 1], color="red")
G.add_nodes_from([2, 3], color="blue")
G.add_edges_from([(1, 3),(2,3)])
G_drawing(G,node_labels='color')
d=nx.attribute_mixing_dict(G, "color")
print(d)
print(d["red"]["blue"])
print(d["blue"]["red"])  # d symmetric for undirected graphs

```

<img src="./imgs/2_8_1/output_170_0.png" height='auto' width='auto' title="caDesign">
    

    


    {'red': {'blue': 1}, 'blue': {'blue': 2, 'red': 1}}
    1
    1
    

`mixing_dict`(xy[, normalized])：给定元组列表的方式计算“顶点”的混合矩阵。


```python
nx.mixing_dict([(1,2),(3,4),(2,1),(3,4)])
```




    {1: {2: 1}, 2: {1: 1}, 3: {4: 2}, 4: {}}



### 2.8.1.10 图中的其它概念

**超图（hypergraph）**是一对不相交的集合$(V,E)$，其中$E$的元素是$V$的（具有任意基数的）非空子集。因此，图是一种特殊的超图。

<img src="./imgs/2_8_1/2_8_1_11.png" height='auto' width='auto' title="caDesign"><img src="./imgs/2_8_1/2_8_1_12.png" height='auto' width='auto' title="caDesign">

上图（左图一般描绘，右图称为PAOH描绘）<sup>②</sup>为一个无向超图的例子，其中集合$V=\{ v_{1},v_{2},v_{3},v_{4},v_{5},v_{6},v_{7} \}$，$E=\{ e_{1}, e_{2}, e_{3}, e_{4} \}=\{\{ v_{1},v_{2},v_{3} \},\{v_{2},v_{3}\},\{v_{3},v_{4},v_{5}\},\{v_{4}\}\}  $。这个超图的阶数（order）为7，大小（size）为4。

**有向图（directed graph or digraph）**是由一对不相交的集合$(V,E)$（分别称作顶点和边）以及两个映射$init:E \longrightarrow V $和$ter:E \longrightarrow V $组成的，其中$init$把每条边$e$映到了一个**初始点（initial vertex）**$init(e)$上，而$ter$把每条边$e$映到一个**终点（terminal vertex）**$ter(e)$上。称边$e$是从$init(e)$**指向（directed to）**$ter(e)$的。注意到在有向图中，两个顶点$x$和$y$之间可以有若干条边，这样的边称为**重边（multiple edge）**；如果边的方向相同（例如从$x$到$y$），则称为**平行边（paralle edge）**。如果$init(e)=ter(e)$，则$e$叫做**环边（loop）**。

对于有向图$D$和（无向）图$G$，如果$V(D)=V(G)$，$E(D)=E(G)$且对每条边$e=xy$有$\{init(e),ter(e)\}=\{x,y\}$，则称$D$是$G$的一个**定向（orientation）**。直观的看，一个**定向图（oriented graph）**就是把一个无向图的每条边从一个端点到另一个端点给出方向而得到的图，也可以把定向图看作没有重边和环边的有向图。

`to_directed`(graph)：返回一个图为有向图。具有相同名称、顶点，且每条边$(u, v, data)$被两条有向边$(u, v, data)$和$ (v, u, data)$替换。类似的方法还有`DiGraph.to_directed(as_view=False)`，`MultiGraph.to_directed(as_view=False)`等。


```python
G=nx.Graph()
G.add_edges_from([(0, 1),(1,2),(2,3),(1,3)])
G_drawing(G)
D=G.to_directed()
G_drawing(D)
```

<img src="./imgs/2_8_1/output_174_0.png" height='auto' width='auto' title="caDesign">
    

    


<img src="./imgs/2_8_1/output_174_1.png" height='auto' width='auto' title="caDesign">
    

    


**多重图（multigraph）**是由一对不相交的集合$(V,E)$（称为顶点和边）以及从$E$到$V \cup  [V]^{2} $的一个映射组成的，这里映射给每条边指定一个或两个顶点（叫做端点（end））。所以，多重图可以有重边和环边（或叫做**双边（double edge）**）。

---

注释（Notes）：

① NetworkX，（<https://networkx.org/documentation/stable/index.html#>）。

② Hypergraph(Wikipedia)，（<https://en.wikipedia.org/wiki/Hypergraph>）。

参考文献（References）:

[1] Estrada, E. & Rodríguez-Velázquez, J. A. Spectral measures of bipartivity in complex networks. Phys Rev E Stat Nonlin Soft Matter Phys 72, (2005).

[2] [德]Reinhard Diestel著, 于青林译.图论（第五版）[M].北京: 科学出版社, 2020.04.

[3] Boppana, R., & Halldórsson, M. M. (1992). Approximating maximum independent sets by excluding subgraphs. BIT Numerical Mathematics, 32(2), 180–196. Springer.

[4] L. P. Cordella, P. Foggia, C. Sansone, M. Vento, “An Improved Algorithm for Matching Large Graphs”, 3rd IAPR-TC15 Workshop on Graph-based Representations in Pattern Recognition, Cuen, pp. 149-159, 2001. https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.101.5342

[5] A. Barrat, M. Barthélemy, R. Pastor-Satorras, and A. Vespignani, “The architecture of complex weighted networks”. PNAS 101 (11): 3747–3752 (2004).

[6] “An algorithm for computing simple k-factors.”, Meijer, Henk, Yurai Núñez-Rodríguez, and David Rappaport, Information processing letters, 2009.

[7] Jin Y. Yen, “Finding the K Shortest Loopless Paths in a Network”, Management Science, Vol. 17, No. 11, Theory Series (Jul., 1971), pp. 712-716.

[8] R. Sedgewick, “Algorithms in C, Part 5: Graph Algorithms”, Addison Wesley Professional, 3rd ed., 2001.

[9] Paton, K. An algorithm for finding a fundamental set of cycles of a graph. Comm. ACM 12, 9 (Sept 1969), 514-518.

[10] Finding all the elementary circuits of a directed graph. D. B. Johnson, SIAM Journal on Computing 4, no. 1, 77-84, 1975. https://doi.org/10.1137/0204007

[11] Enumerating the cycles of a digraph: a new preprocessing strategy. G. Loizou and P. Thanish, Information Sciences, v. 27, 163-182, 1982.

[12] A search strategy for the elementary cycles of a directed graph. J.L. Szwarcfiter and P.E. Lauer, BIT NUMERICAL MATHEMATICS, v. 16, no. 2, 192-204, 1976.

[13] Pearl, J. (2009). Causality. Cambridge: Cambridge University Press.

[14] Darwiche, A. (2009). Modeling and reasoning with Bayesian networks. Cambridge: Cambridge University Press.

[15] Shachter, R. D. (1998). Bayes-ball: rational pastime (for determining irrelevance and requisite information in belief networks and influence diagrams). In , Proceedings of the Fourteenth Conference on Uncertainty in Artificial Intelligence (pp. 480–487). San Francisco, CA, USA: Morgan Kaufmann Publishers Inc.

[16] Koller, D., & Friedman, N. (2009). Probabilistic graphical models: principles and techniques. The MIT Press.

[17] White, Douglas R., and Mark Newman. 2001 A Fast Algorithm for Node-Independent Paths. Santa Fe Institute Working Paper #01-07-035 http://eclectic.ss.uci.edu/~drwhite/working.pdf

[18] Abdol-Hossein Esfahanian. Connectivity Algorithms. http://www.cse.msu.edu/~cse835/Papers/Graph_connectivity_revised.pdf

[19] J. Edmonds, E. L. Johnson. Matching, Euler tours and the Chinese postman. Mathematical programming, Volume 5, Issue 1 (1973), 111-114.
