<tr>
<td> 

__1.1__ XXXXX

</td>
<td>

print
`print('Hello World!') `代码。

</td>
<td>

```python
print('Hello World!') 
```

</td>
<td>

Hello World! 

</td>
<td>
</td>
</tr>


---


<tr>
<td> 

</td>
<td>

print
`print('Hello World!') `代码。

</td>
<td>

```python
print('Hello World!') 
```

</td>
<td>

Hello World! 

</td>
<td>
</td>
</tr>
















> Created on Mon Sep  5 12:45:39 2022 @author: Richie Bao-caDesign设计(cadesign.cn)

<style>
  code {
    white-space : pre-wrap !important;
    word-break: break-word;
  }
</style>

# Python Cheat Sheet-9. (OOP)_Classes_Decorators(装饰器)_Slots

<span style = "color:Teal;background-color:;font-size:20.0pt">PCS_8</span>

<table style="width:100%">
<tr>
<th style="width:10%"> 知识点 </th>
<th style="width:30%"> 描述 </th>
<th style="width:30%"> 代码段 </th> 
<th style="width:20%"> 运算结果 </th>
<th style="width:10%"> 备注</th> 
</tr>




</table>

<span style = "color:Teal;background-color:;font-size:20.0pt">是否完成PCS_9(&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;)</span>


```algorithm
% Firefly Algorithm
\begin{algorithm}
\caption{Firefly Algorithm}
\begin{algorithmic}
\STATE Objective function $f(\boldsymbol{x}), \quad \boldsymbol{x}=\left(x_1, \ldots, x_d\right)^T$.
\STATE Generate an initial population of $n$ fireflies $\boldsymbol{x}_i(i=1,2, \ldots, n)$.
\STATE Light intensity $I_i$ at $\boldsymbol{x}_i$ is determined by $f\left(\boldsymbol{x}_i\right)$.
\STATE Define light absorption coefficient $\gamma$.
\WHILE{$(t<$ MaxGeneration)}
\FOR{$i=1: n$ (all $n$ fireflies)}
\FOR{$j=1: n$ (all $n$ fireflies) (inner loop)}
\IF{$\left(I_i<I_j\right)$}
\STATE Move firefly $i$ towards $j$.
\ENDIF
\STATE Vary attractiveness with distance $r$ via $\exp \left[-\gamma r^2\right]$.
\STATE Evaluate new solutions and update light intensity.
\ENDFOR
\ENDFOR
\STATE Rank the fireflies and find the current global best $\boldsymbol{g}_*$.
\ENDWHILE
\STATE Postprocess results and visualization.
\end{algorithmic}
\end{algorithm}
```

```algorithm
% Cuckoo Search
\begin{algorithm}
\caption{Cuckoo Search}
\begin{algorithmic}
\STATE Objective function $f(\mathbf{x}), \mathbf{x}=\left(x_1, \ldots, x_d\right)^T$
\STATE Generate initial population of $n$ host nests $\mathbf{x}_i(i=1,2, \ldots, n)$
\WHILE{( $i<$ MaxGeneration) or $($ stop criterion)}
\STATE Get a cuckoo randomly by Lévy flights evaluate its quality/fitness $F_i$
\STATE Choose a nest among $n($ say, $j)$ randomly 
\IF{$\left(F_i>F_j\right)$}
\STATE replace $j$ by the new solution;
\ENDIF
\STATE A fraction $\left(p_a\right)$ of worse nests are abandoned and new ones are built;
\STATE Keep the best solutions (or nests with quality solutions);
\STATE Rank the solutions and find the current best
\ENDWHILE
\STATE Postprocess results and visualization
\end{algorithmic}
\end{algorithm}
```