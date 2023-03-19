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
% This quicksort algorithm is extracted from Chapter 7, Introduction to Algorithms (3rd edition)
\begin{algorithm}
\caption{Quicksort}
\begin{algorithmic}
\PROCEDURE{Quicksort}{$A, p, r$}
    \IF{$p < r$} 
        \STATE $q = $ \CALL{Partition}{$A, p, r$}
        \STATE \CALL{Quicksort}{$A, p, q - 1$}
        \STATE \CALL{Quicksort}{$A, q + 1, r$}
    \ENDIF
\ENDPROCEDURE
\PROCEDURE{Partition}{$A, p, r$}
    \STATE $x = A[r]$
    \STATE $i = p - 1$
    \FOR{$j = p$ \TO $r - 1$}
        \IF{$A[j] < x$}
            \STATE $i = i + 1$
            \STATE exchange
            $A[i]$ with $A[j]$
        \endif
        \STATE exchange $A[i]$ with $A[r]$
    \ENDFOR
\ENDPROCEDURE

\end{algorithmic}
\end{algorithm}
```

