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


```algorithm
% GAN
\begin{algorithm}
\caption{Minibatch stochastic gradient descent training of generative adversarial nets. The number of steps to apply to the discriminator, $k$, is a hyperparameter. We used $k=1$, the least expensive option, in our experiments.}
\begin{algorithmic}
\FOR{number of training iterations}
\FOR{$k$ steps}
\STATE • Sample minibatch of $m$ noise samples $\left\{\boldsymbol{z}^{(1)}, \ldots, \boldsymbol{z}^{(m)}\right\}$ from noise prior $p_g(\boldsymbol{z})$.
\STATE • Sample minibatch of $m$ examples $\left\{\boldsymbol{x}^{(1)}, \ldots, \boldsymbol{x}^{(m)}\right\}$ from data generating distribution $p_{\text {data }}(\boldsymbol{x})$.
\STATE • Update the discriminator by ascending its stochastic gradient:$\nabla_{\theta_d} \frac{1}{m} \sum_{i=1}^m\left[\log D\left(\boldsymbol{x}^{(i)}\right)+\log \left(1-D\left(G\left(\boldsymbol{z}^{(i)}\right)\right)\right)\right] .$
\ENDFOR
\STATE • Sample minibatch of $m$ noise samples $\left\{\boldsymbol{z}^{(1)}, \ldots, \boldsymbol{z}^{(m)}\right\}$ from noise prior $p_g(\boldsymbol{z})$.
\STATE • Update the generator by descending its stochastic gradient:$\nabla_{\theta_g} \frac{1}{m} \sum_{i=1}^m \log \left(1-D\left(G\left(\boldsymbol{z}^{(i)}\right)\right)\right) \text {. }$
\ENDFOR
\STATE The gradient-based updates can use any standard gradient-based learning rule. We used momentum in our experiments.
\end{algorithmic}
\end{algorithm}
```


```algorithm
% WGAN
\begin{algorithm}
\caption{WGAN, our proposed algorithm. All experiments in the paper used the default values $\alpha=0.00005, c=0.01, m=64, n_{\text {critic }}=5$}
\begin{algorithmic}
\REQUIRE $\alpha$, the learning rate. $c$, the clipping parameter. $m$, the batch size. $n_{\text {critic }}$, the number of iterations of the critic per generator iteration.
\REQUIRE $w_0$, initial critic parameters. $\theta_0$, initial generator's parameters.
\WHILE{$\theta$ has not converged}
\FOR{$t=0, \ldots, n_{\text {critic }}$}
\STATE Sample $\left\{x^{(i)}\right\}_{i=1}^m \sim \mathbb{P}_r$ a batch from the real data.
\STATE Sample $\left\{z^{(i)}\right\}_{i=1}^m \sim p(z)$ a batch of prior samples.
\STATE $g_w \leftarrow \nabla_w\left[\frac{1}{m} \sum_{i=1}^m f_w\left(x^{(i)}\right)-\frac{1}{m} \sum_{i=1}^m f_w\left(g_\theta\left(z^{(i)}\right)\right)\right] $
\STATE $w \leftarrow w+\alpha \cdot \operatorname{RMSProp}\left(w, g_w\right)$
\STATE $w \leftarrow \operatorname{clip}(w,-c, c)$
\ENDFOR
\STATE Sample $\left\{z^{(i)}\right\}_{i=1}^m \sim p(z)$ a batch of prior samples.
\STATE $g_\theta \leftarrow-\nabla_\theta \frac{1}{m} \sum_{i=1}^m f_w\left(g_\theta\left(z^{(i)}\right)\right) $
\STATE $\theta \leftarrow \theta-\alpha \cdot \operatorname{RMSProp}\left(\theta, g_\theta\right)$
\ENDWHILE
\end{algorithmic}
\end{algorithm}
```


```algorithm
% WGAN
\begin{algorithm}
\caption{Simple version of t-Distributed Stochastic Neighbor Embedding.}
\begin{algorithmic}
\STATE \textbf{Data}: data set $X=\left\{x_1, x_2, \ldots, x_n\right\}$,
\STATE cost function parameters: perplexity Perp,
\STATE optimization parameters: number of iterations $T$, learning rate $\eta$, momentum $\alpha(t)$.
\STATE \textbf{Result}: low-dimensional data representation $\mathcal{Y}^{(T)}=\left\{y_1, y_2, \ldots, y_n\right\}$.
\PROCEDURE{t-SNE}{}
\STATE compute pairwise afﬁnities $p_{j \mid i}$ with perplexity Perp (using Equation 1)
\STATE set $p_{i j}=\frac{p_{j \mid i}+p_{i \mid j}}{2 n}$
\STATE sample initial solution $\mathcal{Y}^{(0)}=\left\{y_1, y_2, \ldots, y_n\right\}$ from $\mathcal{N}\left(0,10^{-4} I\right)$
\FOR{$t=1$ to $T$}
\STATE compute low-dimensional afﬁnities  $q_{i j}$ (using Equation 4)
\STATE compute gradient $\frac{\delta C}{\delta y}$ (using Equation 5)
\STATE set $\mathcal{Y}^{(t)}=\mathcal{Y}^{(t-1)}+\eta \frac{\delta C}{\delta \mathcal{Y}}+\alpha(t)\left(\mathcal{Y}^{(t-1)}-\mathcal{Y}^{(t-2)}\right)$
\ENDFOR
\ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
```


```algorithm
% WGAN
\begin{algorithm}
\caption{A simple bandit algorithm}
\begin{algorithmic}
\FOR{$a = 1$ to $k$(\textbf{Initialize})}
\STATE  $Q(a) \leftarrow 0$
\STATE $N(a) \leftarrow 0$
\ENDFOR
\WHILE{(\textbf{Loop forever})}
\STATE $A \leftarrow \begin{cases}\operatorname{argmax}_a Q(a) & \text { with probability } 1-\varepsilon \quad \text { (breaking ties randomly) } \\ \text { a random action } & \text { with probability } \varepsilon\end{cases} $
\STATE $R \leftarrow \operatorname{bandit}(A)$
\STATE $N(A) \leftarrow N(A)+1$
\STATE $Q(A) \leftarrow Q(A)+\frac{1}{N(A)}[R-Q(A)]$
\ENDWHILE
\end{algorithmic}
\end{algorithm}
```

```algorithm
% WGAN
\begin{algorithm}
\caption{Policy Iteration (using iterative policy evaluation) for estimating }
\begin{algorithmic}
\STATE \textbf{1. Initialization}
\STATE $V(s) \in \mathbb{R}$ and  $\pi(s) \in \mathcal{A}(s)$ arbitrarily for all $s \in \mathcal{S}$; $V(terminal)\doteq 0$
\STATE \textbf{2. Policy Evaluation}
\WHILE{(\textbf{Loop})}
\STATE $\Delta \leftarrow 0$
\FOR{Loop for each $s \in \mathcal{S}$}
\STATE $v \leftarrow V(s)$
\STATE $V(s) \leftarrow \sum_{s^{\prime}, r} p\left(s^{\prime}, r \mid s, \pi(s)\right)\left[r+\gamma V\left(s^{\prime}\right)\right]$
\STATE $\Delta \leftarrow \max (\Delta,|v-V(s)|)$
\ENDFOR 
\STATE until $\Delta<\theta$ (a small positive number determining the accuracy of estimation)
\ENDWHILE 
\STATE \textbf{3. Policy Improvement}
\STATE $policy-stable \leftarrow true$
\FOR{each $s \in \mathcal{S}$}
\STATE $old-action \leftarrow \pi(s)$
\STATE $ \pi(s) \leftarrow \arg \max _a \sum_{s^{\prime}, r} p\left(s^{\prime}, r \mid s, a\right)\left[r+\gamma V\left(s^{\prime}\right)\right]$
\IF{$old-action \neq \pi(s)$}
\STATE $policy-stable \leftarrow false$
\ENDIF
\ENDFOR 
\IF{$policy-stable$}
\STATE stop and return $V \approx v_*$ and $\pi \approx \pi_*$
\ELSE 
\STATE got to 2
\ENDIF
\end{algorithmic}
\end{algorithm}
```

```algorithm
% WGAN
\begin{algorithm}
\caption{First-visit MC prediction, for estimating $V \approx v_\pi$}
\begin{algorithmic}
\STATE \textbf{Input:}  a policy $\pi$ to be evaluated
\STATE \textbf{Initialize:} 
\STATE $V(s) \in \mathbb{R}$, arbitrarily, for all $s \in \mathcal{S}$
\STATE Returns (s) $\leftarrow$ an empty list, for all $s \in \mathcal{S}$
\WHILE{(\textbf{Loop forever},for each episode)}
\STATE Generate an episode following $\pi: S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_{T-1}, A_{T-1}, R_T$
\STATE $G \leftarrow 0$
\FOR{(Loop for each step of episode, $t=T-1, T-2, \ldots, 0$ )}
\STATE  $G \leftarrow \gamma G+R_{t+1}$
\STATE Unless $S_t$ appears in $S_0, S_1, \ldots, S_{t-1}$:
\STATE Append $G$ to Returns $\left(S_t\right)$
\STATE $V\left(S_t\right) \leftarrow \text { average }\left(\operatorname{Returns}\left(S_t\right)\right)$
\ENDFOR 
\ENDWHILE
\end{algorithmic}
\end{algorithm}
```

```algorithm
% WGAN
\begin{algorithm}
\caption{Monte Carlo ES (Exploring Starts), for estimating $\pi \approx \pi_*$}
\begin{algorithmic}
\STATE \textbf{Initialize:} 
\STATE $ \pi(s) \in \mathcal{A}(s) \text { (arbitrarily), for all } s \in \mathcal{S}$
\STATE $ Q(s, a) \in \mathbb{R} \text { (arbitrarily), for all } s \in \mathcal{S}, a \in \mathcal{A}(s)$
\STATE $\text { Returns }(s, a) \leftarrow \text { empty list, for all } s \in \mathcal{S}, a \in \mathcal{A}(s)$
\WHILE{(\textbf{Loop forever},for each episode)}
\STATE Choose $S_0 \in \mathcal{S}, A_0 \in \mathcal{A}\left(S_0\right)$ randomly such that all pairs have probability $>0$ Generate an episode from $S_0, A_0$, following $\pi: S_0, A_0, R_1, \ldots, S_{T-1}, A_{T-1}, R_T$
\STATE $G \leftarrow 0$
\FOR{(Loop for each step of episode, $t=T-1, T-2, \ldots, 0$ )}
\STATE  $G \leftarrow \gamma G+R_{t+1}$
\STATE Unless the pair $S_t, A_t$ appears in $S_0, A_0, S_1, A_1 \ldots, S_{t-1}, A_{t-1}$:
\STATE Append $G$ to Returns $\left(S_t, A_t\right)$
\STATE $Q\left(S_t, A_t\right) \leftarrow \operatorname{average}\left(\operatorname{Returns}\left(S_t, A_t\right)\right)$
\STATE $\pi\left(S_t\right) \leftarrow \arg \max _a Q\left(S_t, a\right)$
\ENDFOR 
\ENDWHILE
\end{algorithmic}
\end{algorithm}
```

```algorithm
% WGAN
\begin{algorithm}
\caption{Tabular $\mathrm{TD}(0)$ for estimating $v_\pi$}
\begin{algorithmic}
\STATE \textbf{Input:} the policy $\pi$ to be evaluated
\STATE \textbf{Algorithm parameter:} step size $\alpha \in(0,1]$
\STATE \textbf{Initialize} $V(s)$, for all $s \in \mathcal{S}^{+}$, arbitrarily except that $V($ terminal $)=0$
\FOR{(Loop for each episode)}
\STATE \textbf{Initialize} $S$
\FOR{(Loop for each step of episode)}
\STATE $\quad A \leftarrow$ action given by $\pi$ for $S$
\STATE Take action $A$, observe $R, S^{\prime}$
\STATE $\quad V(S) \leftarrow V(S)+\alpha\left[R+\gamma V\left(S^{\prime}\right)-V(S)\right]$
\STATE  $S \leftarrow S^{\prime}$
\ENDFOR 
\ENDFOR 
\end{algorithmic}
\end{algorithm}
```


```algorithm
% WGAN
\begin{algorithm}
\caption{Sarsa (on-policy TD control) for estimating $Q \approx q_*$}
\begin{algorithmic}
\STATE \textbf{Algorithm parameter:} step size $\alpha \in(0,1]$, small $\varepsilon>0$
\STATE \textbf{Initialize} $Q(s, a)$, for all $s \in \mathcal{S}^{+}, a \in \mathcal{A}(s)$, arbitrarily except that $Q($ terminal,$\cdot)=0$
\FOR{(Loop for each episode)}
\STATE \textbf{Initialize} $S$
\STATE Choose $A$ from $S$ using policy derived from $Q$ (e.g., $\varepsilon$-greedy)
\FOR{(Loop for each step of episode)}
\STATE Take action $A$, observe $R, S^{\prime}$
\STATE Choose $A^{\prime}$ from $S^{\prime}$ using policy derived from $Q$ (e.g., $\varepsilon$-greedy)
\STATE $Q(S, A) \leftarrow Q(S, A)+\alpha\left[R+\gamma Q\left(S^{\prime}, A^{\prime}\right)-Q(S, A)\right]$
\STATE  $S \leftarrow S^{\prime} ; A \leftarrow A^{\prime} $;
\ENDFOR 
\STATE until $S$ is terminal
\ENDFOR 
\end{algorithmic}
\end{algorithm}
```

```algorithm
% WGAN
\begin{algorithm}
\caption{Q-learning (off-policy TD control) for estimating $\pi \approx \pi_*$}
\begin{algorithmic}
\STATE \textbf{Algorithm parameter:} step size $\alpha \in(0,1]$, small $\varepsilon>0$
\STATE \textbf{Initialize} $Q(s, a)$, for all $s \in \mathcal{S}^{+}, a \in \mathcal{A}(s)$, arbitrarily except that $Q($ terminal,$\cdot)=0$
\FOR{(Loop for each episode)}
\STATE \textbf{Initialize} $S$
\FOR{(Loop for each step of episode)}
\STATE Choose $A$ from $S$ using policy derived from $Q$ (e.g., $\varepsilon$-greedy)
\STATE Take action $A$, observe $R, S^{\prime}$
\STATE $Q(S, A) \leftarrow Q(S, A)+\alpha\left[R+\gamma \max _a Q\left(S^{\prime}, a\right)-Q(S, A)\right]$
\STATE $S \leftarrow S^{\prime}$
\ENDFOR 
\STATE until $S$ is terminal
\ENDFOR 
\end{algorithmic}
\end{algorithm}
```


```algorithm
% WGAN
\begin{algorithm}
\caption{$n$-step TD for estimating $V \approx v_\pi$}
\begin{algorithmic}
\STATE \textbf{Input:} a policy $\pi$
\STATE \textbf{Algorithm parameter:} step size $\alpha \in(0,1]$, a positive integer $n$
\STATE \textbf{Initialize} $V(s)$ arbitrarily, for all $s \in \mathcal{S}$
\STATE All store and access operations (for $S_t$ and $R_t$ ) can take their index $\bmod n+1$
\FOR{(Loop for each episode)}
\STATE \textbf{Initialize and store} $S_0 \neq$ terminal
\STATE $T \leftarrow \infty$
\FOR{(Loop for $t=0,1,2, \ldots$ )}
\IF{$t<T$}
\STATE Take an action according to $\pi\left(\cdot \mid S_t\right)$
\STATE Observe and store the next reward as $R_{t+1}$ and the next state as $S_{t+1}$
\IF{$S_{t+1}$ is terminal}
\STATE $T \leftarrow t+1$
\ENDIF
\ENDIF
\STATE  $\tau \leftarrow t-n+1 \quad$ ( $\tau$ is the time whose state's estimate is being updated)
\IF{$\tau \geq 0$}
\STATE $G \leftarrow \sum_{i=\tau+1}^{\min (\tau+n, T)} \gamma^{i-\tau-1} R_i$
\IF{$\tau+n<T$}
\STATE $G \leftarrow G+\gamma^n V\left(S_{\tau+n}\right)$
\ENDIF
\STATE $V\left(S_\tau\right) \leftarrow V\left(S_\tau\right)+\alpha\left[G-V\left(S_\tau\right)\right]$
\ENDIF
\ENDFOR 
\STATE Until $\tau=T-1$
\ENDFOR 
\end{algorithmic}
\end{algorithm}
```

```algorithm
% WGAN
\begin{algorithm}
\caption{Tabular Dyna-Q}
\begin{algorithmic}
\STATE \textbf{Initialize} $Q(s, a)$ and $\operatorname{Model}(s, a)$ for all $s \in \mathcal{S}$ and $a \in \mathcal{A}(s)$
\WHILE{(\textbf{Loop forever})}
\STATE (a) $S \leftarrow$ current (nonterminal) state
\STATE (b) $A \leftarrow \varepsilon$-greedy $(S, Q)$
\STATE (c) Take action $A$; observe resultant reward, $R$, and state, $S^{\prime}$
\STATE (d) $Q(S, A) \leftarrow Q(S, A)+\alpha\left[R+\gamma \max _a Q\left(S^{\prime}, a\right)-Q(S, A)\right]$
\STATE (e) $\operatorname{Model}(S, A) \leftarrow R, S^{\prime}$ (assuming deterministic environment)
\FOR{(Loop repeat $n$ times)}
\STATE $S \leftarrow$ random previously observed state
\STATE $A \leftarrow$ random action previously taken in $S$
\STATE $R, S^{\prime} \leftarrow \operatorname{Model}(S, A)$
\STATE $Q(S, A) \leftarrow Q(S, A)+\alpha\left[R+\gamma \max _a Q\left(S^{\prime}, a\right)-Q(S, A)\right]$
\ENDFOR 
\ENDWHILE
\end{algorithmic}
\end{algorithm}
```


```algorithm
% WGAN
\begin{algorithm}
\caption{Prioritized sweeping for a deterministic environment}
\begin{algorithmic}
\STATE \textbf{Initialize} $Q(s, a)$ and $\operatorname{Model}(s, a)$ for all $s \in \mathcal{S}$ and $a \in \mathcal{A}(s)$
\WHILE{(\textbf{Loop forever})}
\STATE (a) $S \leftarrow$ current (nonterminal) state
\STATE (b) $A \leftarrow \operatorname{policy}(S, Q)$
\STATE (c) Take action $A$; observe resultant reward, $R$, and state, $S^{\prime}$
\STATE (d) $\operatorname{Model}(S, A) \leftarrow R, S^{\prime}$
\STATE (e) $P \leftarrow\left|R+\gamma \max _a Q\left(S^{\prime}, a\right)-Q(S, A)\right|$.
\STATE (f) if $P>\theta$, then insert $S, A$ into $P Queue$  with priority $P$
\FOR{(Loop repeat $n$ times, while $P Queue$ is not empty)}
\STATE $S, A \leftarrow \operatorname{first}(P Queue )$
\STATE $R, S^{\prime} \leftarrow \operatorname{Model}(S, A)$
\STATE $R, S^{\prime} \leftarrow \operatorname{Model}(S, A)$
\STATE $Q(S, A) \leftarrow Q(S, A)+\alpha\left[R+\gamma \max _a Q\left(S^{\prime}, a\right)-Q(S, A)\right]$
\FOR{(Loop for all $\bar{S}, \bar{A}$ predicted to lead to $S$)}
\STATE $\bar{R} \leftarrow$ predicted reward for $\bar{S}, \bar{A}, S$
\STATE $P \leftarrow\left|\bar{R}+\gamma \max _a Q(S, a)-Q(\bar{S}, \bar{A})\right| .$
\STATE if $P>\theta$ then insert $\bar{S}, \bar{A}$ into $P Q u e u e$ with priority $P$
\ENDFOR 
\ENDFOR 
\ENDWHILE
\end{algorithmic}
\end{algorithm}
```

```algorithm
% WGAN
\begin{algorithm}
\caption{One iteration of the 1-dim. Weisfeiler-Lehman test of graph isomorphism}
\begin{algorithmic}
\STATE \textbf{1: Multiset-label determination} 
\STATE • For $i=0$, set $M_i(v):=l_0(v)=\ell(v) .$
\STATE • For $i>0$, assign a multiset-label $M_i(v)$ to each node $v$ in $G$ and $G^{\prime}$ which consists of the multiset $\left\{l_{i-1}(u) \mid u \in \mathcal{N}(v)\right\}$.
\STATE \textbf{2: Sorting each multiset} 
\STATE • Sort elements in $M_i(v)$ in ascending order and concatenate them into a string $s_i(v)$.
\STATE • Add $l_{i-1}(v)$ as a prefix to $s_i(v)$ and call the resulting string $s_i(v)$.
\STATE \textbf{3: Label compression} 
\STATE • Sort all of the strings $s_i(v)$ for all $v$ from $G$ and $G^{\prime}$ in ascending order.
\STATE • Map each string $s_i(v)$ to a new compressed label, using a function $f: \Sigma^* \rightarrow \Sigma$ such that $f\left(s_i(v)\right)=f\left(s_i(w)\right)$ if and only if $s_i(v)=s_i(w)$.
\STATE \textbf{4: Relabeling} 
\STATE • Set $l_i(v):=f\left(s_i(v)\right)$ for all nodes in $G$ and $G^{\prime}$.
\end{algorithmic}
\end{algorithm}
```
















<!--
| 角色      |      工作     |  修行者 |
|:----------|:-------------|:------|
| 作者 |  完成著作的主体，保证成书的完成； |<img src="./imgs/author/richie.jpg" height="auto" width="120" style="border-radius:50%" title="caDesign"><em>包瑞清(Richie Bao)-西建大（中）</em></a>|
| 章节作者 | 部分对成书有所贡献；  |<img src="./imgs/author/Alexis.jpg" height="auto" width="120" style="border-radius:50%" title="chapter author"><em>Alexis Arias-IIT(美)</em> |
| 手绘君 | 希望能有日本漫画学习书的风格，真的会爱上学习和研究，还有生活； |<img src="./imgs/author/lj.jpg" height="auto" width="120" style="border-radius:50%" title="caDesign"> <em>李静(Jing Li)</em></a>|
| 语言君 | 英语语言纠正，修正审核，不含翻译（翻译由作者初翻）； | <img src="./imgs/author/Migel.jpg" height="auto" width="120" style="border-radius:50%" title="Migel Santos"> <em>Migel Santos</em><img src="./imgs/author/xutao.jpg" height="auto" width="120" style="border-radius:50%" title="许韬"><em>许韬(Tao Xu)</em> |
| 贡献者(测试君) | [数字营造学社](https://digit-x.github.io/digit_x/#/)，和更大基数的社区伙伴们； |<a href="https://digit-x.github.io/digit_x/#/"><img src="./imgs/author/avatar.png" height="auto" width="120" style="border-radius:50%" title="digti-x"></a> <em>数字营造学社(digit-x):王育辉、刘航宇、张旭阳、柴金玉、戴礽祁、许保平、赵丽璠、张卜予</em> |
|技术审稿人 |负责代码审核，保证代码质量。 |<img src="./imgs/author/ChengHong.jfif" height="auto" width="120" style="border-radius:50%" title="Vacant Position"> <em>程宏 (Hong Cheng)-Kwangwoon University光云大学（韩）</em> |
-->


    * [PCS_X Python新特性更新](./markdown/pcs_quize.md)  


   * [+ 专项代码结构](./markdown/3_1_项目代码结构.md)

    * [TEMP](./markdown/temp.md)
    * [+ 城市空间数据计算](./markdown/城市空间数据计算.md) 