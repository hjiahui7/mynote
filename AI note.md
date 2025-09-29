---
number headings: auto, first-level 1, max 6, 1.1
---


# 1 RL 知识梳理


 1.1 Top Down
![[Pasted image 20250927014929.png]]

# 2 统一框架（总览）

> 目标：在马尔可夫决策过程（MDP）中最大化期望回报  
> \(J(\pi)=\mathbb{E}_{\tau\sim \pi}\!\left[\sum_{t=0}^{T-1}\gamma^{t}\,r_{t+1}\right]\)

- 两条主线：Value-based（c2a：先学值后出策）与 Policy-based（a2c：先建策再用值降方差）。
- 三条横轴：MC vs TD、On-policy vs Off-policy、稳定性（KL/熵/致命三角）。
- 核心循环：广义策略迭代（GPI）= 评估（学 V/Q/Adv）→ 改进（greedy/softmax 或 策略梯度）→ 重复。


 2.1 Value-based（**c2a：Critic → Actor**）
- **思路**：先学习值函数（Q 或 VVV），再用 **greedy / ϵ\epsilonϵ-greedy / softmax** 从值函数**导出**策略。
- **代表**：MC Control、SARSA、Expected SARSA、Q-learning、DQN（+ 目标网络/重放/Double-Q）。
- **优点**：离散动作简单高效。
- **局限**：连续动作 arg⁡max⁡aQ\arg\max_a Qargmaxa​Q 困难；策略不可直接正则化。

 2.2 Policy-based（**a2c：Actor → Critic**）

 2.3 Markov decision process








# 3 RL 基础 第二版本

https://www.bilibili.com/video/BV1rooaYVEk8/?spm_id_from=333.1387.homepage.video_card.click&vd_source=7edf748383cf2774ace9f08c7aed1476
## 3.1 Top down
![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=OTg4YmU4M2ZiMDIyZjZlNGRkOTUxMDlkOWRkOWIwOWRfNGNqa2h4dlhEQWZaeTZ4OHEwSnpYRnEwYnNsYXlUbzhfVG9rZW46THZNc2JmeFZ0b2tzeUV4SDdBWmxzQ3NPZ1FkXzE3NTkwNTY1NDA6MTc1OTA2MDE0MF9WNA)

## 3.2 Markov decision process
![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=MWI2ZTdlMDBlYTBjMDE2ODAzM2IwZWQ0NTVlMGQ3ZjFfVXUwOHhMSXE3NU5laDZGT0lPVUs0VTl5YjNBMUpUc3VfVG9rZW46RW55SGJFRVdOb2ZFdHN4R3ROQmxDR3YzZ1ZmXzE3NTkwNTY1NDA6MTc1OTA2MDE0MF9WNA)

## 3.3 State Value & Action Value
    

![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=Yjc3NmZjNGVjMjQyMGY4ODE4ZmQwM2JkYTBmNTdjMGRfMzBUQ2x3WmU0NU9MUWhmWE5Od0pHdXFHb0FuUURiQ0ZfVG9rZW46WE9DOWJEOTBIbzBMQlB4RlQ3emxhcmkxZ2ZjXzE3NTkwNTY1NDA6MTc1OTA2MDE0MF9WNA)

**Note:**

1. **价值函数 (Value) V**： 在状态 s 下，按照策略走下去的**总回报**。
	V(s)=R(s 起点往后算的总回报)
2. **动作价值函数 (Q-value) Q**： 在状态 s 下，先选动作 a，再按照策略走下去的**总回报**。
	Q(s,a)=R(s,a 起点往后算的总回报)
3. **优势函数 (Advantage) A**： 动作 a 相比于该状态平均水平的好坏。
	A(s,a)=Q(s,a)−V(s)

4. 直观关系
    
    1. V(s)可以看成“平均水平”。
    2. Q(s,a) 是“指定动作的分数”。
        
    3. A(s,a)就是“指定动作分数 − 平均水平”。
        
    4. 其中的每一步： $Q(s_t,a_t) = r_t + \gamma V(s_{t+1})$
        

![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=MWJiMzcyN2VjY2MxNzFiMmJjNjhhN2RhM2M1MTBjZjRfaG1oYWhwRFVzR0pneGxoc0prNFQ5VWVnRlhjV1VoZXBfVG9rZW46U1o2MWJ5bTdnbzdrM3R4czVWTWxGVmRkZ3lkXzE3NTkwNTY1NDA6MTc1OTA2MDE0MF9WNA)

4.  Value based：MC & TD
    

![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=OTgxYjE5MmY3MDE1ODQzNTAwOGUzNDA5ODAyZTE1NmNfQTRBR2pmWkE4SklxcXdEaFlIN2EyWHpQSk42ZVZXUndfVG9rZW46S1R2dWJQYlZLb0tXQ0R4NmNjYmx1OFdoZ3hnXzE3NTkwNTY1NDA6MTc1OTA2MDE0MF9WNA)

1.  Monte Carlo
    

2. # REINFORCE（policy based）
    

**REINFORCE 就是最原生的 Monte Carlo 方法**——它用整段回报（return）做无偏的梯度估计、没有 critic、也不做 bootstrapping。

1. 优化目标
    

在一个 episodic MDP 里，策略 πθ 的**轨迹**为 $\tau=(s_0,a_0,r_1,\ldots,s_{T-1},a_{T-1},r_T)$ 目标是最大化**期望总回报**（也可含折扣）：

$J_{\text{true}}(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}\big[R(\tau)\big], \quad R(\tau)=\sum_{t=0}^{T-1}\gamma^t r_{t+1}$

这里 pθ(τ) 是在当前策略与环境转移下生成该轨迹的**概率密度**：

$p_\theta(\tau)=\rho(s_0)\prod_{t=0}^{T-1}\pi_\theta(a_t|s_t)\,P(s_{t+1}|s_t,a_t)$

所以最终我们要让这个最大，所以对其求导即可：

$J_{\text{true}}(\theta) = \sum_{index=0}^{N}p_\theta(\tau_{index})\big[R(\tau_{index})\big]$

2. 求导过程：目标的梯度（log-derivative trick）
    

  

我们要 $\nabla_\theta J_{\text{true}}(\theta)$，用**似然比技巧**：

$\nabla_\theta J_{\text{true}} =\nabla_\theta \int p_\theta(\tau)R(\tau)\,d\tau =\int p_\theta(\tau)\,\nabla_\theta \log p_\theta(\tau)\,R(\tau)\,d\tau =\mathbb{E}_{\tau\sim p_\theta}\!\big[\nabla_\theta \log p_\theta(\tau)\,R(\tau)\big]$注意是 $\mathbb{E}_{\tau\sim p_\theta}$

而

$\log p_\theta(\tau)=\log\rho(s_0)+\sum_{t=0}^{T-1}\log\pi_\theta(a_t|s_t)+\sum_{t=0}^{T-1}\log P(s_{t+1}|s_t,a_t)$

对 θ 求导时只有策略项留下：

$\nabla_\theta \log p_\theta(\tau)=\sum_{t=0}^{T-1}\nabla_\theta \log\pi_\theta(a_t|s_t)$

代回去：

$\nabla_\theta J_{\text{true}} =\mathbb{E}_{\tau\sim p_\theta}\!\Big[\sum_{t=0}^{T-1}\nabla_\theta \log\pi_\theta(a_t|s_t)\,R(\tau)\Big]$

注意后续计算可能会忽略最外层的E，因为我们的数据都是通过P（这里不是状态转移函数，是上面的这个轨迹的**概率密度函数**）这个函数的概率分布来取样的，所以我们就可以忽略他了

这就是 **REINFORCE 梯度**的“轨迹级”形式。为了**降方差**，把整段 R(τ)换成“从 t 开始的 reward-to-go 就是步数越远γ越大”：

$G_t=\sum_{k=t}^{T-1}\gamma^{k-t} r_{k+1}, \quad \nabla_\theta J_{\text{true}} =\mathbb{E}_{\tau\sim p_\theta}\!\Big[\sum_{t=0}^{T-1}\nabla_\theta \log\pi_\theta(a_t|s_t)\,G_t\Big]$

于是我们可以把

$\boxed{\ J(\theta)\;\;\text{定义为其无偏 MC 估计对应的目标：}\;\; J(\theta)=\mathbb{E}\!\Big[\sum_t G_t\,\log\pi_\theta(a_t|s_t)\Big]\ }$

为什么我们的目标函数直接变成了这样子呢？

因为我们发现通过**某一个式子**利用**似然比技巧**求导后的式子为 $\quad \nabla_\theta J_{\text{true}} =\mathbb{E}_{\tau\sim p_\theta}\!\Big[\sum_{t=0}^{T-1}\nabla_\theta \log\pi_\theta(a_t|s_t)\,G_t\Big]$，那么这个**某一个式子为** $J(\theta)=\mathbb{E}\!\Big[\sum_t G_t\,\log\pi_\theta(a_t|s_t)\Big]$，简单来说就是求了半天发现这个J(θ)可以由这个简单形式表达，并且他和最初的他是等价的

我们此时此刻求出了导数后，就可以用优化函数更新参数了

  

3. 实际怎么做（REINFORCE 一轮）
    
    1. **采样** N 条轨迹 {τi}（按当前策略）
        
    2. **回放**：对每条 τi 倒序算 $G_t^i$
        
    3. **（可选）基线**：用 $G_t^i-b(s_t^i)$ 降方差
        
    4. **估计梯度**：
        
            $\widehat{\nabla_\theta J} =\frac{1}{N}\sum_{i=1}^N\sum_{t}(G_t^i-b(s_t^i))\,\nabla_\theta\log\pi_\theta(a_t^i|s_t^i)$
        
    1. **更新参数**： $\theta\leftarrow\theta+\alpha\,\widehat{\nabla_\theta J}$
        

全程没有显式出现 pθ(τ)的数值计算。

  

  

2.  降方差：Baseline → Advantage → GAE
    

3.  Advantage 的由来
    

REINFORCE 无偏但方差大，学习抖。说白了就是G一般情况下可能是一个非常大的值，我们希望降低梯度的幅度，所以需要对他进行normalization，所以才有了baseline这个东西。也就是Advantage= Gt−V(st)。我们用V(st)来估计未来的期望奖励是多少，也就是平均值，减掉了之后就是X-E[X]，看到没有，非常像是normalization了一下。这里我们才第一次引入了A

**REINFORCE（纯 MC）**：即便有 baseline（哪怕 Gt−V(st)，只要优势里的 Gt是**整段回报**，你仍然**需要等到 episode 结束**才能算完 Gt 再更新（可以逐步累积，但目标依赖未来完整回报）。

**Actor–Critic（TD）**：**一旦把 Gt 换成 TD 目标（例如用 δt 或 n-step/GAE 近似），你就进入了 actor–critic 范式，能够k 步一更，甚至步步更新。**。关键是把优势用**TD 残差**近似，完全不必等 episode 结束。

相当于把Gt换成 $r_{t+1} \;+\; \gamma\,V_\phi(s_{t+1})$

2.  (GAE)Generalized Advantage Estimation
    

![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=MGNhZWEwMThkNzZmNWRiOWQ5ZGZhNDlkMTY5YTQ0ZTJfU0Z2ZGNJYlVTYUVJV2RaZkZ5bUtZbHdkM3dZeHRnN2tfVG9rZW46RUh3S2JUanQzb1Z5cXR4bGNPZWxBSkE5Z3ZnXzE3NTkwNTY1NDA6MTc1OTA2MDE0MF9WNA)

实际值-期望值=At advantage，优势，如果>0，说明在当前s的情况下，选择action是有利的，如果<0，则是由penalty

3.  A，V，Q的关系
    

本质就是A是由G - baseline得出的，其中G,baseline可以是r+Q(st+1,at+1), Q(st,at)或者r+V(st+1), V(s)都行.

![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=MThiMmUyOTMwN2M1NTBiMmRlMWIwYzY3OWUzY2Q5ZjRfUXl1aU9FcUFaMU8zRU43R1lENDRxa0hOeWFYaDBoZmRfVG9rZW46WTZ3N2JMYXo2b0dLaEt4UmUya2x4NUdSZ1pmXzE3NTkwNTY1NDA6MTc1OTA2MDE0MF9WNA)

4.  如何理解方差和偏差
    

- **Monte Carlo**：
    
    - Q 是通过完整轨迹的 return U 来估计的。
        
    - 每条轨迹可能很不同 → **方差高**。
        
    - 但期望值等于真实值 → **无偏**。
        
- **TD**：
    
    - Q是通过 bootstrapping估计每一步的Q(st,at)，它不是“真实的未来回报”，而是**模型自己对未来的估计**。
        
        - rt+γQ(st+1,at+1)
            
    - 由于用的是自己的估计 Q，所以期望值和真值之间可能有偏差。
        
    - 但因为只依赖一步的采样，随机性小 → **方差低**。
        

---

1. 举个例子
    

假设真实 Q(s,a)=5。

- **Monte Carlo**：跑 3 条轨迹，得到回报：2,10,32
    
    - 平均值 = 5（无偏差）
        
    - 方差很大（数值波动大）。
        
- **TD**：一步预测：4.8,5.1,5.2
    
    - 平均值 ≈ 5.03，有一点点偏差。
        
    - 但方差很小（结果都接近 5）
        

3.  Temporal Difference
    

note：

1. TD里面应该是Q(st+1,at+1)。
    
2. Qt和Qt+1是如何演变的
    
    ![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=ZWRhMTY3MTk5Y2FlMWY2NjBlZjQwMzAzZWFlYWJmYzdfRkZLRnJNQXpaUEphNXdrQmN1UjJYdDRhR3dKN2w4aE1fVG9rZW46T29YcGI5bmJJb3NFTkR4a0lFS2wzRFBjZ3hnXzE3NTkwNTY1NDA6MTc1OTA2MDE0MF9WNA)
    
3. 更新Q（st，at），我们可以看到
    
    1. 小汽车一开始的Q(st,at)=30
        
    2. 开了10分钟（r(st,at)）=10，Q(st+1,at+1) = 18
        
    3. 我们希望 $rt+γQπ(st+1,at+1)−Qπ(st,at) = 0$ ，所以我们求导，然后得到梯度。然后基于优化函数（把他当作adam，sgd等看待就行） $Qπ(st,at)←Qπ(st,at)+α[rt+γQπ(st+1,at+1)−Qπ(st,at)]$, 我们的更新公式为30 + α（10 + 18 - 30），然后得到的值来更新table。
        
    4. 如果是网络则用损失函数更新，得到梯度的方式为最大化 $L(\theta) = \Big( r_t + \gamma Q(s_{t+1},a_{t+1};\theta) - Q(s_t,a_t;\theta) \Big)^2$
        

![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=ZjJjNmQ2NWRjNzg2OTU4ODQ5YjI0NzE1ZDk4MDQ5M2ZfNmw4TEx4RzJUTDJadGEza0xGMUM0RWRHNGR6SE02cWxfVG9rZW46TTAxY2JUZlhmb05iNU54ZXlHQmx5UUNLZ1ZoXzE3NTkwNTY1NDA6MTc1OTA2MDE0MF9WNA)

```C++
极简伪代码（每步学习）
init θ, φ
loop for t = 0,1,2,...:
    observe s_t
    sample a_t ~ π_θ(·|s_t)
    execute a_t → get r_{t+1}, s_{t+1}

    δ_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)

    # critic update (one step)
    φ ← φ + α_V * δ_t * ∇_φ V_φ(s_t)

    # actor update (one step)
    θ ← θ + α_π * δ_t * ∇_θ log π_θ(a_t | s_t)
    # (+ optional entropy bonus on θ)
```

4.  SARSA and Q learning（TD）
    

TD的算法有SARSA and Q learning

![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=NzUwMGE5M2EyZTZmOTM4NTEzNGEyOTRmNzJjMDkzYzNfd2MyS0FKNE1JQUhrelBXY3BqY0dMNmtKZ29iV1laYldfVG9rZW46U0E3RGJ3VmU2b2E2TWJ4ZTZlR2xycjlrZ2htXzE3NTkwNTY1NDA6MTc1OTA2MDE0MF9WNA)

1. Sarsa 为一个greedy算法，给定s1，然后找最大价值的a，并返还Q（价值，比如图中的23）和a，但是为了防止次次都算最大，我们以概率ε选择其他的action
    
    1. 如果 ε=0，一定选 a1 （greedy）
        
    2. 果 ε=0.1，那么：
        
        1. 90% 概率选 a1，
            
        2. 10% 概率在 {a1,a2,a3}里随机选一个（可能选到 a2 或 a3）。
            
2. 不管是DQN还是Table的形式，本质都是查表，只不过网络是一次性输出该状态下st **所有可能动作a1,a2,a3, 最终得到所有的**的 Q 值向量，也就是discounted return，然后我们greedy的获得a，然后再通过环境获得r
    
3. **Behavior policy** = 你实际在环境里怎么选动作的方式。
    

**Target policy** = 你更新时假设未来会怎么选动作的方式。

4. DQN 工作流程：
    
      DQN为Q函数的神经网络版本，SARSA，Qlearning都是用table来做Q函数
    
    2. 输入状态
        
        1. 神经网络输入当前环境的状态 st（比如一张游戏画面）。
            
    3. 输出所有动作的 Q 值
        
        1. 网络一次性输出该状态下 **所有可能动作** 的 Q 值向量：
            
        2. $[Q(s_t,a_1), Q(s_t,a_2), \dots, Q(s_t,a_n)]$
            
        
    📌 注意：不用一个一个传入 action，而是一次前向传播就得到所有动作的 Q 值。
        
    4. 动作选择 (ε-greedy)
        
        1. 以概率 1−ε：选 Q 值最大的动作（greedy）。
            
        2. 以概率 ε：随机选一个动作（探索）。
            
    5. 执行动作，得到奖励和下一个状态
        
        1. 执行动作 a，环境返回奖励 rt 和新状态 st+1。
            
    6. 存储经验 (Replay Buffer)
        
        1. 把转移样本 (st,at,rt,st+1,done)存入经验回放池。
            
    7. 采样训练
        
        1. 从回放池里随机采样一批数据，用来训练神经网络。
            
        2. 目标值 (TD target)：
            
            - $y_t = r_t + \gamma \max_{a'} Q_{\theta^-}(s_{t+1}, a')$
                
            - （这里的 $Q_{\theta^-}$是 target network）
                
        3. 损失函数：
            
            - $L(\theta) = \frac{1}{2}\big(y_t - Q_\theta(s_t,a_t)\big)^2$or $L(\theta) = \Big( r_t + \gamma Q(s_{t+1},a_{t+1};\theta) - Q(s_t,a_t;\theta) \Big)^2$
                
    8. 更新参数
        
        1. 用梯度下降更新神经网络参数 θ。
            

4.  On policy and Off policy
    

如果behavior和target policy是一样的方法，比如SARSA，那么就是on policy，如果不一样那么就是off policy

  

  

1. On policy的本质就是Π是不是新的Π，会不会产生新的不同分布的a
    
    ![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=YWE3ODFkNDg1ZDkwMDYwMDZiOWI1ZDIyMjQyYmE4YmVfSmJTZFJ6cVZOVjlpQ0U4T0pjRktQY0d1dnJISzh5UEpfVG9rZW46VHdpQ2JKQkVBb1pTd3p4TWVKN2xuTlRIZ0toXzE3NTkwNTY1NDA6MTc1OTA2MDE0MF9WNA)
    
    2. 以SARSA为例：
        
        1. 我们首先采样N个st,at,rt,st+1
            
        2. 以上面的公式为例子，其中在更新Q的过程中，右边的Q(st,at)和Q(st+1,at+1)的参数都是一样的，所以action a的分布是一样的，所以不需要重要性采样，并且用的就是老数据更新的。
            
        3. 这里一直冻结Qθ直到所有N更新完，这里可以是一次性直接全部更新完，或者mini batch都行
            
        4. 最终更新Q
            
        
            总结：和critic不一样的是，这里是参数冻结的情况下，更新完N个data point
        
    3. 以Q learning 为例
        
        ![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=NmEyNGE4NmY1ZTY5NzdkMTE4YWI4YWM0NTllYzlkMDRfdndTbVJmUFlqU055bTVlM0JQa0o0eWdGWXVpVm4ydElfVG9rZW46RUhBMmJKajVCbzUwUmJ4UWxkeWxOeldOZ3BiXzE3NTkwNTY1NDA6MTc1OTA2MDE0MF9WNA)
        
        ![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=ODhkNWE1NmJhZjk0MmRlOGJmMDliNzEwMDE2MWQxZWRfa1pNRVRUaXNPNWRFUVRqcEprMzlWQ2hUQmdPYlJ2RnBfVG9rZW46WkphbWJxMGNOb2hyRlR4NzBNb2xNYzdqZ3BjXzE3NTkwNTY1NDA6MTc1OTA2MDE0MF9WNA)
        
        3. 我们首先用behavior采样N个st,at,rt,st+1, target behavior也可以去采样一些点。这里的policy Π是不一样的，所以可以分开采样。相当于behavior用了一个网络（random），或者table来进行采样然后获得数据，target也是一样，只不过他们用的网络或者table不一样。虽然Q learning用的还是之前的Q的table或者网络，但是最终的决策过程Π是greedy 不是random。比如，Q(st,at)和Q(st+1,at+1)的决策方法是不一样的，因为一个用max （greedy）一个random，策略不同，所以会直到后续的采样数据是不一样的，比如数据a，st+1, at+1...sT分布是不一样的，所以我们说两者behavior数据分布不同，那么就是off policy
            
    4. PPO为例（onpolicy）
		PPO是一个看起来很像off-policy（因为他是复制了老的，然后更新，过程中会出现两个Π）的on-policy算法（**PPO 要“新采样→在这批上训练→丢弃”，不能像 off-policy 那样长期吃旧/异策略数据，这才是它 on-policy 的本质**）。“丢不丢弃数据”只是**现象**而不是定义：**on-policy**要求用与目标策略（当前/刚冻结的策略）**一致或近邻**分布的数据训练（所以旧数据常被丢弃以避免分布漂移）；**off-policy**则能在**行为≠目标**时依然有效学习（靠最优/软最优备份如 `max`，或 IS/截断-IS 等纠偏），因此可以长期复用回放数据。
        

  

2.  Policy based：Policy Gradient
    

![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=OWViYmVhNDIxNTM3NzAxNmQyYThlMGYyYzJiYTlkNzZfWW1pMDI1SzBDWnliT2tVSmpDT09hd0NKYWk4WTZyOWVfVG9rZW46SlhDWmJZSjY5b1dzeFV4c2tIemxjZ2RzZ1RkXzE3NTkwNTY1NDA6MTc1OTA2MDE0MF9WNA)

1. 我们希望当前s出现动作a的概率增高，然后Q(s,a)的价值最大
2. 目标函数 J(θ)（对所有轨迹求和）
	- 设一条轨迹 $τ=(s0,a0,r0,…,sT)$，它的累计回报$R(\tau)=\sum_{t=0}^{T-1}\gamma^t r_t$ 
	- 轨迹在策略 πθ 下出现的概率
	- $p_\theta(\tau)=\rho(s_0)\prod_{t=0}^{T-1}\pi_\theta(a_t|s_t)\,P(s_{t+1}|s_t,a_t)$
	- （初始分布 ρ 和环境转移 P 与 θ 无关）。
	- 于是 $J(\theta)=\sum_{\tau} p_\theta(\tau)\,R(\tau)$

3.  Reinforce and ACtor Critic
    

![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=ODQ5Njc5ZGM5ODFiOGY4YjAzMWYwNDM1MjM0MWY1NWVfUGxjUTl3Mlk1RUV4RVpPbzFLNDZhVUJYOHkzZDg2aHpfVG9rZW46UDJWbmJUOFFCb1RvTWd4OVNjNmxuQUJYZ3hnXzE3NTkwNTY1NDA6MTc1OTA2MDE0MF9WNA)

1. 左上角如何求解Q是个问题，我们可以使用
    
    1. Actor critic 得到一个Q的网络，或者table
        
    2. reinforce，就是直接把所有的r加起来，但是它做不到中途训练，**完整一条轨迹 episode 结束**，才能算每个时刻的回报 Gt
        
    3. baseline就是李宏毅actor critic的方式
        

2.  The problem of policy Gradient
    

![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=YTNiZGYyODlkZjg3NDZlMzdiM2ZkMTVlOWMzNzA0NmJfM1p0N2lqeE1ZbVlwSXJadmJrcUlteXlXSTBWQzhHczdfVG9rZW46VFRHc2JsUEc3bzlpMjV4YjAwMWw5SW1MZ0xmXzE3NTkwNTY1NDA6MTc1OTA2MDE0MF9WNA)

我们不希望参数一次性更新的太大，所以我们希望参数更新的值小于一个阈值

1.  Important sampling
    

我们有p(x),f(x), 我们想要取从p(x)采样很困难的话，我们可以引入一个q(x)然后，然后对x求积分。

![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=OGIwN2JhOTNmYzBmYjdiY2U3NGUyM2IxM2JmMDNlMzdfNDhBa2ppc01kSE1jcldVdlFsbFJxQk05MWRUM1g0eUVfVG9rZW46Szd1YmIxWGVqb0pZb2N4VjdGZ2xuRjh6Z0pjXzE3NTkwNTY1NDA6MTc1OTA2MDE0MF9WNA)

意思就是多次采样之后，得到的平均值就是p(x)的概率密度函数的情况下f(x)的期望/平均值。

现在我们思考如何才能应用到RL中。

1. 我们获取到了一堆st,at,rt,st+1 * N，并最终计算A_old
    
2. 我们开始更新policy Π_old 得到Π_new，那么Π_new就是新的分布，我们又不想重新计算A_new，如何才能继续使用A_old呢？
    
3. 我们把A_old当作f(x)，p(x)当作Π_new，我们从老Π_old 采样得到的数据是不是就可以用了？
    
4. 所以最终我们会使用 Π_new/Π_old 的形式来表示p(x)/q(x)
    

  

思考：

1. # 为什么Q learning不用这个？（说实话没搞懂这个）https://zhuanlan.zhihu.com/p/346433931
    
    1. 直觉上想着，我通过不同的policy采样，那么我的Q值也是不一样的呀，这样不会影响其在更新时的分布吗？ $Q_t - (r + Q_{t+1})$比如Vt+1很大，Vt很小，我们让他们分布一样不好吗？答案是同分布”没意义，甚至有害。**Bellman 不动点会被改写**：如果你对 Qt 或 yt 施加与样本相关的非线性“归一化”，就相当于改了目标函数，可能不再收敛到 Q
        
    
      $\mathbb{E}_{(s,a)\sim d_\mu}\big[\big(y(s,a)-Q_\theta(s,a)\big)^2\big], \quad y=r+\gamma \max_{a'}Q_{\bar\theta}(s',a')$
    

2.  Trust region policy optimization(细节还没有研究)
    

![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=Nzk1YmJkN2RiZjYzM2U3MGYxNmNiZjI3MTE1NzAzZjJfSVdYaUo4MzU5QTU5RFEwOENUeVdoNU1JUjhiOW5USjJfVG9rZW46WjF1eWJpdThub2MyRHB4TWdCd2xPYzd2Z0lkXzE3NTkwNTY1NDA6MTc1OTA2MDE0MF9WNA)

Delve in 研究

假设我们采样了 N 条轨迹，每条轨迹长度 Ti。那么期望就可以近似为：

$J(\theta') - J(\theta) \;\approx\; \frac{1}{N} \sum_{i=1}^{N} \;\sum_{t=0}^{T_i-1} \Bigg[ \frac{\pi_{\theta'}(a_t^{(i)} \mid s_t^{(i)})}{\pi_\theta(a_t^{(i)} \mid s_t^{(i)})} \;\gamma^t \; A_{\pi_\theta}(s_t^{(i)},a_t^{(i)}) \Bigg]$

注意：

1. 为什么使用了重要性采样之后，式子感觉少了一个Πθ？也就是老的策略
    

![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=OTkwOTkyOGQyMGY2NGMxZGM1MGRkY2Y1NjEwZDMwMGNfT0o4d3FpVnpJdU9UQkhtR3JyaE9VeFZPT3JYZ0F2NUhfVG9rZW46RjBSeWJLOTd2bzhRUG94SGFsNGxCNHFWZ1plXzE3NTkwNTY1NDA6MTc1OTA2MDE0MF9WNA)

因为这个求和是以数据维度求和，数据为一堆st,at,r,st+1，并且这些数据已经是Πθ的概率的分布了，所以不需要乘Πθ

![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=ZTkzMDMzZDE5MjcwY2JmZTBlYjMyZjg4OWFkYzUyY2NfckZ4MDBRSUNXSzhHUlY3YUFCazd1U01uem9Jd2pyRVdfVG9rZW46QlpkQmJJM1pjb3RkNW94QTRvYWwzcGNKZzJnXzE3NTkwNTY1NDA6MTc1OTA2MDE0MF9WNA)

  A这个部分为直接把期望写成公式的形式，下面为等价转换的形式，也就是我们公式中使用给的形式。

7.  PPO
    

  

8.  公式
    

![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=MTEwMTFlNTI2NTczNTc2Njg5MTFhMjc2MmU0OGYzM2VfNWdMTkFyMUdWeUZROTM5cEVWTU9LZEFndjJvdURZMWhfVG9rZW46UWdJUGI5VXRmb3ZwbTR4TXducGxMV1A5Z0tiXzE3NTkwNTY1NDA6MTc1OTA2MDE0MF9WNA)

1. 非期望形式
    
    1. PPO-penalty
        
    
      $\begin{equation} L^{\text{PPO-penalty}}(\theta') \approx \frac{1}{N}\sum_{i=1}^N \sum_{t=0}^{T_i-1} \left[ \frac{\pi_{\theta'}(a_t^{(i)} \mid s_t^{(i)})}{\pi_{\theta}(a_t^{(i)} \mid s_t^{(i)})} \, \hat{A}_t^{(i)} - \beta \, D_{\text{KL}}\!\Big(\pi_{\theta}(\cdot \mid s_t^{(i)}) \,\|\, \pi_{\theta'}(\cdot \mid s_t^{(i)})\Big) \right] \end{equation}$
    
    2. PPO-clip
        
            $\begin{equation} L^{\text{PPO-clip}}(\theta') \approx \frac{1}{N}\sum_{i=1}^N \sum_{t=0}^{T_i-1} \min\!\Bigg( \frac{\pi_{\theta'}(a_t^{(i)} \mid s_t^{(i)})}{\pi_{\theta}(a_t^{(i)} \mid s_t^{(i)})} \, \hat{A}_t^{(i)}, \; \text{clip}\!\left( \frac{\pi_{\theta'}(a_t^{(i)} \mid s_t^{(i)})}{\pi_{\theta}(a_t^{(i)} \mid s_t^{(i)})}, \, 1-\epsilon, \, 1+\epsilon \right)\hat{A}_t^{(i)} \Bigg) \end{equation}$
        
    3. 我们目标就是让这俩L最大
        
2. 这里的A_hat就是GAE
    
3. penalty
    
    1. 如果kl小于阈值，那么我们希望多更新，所以减少惩罚
        
    2. 如果kl>阈值，那么我们希望少更新，所以增加惩罚
        
    
      PPO-penalty 动态调节 β 的目的 = 控制训练的平稳性，减少震荡。强化学习里非常重要的 **稳定性优先** 原则：比起学得快，更怕学坏。
    
4. Clip
    
    1. 如果超出了一个范围就直接截断，也是为了稳定性
        

5.  训练
    

![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=Zjc1NWQ3ZWQ1MDBiZTFkMWY2N2U3ZTAwYjU5YmU5YjVfZWM2ZnFrV3p1S0xWMHZDcm1Wekk5VjVHaHNUMmNYVFNfVG9rZW46REJ5UmJPR050b052V1d4enBqTGxOOWxlZ3ZoXzE3NTkwNTY1NDA6MTc1OTA2MDE0MF9WNA)

1. 数据准备，policy网络，value网络
    
    ![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=NmRkMGU3YThmZDJlM2QxZjM2ZjQ0NGI3OWQ2MWY4M2RfajRDRHpwVmNkeGZ6N3NLRkJ3OVhiZDc5V1FVWUFUdlVfVG9rZW46QWFPMGJ1Tno2b1BxTTR4cllnQ2xtQ095Z3FnXzE3NTkwNTY1NDA6MTc1OTA2MDE0MF9WNA)![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=NzY5NjdhYjdmZmM2ZmU4OTUwOWZiOTEzYTBiY2QxYzBfSWJ6UHpIek4yQmdHNXc4WHdLTE5zbFlSMllPdGdaZGFfVG9rZW46VHVXUGJ0em1hb21SNTh4cVNnUWxpU2trZ2ZnXzE3NTkwNTY1NDA6MTc1OTA2MDE0MF9WNA)![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=ZjU5OTFiMDg3Y2ViYjgzYjVjOGNiZmI4MDc2YjVhZWRfbXhTRUdZUFJ2OXVpaFNlVWJ0QVU5U3dkREw5MHNoU1pfVG9rZW46WmlwUGJjdlNYb1pOQVN4OG9ScWxaVlBEZzhiXzE3NTkwNTY1NDA6MTc1OTA2MDE0MF9WNA)
    
    3. 收集数据（使用旧策略）在当前策略参数 θ下，跑环境，收集一批轨迹： (st,at,rt,st+1)。
        
    4. 用这些数据用Vθ估计 **优势函数** $\hat{A}_t$（比如用 GAE）。
        
        ![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=OWZlNzQ5ODk1NzIzNDk5MjYwZWJjNjU5N2NjOWJmN2Nfc2FOSk9SVzNNbVhSUGdMWXhDNmlvNkg5T2pxd3NkQmdfVG9rZW46TWFJdWJDN0E5b2NINml4dGR4TmxKVlNrZzBlXzE3NTkwNTY1NDA6MTc1OTA2MDE0MF9WNA)
        
        ![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=YTU1NTJlM2JhNmVhYzE0MTZlNjZiMDJhMWI2OTMxMjFfV2tKclkyS2VMU1VEblFCSUN0TTF1WHlnbm1FOU10bDhfVG9rZW46QXVvTWJtRUZKb3RXR3N4dmxSUWxXbmhVZ3pnXzE3NTkwNTY1NDA6MTc1OTA2MDE0MF9WNA)
        
    5. 这里的策略就是 **旧策略** πθ。这一步的作用：生成样本，固定下来，接下来训练时不再更新它。
        
2. 计算比率（新/老策略）
    
      优化时，我们引入一个新的参数 θ′（训练时会逐渐更新）。
    
       对每个样本计算：
    
      $r_t(\theta') \;=\; \frac{\pi_{\theta'}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)}$
    
    4. 分子：**新策略** πθ′ 对样本的概率（随着训练更新）。
        
    5. 分母：**旧策略** πθ 对样本的概率（固定不变）。
        
    6. 如果 >1，说明新策略更倾向于这个动作；
        
    7. 如果 <1，说明新策略更不倾向于这个动作。
        
    8. 这样做的原因：虽然样本是用旧策略生成的，但我们希望评估如果换成新策略，它的表现如何。这个比率就是 **重要性采样 (importance sampling)**。
        
3. 构造 PPO-clip 的目标
    
    1. PPO-penalty or PPO-clip （最大化价值）
        
            $\begin{equation} L^{\text{PPO-penalty}}(\theta') \approx \frac{1}{N}\sum_{i=1}^N \sum_{t=0}^{T_i-1} \left[ \frac{\pi_{\theta'}(a_t^{(i)} \mid s_t^{(i)})}{\pi_{\theta}(a_t^{(i)} \mid s_t^{(i)})} \, \hat{A}_t^{(i)} - \beta \, D_{\text{KL}}\!\Big(\pi_{\theta}(\cdot \mid s_t^{(i)}) \,\|\, \pi_{\theta'}(\cdot \mid s_t^{(i)})\Big) \right] \end{equation}$$\begin{equation} L^{\text{PPO-clip}}(\theta') \approx \frac{1}{N}\sum_{i=1}^N \sum_{t=0}^{T_i-1} \min\!\Bigg( \frac{\pi_{\theta'}(a_t^{(i)} \mid s_t^{(i)})}{\pi_{\theta}(a_t^{(i)} \mid s_t^{(i)})} \, \hat{A}_t^{(i)}, \; \text{clip}\!\left( \frac{\pi_{\theta'}(a_t^{(i)} \mid s_t^{(i)})}{\pi_{\theta}(a_t^{(i)} \mid s_t^{(i)})}, \, 1-\epsilon, \, 1+\epsilon \right)\hat{A}_t^{(i)} \Bigg) \end{equation}$
        
            我们目标就是让这俩L最大
        
    2. Value 目标（最小化误差）
        
            $y_t =\hat R_t = \hat A_t + V_{\phi_{\text{old}}}(s_t) \;\;\approx\; Q(s_t,a_t)$
        
            然后让 Vθ(st) 去回归这个目标：
        
            $L_{(\theta)} = \frac{1}{N}\sum_t \big(V_\theta(s_t) - y_t\big)^2$
        
            note: 和Value based：MC & TD中更新Q的方式是一样的
        
    3. −c2 Entropy(πθ)熵正则项
        
            $H(\pi_\theta(\cdot|s_t)) = -\sum_a \pi_\theta(a|s_t) \,\log \pi_\theta(a|s_t)$
        
        2. 策略的熵定义为：
            
        
            $H(\pi_\theta(\cdot|s)) = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)$
        
        3. 熵越大，策略越随机；熵越小，策略越确定（贪心）。
            
        4. 我们希望在训练初期**鼓励探索**，让策略不要太快变得确定，所以要**最大化熵**。
            
        5. 因为整体是最小化问题，所以写成 −c2 Entropy。
            
    4. 实际
        
        1. 在一个 epoch 的 mini-batch 里，loss 一般写成：
            
            1. 期望
                
            
                  $\begin{aligned} L(\theta,\phi) &= \mathbb{E}_t \Bigg[ \underbrace{-\min\Bigg( r_t(\theta)\,\hat{A}_t, \; \text{clip}\!\big(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon\big)\,\hat{A}_t \Bigg)}_{\text{Policy Loss (Actor)}} \\ &\quad\quad + \; \underbrace{c_1 \big( V_\phi(s_t) - \hat R_t \big)^2}_{\text{Value Loss (Critic)}} \; - \; \underbrace{c_2 \, H\!\big(\pi_\theta(\cdot|s_t)\big)}_{\text{Entropy Bonus}} \Bigg] \end{aligned}$
            
            2. Batch 形式
                
                        设一个训练批次包含若干条序列，用索引集合 M={(i,t)}表示本次用于优化的所有样本（第 i 条轨迹在时刻 t 的一条样本）。PPO 的**要最小化**的总损失：
                
                        $\boxed{ L(\theta,\phi) = \frac{1}{|\mathcal{M}|}\sum_{(i,t)\in\mathcal{M}} \Big[ -\min\!\big(\, r_{i,t}(\theta)\,\hat A_{i,t},\ \text{clip}(r_{i,t}(\theta),\,1-\epsilon,\,1+\epsilon)\,\hat A_{i,t}\big) \;+\; c_1\,(V_\phi(s_{i,t})-\hat R_{i,t})^2 \;-\; c_2\,H(\pi_\theta(\cdot|s_{i,t})) \Big] }$
                
        2. 各部分定义
            
            1. 策略比率
                
            
                  $r_{i,t}(\theta) = \frac{\pi_\theta(a_{i,t}|s_{i,t})}{\pi_{\text{old}}(a_{i,t}|s_{i,t})}$
            
            1. 优势 $\hat A_{i,t}$（GAE 的展开/递推，均为有限和）
                
                        先定义一步 TD 残差（带终止遮罩）：
                
                        $ \delta_{i,t} \;=\; r_{i,t} + \gamma(1-\text{done}_{i,t+1})\,V_\phi(s_{i,t+1}) \;-\; V_\phi(s_{i,t})$
                
            
                  向后递推计算 GAE：
            
                    $\hat A_{i,T_i-1} \;=\; \delta_{i,T_i-1}, \quad \hat A_{i,t} \;=\; \delta_{i,t} + \gamma\lambda(1-\text{done}_{i,t+1})\,\hat A_{i,t+1}$
            
                  或写成有限项显式求和：
            
                  $\hat A_{i,t} \;=\; \sum_{l=0}^{T_i-t-1} (\gamma\lambda)^l \left[\, r_{i,t+l} + \gamma(1-\text{done}_{i,t+l+1})\,V_\phi(s_{i,t+l+1}) - V_\phi(s_{i,t+l}) \right]$
            
            2. 回报估计
                
            
                  $\hat{R}_{i,t} = \hat{A}_{i,t} + V_\phi(s_{i,t})$
            
            3. 熵正则项
                
            
                  $H(\pi_\theta(\cdot|s_{i,t})) = -\sum_a \pi_\theta(a|s_{i,t}) \,\log \pi_\theta(a|s_{i,t})$
            
            4. 超参数
                
                1. ϵ：clip 范围（如 0.1 或 0.2）。
                    
                2. c1：value loss 的权重。
                    
                3. c2：熵项的权重。
                    
                4. γ：折扣因子。
                    
                5. λ：GAE 衰减参数。
                    
4. 优化与更新
    
    1. **收集数据**（用冻结的 $\pi_{\text{old}}$）得到 $(s_{i,t},a_{i,t},r_{i,t},\text{done}_{i,t})$
        
    2. 用当前的 Vϕ 计算 $\delta_{i,t}$,再**向后递推**得 $\hat A_{i,t}$，并令 $\hat R_{i,t}=\hat A_{i,t}+V_\phi(s_{i,t})$
        
    3. 初始化新参数：θ′←θ（旧策略参数的拷贝）。
        
    4. 在这同一批数据上，做 **K 个 epoch**、若干 mini-batch：
        
        1. 计算 $r_{i,t}(\theta)$、clip 后的策略最大Advantage；
            
            - **旧策略分母** $\pi_\theta(a_t|s_t)$是固定的（旧策略，来自采样）。
                
            - **新策略分子** $\pi_{\theta'}(a_t|s_t)$每次都会随着 θ′ 更新而改变。
                
        2. 计算价值 MSE 项 $(V_\phi-\hat R)^2$
            
        3. 计算熵项；
            
        4. 按上面的 **经验损失** $L(\theta,\phi)$ 反传更新。
            
    5. 结束后把 $\pi_{\text{old}}\leftarrow \pi_{\theta}$，进入下一批。
        

## 3.4 PPO LLM

┌──────────────────────────────┐

│ 1. Prompt 数据（用户输入） │

└──────────────┬───────────────┘

│

▼

┌──────────────────────────────┐

│ 2. Policy 模型 (LLM, πθ) │

│ 生成多个 candidate 回答 y │

└──────────────┬───────────────┘

│

▼

┌──────────────────────────────┐

│ 3. 奖励模型 RM(x,y) │

│ 根据人类偏好训练得到 │

│ 给每个回答打分 reward │

└──────────────┬───────────────┘

│

▼

┌──────────────────────────────┐

│ 4. 加 KL penalty │

│ R(x,y) = RM(x,y) - λ·KL(...)│

│ 约束新策略别偏离参考模型 │

└──────────────┬───────────────┘

│

▼

┌──────────────────────────────┐

│ 5. PPO 更新 │

│ - 计算概率比 r_t │

│ - 用 clip 限制更新幅度 │

│ - 让好回答概率↑，坏回答↓ │

└──────────────┬───────────────┘

│

▼

┌──────────────────────────────┐

│ 6. 更新后的 Policy 模型 │

│ πθ' 生成更符合人类偏好的输出 │

└──────────────────────────────┘

### 3.4.1 PPO
#### 3.4.1.1 图解

![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=MmUyZDE5MWE3ZTNmOGFkY2M2OGRiYTkyMjBhZDY4ZDVfS3dBYlZ5bEltZE84cW0wb2ljQlYyTFpjbEtzY3hvNE9fVG9rZW46STFWdGJ5ZXZlb1kyb0l4Z2lBYmw1VzkxZzZiXzE3NTkwNTY1NDA6MTc1OTA2MDE0MF9WNA)

![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=YzI5MTNmOGNlMTg3OGYxZWZjMGRmZWY5Zjc1YWY4ZDhfTHRGZkg2T2FHSmV2cVZpNjh0SzBCVnA5QU5JZVZoRDdfVG9rZW46WXN0emJsZlFob09PTEd4MXhjeGxBWmdtZ1JmXzE3NTkwNTY1NDA6MTc1OTA2MDE0MF9WNA)

![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=OWI2NzAzNWFmOWJhNjExZWEzNTEwYWY3YTExYjYyOGFfMlhwTjVldGRiUWtMd09DSmFrMDFZMUQyd2dXVFVsU2tfVG9rZW46RXNhQ2JCMm9ab2VJQVd4OFV0OWwzbkRVZ2NnXzE3NTkwNTY1NDA6MTc1OTA2MDE0MF9WNA)



#### 3.4.1.2 PPO 一轮训练

1. 准备数据
	- 提示（问题）  
	    - $q=$ “eric has one banan”
	    
	- 由 **行为策略**（本轮冻结快照） $\pi_{\text{old}}$ 生成的回复 token 列表  
	    - $o_{1:T}=$ “␠No”, “,”, “␠yuxuan”, “␠steal”, “␠it”, “,”, “␠so”, “␠eric”, “␠has”, “␠zero”, “.” …  
	    - 说明：LLM 是按 **token** 发射概率，**每个 token** 的条件是“提示+先前已生成的所有 token 前缀”。  
	    - 记每步状态 $s_t=(q, o_{<t})$，动作 $a_t=o_t$。


2. 采样（rollout，来自 $\pi_{\text{old}}$）
	- 逐步缓存 **对数概率**（只对生成段，mask 掉提示 token）：
	
	- 第 1 步（发第一个 token “␠No”）：  
	    - $\log \pi_{\text{old}}(\text{“␠No”}\mid s_1{=}(q))$
	    
	- 第 2 步（发 “,”）：  
	    - $\log \pi_{\text{old}}(\text{“,”}\mid s_2{=}(q,\text{“␠No”}))$
	    
	- … 每步都存一份；同时再前向一次 **参考模型** $\pi_{\text{ref}}$ 得  
	    - $\log \pi_{\text{ref}}(a_t\mid s_t)$（用来做 KL）。
		- 注：整句的序列概率是乘积 $\prod_{t=1}^{T}\pi(\,o_t\mid s_t\,)$，但 PPO 的更新都发生在 **逐 token** 的这些条件概率上。

* * *

# 4 即时奖励（reward shaping）

常见做法：**把 KL 放进奖励**（也有人放进 loss，见后述等价项）。

* token 级 KL 估计（单样本）：  
    $\mathrm{kl}_t \approx \log \pi_{\text{old}}(a_t\mid s_t)-\log \pi_{\text{ref}}(a_t\mid s_t)$
    
* 逐步即时奖励
    

$$r_t=\begin{cases}  
-\beta\cdot \mathrm{kl}_t, & t<T \\  
R_\psi(q,o_{1:T})\;-\beta\cdot \mathrm{kl}_T, & t=T  
\end{cases}$$

其中 $R_\psi$ 是奖励模型（对整句打分）。

> 例子里，最后一句“␠zero.” 若被 RM 判为“事实更好+风格更好”，则 $R_\psi$ 较高；前面各步仅有 KL 惩罚，鼓励输出别偏离参考模型太远。

* * *

# 5 Critic 目标（GAE 基于本批数据）

用价值头 $V_\phi$（同一 Transformer 干线或独立）：

* TD 残差：  
    $\delta_t=r_t + \gamma V_\phi(s_{t+1})-V_\phi(s_t)$（文本里常 $\gamma=1$）
    
* GAE 逆序递推：  
    $\hat A_t=\delta_t+\gamma\lambda\,\hat A_{t+1}$, $\hat A_{T+1}=0$
    
* 回报：  
    $\hat G_t=\hat A_t+V_\phi(s_t)$
    
* **只对本批**标准化优势：  
    $\hat A \leftarrow (\hat A-\mathrm{mean})/(\mathrm{std}+\epsilon)$
    

* * *

# 6 Actor（PPO-clip）逐 token 更新

在线策略（待优化） $\pi_\theta$ 前向得到 $\log \pi_\theta(a_t\mid s_t)$，构造 **概率比**：

$$\underbrace{\rho_t}_{\text{IS 比值}} \;=\;  
\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{old}}(a_t\mid s_t)}  
=\exp\!\big(\log\pi_\theta(a_t\mid s_t)-\log\pi_{\text{old}}(a_t\mid s_t)\big)$$

* 以 **“␠No”** 那步为例：  
    $\rho_1=\dfrac{\pi_\theta(\text{“␠No”}\mid q)}{\pi_{\text{old}}(\text{“␠No”}\mid q)}$
    
* 以 **“␠yuxuan”** 那步为例（已含前缀 “␠No,”）：  
    $\rho_3=\dfrac{\pi_\theta(\text{“␠yuxuan”}\mid q,\text{“␠No”},\text{“,”})}{\pi_{\text{old}}(\text{“␠yuxuan”}\mid q,\text{“␠No”},\text{“,”})}$
    

**PPO-clip 的策略目标（对生成 token 求平均）：**

$$L_{\text{policy}}(\theta)=  
\frac{1}{M}\sum_{t}\min\Big(\rho_t\hat A_t,\;  
\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\,\hat A_t\Big)$$

直觉：若“␠zero”那几步的 $\hat A_t>0$，就提高其 $\pi_\theta(a_t\mid s_t)$；若某步 $\hat A_t<0$，就降低；**clip** 防止离 $\pi_{\text{old}}$ 太远。

* * *

# 7 Critic/熵/KL 等辅助项

* 价值损失：$\displaystyle L_{\text{value}}(\phi)=\frac{1}{M}\sum_t \tfrac12\big(V_\phi(s_t)-\hat G_t\big)^2$  
    （可用 value-clip 稳定）
    
* 熵奖励（鼓励多样性）：$\displaystyle L_{\text{ent}}(\theta)=-\frac{1}{M}\sum_t \mathcal H(\pi_\theta(\cdot\mid s_t))$
    
* （可选）把 **参考 KL** 放到 loss：  
    $\displaystyle L_{\text{KL}}(\theta)=\frac{1}{M}\sum_t \mathrm{KL}\big(\pi_\theta(\cdot\mid s_t)\,\|\,\pi_{\text{ref}}(\cdot\mid s_t)\big)$
    

**总损失（最小化形式）**

$$\min_{\theta,\phi}\;\;-\;L_{\text{policy}}  
+\;c_v\,L_{\text{value}}  
-\;c_H\,L_{\text{ent}}  
+\;\beta\,L_{\text{KL}}\;.$$

> 若你选择 **“KL 进奖励”** 的方案（第 2 步），就**不要**再把同一 KL 重复加进 loss；两者效果等价，择一即可。

* * *

# 8 小批多轮 & 刷新行为策略

* 把本批生成 token 展平、打乱，做 $K$ 个 epoch 的小批 SGD（AdamW，梯度裁剪等）。
    
* 结束后 **丢弃这批数据**，更新行为策略：$\pi_{\text{old}}\leftarrow \pi_\theta$，进入下一轮采样。
    

* * *

## 8.1 关键点回顾（对你关心的“条件到底是谁”）

* $\pi(\text{“␠No”}\mid q)$：**单个 token** 的条件概率（第一步只依赖提示）。
    
* $\pi(\text{“␠yuxuan”}\mid q,\text{“␠No”},\text{“,”})$：**单个 token**，但条件是**整段前缀**（一堆词/子词）。
    
* 整句的概率是这些 **单步条件概率的乘积**；PPO 的比值 $\rho_t$ 与优势 $\hat A_t$ 都是 **逐 token** 地作用。
    
* 这就是为什么我们在例子里按 “No → , → yuxuan → … → zero” 每一步都能写出对应的  
    $\pi_{\text{old}}(a_t\mid s_t),\ \pi_{\text{ref}}(a_t\mid s_t),\ \pi_\theta(a_t\mid s_t)$、$\rho_t$、$\hat A_t$ 与损失项。

* * *