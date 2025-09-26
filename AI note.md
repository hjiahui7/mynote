---
number headings: auto, first-level 1, max 6, 1.1
---


# 1 RL 知识梳理


## 1.1 Top Down
![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=MzhiMmZlMzQwNDBhOTQxN2M0YjJhMTE3ZmVlNzk3Y2RfcDc0eFVMR1hYSFBUQ284MEYwcnZlWHpVd20yQnhMdlhfVG9rZW46THZNc2JmeFZ0b2tzeUV4SDdBWmxzQ3NPZ1FkXzE3NTg4NzYzNDA6MTc1ODg3OTk0MF9WNA)

# 2 统一框架（总览）

> 目标：在马尔可夫决策过程（MDP）中最大化期望回报  
> \(J(\pi)=\mathbb{E}_{\tau\sim \pi}\!\left[\sum_{t=0}^{T-1}\gamma^{t}\,r_{t+1}\right]\)

- 两条主线：Value-based（c2a：先学值后出策）与 Policy-based（a2c：先建策再用值降方差）。
- 三条横轴：MC vs TD、On-policy vs Off-policy、稳定性（KL/熵/致命三角）。
- 核心循环：广义策略迭代（GPI）= 评估（学 V/Q/Adv）→ 改进（greedy/softmax 或 策略梯度）→ 重复。


## 2.1 Value-based（**c2a：Critic → Actor**）
- **思路**：先学习值函数（Q 或 VVV），再用 **greedy / ϵ\epsilonϵ-greedy / softmax** 从值函数**导出**策略。
- **代表**：MC Control、SARSA、Expected SARSA、Q-learning、DQN（+ 目标网络/重放/Double-Q）。
- **优点**：离散动作简单高效。
- **局限**：连续动作 arg⁡max⁡aQ\arg\max_a Qargmaxa​Q 困难；策略不可直接正则化。

## 2.2 Policy-based（**a2c：Actor → Critic**）

## 2.3 Markov decision process


