---
number headings: auto, first-level 1, max 6, 1.1
---


# 1 RL 知识梳理


## 1.1 Top Down
![[Pasted image 20250927014929.png]]

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
- **思路**：- 直接学习一个参数化的策略函数 \(\pi_\theta(a|s)\)，用梯度上升最大化期望回报。  策略可以是概率分布（分类/高斯），所以能自然地输出离散或连续动作。 不依赖“从值函数再推导”，而是直接调整“行为概率”。  
- **代表** - REINFORCE（蒙特卡洛策略梯度) , Actor-Critic 框架（结合价值函数基准）, TRPO（Trust Region Policy Optimization), PPO（Proximal Policy Optimization）  
- **优点**： **连续动作空间**处理自然（例如输出高斯分布均值/方差）。  策略可直接加正则或熵项，鼓励探索，避免过拟合。 学到的是**随机策略**，能在不确定性场景中表现更好。 梯度目标直接与“最终回报”挂钩，不需要显式 \(\arg\max_a Q\)。
- **局限**：**方差高** —— 采样到的回报噪声大，更新不稳定。  **样本效率低** —— 每一步数据往往只能用一次（除非特别设计，比如 PPO 可以多 epoch）。  **局部最优**风险更大（策略直接参数化，可能陷入 suboptimal 策略）。  对**学习率、熵系数**等超参数敏感，需要调参。  

## 2.3 Markov decision process

- **马尔可夫决策过程**（Markov Decision Process, **MDP**）是在马尔可夫链的基础上引入了动作（A）和奖励（R）的概念。





![[Drawing 2025-09-26 02.12.13.excalidraw]]
