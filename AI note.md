---
number headings: auto, first-level 1, max 6, 1.1
---


# 1 RL 知识梳理


## 1.1 Top Down
![](https://susfq45zc9c0.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=MzhiMmZlMzQwNDBhOTQxN2M0YjJhMTE3ZmVlNzk3Y2RfcDc0eFVMR1hYSFBUQ284MEYwcnZlWHpVd20yQnhMdlhfVG9rZW46THZNc2JmeFZ0b2tzeUV4SDdBWmxzQ3NPZ1FkXzE3NTg4NzYzNDA6MTc1ODg3OTk0MF9WNA)

## 1.2 Value-based（**c2a：Critic → Actor**）
- **思路**：先学习值函数（QQQ 或 VVV），再用 **greedy / ϵ\epsilonϵ-greedy / softmax** 从值函数**导出**策略。
- **代表**：MC Control、SARSA、Expected SARSA、Q-learning、DQN（+ 目标网络/重放/Double-Q）。
- **优点**：离散动作简单高效。
- **局限**：连续动作 arg⁡max⁡aQ\arg\max_a Qargmaxa​Q 困难；策略不可直接正则化。

## 1.3 Policy-based（**a2c：Actor → Critic**）

## 1.4 Markov decision process


