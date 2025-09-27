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

## 2.3 Markov decision process





![[Drawing 2025-09-26 02.12.13.excalidraw]]



\documentclass[11pt]{article}

\usepackage{amsmath,amssymb,amsfonts}

\usepackage[margin=1in]{geometry}

\usepackage{hyperref}

\usepackage{mathtools}

\title{PPO Loss (No Expectation) --- Full LaTeX Formulas}

\author{}

\date{}

  

\begin{document}

\maketitle

  

\section*{Notation}

Let a training batch index set be $\mathcal{M}=\{(i,t)\}$ where $i$ indexes a trajectory in the batch and $t$ a timestep. For each $(i,t)$ we have $(s_{i,t}, a_{i,t}, r_{i,t}, \mathrm{done}_{i,t})$. The old policy used to collect the batch is $\pi_{\mathrm{old}}$, the current trainable policy is $\pi_\theta$, and the value function is $V_\phi$. PPO hyperparameters: clipping $\epsilon>0$, discount $\gamma\in(0,1)$, and GAE parameter $\lambda\in[0,1]$. Weights $c_1,c_2\ge 0$ balance losses.

  

\section*{Empirical PPO Objective (Minimization Form, No Expectation)}

\begin{equation}

\label{eq:ppo_empirical}

\begin{aligned}

L(\theta,\phi)

&= \frac{1}{|\mathcal{M}|}\sum_{(i,t)\in\mathcal{M}}

\Big[

-\min\!\big(\, r_{i,t}(\theta)\,\hat A_{i,t},\ \operatorname{clip}(r_{i,t}(\theta),\,1-\epsilon,\,1+\epsilon)\,\hat A_{i,t}\big) \\[-2pt]

&\qquad\qquad\qquad\quad +\; c_1\,\big(V_\phi(s_{i,t})-\hat R_{i,t}\big)^2\;-\; c_2\,H\!\big(\pi_\theta(\cdot\mid s_{i,t})\big)

\Big].

\end{aligned}

\end{equation}

  

\section*{Components}

  

\paragraph{Policy Ratio.}

\begin{equation}

\label{eq:ratio}

r_{i,t}(\theta) \;=\; \frac{\pi_\theta(a_{i,t}\mid s_{i,t})}{\pi_{\mathrm{old}}(a_{i,t}\mid s_{i,t})}.

\end{equation}

  

\paragraph{TD Residual (with termination mask).}

\begin{equation}

\label{eq:delta}

\delta_{i,t} \;=\; r_{i,t} + \gamma\,\big(1-\mathrm{done}_{i,t+1}\big)\,V_\phi(s_{i,t+1}) \;-\; V_\phi(s_{i,t}).

\end{equation}

  

\paragraph{Generalized Advantage Estimation (backward recursion).}

\begin{equation}

\label{eq:gae_recursion}

\hat A_{i,T_i-1} \;=\; \delta_{i,T_i-1},

\qquad

\hat A_{i,t} \;=\; \delta_{i,t} + \gamma\lambda\,\big(1-\mathrm{done}_{i,t+1}\big)\,\hat A_{i,t+1}.

\end{equation}

  

\paragraph{Generalized Advantage Estimation (finite sum, no expectation).}

\begin{equation}

\label{eq:gae_sum}

\hat A_{i,t} \;=\; \sum_{l=0}^{T_i-t-1}

(\gamma\lambda)^l \left[\,

r_{i,t+l} + \gamma\,\big(1-\mathrm{done}_{i,t+l+1}\big)\,V_\phi(s_{i,t+l+1})

- V_\phi(s_{i,t+l})

\right].

\end{equation}

  

\paragraph{Return Target for Value Fitting.}

\begin{equation}

\label{eq:return_target}

\hat R_{i,t} \;=\; \hat A_{i,t} + V_\phi(s_{i,t}).

\end{equation}

  

\paragraph{Policy Entropy (discrete actions).}

\begin{equation}

\label{eq:entropy}

H\!\big(\pi_\theta(\cdot\mid s_{i,t})\big) \;=\; -\sum_{a}\pi_\theta(a\mid s_{i,t})\,\log \pi_\theta(a\mid s_{i,t}).

\end{equation}

For continuous actions (e.g., Gaussian policies), replace \eqref{eq:entropy} with the analytical entropy of the chosen distribution.

  

\section*{One Training Iteration (No Expectations)}

\begin{enumerate}

  \item Collect a batch $\{(s_{i,t},a_{i,t},r_{i,t},\mathrm{done}_{i,t})\}_{(i,t)\in\mathcal{M}}$ using frozen $\pi_{\mathrm{old}}$.

  \item Compute $\delta_{i,t}$ by \eqref{eq:delta}; then compute $\hat A_{i,t}$ via \eqref{eq:gae_recursion} (or \eqref{eq:gae_sum}); set $\hat R_{i,t}$ by \eqref{eq:return_target}.

  \item For $K$ epochs, iterate mini-batches $\subset\mathcal{M}$ and minimize \eqref{eq:ppo_empirical}.

  \item Set $\pi_{\mathrm{old}}\leftarrow \pi_\theta$ and repeat.

\end{enumerate}

  

\end{document}