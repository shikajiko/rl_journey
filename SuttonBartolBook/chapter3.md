## Agent - Environment

Agent is what we train. Environment is everything around the agent that gives the agent information on its current state. Agent do actions. The environment then return a state and a reward for the agent based on the action it picks. 

The environment needs to contain things that the agent needs to know, but not to the full extend. For example if the agent is learning how to play blackjack, it shouldn't know what card comes next. 

In the context of training a robot, each joint states, sensor readings, positions, and orientations are part of the *environment* and not the agent. What count as part of the agent is limited to only the *interface* in which the agent use to communicate or affect the environment based on the action it picks. 

## Reward

A reward is what the agent earn after picking certain actions. We must define reward with the broader goal in mind. Reward should specify what we *want* the agent to achieve, not *how* the agent should achieve it. For example if we're training an agent to play chess, we shouldn't make taking enemy's piece a reward, because then the agent could choose to take the enemy's piece even if that can lead to a checkmate by the enemy. The reward in this context should ultimately be winning the game, not the process of how to win the game. 

## Return and Discount

A learning environment can have finite steps with clear endings (episode) like winning or losing in the game of chess. But it can also be a continuous task, for example on a self-balancing robot. For episodic environment, we can define the Return $G_t$ as the sum of all rewards after time $t$, $G_t = R_{t+1} + R_{t+2} + \dots + R_T$, where $T$ is the total step until it reach a terminal state (e.g. winning or losing). However for a continuous task, $T = \infty$ 

To account for continuous task, we define discount rate $\gamma$ as the decaying value of future reward, where $\gamma < 1$. Formally, the return $G_t$ is the sum of $R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$

This allow us to get finite value of $G_t$ even when the task continues forever (as long as the rewards are bounded). Moreover, having future reward has decreasing value makes the agent prioritize immediate reward more than reward far from current step. The greater the discount rate, the more far-sighted the agent become while smaller discount rate make the agent more short-sighted. 

## Markov Decision Process

It is logical to think that each prior state, action, and reward taken by the agent should affect the future state, action, and reward. However, it is quite impractical if we need to store every state, action, and reward the agent taken up until the current point in order to know what happens next. 

A better approach would be to define a state in which the information of the current state is enough to determine the current situation and what happens next. For example, on a game of chess, in order to make a decision, we don't need to know each steps chosen by both players up to the current point; we just need to know how the current position of the pieces on the board. A state that satisfy this criteria is said to have *Markov Property*. 

Furthermore, since current state is enough to tell us about the overall condition of the environment, we can derive couple of things:

1. The whole environment:

$$p(s',r \mid s,a) = \Pr\{S_{t+1}=s',\,R_{t+1}=r \mid S_t=s,\,A_t=a\}$$

2. Expected reward for a state-action pair. The average reward from taking $a$ in $s$, ignoring where the agent lands:

$$r(s,a) = \mathbb{E}[R_{t+1} \mid S_t=s,A_t=a] = \sum_{r\in\mathcal{R}} r \sum_{s'\in\mathcal{S}} p(s',r \mid s,a)$$

3. State-transition probabilities. The probability of landing in $s'$, ignoring the reward. 
$$p(s' \mid s,a) = \Pr\{S_{t+1}=s' \mid S_t=s,A_t=a\} = \sum_{r\in\mathcal{R}} p(s',r \mid s,a)$$

4. Expected reward for a state-action-next-state triple. The average reward given that the agent landed in $s'$:

$$r(s,a,s') = \mathbb{E}[R_{t+1} \mid S_t=s,A_t=a,S_{t+1}=s'] = \frac{\sum_{r\in\mathcal{R}} r\,p(s',r \mid s,a)}{p(s' \mid s,a)}$$

## Value Function

From the MDP desribed earlier, we already know the probability of ending on a certain state $s'$, but we don't have any way of knowing how "good" is that state for the agent. A value function is a function that precisely answer this problem.

We know that $G_t$ is the return value that we received after certain step $t$. Or more precisely, its the sum of the discounted rewards from step $t+1$ onward. 

Now, it may be tempting to use $G_t$ directly to calculate the value of a state. However, since the environment is stochastic, a state $s$ doesn't necessarily always gives the same return $G_t$. So instead of using $G_t$ directly, we denote $v_\pi$ as the expected *return* of $G_t$ under certain state $s$. 

$$v_\pi(s) = \mathbb{E}_\pi\left[ G_t \mid S_t = s \right]$$

Furthermore, with our knowledge of the environment, it can be expanded into:
$$v_\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma v_\pi(s') \right]$$

Which also known as the *Bellman Equation*

However, since the agent doesn't have the full knowledge of the environment $p(s', r \mid s, a)$, the part it can use to update the state value estimate $V(s)$ is the part inside the bracket on the right hand side: $$[ r + \gamma V(s')]$$ which roughly translate to "add the reward $r$ just received to the discounted estimated value of the next state $s'$". The agent then moves $V(s)$ a little toward this target (this is TD learning, covered later): $$V(s) \leftarrow V(s) + \alpha\left[ r + \gamma V(s') - V(s) \right]$$ Given enough step, $V(s)$ would eventually converge into $v_\pi(s)$