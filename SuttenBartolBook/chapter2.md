## n-Armed Bandit Problem

Picture a slot machine with a lever. Imagine that pulling the lever yields random amount of money, e.g: first pull give you 10$, the next one 2$, etc. And let's say that, despite the random reward given, each lever has a hidden value $q$ that denotes the true value of pulling it. 

Now let's say there are n number of slot machine, each with varying amount of rewards. We want to maximize the amount of money we get from pulling the levers, say, 100 times. How do we decide which lever to pull?

## Action-Value Methods

In order get the most amount of money, we want to pull the lever with the highest reward. But since each lever's reward is random, and we don't know the hidden value $q$ of each lever, how do we know which one yields highest result?

Let's call the act of pulling a lever an *action*. We denote $q(a)$ as the *actual* value of the lever a. We don't exactly know what $q(a)$ is, but one way to approximate it is by averaging the reward we get. $Qt(a)$ is our approximation of the true value of lever $a$ after $t$ times of pulling it.

$$
Q_t(a) = \frac{R_1 + R_2 + \cdots + R_{N_t(a)}}{N_t(a)}.
$$

Now we have a way to approximate a true value of the lever. The question now become:
1. how do we pick the lever initially when we don't have any estimate? 
2. how do we update our choice based on what we know (a.k.a actually learning)?

The first problem can be solved by assigning the same value to each lever and pick one randomly. Then, we update the value of that lever with the actual reward that we get and average it. We can use a greedy method by keep picking the same lever if we find that the reward is higher than our initial reward. But what if there might be another lever with higher reward?

To avoid getting stuck on a local optimum, we introduce a small probability of choosing a sub-optimal lever, for example 0.9 for picking current optimal and 0,1 for picking another lever in hopes of finding a better one. Do this with large enough of number of steps and we will be closer to finding the lever with the most optimal value. 

## Nonstationary Problem

The above method work with the assumption of the actual value `q(a)` as a constant during the whole simulation. More often than not, the actual reward in a reinforcement learning environment is not constant; it may fluctuate between steps. To accommodate this, we introduce step size `α` that decide how much new reward affect our estimate, written as:

$$
Q_{k+a} = Q_k + α[R_k - Q_k]
$$

where $ R_k $ is the current reward of pulling the lever

## Gradient Bandits

Instead of using $q(a)$, another way of determining how much we want to pick a certain lever is by using preference $H_t(a)$. Since preference is a score, we need to normalize it into probability with the sum of 1. We do this with softmax equation, which basically work by putting all $H_t(a)$ as the exponent of $e$ and averaging them by the sum of all $e^H_t$, or formally:

$$
\frac{e^{H_t(a)}}{\sum_{b=1}^{n} e^{H_t(b)}} = \pi_t(a)
$$

Where we can then update the value of our preference with gradient ascent

$$
H_{t+1}(a) = H_t(a) + \alpha \frac{\partial \mathbb{E}[R_t]}{\partial H_t(a)}
$$

With 
$$
\mathbb{E}[R_t] = \sum_{b} \pi_t(b) q(b)
$$

How i understand this equation is basically, the $\mathbb{E}[R_t]$ is there to provide a way to measure how much do we need to increase the preference. the derivative of $E$ with respect of $H$ measure how much $E$ change when the preference change, which in turn also affect how much the probability change when the preference change. Note that since all probability sum to 1, changing any one preference in turn affect the probability of all levers. $α$ controls how much fraction from the rate of change of $E$ that we actually use to update the $H_{t(a)}$
