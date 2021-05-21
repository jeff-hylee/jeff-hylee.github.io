---
layout: post
title:  "Implementing SARSA vs Q-learning (for Cliff Walking)"
date:   2021-05-20 21:40:00 -0400
categories: ML RL
permalink: /:categories/:title:output_ext
uselatex: true
---

## SARSA vs Q-Learning: on-policy vs off-policy
This is a continuation of the previous entry about the difference between SARSA and Q-Learning ([here](/ml/rl/sarsa-vs-qlearning.html)). It's a simple python implementation of these 2 algorithms for the cliff walking example.

### The environment
For the environment, we use [gym](https://gym.openai.com/) from OpenAI. Basically, it allows us to instantiate an object that abstract the environment. 

```python
env = gym.make('CliffWalking-v0')
```

![Cliff walking]({{ 'assets/images/cliff_walking_grid.png' | relative_url }})

*Fig 1. Cliff walking. Image source: [Sutton & Barto (2018)](http://incompleteideas.net/book/the-book-2nd.html)*

Here, we only really need to know about 2 methods:
 - `reset()`: resets the environment (to its initial state)
 - `step(action)`: performs transition of `env` as if agent took action `action`. It returns a tuple with the 1. next state, 2. reward, 3. done (terminal state reached?), 4. info; we will only use the first 3, info is auxiliary diagnostic information.

```python
# Reset env back to state: 'S'
env.reset()
# Advances env and observe the next_state, reward 
# (and also whether a terminal state has been reached)
next_state, reward, done, _ = env.step(some_action)
```

### $$\epsilon$$-greedy policy
For convenience, here are a couple of functions. `create_policy(Q)` returns a policy function for a given Q; `e_greedyfy(policy, epsilon, n_actions)` returns the e-greedy version of the provided policy.

```python
# Q: (S,A) -> Value
# We will represent Q with a 2d np array Q[s,a] -> Value

def create_policy(Q):
    """
    Creates the optimal policy for a given Q-value function.
    """
    def policy(state):
        return np.argmax(Q[state])
    
    return policy

def e_greedyfy(policy, epsilon, n_actions):
    """
    E-greedy version of the provided policy.
    epsilon is the probability of exploration, n_actions is the cardinality of the action space
    """
    def e_greedy_policy(state):
        # This could have been written in a more compact and efficient way, but I like this way
        # better in examples, as it shows clearly the exploration vs exploitation
        if np.random.uniform() <= epsilon:  # explore!
            return np.random.randint(n_actions)
        else:                               # exploit!
            return policy(state)
    
    return e_greedy_policy
```


### (Finally) SARSA/Q-learning
Finally, we implement SARSA and Q-learning. Here is a rough sketch of what happens for each *step* iteration (inside a rollout/episode):
1. Take action $$A$$, observe $$R$$ and $$S'$$
2. Select next action $$S'$$ ($$\epsilon$$-greedy policy) 
3. Q-value (policy) update
4. Setup for next step, if state not terminal

Here is the SARSA implementation, Q-learning is very similar with the main difference being the Q-value update step:

```python
for episode in range(num_episodes):
    # Start a new episode
    env.reset()
    e_greedy_policy = e_greedyfy(create_policy(Q), epsilon, n_actions)
    state = env.s
    action = e_greedy_policy(state)
    for step in itertools.count():
        # Take action A, observe R, S'
        next_state, reward, done, _ = env.step(action)
        # Select next action - S'
        next_action = e_greedy_policy(next_state)
        # Q-value/policy update
        q_target = reward + gamma * Q[next_state][next_action]
        Q[state][action] += alpha * (q_target - Q[state][action])
        e_greedy_policy = e_greedyfy(create_policy(Q), epsilon, n_actions)
        # Check for terminal state
        if done:
            break
        # prepare for next loop
        state, action = next_state, next_action
```

I have skipped a lot of details, these have been mostly high-ish level explanations as a pre-amble to the actual implementation.

### Code
Google colab is great, so I am providing the code via google colab jupyter notebooks:

- [SARSA code](https://colab.research.google.com/drive/1N6_1bNqVcyMYGNBThzRm46XDLH4K8otv?usp=sharing)
- [Q-learning code](https://colab.research.google.com/drive/12WUkeyqqTfGI8WHHkFbx_y7Tx3WHD-kI?usp=sharing)