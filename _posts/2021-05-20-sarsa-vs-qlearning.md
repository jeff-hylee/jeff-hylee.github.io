---
layout: post
title:  "SARSA vs Q-learning"
date:   2021-05-20 18:43:00 -0400
categories: ML RL
permalink: /:categories/:title:output_ext
uselatex: true
---
## SARSA vs Q-Learning: on-policy vs off-policy
If you put the SARSA and Q-learning algorithm side-by-side, they look very similar. In fact, in some pseudo-code descriptions, the only difference seems to be how the TD target is calculated. When asked about their main differences, most often, the answer that we get is that “SARSA is **on-policy** and Q-Learning is **off-policy**”. But how do we interpret that?

First, we will look at the algorithms and look at the differences
Second, we will look at the policies learned from SARSA and Q-learning for a great example, "cliff walking" (from [Sutton & Barto (2018)](http://incompleteideas.net/book/the-book-2nd.html)). This should allow us to understand the difference much better.

### Comparing algorithms:
Here are the SARSA and Q-learning algorithms as described by [Sutton & Barto (2018)](http://incompleteideas.net/book/the-book-2nd.html).

![SARSA and  Q-learning algorithms]({{ 'assets/images/sarsa-qlearning-algos.png' | relative_url }})

*Fig 1. Annotated SARSA and Q-learning. Image source: [Sutton & Barto (2018)](http://incompleteideas.net/book/the-book-2nd.html)*

- Common:
  - Blue box: both are iterative over steps (episodes are not collected/generated externally)
  - Red box: At the beginning of each step of the episode, the selection of the action is done according to the current e-greedy policy. There is a minor difference in how action selection is performed. Notably for SARSA, the action is selected using an $$\epsilon$$-greedy policy derived from the previous $$Q$$; for Q-learning the action is selected using an $$\epsilon$$-greedy policy derived from the current $$Q$$. But this is *not* quite what is meant by “on-policy vs off-policy”
- Different:
  - Green box: the value function update (in specific, the td target q-value) are different.
    - In SARSA, $$Q(S’,A’)$$ is used => td target calculation uses actual next action chosen by the current $$\epsilon$$-greedy policy (stochastic)
    -  In Q-learning, $$\max_a Q(S’, a)$$ is used => td target calculation does not necessarily use the actual next action chosen by the $$\epsilon$$-greedy policy. It uses the optimal action chosen by a non e-greedy policy (deterministic) 

In summary, the main difference is whether the action used for the TD target calculation matches (or not) the actual next action taken. For SARSA, it matches => on-policy; for Q-learning, it does NOT => off-policy.

### Cliff walking example
If you are still having difficulties grasping exactly what this means, you are not alone. Let’s look at the hilariously good example provided by Sutton and Barto - the cliff walking problem.
In a grid world, composed of a 4x12 grid, there exists a cliff. Failing off the cliff results in a significant penalty and being moved back to the starting position. The goal is to find the shortest path from the starting point to the end point. I am emitting the exact details, but looking at the figure below should be enough explanation.

![Cliff walking]({{ 'assets/images/cliff_walking_grid.png' | relative_url }})

*Fig 2. Cliff walking. Image source: [Sutton & Barto (2018)](http://incompleteideas.net/book/the-book-2nd.html)*

For this problem, what kind of learned policies should we expect from SARSA and Q-learning?

- SARSA: longer path that “avoids” getting too close to the cliff
- Q-learning: shorter path that fearlessly walks by the cliff

Now the difference between SARSA and Q-learning should be a bit more intuitive. SARSA ends up with a “safer” and longer path because it’s taking into account the risk of falling over the cliff - remember that the policy is $$\epsilon$$-greedy and stochastic. This means that walking by the cliff could result in the agent inadvertently falling off the cliff 
Q-learning ends up with a short path because it’s learning a policy that does not include the $$\epsilon$$-greedy randomness.

![Cliff walking: SARSA vs Q-learning]({{ 'assets/images/cliff_walking_sarsa_qlearning_comparison.png' | relative_url }})

*Fig 3. Cliff walking: SARSA vs Q-learning. Image source: [Sutton & Barto (2018)](http://incompleteideas.net/book/the-book-2nd.html)*

As a last word, if you are interesting with seeing these algorithms in the flesh, [my next entry/post](/ml/rl/cliff-walking-impl-sarsa-vs-qlearning.html) has a google colab jupyter implementation for both!
