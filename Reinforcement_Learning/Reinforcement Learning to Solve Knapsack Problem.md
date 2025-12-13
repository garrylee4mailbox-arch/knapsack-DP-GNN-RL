 # 1. Introduction to Reinforcement Learning Basics
**Reinforcement Learning (RL)** is one of the three fundamental machine learning paradigms (alongside Supervised Learning and Unsupervised Learning) which especially focuses on how an intelligent agent can take actions in a dynamic and variational environment in order to maximize the cumulative rewards. Unlike Supervised Learning which relies on labeled data, or Unsupervised Learning which seeks patterns in data, RL is concerned with  finding a balance between **exploration** (of undiscovered territory) and **exploitation** (of current knowledge) with the goal of maximizing the cumulative reward (the feedback of action taking). The search for this balance is known as the *Exploration-Exploitation Dilemma*. This section introduces the core components and concepts of Reinforcement Learning, including agents, environments, states, actions, rewards and policies, with explanation of how these elements interact within the **Markov Decision Process (MDP)** framework.
## 1.1. Core Components in RL
Reinforcement Learning focuses on the idea that an agent (the learner or decision-maker) interacts with an environment to finally achieve a goal. The agent make actions and receive feedback to estimate its performance and optimize its decision-making over time. Here are some components:
- **Agent**: The learner or decision-maker that performs actions and interacts with the environment
- **Environment:** The world or system in which the agent interact, provides rewards and feedback in response to agent's actions
- **State (S)**: The situation or condition the agent is currently in. State can be *discrete* (e.g. board positions in chess) or *continuous* (e.g. a robot's position in a space)
- **Action (A):** The possible move or decision the agent could make. Action also can be *discrete* (e.g. left/right) or *continuous* (e.g. steering angles)
- **Reward (R):** The feedback or result from the environment based on the agent's action taken in a state. The reward function is the only feedback the agent receives about its performance.
- **Policy (π):** The agent's strategy, mapping states to actions. Policies can be *deterministic* (always choosing the same action in a state) or *stochastic* (choosing actions according to a probability distribution)
- **Value Function (V(s), Q(s, a)):** Functions estimating the expected cumulative reward from a state (or state-action pair), under a given policy. These are central to many RL algorithm.
## 1.2. Working of Reinforcement Learning
The agent interacts iteratively with its environment in a feedback loop.

1. The agent observes the current state of the environment.
2. It chooses and performs an action based on its policy.
3. The environment responds by transitioning to a new state and providing a reward or penalty.
4. The agent updates its knowledge (policy, value function) based on the reward received and the new state.
This cycle repeats with the agent balancing **exploration** (trying new actions) and **exploitation** (using known good actions) to maxing the cumulative reward over time.
## 1.3. Markov Decision Process (MDP) Framework
Most RL problems are formalized as **Markov Decision Processes (MDPs)**, which provide a mathematical structure for sequential decision-making under uncertainty.
A **Markov Decision Process** is defined by the tuple:
- $S$: is a set of states called the *state space*. The state space can be discrete or continuous like the set of real numbers.
- $A$: is a set of actions called the *action space* (alternatively, $A_s$ is the set of actions available from state $s$). As for the state, actions can be discrete or continuous.
- $P_a(s,s')$: is the transition probability function, providing the probability of moving to the state $s'$ from state $s$ with action $a$ 
- $R_a(s,s')$: is the immediate reward (or expected immediate reward) received after action $a$ is taken to transaction from state $s$ to state $s'$ 
- $\gamma \in [0, 1]$: Discount factor which determines the importance of future rewards relative to immediate rewards. A lower discount factor makes the decision maker more short-sighted.

A particular MDP may have multiple distinct optimal policies. Because of the Markov property, it can be shown that the optimal policy is a function of the current state.
## 1.4. Policies: Deterministic & Stochastic
A policy $\pi$ defines the agent's behavior. In deterministic policies, $\pi(s)$ yields a specific action for each state. In Stochastic policies, $\pi(as)$ gives the probability of taking action $a$ in state $s$. Stochastic policies are particularly useful in environments with inherent randomness or partial observability, and they facilitate exploration during learning.

The central objective in RL is to find an optimal policy $\pi^*$ , which maximizes the expected cumulative reward (return) from any starting state.
## 1.5. Value Functions & Bellman Equations
The state-value function $V\pi(s)$ is the expected returen when starting from state $s$ and following policy $\pi$ thereafter:

$$V\pi(s)=E\pi[∑_{k=0}k r_{t+k+1} s_t = s]$$ 

The action-value function $Q\pi(s,a)$ is the expected return from state $s$, taking action a, and then following $\pi$:

$$Q\pi(s,a)=Eπ[∑_{k=0}k r_{t+k+1} s_t = s, a_t = a]$$ 
## 1.6. Q-Learning Concepts
Q-Learning is a reinforcement learning algorithm that trains an agent assign values to its possible actions based on its current state, without requiring a model of the environment (model-free), it directly update estimation of "state-action" values from experience with goal of maximize long-term cumulative feedbacks and rewards.

Several core components(similar with **Markov Decision Process** but added $Q-value$ and $\alpha$ ):
- $S$: is a set of states called the *state space*. The state space can be discrete or continuous like the set of real numbers.
- $A$: is a set of actions called the *action space* (alternatively, $A_s$ is the set of actions available from state $s$). As for the state, actions can be discrete or continuous.
- $P_a(s,s')$: is the transition probability function, providing the probability of moving to the state $s'$ from state $s$ with action $a$ 
- $R_a(s,s')$: is the immediate reward (or expected immediate reward) received after action $a$ is taken to transaction from state $s$ to state $s'$ 
- $\gamma \in [0, 1]$: Discount factor which determines the importance of future rewards relative to immediate rewards. A lower discount factor makes the decision maker more short-sighted, More closer to 1 discount factor make decision maker more focusing on long-term future.
- $Q-value$: is the expected cumulative rewards after taking action $A$ in the state $S$. Q-Learning finds optimal policy from updating $Q-value$.
- $\alpha\in[0,1]$: is the learning rate which controls the $Q-value$ updates based on current received experience. 
### 1.6.1. Steps of Q-Learning Algorithm
1. **Initialization:** Set an initial $Q-value$ for all state-action (generally 0)
2. **Exploration & Exploitation:** within each episode:
- begin with initial state.
- using $\epsilon-greedy$ policy to take action: using $\epsilon$ possibility to choose random exploration (test new actions), using $1-\epsilon$ possibility to choose current actions with the highest $Q-value$.
1. **Update $Q-value$:** After taking action, check new state $S'$ and reward $R$, using **Bellman Equation (*Weighted Average Form*)** to update:

$$Q(S_t,A_t)_{new} \leftarrow (1-\alpha)Q(S_t,A_t) + \alpha \cdot (R_{t+1} + \gamma \cdot \max Q(S_{t+1},A_{t+1}))$$

where $R_{t+1}$ is the reward received when moving from the state $S_t$ to the state $S_{t+1}$, and $\alpha$ is the learning rate within [0, 1]

Note that $Q(S_t,A_t)_{new}$ is the sum of three terms:

- $(1-\alpha)Q(S_t,A_t)$: the current value (weighted by one minus the learning rate)
- $\alpha R_{t+1}$: the reward $R_{t+1}$ to obtain it action $A_t$ is taken when in state $S_t$(weighted by learning rate)
- $\alpha\gamma maxQ(S_{t+1},A_{t+1})$: the maximum reward that can be obtained from state $S_{t+1}$ (weighted by learing rate and discount factor)

Notice that the **Bellman Equation** also has an ***Incremental Form***:

$$Q(S_t,A_t)_{new}\leftarrow Q(S_t,A_t)+ \alpha\cdot(R_{t+1} + \gamma\cdot maxQ(S_{t+1},A_{t+1}) - Q(S_t,A_t))$$

These two forms are the same and can be transformed to each other.

4. **Recursive:** running multiple episodes, until $Q-value$ converges (agent finds the optimal policy: always chooses the action with the highest $Q-value$)
## 1.7. Exploration & Exploitation
A fundamental challenge in RL is to balance exploration (trying new actions to discover their effects) and exploitation (choosing actions known to yield high rewards). Strategies such as $\epsilon$ - greedy (choosing a random action with probability $\epsilon$, otherwise the best-known action) are commonly used to manage this trade-off.
## 1.8. Categories of RL Algorithm
RL algorithm can be broadly categorized as:
- **Value-based methods:** Learn value functions (e.g. Q-learning, SARSA) and derive policies from them.
- **Policy-based methods:** Directly optimize the policy.
- **Actor-Critic methods:** Combine value-based and policy-based approaches, with an actor (policy) and a critic (value function).
Recent advances in Deep Reinforcement Learning (DRL) integrate deep neural networks as function approximators, enabling RL to tackle high-dimensional and complex environments.

# 2. Modeling the Knapsack Problem with Reinforcement Learning
The Knapsack Problem (KP) is a classic combinatorial optimization problem, widely studied due to its theoretical significance and practical applications in logistics, finance and resource allocation. In its 0-1 variant, the problem is to select a subset of items, each with a value and weight, to maximize total value without exceeding the knapsack's capacity.
## 2.1. The Definition of 0/1 Knapsack Problem
Given $n$ items where each item has some weight and profit associated with it and also given a bag with capacity $W$(i.e. the bag can hold at most $W$ weight in it). The task is to put the items into the bag such that the sum of profits associated with them is the maximum possible.

**Note:** The constraint here is we can either put an item completely into the bag or cannot put it at all (It is not possible to put a part of an item into the bag).

> [Here is an example]
> **Input:** W = 4, **profit**[ ] = [1, 2, 3], **weight**[ ] = [4, 5, 1]
> **Output:** 3
> **Explanation:** There are two items which have weight less than or equal to 4. If we selec the item with weight 4, the possible value is 1. And if we select the item with weight 1, the possible value is 3. So the maximum possible profit is 3. Note that we cannot put both the items with weight 4 and 1 together as the capacity of the bag is 4.


## 2.2. Formulating the Knapsack Problem as an MDP
To apply **Reinforcement Learning (RL)**, the **Knapsack Problem (KP)** must be cast as a **Markov Decision Process (MDP)**. This fundamental concept involves defining the state space, action space, transition dynamics, and reward function.

### 2.2.1. State Representation
The state should encapsulate all information necessary for decision-making. Common representations include:

- **Binary vector:** Each element indicates whether an item is included or not. 
- **Tuple:** (current total weight, current total value, items selected so for).
- **Aggregated features:** For large instances, state aggregation or embedding techniques can reduce dimensionality and improve scalability.

### 2.2.2. Action Representation
At each step, the agent decides which item to include next (or to skip). Actions can be:
- **Discrete:** Select item $j$ (if not already chosen and if feasible), or skip.
- **Masking:** Actions corresponding to infeasible or already-selected items are masked out to prevent illegal moves.

### 2.2.3. Transition Dynamics
The environment transitions deterministically: selecting an item updates the state by marking the item as chosen and updating the total weight and value. If the capacity is exceeded, the episode may terminate or a penalty may be implemented.

### 2.2.4. Example Reward Function
A typical reward function for the KP might be:
- $+v_j$ if item $j$ is added and the solution remains feasible
- $-w_j$ if adding item $j$ exceeds capacity
- $-W$ if an illegal action is taken (e.g. select an already-chosen item)
- Final reward: total value if feasible, large penalty if not.

## 2.3. RL Algorithm for the Knapsack Problem
Several RL algorithms have been successfully applied to the KP:

- **Tabular Q-learning:** Suitable for small instances with manageable state-action spaces. Learns $Q(s, a)$ values for each state-action pair.
- **SARSA:** On-policy variant of Q-learning, updates Q-values based on the action actually taken.
- **Deep Q-Networks(DQN):** Used neural networks to approximate $Q(s, a)$, enabling scalability to larger instances.
- **Policy Gradient Methods (REINFORCE, A2C, PPO):** Directly optimize the policy, often with neural network parameterization. Actor-Critic methods (A2C, PPO) combine policy and value function learning for improved stability and efficiency.

**Algorithm selection Considerations**
- Tabular methods are simple and interpretable but do not scale well.
- Deep RL methods handle high-dimensional state spaces and generalize across instances.
- Hybrid and meta-learning approaches can further enhance performance, especially in real-world, variable-sized problems.

# 3. Code Implementation & Demonstration
## 3.1. Q-Learning Algorithm for 0/1 Knapsack Problem
This section provides practical Python code for implementing RL solution to the Knapsack Problem, using...(*mostly used libraries*), we cover...(*RL algorithms*) methods, illustrating key implementation details and best practices.

According to the **Steps of Q-Learning Algorithm** section as we mentioned before, Here is an explanation of the step-by-step approach to formulate and implement Q-Learning for Knapsack Problem.
### Step 1: Formulating the Problem as an MDP in Q-Learning
**States (S):** represents the current state of decision point. A general way is to use a tuple `(current_item_idx, current_cap)`.
`current_item_idx`: An index ranges from [0, N] where N is number of item. The episode will stop when it reaches N.
`current_cap`: The knapsack's current capacity, which is discretized to integers from 0 to capacity W.

**Action (A):** represents the binary choice within per state, 0 means skipping the item and 1 means taking the item (without exceeding capacity W)

**Transitions:** Deterministic.
If action = 0, move to `(current_item_idx + 1, current_cap)`.
If action = 1 and `weights of current_item_idx <= current_cap`, move to 
`(current_item_idx + 1, current_cap + item_weight)`.
If action = 1 but it exceeds W, Either invalid (force taking action = 0) or penalize heavily.

**Reward (R):** Sparse or immediate.
- `item_value++` if taking taking this item (action = 1) and it fits.
- 0 if skipping this item (`action = 0`).
- Negative penalty if trying to take but it doesn't fit, which means discourage invalid actions.
- Optionally, defer all rewards to the terminal state (when `index == N` ), giving the total value there.

**Discount Factor ($\gamma$):** Typically $\gamma \in [0, 1]$ , to value future rewards.
Episodes: Start from state (0, 0), Run massive episodes (e.g. 1000-5000) to train the Q-table.
**Exploration or Exploitation:** Use $\epsilon$ - greedy policy (e.g. set $\epsilon$ = 1, then decaying over time) to balance decision between **random actions** (exploration) or **known best actions** (exploitation).

**Q-value Update:**

$Q(S,a)_{new}\leftarrow Q(S,a)+ \alpha\cdot(R + \gamma\cdot maxQ(S',a') - Q(S,a))$

### Step2: Implementation Overview
We'll need:
- **Q-table:** A dictionary or NumPy array of shap (N+1, W+1, 2) for states and actions.
- **Item lists:** Arrays for weights and values.
- **Training loop:** Simulate episodes, update Q-values.
- **Inference:** After training, start from (0, 0) and greedily choose actions with max Q-value (handling invalid takes).

**Potential challenges:**
- **Large state space:** For N = 10, W = 100, table is ... entries (manageable), but scales poorly.
- **Convergence:** Tune hyperparameters ($\alpha$, $\gamma$, $\epsilon$, episodes).
- **Invalid actions:** Mask them or penalize.

### Step3: Code Implementation
#### 3.1. Disposition of Initial Environment:
```python
    def __init__(self, weights, values, capacity):

        self.weights = weights  # list of item weights

        self.values = values  # list of item values

        self.capacity = capacity  # list of knapsack capacity

        self.n_items = len(weights)  # number of items

```

#### 3.2. Implementation of Action:
```python
    def step(self, state, action):

        """

        Take action with return (new state, reward, if down or not)

        state: (item_index, current_capacity)

        action: 0 (skip) or 1 (take)

        """

        item_idx, current_cap = state

        # Check if all items are already in consideration

        if item_idx >= self.n_items:

            return state, 0, True

        reward = 0

        # Pattern of action taking

        if action == 1:  # try to take current item

            if self.weights[item_idx] <= current_cap:

                # if the item can be taken: then take it and get value, meanwhile the capacity decrease

                reward = self.values[item_idx]

                next_cap = current_cap - self.weights[item_idx]

            else:

                # if the item cannot be taken: punishment will be implemented, meanwhile capacity stays the same (or the action itself can be regarded as illegal)


                reward = -100  # factor of punishment, for telling agent not try to take item which is larger than capacity

                next_cap = current_cap

        else:  # action == 0 (skip)

            reward = 0

            next_cap = current_cap

  

        # turn to the next item

        next_state = (item_idx + 1, next_cap)

  

        # check if this is the last step

        done = (item_idx + 1 == self.n_items)

        return next_state, reward, done

```


#### 3.3. Q-Table:
```python
def train_q_learning(weights, values, capacity, episodes=10000, alpha=0.1, gamma=0.95):

    env = KnapsackEnvironment(weights, values, capacity)

    q_table = np.zeros((env.n_items + 1, capacity + 1, 2))

```


#### 3.4. Implementation of Dynamic Greedy Policy $\epsilon$ :
```python
    epsilon = 1.0  # initial value of greedy policy is set as 1, which means totally random

    epsilon_min = 0.01  # minimum of greedy policy which keeps 1% randomization

    epsilon_decay = 0.9995  # discount factor: decrease the greedy policy as each time of training finished

    for episode in range(episodes):

        state = env.reset()

        done = False
        

        while not done:

            item_idx, current_cap = state

            # strategy: randomly explore on early stage, utilize experience in later period

            if random.uniform(0, 1) < epsilon:

                action = random.choice([0, 1])  # randomly choose

            else:

                action = np.argmax(q_table[item_idx, current_cap])  # take the action with maximum of Q-value

  

            next_state, reward, done = env.step(state, action)

            next_item_idx, next_cap = next_state

  

            old_value = q_table[item_idx, current_cap, action]

  

            # Get the maximum of Q-value of the next step

            if next_item_idx < env.n_items:

                next_max = np.max(q_table[next_item_idx, next_cap])

            else:

                next_max = 0

  

            # Update the formula of Q-table

            new_value = old_value + alpha * (reward + gamma * next_max - old_value)

            q_table[item_idx, current_cap, action] = new_value

            state = next_state

  

        # decrease the greedy policy as each time of training finished

        if epsilon > epsilon_min:

            epsilon *= epsilon_decay

  

    return q_table

```

#### 3.5. Solve the Knapsack Problem with Q-table:
```python
def solve_knapsack(weights, values, capacity, q_table):

    """

    Solve the Knapsack Problem with well trained Q-table
    
    """

    env = KnapsackEnvironment(weights, values, capacity)

    state = env.reset()

    done = False

  

    total_value = 0

    total_weight = 0

    selected_items = []

  

    print("\n--- Agent Decision Making Process ---")

  

    while not done:

        item_idx, current_cap = state

  

        # directly choose the action with maximum of Q-value (greedy policy)

        action = np.argmax(q_table[item_idx, current_cap])

  

        # logic judgement: even though Q-table suggests taking the item, we'd better check substantively if the capacity is enough or not

        # (since if the training is not surficient, Q-table may suggest illegal action)

        if action == 1 and weights[item_idx] > current_cap:

            action = 0  # compulsorily modify (or let it fail)

  

        if action == 1:

            print(f"items {item_idx} (weights{weights[item_idx]}, values{values[item_idx]}): -> 1")

            total_value += values[item_idx]

            total_weight += weights[item_idx]

            selected_items.append(item_idx)

        else:

            print(f"items {item_idx} (weights{weights[item_idx]}, values{values[item_idx]}): -> 0")

  

        next_state, _, done = env.step(state, action)

        state = next_state

  

    return total_value, total_weight, selected_items

```
## 3.2. RL Algorithm for Multi-Dimensional Knapsack Problem (MDKP)

### Step 1: Formulating the Environment as a MDP in RL
The **MDKP** is framed as a finite-horizon, episodic **MDP** where the agent decides for each item ***in a fixed order*** (items 0 to n-1).

- **States (S)**: Information about the current decision point.
    - In code: `get_state()` method in `MDKPEnv`.
    - State vector (5-dimensional, normalized for better NN training):
        - Value of the current item (***normalized by ~100***).
        - Weight of current item in dimension 1 (***normalized by cap1***).
        - Weight of current item in dimension 2 (***normalized by cap2***).
        - Remaining capacity in dimension 1 (***fraction of cap1***).
        - Remaining capacity in dimension 2 (***fraction of cap2***).
    - Terminal state: When all items processed (`current_item >= n_items`), returns zeros.
- **Actions (A)**: Binary decision per item.
    - In code: `action` in `step()` — 0 = skip, 1 = take.
    - Discrete action space with 2 options.
- **Transition Dynamics** ( $P(s'|s,a)$ ): Deterministic.
    - In code: `step()` method.
    - After action:
        - Always advance to next item (`current_item += 1`).
        - If action=1 and item fits both constraints → subtract weights from remaining capacities, add value.
        - If action=1 but item doesn't fit → no change to capacities/value, but apply penalty.
        - Next state computed based on new remaining capacities and next item.
    - The environment enforces hard constraints partially (via penalties for violations) while allowing learning.
- **Episode**: One full pass through all items.
    - In code: `reset()` starts a new episode (full capacities, item 0).
    - Ends when `done=True` (all items processed).
    - Each episode = one complete solution (selection of subset of items).
### Step2: Implementation Overview
#### Reward Function ( $R(s,a,s')$ )
The reward signal guides the agent toward maximizing total value while respecting constraints.

- In code: Inside `step()`.
    - Sparse + shaped reward:
        - If take (action=1) and item fits → + (item value / 10.0) — positive, scaled to prevent exploding gradients.
        - If take but doesn't fit → -5.0 — strong negative penalty to discourage invalid selections.
        - If skip (action=0) → 0 reward.
    - No reward at terminal state.
    - **Objective alignment**: The undiscounted sum of rewards ≈ total value (scaled), minus penalties. Since penalties are avoided in good policies, the agent learns to maximize the knapsack objective (total value) subject to constraints.
#### **Policy ( $π(a|s)$ )**
The "brain" that decides actions — here, a stochastic policy learned by a neural network.

- In code: `PolicyNetwork` class.
    - Input: State vector (5 dims).
    - Output: Probability distribution over actions — $softmax$ → $[P(skip), P(take)]$.
    - During training: Sample actions stochastically (`Categorical` distribution) for exploration.
    - During testing: Greedy — choose argmax (most probable action).
    - This is a **parametric policy** $π_θ(a|s)$, where $θ$ are the network weights.

#### **Training: REINFORCE (Policy Gradient)**
Optimizes the policy to maximize expected cumulative reward.

- In code: `train()` function loop.
    - For each episode:
        - Rollout a full trajectory (sequence of states, actions, rewards) using current stochastic policy.
        - Compute **discounted returns** $(G_t)$ for each timestep: Future rewards discounted by $γ=0.99$.
            - Code: Backward pass accumulating $R = r + γ*R$.
            - Normalized for lower variance.
        - Update policy via REINFORCE gradient:
            - Loss = $-∑ log π(a_t|s_t) * G_t$
            - Gradient ascent to increase probability of actions leading to high returns.
    - Goal: Maximize expected total value over many episodes.

### Step3: Code Implementation
#### 3.1. MDKP Environment (`MDKPnv` Class)
This class simulates the knapsack problem as a sequential decision environment:

```python
class MDKPEnv:
    def __init__(self, values, weights_dim1, weights_dim2, cap1, cap2):
        self.values = values                  # list/array of item values
        self.w1 = weights_dim1                # weights for constraint 1
        self.w2 = weights_dim2                # weights for constraint 2
        self.cap1_max = cap1                  # maximum capacity 1
        self.cap2_max = cap2                  # maximum capacity 2
        self.n_items = len(values)
        self.reset()
```
- Constructor stores problem data and resets the environment.

```python
def reset(self):
	    self.current_item = 0                 # start at first item
        self.current_cap1 = self.cap1_max     # full capacities
        self.current_cap2 = self.cap2_max
        self.total_value = 0                  # accumulated value
        return self.get_state()
```
- `reset()`: prepares for a new episode (new knapsack filling).

```python
def get_state(self):
        if self.current_item >= self.n_items:
            return np.zeros(5)                # terminal state
        
        v = self.values[self.current_item] / 100.0           # normalize value (assume max ~100)
        w1 = self.w1[self.current_item] / self.cap1_max      # normalize weight 1
        w2 = self.w2[self.current_item] / self.cap2_max      # normalize weight 2
        rem1 = self.current_cap1 / self.cap1_max             # remaining capacity 1
        rem2 = self.current_cap2 / self.cap2_max             # remaining capacity 2
        
        return np.array([v, w1, w2, rem1, rem2], dtype=np.float32)
```
- Returns a 5-dimensional state vector for the current item:
    1. Normalized value of current item
    2. Normalized weight in dimension 1
    3. Normalized weight in dimension 2
    4. Fraction of capacity 1 still available
    5. Fraction of capacity 2 still available
- Normalization helps the neural network learn faster.

```python
def step(self, action):
        reward = 0
        done = False
        
        if action == 1:  # take the item
            item_w1 = self.w1[self.current_item]
            item_w2 = self.w2[self.current_item]
            
            if item_w1 <= self.current_cap1 and item_w2 <= self.current_cap2:
                # Can take it → update capacities and reward
                self.current_cap1 -= item_w1
                self.current_cap2 -= item_w2
                r = self.values[self.current_item]
                self.total_value += r
                reward = r / 10.0                     # scale reward to avoid huge gradients
            else:
                # Invalid action → give negative reward (penalty)
                reward = -5.0
                # Item is NOT taken (capacities unchanged)
                
        # Always move to next item
        self.current_item += 1
        
        if self.current_item >= self.n_items:
            done = True
            
        next_state = self.get_state()
        return next_state, reward, done
```
- `step(action)`: executes action (0 = skip, 1 = take).
- Gives positive reward only if item fits.
- Penalizes heavily (-5) if trying to take an item that doesn't fit.
- Moves to next item regardless.
- Returns next state, reward, and whether episode is finished.

#### 3.2. Policy Network
A small neural network that outputs probabilities of taking or skipping the current item.

```python
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)   # 5 → 32
        self.fc2 = nn.Linear(hidden_dim, output_dim)   # 32 → 2 (skip/take)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)  # probabilities summing to 1
```
- Simple 2-layer MLP.
- Output: `[prob_skip, prob_take]`.

#### 3.2. Training with REINFORCE (`train()` function)
```python
def train():
    # Generate random problem instance
    N_ITEMS = 20
    np.random.seed(42)
    values = np.random.randint(10, 100, N_ITEMS)
    w1s = np.random.randint(1, 15, N_ITEMS)
    w2s = np.random.randint(1, 15, N_ITEMS)
    CAP1 = 50
    CAP2 = 50
```
- Fixed seed for reproducibility.
- 20 items, two capacities of 50 each.

```python
env = MDKPEnv(values, w1s, w2s, CAP1, CAP2)
    policy = PolicyNetwork(input_dim=5, hidden_dim=32, output_dim=2)
    optimizer = optim.Adam(policy.parameters(), lr=0.01)
    
    gamma = 0.99
    num_episodes = 5000
```
- Create environment and policy network.
- Adam optimizer with learning rate 0.01.
- Discount factor $γ = 0.99$.

```python
episode_rewards_history = []

    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []   # store log probabilities of chosen actions
        rewards = []     # store rewards

        # Collect one full trajectory (one complete knapsack filling)
        while True:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            probs = policy(state_tensor)
            
            # Sample action from probability distribution
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            
            next_state, reward, done = env.step(action.item())
            
            log_probs.append(m.log_prob(action))  # needed for gradient
            rewards.append(reward)
            state = next_state
            
            if done:
                break
```
For each episode:
- Reset environment
- Roll out a full sequence of decisions
- Save log-probability of each chosen action and corresponding reward

```python
episode_rewards_history.append(env.total_value)  # actual objective value
        
        # Compute discounted returns (future rewards)
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        
        # Normalize returns (very important for stable training)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
```
- Calculate discounted cumulative rewards (returns) from the end.
- Normalize them (zero mean, unit variance) → reduces variance in policy gradient.

```python
# REINFORCE loss: -log_prob * return (we want to maximize return)
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
```
- Core of REINFORCE: gradient ascent on expected return.
- Negative sign because we minimize loss, but want to increase probability of good actions.

```python
if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards_history[-100:])
            print(f"Episode {episode+1}, Average Value: {avg_reward:.2f}")
```
- Print average total value over last 100 episodes every 100 episodes.

#### 3.3. Testing the Trained Policy (`test()` function)
```python
def test(policy, env):
    state = env.reset()
    print("\n--- test the final policy ---")
    actions_taken = []
    
    while True:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():                     # no gradients during testing
            probs = policy(state_tensor)
            action = torch.argmax(probs).item()    # greedy: pick most probable action
            
        actions_taken.append(action)
        state, _, done = env.step(action)
        if done:
            break
            
    print(f"final sequence of actions_taken: {actions_taken}")
    print(f"final total_value: {env.total_value}")
    print(f"remaining capacity: {env.current_cap1}, {env.current_cap2}")
```
- Runs one episode using **greedy** policy (always choose highest-probability action).
- Prints which items were taken (1) or skipped (0), final value, and remaining capacities.

#### 3.4. Verification and Benchmarking (`verify_result()`)
```python
def verify_result(env, actions, final_value):
```
- Takes the sequence of actions from testing and compares results.

**Part 1: Check validity**
- Recomputes total value and weights from selected items.
- Confirms no capacity violation.

**Part 2: Greedy benchmark**
```python
ratios = []
    for i in range(env.n_items):
        w_sum = env.w1[i] + env.w2[i]
        ratio = env.values[i] / w_sum if w_sum > 0 else 0
        ratios.append((ratio, i))
    
    ratios.sort(key=lambda x: x[0], reverse=True)
```
- Simple heuristic: sort items by value / (w1 + w2).
- Fill knapsack greedily with highest ratio items first.


**Part 3: Exact optimal solution using `PuLP`**
```python
prob = pulp.LpProblem("MDKP_Optimal", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("item", range(env.n_items), cat='Binary')

    prob += pulp.lpSum([env.values[i] * x[i] for i in range(env.n_items)])

    prob += pulp.lpSum([env.w1[i] * x[i] for i in range(env.n_items)]) <= env.cap1_max
    prob += pulp.lpSum([env.w2[i] * x[i] for i in range(env.n_items)]) <= env.cap2_max

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    optimal_val = pulp.value(prob.objective)
```
- Formulates the problem as 0/1 integer linear program.
- Solves it exactly using CBC solver (included with PuLP).
- Compares RL result to true optimum and computes gap percentage.

#### 3.5. Main Execution Block
```python
if __name__ == "__main__":
    trained_policy, env = train()
    test(trained_policy, env)
    
    # Re-run test to capture actions again (since test() doesn't return them)
    state = env.reset()
    actions_taken = []
    while True:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            probs = trained_policy(state_tensor)
            action = torch.argmax(probs).item()
        actions_taken.append(action)
        state, _, done = env.step(action)
        if done: break

    verify_result(env, actions_taken, env.total_value)
```
- Trains the agent.
- Tests it.
- Repeats test to get action list.
- Runs full verification.


# 4. Performance Evaluation and Conclusion











