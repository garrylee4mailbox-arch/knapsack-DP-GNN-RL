import numpy as np
import random
import time
import pandas as pd

class KnapsackEnvironment:
    """
    Knapsack environment class
    Simulates state transitions and reward feedback for the knapsack problem
    """

    def __init__(self, weights, values, capacity):
        self.weights = weights  # list of item weights
        self.values = values  # list of item values
        self.capacity = capacity  # maximum knapsack capacity
        self.n_items = len(weights)  # number of items

    def reset(self):
        """Reset the environment, return the initial state (current item index, remaining capacity)"""
        # State definition: (current item index, current remaining capacity)
        return (0, self.capacity)

    def step(self, state, action):
        """
        Execute an action, return (new_state, reward, done)
        state: (item_index, current_capacity)
        action: 0 (do not take) or 1 (take)
        """
        item_idx, current_cap = state

        # Check if all items have been considered
        if item_idx >= self.n_items:
            return state, 0, True

        reward = 0

        # Action logic
        if action == 1:  # attempt to take the current item
            if self.weights[item_idx] <= current_cap:
                # Fits: receive value, capacity decreases
                reward = self.values[item_idx]
                next_cap = current_cap - self.weights[item_idx]
            else:
                # Does not fit: give a severe penalty, capacity remains unchanged
                # (or this could be considered an illegal action)
                # For simplicity we allow the action but give a penalty and do not change capacity
                reward = -100  # penalty factor to teach the agent not to try illegal picks
                next_cap = current_cap
        else:  # action == 0 (do not take)
            reward = 0
            next_cap = current_cap

        # Move to the next item
        next_state = (item_idx + 1, next_cap)

        # Check whether this is the last step
        done = (item_idx + 1 == self.n_items)

        return next_state, reward, done


def train_q_learning(weights, values, capacity, episodes=10000, alpha=0.1, gamma=0.95):
    env = KnapsackEnvironment(weights, values, capacity)
    q_table = np.zeros((env.n_items + 1, capacity + 1, 2))

    # --- Core change: introduce dynamic exploration rate ---
    epsilon = 1.0  # initial exploration rate: 100% random actions
    epsilon_min = 0.01  # minimum exploration rate: keep 1% randomness
    epsilon_decay = 0.9995  # decay factor: reduce exploration slightly each episode

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            item_idx, current_cap = state

            # Policy: random exploration early, exploit experience later
            if random.uniform(0, 1) < epsilon:
                action = random.choice([0, 1])  # random action
            else:
                action = np.argmax(q_table[item_idx, current_cap])  # choose action with highest Q value

            next_state, reward, done = env.step(state, action)
            next_item_idx, next_cap = next_state

            old_value = q_table[item_idx, current_cap, action]

            # Get the max Q value for the next step
            if next_item_idx < env.n_items:
                next_max = np.max(q_table[next_item_idx, next_cap])
            else:
                next_max = 0

            # Q-table update rule
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            q_table[item_idx, current_cap, action] = new_value
            state = next_state

        # After each episode, reduce exploration rate slightly
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    return q_table


def solve_knapsack(weights, values, capacity, q_table):
    """
    Use the trained Q-table to solve the problem
    """
    env = KnapsackEnvironment(weights, values, capacity)
    state = env.reset()
    done = False

    total_value = 0
    total_weight = 0
    selected_items = []

    print("\n--- Agent decision process ---")

    while not done:
        item_idx, current_cap = state

        # Directly choose the action with the highest Q value (greedy policy)
        action = np.argmax(q_table[item_idx, current_cap])

        # Logical check: if Q suggests taking but it doesn't fit, we correct it
        # (because if training is insufficient Q-table may suggest illegal actions)
        if action == 1 and weights[item_idx] > current_cap:
            action = 0  # force correction (or let it fail)

        if action == 1:
            print(f"Item {item_idx} (wt {weights[item_idx]}, val {values[item_idx]}): -> 1")
            total_value += values[item_idx]
            total_weight += weights[item_idx]
            selected_items.append(item_idx)
        else:
            print(f"Item {item_idx} (wt {weights[item_idx]}, val {values[item_idx]}): -> 0")

        next_state, _, done = env.step(state, action)
        state = next_state

    return total_value, total_weight, selected_items


if __name__ == "__main__":


    # --- 1. Read data ---
    # Assume your file name is 'data.xlsx'; please change to your actual path
    data = pd.read_excel(r"C:\Users\lcg\Desktop\data.xlsx")
    data = data.dropna(subset=['weight3', 'value3'])
    # Assume your column names are 'weight3' and 'value3' (modify according to actual names)
    capacity = int(data.at[0,'cap3'])
    weights = data['weight3'].astype(int).tolist()
    values = data['value3'].astype(int).tolist()
    print(f"Total knapsack capacity: {capacity}")
    print(f"Item weights: {weights}")
    print(f"Item values: {values}")
    print("-" * 30)
    start = time.perf_counter()
    # 1. Training
    print("Q-Learning agent...")
    # Increasing the number of episodes can improve accuracy
    q_table = train_q_learning(weights, values, capacity, episodes=5000)
    print("Training completed.")

    # 2. Validate result
    final_val, final_weight, items = solve_knapsack(weights, values, capacity, q_table)
    end = time.perf_counter()
    print("-" * 30)
    print(f"Final selected item indices: {items}")
    print(f"Total weight: {final_weight}/{capacity}")
    print(f"Total value: {final_val}")
    print(f"Elapsed time: {end - start:.8f} seconds")