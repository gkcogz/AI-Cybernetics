#!/usr/bin/env python3
import random
from kuimaze2 import MDPProblem, Map
from kuimaze2.typing import Policy

class MDPAgent:
    """
    A base agent for solving Markov Decision Processes.
    """
    def __init__(self, env: MDPProblem, gamma: float = 0.9, epsilon: float = 0.01):
        """
        Initializes the MDP Agent with an environment, gamma discount factor, and epsilon threshold.
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon

    def get_action_values(self, state, V):
        """
        Calculates the expected utility of all possible actions from a given state using the current value estimates.
        """
        action_values = {}
        for action in self.env.get_actions(state):
            total = sum(prob * (self.env.get_reward(next_state) + self.gamma * V[next_state])
                        for next_state, prob in self.env.get_next_states_and_probs(state, action))
            action_values[action] = total
        return action_values

    def render(self, policy: Policy):
        """
        Visualizes the policy on the MDP environment. Colors and texts represent different actions at each state.
        """
        action_colors = {
            'UP': 0.25,  # Light green
            'DOWN': 0.50,  # Light red
            'LEFT': 0.75,  # Light blue
            'RIGHT': 1.00  # Light yellow
        }
        state_colors = {}
        state_texts = {}
        for state, action in policy.items():
            state_colors[state] = action_colors[action.name]
            state_texts[state] = action.name
        if hasattr(self.env, 'render'):
            self.env.render(square_colors=state_colors, square_texts=state_texts)
        else:
            print("Rendering not implemented or render method not available in the environment.")

class ValueIterationAgent(MDPAgent):
    def find_policy(self) -> Policy:
        """
        Implements the value iteration algorithm to find the optimal policy.
        """
        V = {state: 0 for state in self.env.get_states()}
        policy = {}
        while True:
            delta = 0
            for state in self.env.get_states():
                if self.env.is_terminal(state):
                    continue
                action_values = self.get_action_values(state, V)
                max_value = max(action_values.values())
                delta = max(delta, abs(V[state] - max_value))
                V[state] = max_value
                policy[state] = max(action_values, key=action_values.get)
            if delta < self.epsilon:
                break
        return policy

class PolicyIterationAgent(MDPAgent):
    def find_policy(self) -> Policy:
        """
        Implements the policy iteration algorithm to compute the optimal policy.
        """
        policy = {state: random.choice(self.env.get_actions(state)) for state in self.env.get_states() if not self.env.is_terminal(state)}
        V = {state: 0 for state in self.env.get_states()}
        is_policy_stable = False
        while not is_policy_stable:
            while True:
                delta = 0
                for state in V:
                    if self.env.is_terminal(state):
                        continue
                    v = V[state]
                    V[state] = sum(prob * (self.env.get_reward(next_state) + self.gamma * V[next_state])
                                   for next_state, prob in self.env.get_next_states_and_probs(state, policy[state]))
                    delta = max(delta, abs(v - V[state]))
                if delta < self.epsilon:
                    break
            is_policy_stable = True
            for state in V:
                if self.env.is_terminal(state):
                    continue
                old_action = policy[state]
                action_values = self.get_action_values(state, V)
                best_action = max(action_values, key=action_values.get)
                if action_values[best_action] > action_values[old_action]:
                    policy[state] = best_action
                    is_policy_stable = False
        return policy

if __name__ == "__main__":
    MAP = """
    ...G
    .#.D
    S...
    """
    map_instance = Map.from_string(MAP)
    env = MDPProblem(map_instance, graphics=True)
    agent = ValueIterationAgent(env)
    policy = agent.find_policy()
    print("Policy found:", policy)
    
    agent.render(policy=policy)

    
    input("Press Enter to exit...")  # Keeps the window open until Enter is pressed.
