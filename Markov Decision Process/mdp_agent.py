#!/usr/bin/env python3 # --> This line is called "shebang".
# It tells the os to use the Python interpreter at the specified 
# path to run the script. 

'''The code defines three classes to solve a MDP using value iteration
and policy iteration algortihms. Let's break it down line by line'''

import random
from kuimaze2 import MDPProblem, Map
from kuimaze2.typing import Policy

# In this class, the common funcitonalities needed by both agents are encapsulated.
class MDPAgent:
    """
    A base agent for solving Markov Decision Processes.
    """

    # We here initialize the agent with a specific MDP environment, we set discount factor gamma to 0.9; and epsilon value to 0.01, which 
    # represents the difference between scores of subsequent steps. Epsilen determines, when the iterative algorithm has converged.
    def __init__(self, env: MDPProblem, gamma: float = 0.9, epsilon: float = 0.01):
        """
        Initializes the MDP Agent with an environment, gamma discount factor, and epsilon threshold.
        """

        # env, here, is an instance of MDDProblem, which provides details such as states, actions, rewards, and state transitions.
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon

    # For each action possible in the given state, it calculates the total expected utility by summing over all possible next states. 
    # This sum incorporates the probability of each next state given the action, the reward associated with moving to the next state, 
    # and the discounted value of the next state as estimated in V.
    def get_action_values(self, state, V):
        """
        Calculates the expected utility of all possible actions from a given state using the current value estimates.
        """

        # We initialize an empty dictionary to store the computed values
        # for each action.
        action_values = {}

        # Loop through each possible action for the given state.
        for action in self.env.get_actions(state):
            
            # V here is a dictionary mapping states to their current estimated values.
            # The sum of prob multip. by (reward + discount factor multip. by the value of
            # state.
            
            total = sum(prob * (self.env.get_reward(next_state) + self.gamma * V[next_state])
                        for next_state, prob in self.env.get_next_states_and_probs(state, action))
            action_values[action] = total

        # We return a dictionary, where each key is an action and the value is the calculated expected utility for that action.
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

        # We create two dictionaries to store the colors and texts 
        # as we iterate over the states. Here, states are stored as a 'key' 
        # and name of the action and color related to it are the values.
        state_colors = {}
        state_texts = {}
        for state, action in policy.items():
            if action is not None:
                state_colors[state] = action_colors[action.name]
                state_texts[state] = action.name
            else:
                state_colors[state] = 0  # Default color for terminal or undefined states
                state_texts[state] = 'None'
                
        if hasattr(self.env, 'render'):
            self.env.render(square_colors=state_colors, square_texts=state_texts)
        else:
            print("Rendering not implemented or render method not available in the environment.")



# The first method used is value iteration. 
class ValueIterationAgent(MDPAgent):
    def find_policy(self) -> Policy:
        """
        Implements the value iteration algorithm to find the optimal policy.
        """

        # Again, two dictionaries are created. 
        # We first initialize a dictionary 'V', where each key is a state from the set of all states in the MDP.
        # Each state is initially mapped to a value of 0. 
        V = {state: 0 for state in self.env.get_states()}
        # This dictionary is initilalized to store the optimal action for each state as the algorithm progresses.
        policy = {state: None for state in self.env.get_states()}  # Initialize all states in policy
        while True:
            # We will use delta to see if the algorithm has converged.
            delta = 0
            for state in self.env.get_states():
                # If the state is a terminal state, do not break, but continue.
                if self.env.is_terminal(state):
                    continue
                # We call the function to create a dictionary for each action.
                action_values = self.get_action_values(state, V)
                # This line finds the highest value among the expected utilities of all actions available from the current state.
                # .values() can be found under the class action_values: self.values = list the values.
                max_value = max(action_values.values())
                # Here, we measure how much the values of the state has changed in this iteration.
                delta = max(delta, abs(V[state] - max_value))
                # The value of the state is set to max_value. Thus, we update the value of the state.
                V[state] = max_value
                # key = ... , it lets you tell which part of the elemnents to consider for determining "maximality".
                # .get() is a function available in default python libraries, which returns the value for a given key from the 
                # dictionary.
                policy[state] = max(action_values, key=action_values.get)
                # Check for the convergence.
            if delta < self.epsilon:
                break
        return policy

# The second method used is policy iteration.
# 
class PolicyIterationAgent(MDPAgent):
    def find_policy(self) -> Policy:
        """
        Implements the policy iteration algorithm to compute the optimal policy.
        """
        # random.choice() function is used.
        policy = {state: (random.choice(self.env.get_actions(state)) if not self.env.is_terminal(state) else None)
                  for state in self.env.get_states()}
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
            # We leave the loop, when the algorithm has converged and is_policy_stable is set to true.
            is_policy_stable = True
            for state in V:
                if self.env.is_terminal(state):
                    continue
                old_action = policy[state]
                action_values = self.get_action_values(state, V)
                best_action = max(action_values, key=action_values.get)
                # Policy optimization step. The loop continues, till we find the best possible policy.
                if action_values[best_action] > action_values[old_action]:
                    policy[state] = best_action
                    is_policy_stable = False
        return policy
    
    '''

    Q: How does policy and value iteration methods differ?
    A: VI iteratively updates the values of each state until they converge, and the policy is derived from these values.
       For each state, the maximum expected utility is calculated, using the Bellman Optimality Equation. Then we update the state value to
       this maximum value. Repeat until the changes in values are below a small threshold. Once, the state values are stabilized, we select for the
       each state the action, that maximizes the expected utility. Each iteration involves computation over all actions for all states!

       PI alternates between evaluating a given policy and improving it. It starts with an arbitrary (keyfi) policy and iteratively makes 
       this policy optimal by using a two-step process: policy evaluation and policy improvement. Policy evaluation is done by
       solving the system of linear equations defined by the Bellman Equation for the current policy. We repeat the evaluation and improvement,
       until the policy no longer changes. Policy Iteration often requires fewer iterations than value iteration because each policy improvement 
       step is guaranteed to return a policy that is as good or better than the previous one. However, each policy evaluation phase can be 
       computationally intense, especially in environments with many states, as it may involve solving many linear equations.

       VI is a quick & direct method.
       PI is overall faster, but may require higher compuational force.
       
    '''

# A map instance is created.
if __name__ == "__main__":
    MAP = """
    ...G
    .#.D
    S...
    """
    
    # We render the final policy.
    map_instance = Map.from_string(MAP)
    env = MDPProblem(map_instance, graphics=True)
    agent = ValueIterationAgent(env)
    policy = agent.find_policy()
    print("Policy found:", policy)
    agent.render(policy=policy)
