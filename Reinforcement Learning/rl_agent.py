
import random
from typing import Optional

# Optional for optional type hints.
from kuimaze2 import RLProblem, State, Action
from kuimaze2.typing import Policy, QTable, VTable

T_MAX = 200  # Max steps allowed per episode

class RLAgent:
    """Implementation of the Q-learning algorithm for reinforcement learning.

    Attributes:
        env (RLProblem): The reinforcement learning problem environment.
        gamma (float): The discount factor, influencing the agent's view on future rewards.
        alpha (float): The learning rate, determining how much new information affects learned information.
        q_table (dict): Nested dictionary that stores Q-values for each state-action pair.
    """

    def __init__(self, env: RLProblem, gamma: float = 0.9, alpha: float = 0.1):
        # env is an instance of RLProblem class.
        """Initialize the RLAgent with an environment, discount factor, and learning rate."""
        self.env = env
        self.gamma = gamma  # Discount factor for the future rewards
        self.alpha = alpha  # Learning rate for Q-value updates
        self.init_q_table()  # Initialize the Q-table

    def init_q_table(self) -> None:
        """Create and initialize the Q-table with zero values for all state-action pairs."""
        self.q_table = {
            # for all directions we set the initial Q-value to zero.
            state: {action: 0.0 for action in self.env.get_action_space()}
            for state in self.env.get_states()
        }
        
    
    # This method selects an action using the epsilon-greedy strategy. With probability epsilon, it chooses a random action (exploration); 
    # otherwise, it selects the action with the highest Q-value for the given state (exploitation).
    def epsilon_greedy_action(self, state, epsilon=0.1):
        """Choose an action based on epsilon-greedy strategy to balance exploration and exploitation.

        Args:
            state (State): The current state from which the action is chosen.
            epsilon (float): The probability of selecting a random action for exploration.

        Returns:
            Action: The chosen action.
        """
        if random.random() < epsilon:
            return random.choice(list(self.env.get_action_space()))
        else:
            return max(self.q_table[state], key=self.q_table[state].get)

    def learn_policy(self) -> Policy:
        """Learn the optimal policy using the Q-learning algorithm through multiple episodes.

        Returns:
            Policy: A dictionary mapping each state to an optimal action based on learned Q-values.
        """
        epsilon = 1.0  # Initial epsilon for exploration
        min_epsilon = 0.01  # Minimum epsilon value after decay
        decay_rate = 0.995  # Decay rate of epsilon per episode

        # This is purely for the purpose of repeating the process 100 times without the need to reference the loop counter.
        # Using _ is a clear way to convey that the actual value of the loop variable is irrelevant.
        for _ in range(100):  # Number of episodes
            state = self.env.reset()
            for _ in range(T_MAX):
                action = self.epsilon_greedy_action(state, epsilon)  # Choose action
                next_state, reward, episode_finished = self.env.step(action)  # Take action
                old_q = self.q_table[state][action]  # Store the old Q-value
                epsilon = max(min_epsilon, epsilon * decay_rate)  # Decay epsilon

                if next_state is not None:
                    next_max = max(self.q_table[next_state].values())
                else:
                    next_max = 0  # No future rewards if next state is terminal

                # Update Q-value
                # self.alpha = learning rate.
                # self.gamma = discount factor.
                self.q_table[state][action] = old_q + self.alpha * (reward + self.gamma * next_max - old_q)

                if episode_finished:
                    break
                state = next_state

        return self.extract_policy()

    def extract_policy(self) -> Policy:
        """Extract the policy from Q-values by choosing the best action for each state.

        Returns:
            Policy: The extracted policy mapping each state to an optimal action.
        """
        # key=actions.get tells the max function to use the Q-values (i.e., the values in the dictionary)
        # as the basis for comparison rather than the dictionary keys (which are actions).
        # In other words, max(actions, key=actions.get) finds the action that maximizes the Q-value in the actions dictionary.
        # So, here we create a new dictionary called 'Policy' where the key are the states, the values are the actions that have the highest Q-values for those
        # states.
        return {
            state: max(actions, key=actions.get)
            for state, actions in self.q_table.items()
        }

    # This method is a placeholder for rendering the state of the algorithm, the environment, and the policy. 
    # The actual rendering implementation is not provided here.
    def render(
        # self refers to the instance of the 'RLAgent' class. 
        self,
        # It can be either an instance of State class or None.
        # This allows for flexibility in how the 'render' method is called, making it possible to omit the
        # 'current_state' argument if it is not necessary for the rendering process.
        current_state: Optional[State] = None,
        action: Optional[Action] = None,
        values: Optional[VTable] = None,
        q_values: Optional[QTable] = None,
        policy: Optional[Policy] = None,
        # *args collects any additional positional arguments that are not explicitly defined in the method signature.
        # It allows for passing a variable number of positional arguments to the method.
        # **kwargs collects any additional keyword arguments that are not explicitly defined in the method signature.
        *args, **kwargs,
    ) -> None:
        """Visualize the state of the algorithm, the current environment, and the policy."""
        """
        You might want to allow users to specify additional visualization settings, like the color of the state representation or whether to display a grid.
        debug_info = kwargs.get('debug_info', None)  # Additional debug info, default to None

        # Example usage for **kwargs
        agent.render(policy=policy, debug_info={"steps": 150, "rewards": 200})

        In this example, render looks for debug_info in kwargs and prints it if available, allowing for flexible debugging 
        or additional information display during rendering.

        # Example usage for *args
        state1 = State()  # Assuming State() creates a state object
        state2 = State()
        state3 = State()
        agent.render(current_state=state1, state2, state3, title="State Comparison")

        *args allows you to pass any number of positional arguments after the explicitly named parameters.
        
        In the usage example, state2 and state3 are passed as additional states to the render method.
        The render method processes these states by iterating over args, enabling flexible handling of multiple 
        states without modifying the method signature.
        """

if __name__ == "__main__":
    from kuimaze2 import Map

    MAP = """
    ...G
    .#.D
    S...
    """
    map = Map.from_string(MAP)
    env = RLProblem(map, action_probs=dict(forward=0.8, left=0.1, right=0.1, backward=0.0), graphics=True)

    agent = RLAgent(env, gamma=0.9, alpha=0.1)
    policy = agent.learn_policy()
    print("Policy found:", policy)
    agent.render(policy=policy, use_keyboard=True)


