import random
from typing import Optional

from kuimaze2 import RLProblem, State, Action
from kuimaze2.typing import Policy, QTable, VTable


T_MAX = 200  # Max steps in episode


class RLAgent:
    """Implementation of Q-learning algorithm.

    With the provided code, the agent just walks randomly
    and does not update q-values correctly
    """

    def __init__(
        self,
        env: RLProblem,
        gamma: float = 0.9,
        alpha: float = 0.1,
    ):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.init_q_table()

    def init_q_table(self) -> None:
        """Create and initialize the q-table

        It is initialized as a dictionary of dictionaries;
        it can be used as 'self.q_table[state][action]'.
        """
        self.q_table = {
            state: {action: 0.0 for action in self.env.get_action_space()}
            for state in self.env.get_states()
        }

    def get_values(self) -> VTable:
        """Return the state values derived from the q-table"""
        return {
            state: max(q_values.values()) for state, q_values in self.q_table.items()
        }

    def render(
        self,
        current_state: Optional[State] = None,
        action: Optional[Action] = None,
        values: Optional[VTable] = None,
        q_values: Optional[QTable] = None,
        policy: Optional[Policy] = None,
        *args,
        **kwargs,
    ) -> None:
        """Visualize the state of the algorithm"""
        values = values or self.get_values()
        q_values = q_values or self.q_table
        # State values will be displayed in the squares
        sq_texts = (
            {state: f"{value:.2f}" for state, value in values.items()} if values else {}
        )
        # State-action value will be displayed in the triangles
        tr_texts = {
            (state, action): f"{value:.2f}"
            for state, action_values in q_values.items()
            for action, value in action_values.items()
        }
        # If policy is given, it will be displayed in the middle
        # of the squares in the "triangular" view
        actions = {}
        if policy:
            actions = {state: str(action) for state, action in policy.items()}
        # The current state and chosen action will be displayed as an arrow
        state_action = (current_state, action)
        if current_state is None or action is None:
            state_action = None
        self.env.render(
            *args,
            square_texts=sq_texts,
            square_colors=values,
            triangle_texts=tr_texts,
            triangle_colors=q_values,
            middle_texts=actions,
            state_action_arrow=state_action,
            wait=True,
            **kwargs,
        )

    def extract_policy(self) -> Policy:
        """Extract policy from Q-values"""
        # The provided implementation just chooses a random action for each state!
        # TODO: Do something reasonable to extract the policy from q_table
        # Maybe it would help to introduce a method 'get_best_action_for_state(state)'
        # and use it here?
        print("THIS IS JUST A RANDOM POLICY!!! WRITE A CORRECT VERSION YOURSELF!!!")
        policy = {
            state: random.choice(self.env.get_action_space())
            for state, action_values in self.q_table.items()
        }
        return policy

    def learn_policy(self) -> Policy:
        """Run Q-learning algoritm to learn a policy"""
        # The provided implementation runs only a single episode of random walk;
        # the actual q-learning will have to run many episodes.
        # It is a good idea to create a method that runs a single episode.

        total_reward = 0
        episode_finished = False
        t = 0
        # Print a table header
        print(
            f"{'State':^9}{'Action':^9}{'Next state':^11}{'Reward':>9}"
            f"{'Old Q':>9}{'Trial':>9}{'New Q':>9}",
        )
        # Reset the environment and get the initial state
        next_state = self.env.reset()
        path = [next_state]
        while not episode_finished and t < T_MAX:
            t += 1
            state = next_state
            # TODO: Choose an action using certain strategy
            # Here, we just move randomly...
            action = self.env.sample_action()
            next_state, reward, episode_finished = self.env.step(action)
            total_reward += reward
            if next_state is not None:
                path.append(next_state)

            # Remember the old q-value for printing it in the table
            old_q = self.q_table[state][action]
            # TODO: Implement the right way to determine the trial
            # This is not the right thing to do, just to show something
            trial = reward
            # Another silly idea
            # trial = t
            # TODO: Implement the TD update for q-value
            # And this is not the right way to update q-value
            self.q_table[state][action] = trial

            print(
                f"{str(state):^9}{str(action):^9}{str(next_state):^9}{reward:>9.2f}"
                f"{old_q:>9.2f}{trial:>9.2f}{self.q_table[state][action]:>9.2f}"
            )

            # Extract the current policy such that we can visualize it
            policy = self.extract_policy()
            self.render(current_state=state, action=action, path=path, policy=policy)

        if episode_finished:
            print(f"Episode finished in a terminal state after {t} steps.")
        else:
            print(
                "Episode did not reach any terminal state. Maximum time horizon reached!"
            )
        print("Total reward:", total_reward)
        return self.extract_policy()


if __name__ == "__main__":

    from kuimaze2 import Map
    from kuimaze2.map_image import map_from_image

    MAP = """
    ...G
    .#.D
    S...
    """
    map = Map.from_string(MAP)
    # map = map_from_image("./maps/normal/normal3.png")
    env = RLProblem(
        map,
        action_probs=dict(forward=0.8, left=0.1, right=0.1, backward=0.0),
        graphics=True,
    )

    agent = RLAgent(env, gamma=0.9, alpha=0.1)
    policy = agent.learn_policy()
    print("Policy found:", policy)
    agent.render(policy=policy, use_keyboard=True)
