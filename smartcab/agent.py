import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

import random


class QLearner:

    def __init__(self, actions, learning_rate=0.5, discount_rate=0.3, default_q_value=0, epsilon=0.1):
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.default_q_value = default_q_value
        self.epsilon = epsilon
        self.Q = {}

    def select_action(self, state):
        if QLearner._should_be_random(self.epsilon):
            best_action = random.choice(self.actions)
            return best_action

        return self._select_best_action(state)

    def learn(self, state, action, reward):
        if (state, action) not in self.Q:
            self.Q[(state, action)] = self.default_q_value

        residual_q = (1 - self.learning_rate) * self.Q[state, action]
        learned_q = self.learning_rate * (reward + self.discount_rate * self._get_max_q_value(state))

        self.Q[state, action] = residual_q + learned_q

    @staticmethod
    def _should_be_random(probability):
        return random.random() < probability

    def _get_q_value(self, state, action):
        return self.Q.get((state, action), self.default_q_value)

    def _get_max_q_value(self, state):
        best_action = self._select_best_action(state)
        max_q = self._get_q_value(state, best_action)
        return max_q

    def _select_best_action(self, state):
        max_q = self.default_q_value - 1
        best_action = None
        for action in self.actions:
            q = self._get_q_value(state, action)
            if q > max_q:
                max_q = q
                best_action = action
        return best_action


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        learning_rate = 0.5
        discount_rate = 0.3
        default_q_value = 0
        epsilon = 0.1
        self.q_learner = QLearner(Environment.valid_actions,
                                  learning_rate,
                                  discount_rate,
                                  default_q_value,
                                  epsilon)

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = tuple(self._get_state(inputs))
        
        # TODO: Select action according to your policy
        action = self.q_learner.select_action(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        self.q_learner.learn(self.state, action, reward)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def _get_state(self, inputs):
        states = inputs.items()
        states.append(("next_waypoint", self.next_waypoint))
        return states


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
