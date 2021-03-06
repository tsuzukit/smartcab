from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

import math
import random
import csv


class QLearner:

    DEFAULT_Q_VALUE = 0
    EXPLORATION_RATE_HALF_LIFE_STEP = 300

    def __init__(self, actions, learning_rate=0.5, discount_factor=0.3, exploration_rate=0.1, greedy_exploration=False):
        self.previous_action = None
        self.previous_state = None
        self.previous_reward = None

        self.num_learn = 0
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.greedy_exploration = greedy_exploration
        self.Q = {}

    def select_action(self, state):
        exploration_rate = self._get_exploration_rate()
        if QLearner._should_be_random(exploration_rate):
            best_action = random.choice(self.actions)
            return best_action

        return self._select_best_action(state)

    def learn(self, state, action, reward):
        if not self._is_previous_parameter_set():
            self._set_previous_parameters(action, state, reward)
            return

        previous_q = self._calculate_previous_q(state)
        self.Q[(self.previous_state, self.previous_action)] = previous_q

        self.num_learn += 1.0
        self._set_previous_parameters(action, state, reward)

    @staticmethod
    def _should_be_random(probability):
        return random.random() < probability

    def _get_exploration_rate(self):
        if self.greedy_exploration:
            return self.exploration_rate * 1 / (1 + math.exp(self.num_learn - self.EXPLORATION_RATE_HALF_LIFE_STEP))

        return self.exploration_rate

    def _get_q_value(self, state, action):
        return self.Q.get((state, action), QLearner.DEFAULT_Q_VALUE)

    def _get_max_q_value(self, state):
        best_action = self._select_best_action(state)
        return self._get_q_value(state, best_action)

    def _calculate_previous_q(self, state):
        if (self.previous_state, self.previous_action) not in self.Q:
            self.Q[(self.previous_state, self.previous_action)] = QLearner.DEFAULT_Q_VALUE

        residual_q = (1 - self.learning_rate) * self.Q[(self.previous_state, self.previous_action)]
        learned_q = self.learning_rate * (self.previous_reward + self.discount_factor * self._get_max_q_value(state))
        return residual_q + learned_q

    def _select_best_action(self, state):
        max_q = QLearner.DEFAULT_Q_VALUE
        best_action = None
        for action in self.actions:
            q = self._get_q_value(state, action)
            if q > max_q:
                max_q = q
                best_action = action
        return best_action

    def _is_previous_parameter_set(self):
        return self.previous_action is not None and \
               self.previous_state is not None and \
               self.previous_reward is not None

    def _set_previous_parameters(self, action, state, reward):
        self.previous_action = action
        self.previous_state = state
        self.previous_reward = reward


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        learning_rate = 0.9
        discount_factor = 0.3
        exploration_rate = 0.1
        self.q_learner = QLearner(actions=Environment.valid_actions,
                                  learning_rate=learning_rate,
                                  discount_factor=discount_factor,
                                  exploration_rate=exploration_rate,
                                  greedy_exploration=True)

        # For statistics
        self.stats = []
        self.trial = 0
        self.net_reward = 0
        self.penalty = 0
        self.move = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

        # For statistics
        self.net_reward = 0
        self.penalty = 0
        self.move = 0
        self.trial += 1

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = self._get_state(inputs)
        
        # TODO: Select action according to your policy
        #action = random.choice(Environment.valid_actions)
        action = self.q_learner.select_action(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # record stats
        self._record_stats(reward, deadline)

        # TODO: Learn policy based on state, action, reward
        self.q_learner.learn(self.state, action, reward)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def _get_state(self, inputs):
        state_variables = inputs.items()
        state_variables.append(("next_waypoint", self.next_waypoint))
        state_variables.remove(("right", inputs["right"]))
        state = tuple(state_variables)
        return state

    def _record_stats(self, reward, deadline):
        self.net_reward += reward
        self.move += 1

        if LearningAgent._has_recieved_penalty(reward):
            self.penalty += 1

        if LearningAgent._has_reached_deadline(deadline):
            stats_data = self._create_stats_data(success=0, timeup=1)
            self.stats.append(stats_data)

        if LearningAgent._has_reached_destination(reward):
            stats_data = self._create_stats_data(success=1, timeup=0)
            self.stats.append(stats_data)

    def _create_stats_data(self, success, timeup):
        stats_data = {"trial": self.trial,
                      "move": self.move,
                      "reward": self.net_reward,
                      "penalty": self.penalty,
                      "success": success,
                      "timeup": timeup}
        return stats_data

    @staticmethod
    def _has_reached_deadline(deadline):
        return deadline == 0

    @staticmethod
    def _has_reached_destination(reward):
        return reward >= 12.0

    @staticmethod
    def _has_recieved_penalty(reward):
        return reward < 0

    def write_stats(self):
        keys = self.stats[0].keys()
        with open('result.csv', 'wb') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.stats)


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.01, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    # when everything finished, dump statistics to csv
    a.write_stats()


if __name__ == '__main__':
    run()
