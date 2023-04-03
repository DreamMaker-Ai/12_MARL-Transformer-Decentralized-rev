import random

import ray
import gym
import tensorflow as tf
import numpy as np
from collections import deque

from battlefield_strategy import BattleFieldStrategy

from model import MarlTransformerDecentralized as MarlTransformerModel
from sub_models import CNNModel as CNNModel
from sub_models import MultiHeadAttentionModel as MultiHeadAttentionModel
from sub_models import QLogitModel as QLogitModel

from utils_gnn import get_alive_agents_ids
from utils_transformer import make_id_mask as make_mask
from utils_transformer import buffer2per_agent_list, per_agent_list2input_list, get_td_mask


# @ray.remote
@ray.remote(num_cpus=1, num_gpus=0)
class Actor:
    def __init__(self, pid, epsilon):
        self.pid = pid

        # Make a copy of environment
        self.env = BattleFieldStrategy()
        self.action_space_dim = self.env.action_space.n

        # Make a q_network
        self.policy = MarlTransformerModel(config=self.env.config,
                                           cnn_model=CNNModel,
                                           multihead_attention_model=MultiHeadAttentionModel,
                                           qlogit_model=QLogitModel)

        self.epsilon = epsilon
        self.gamma = self.env.config.gamma
        self.n_frames = self.env.config.n_frames
        self.state_shape = (self.env.config.grid_size,
                            self.env.config.grid_size,
                            self.env.config.observation_channels * self.n_frames)

        # Define local buffer
        self.buffer = []

        # Initialize environment
        ### The followings are reset in 'reset_states'
        self.frames = None  # For each agent in env
        self.prev_actions = None

        self.alive_agents_ids = None  # For all agents, including dummy ones
        self.padded_states = None
        self.padded_prev_actions = None
        self.masks = None

        self.episode_reward = None
        self.step = None

        ### Initialize above Nones
        observations = self.env.reset()
        self.reset_states(observations)

        self.policy(self.padded_states, self.masks, training=True)  # build

    def reset_states(self, observations):
        # TODO prev_actions
        """
        Note: batch=1 for the test

        alive_agents_ids: list of alive agent id

        For agents in Env (Alibe egent)
             each agent stacks observations n-frames in channel-dims
             -> observations[red.id]: (grid_size,grid_size,channels)

             -> generate deque of length=n_frames
             self.frames[red.id]: deque[(grid_size,grid_size,channels),...]

         self.padded_states: (For alive, dead, and dummy agents)
            [(1,grid_size,grid_size,channels*n_frames),...]=[(1,15,15,6),...], len=n=15

         self.prev_actions: int (TODO)

         self.masks: [(1,1,n),...]=[(1,1,15),...], len=n=15
        """

        # alive_agents_ids: list of alive agent id, [int,...], len=num_alive_agents
        self.alive_agents_ids = get_alive_agents_ids(env=self.env)

        # States with padding
        self.frames = {}
        self.padded_states = []
        self.prev_actions = {}

        for idx in range(self.env.config.max_num_red_agents):
            agent_id = 'red_' + str(idx)  # 'red_i', i=0,...,n-1

            if idx in self.alive_agents_ids:
                self.frames[agent_id] = \
                    deque([observations[agent_id]] * self.n_frames, maxlen=self.n_frames)

                padded_state = \
                    np.concatenate(self.frames[agent_id], axis=2).astype(np.float32)

            else:
                padded_state = np.zeros((self.state_shape[0],
                                         self.state_shape[1],
                                         self.state_shape[2])).astype(np.float32)

            padded_state = \
                np.expand_dims(padded_state, axis=0)  # add batch_dim, (1,g,g,ch*n_frames)

            self.padded_states.append(padded_state)

        # Get mask for the padding
        self.masks = make_mask(alive_agents_ids=self.alive_agents_ids,
                               max_num_agents=self.env.config.max_num_red_agents)
        # [(1,1,n),...], len=n

        # reset episode variables
        self.episode_reward = 0
        self.step = 0

    def rollout(self, current_weights):
        """
        rolloutを,self.env.config.actor_rollout_steps回（=10回）実施し、
        各経験の優先度（TD error）と経験（transitions）を格納

        :return td_errors, transitions, self.pid (process id)
                td_errors: (self.env.config.actor_rollout_steps,)=(10,)
                transitions=[transition,...], len=self.env.config.actor_rollout_steps=100
                    transition =
                        (
                            self.padded_states,  # [(1,g,g,ch*n_frames), ...], len=n=15
                            padded_actions,  # [(1,), ...], len=n
                            padded_rewards,  # [(1,), ...], len=n
                            next_padded_states,  # [(1,g,g,ch*n_frames), ...], len=n
                            padded_dones,  # [(1,), ...], len=n, bool
                            self.masks,  # [(1,1,n), ...], len=n, bool
                        )
        """
        # 重みを更新
        self.policy.set_weights(weights=current_weights)

        # Rolloutをenv.config.actor_rollout_steps回実施し、local bufferにtransitionを一時保存

        for _ in range(self.env.config.actor_rollout_steps):

            q_logits, _ = self.policy(self.padded_states, self.masks, training=False)
            # q_logits: [(1,action_dim),...], len=n=15
            # scores: [score_0, score_2]=[[(1,num_heads,n),...], [(1,num_heads,n,...)]

            # get alive_agents & all agents actions. action=0 <- do nothing
            actions = {}  # For env()
            padded_actions = []  # For transition()

            for i in range(self.env.config.max_num_red_agents):
                agent_id = 'red_' + str(i)

                q_logit = q_logits[i]  # (1,action_dim)
                act = np.argmax(q_logit[0]).astype(np.uint8)  # int

                if (i in self.alive_agents_ids) and (np.random.rand() < self.epsilon):
                    act = np.random.randint(low=0,
                                            high=self.action_space_dim,
                                            dtype=np.uint8)

                actions[agent_id] = act

                padded_actions.append(np.array([actions[agent_id]]))  # append (1,)

            # One step of Lanchester simulation, for alive agents in env
            next_obserations, rewards, dones, infos = self.env.step(actions)

            # Make next_agents_states and next_alive_agents_ids,
            # including dummy ones
            next_alive_agents_ids = get_alive_agents_ids(env=self.env)

            next_padded_states = []

            for idx in range(self.env.config.max_num_red_agents):
                agent_id = 'red_' + str(idx)

                if idx in next_alive_agents_ids:
                    self.frames[agent_id].append(next_obserations[agent_id])
                    next_padded_state = \
                        np.concatenate(self.frames[agent_id], axis=2).astype(np.float32)
                    # (g,g,ch*n_frames)

                else:
                    next_padded_state = np.zeros((self.state_shape[0],
                                                  self.state_shape[1],
                                                  self.state_shape[2])).astype(np.float32)
                    # (g,g,ch*n_frames)

                next_padded_state = \
                    np.expand_dims(next_padded_state, axis=0)  # (1,g,g,ch*n_frames)

                next_padded_states.append(next_padded_state)

            # Get next mask for the padding
            next_masks = \
                make_mask(
                    alive_agents_ids=next_alive_agents_ids,
                    max_num_agents=self.env.config.max_num_red_agents
                )  # [(1,,n), ...], len=n

            # 終了判定
            if self.step > self.env.config.max_steps:

                for idx in self.alive_agents_ids:
                    agent_id = 'red_' + str(idx)
                    dones[agent_id] = True

                dones['all_dones'] = True

            # agents_rewards and agents_dones, including dead and dummy ones
            agents_rewards = []
            agents_dones = []

            for idx in range(self.env.config.max_num_red_agents):

                if idx in self.alive_agents_ids:
                    agent_id = 'red_' + str(idx)
                    agents_rewards.append(float(rewards[agent_id]))
                    agents_dones.append(dones[agent_id])
                else:
                    agents_rewards.append(0.0)
                    agents_dones.append(True)

            # Update episode rewards
            self.episode_reward += np.sum(agents_rewards)

            padded_rewards = []
            padded_dones = []

            for idx in range(self.env.config.max_num_red_agents):
                padded_rewards.append(np.array([agents_rewards[idx]]))  # append (1,)
                padded_dones.append(np.array([agents_dones[idx]]))  # append (1,). bool

            # Append to buffer
            transition = (
                self.padded_states,  # [(1,g,g,ch*n_frames), ...], len=n
                padded_actions,  # [(1,), ...], len=n
                padded_rewards,  # [(1,), ...], len=n
                next_padded_states,  # [(1,g,g,ch*n_frames), ...], len=n
                padded_dones,  # [(1,), ...], len=n, bool
                self.masks,  # [(1,1,n), ...], len=n, bool
            )

            self.buffer.append(transition)  # append to local buffer

            if dones['all_dones']:
                # Reset env()
                observations = self.env.reset()
                self.reset_states(observations)
            else:
                self.alive_agents_ids = next_alive_agents_ids
                self.padded_states = next_padded_states
                self.masks = next_masks

                self.step += 1

        if self.env.config.prioritized_replay:
            """ 各transitionの初期優先度（td_error）の計算. """

            # transition in buffer to per_agent_list
            # agent_state = [state list of agent_1=[(1,15,15,6), ...], len=100,
            #                state list of agent_2=[(1,15,15,6), ...], len=100,
            #                       ... ], len=n
            # agent_action = [action list of agent_1=[(1,), ..., len=100,
            #                 action list of agent_2=[(1,), ..., len=100,
            #                       ... ], len=n
            # agent_mask = [mask list of agent_1= [(1,1,15), ...], len=100,
            #               mask list of agent_2= [(1,1,15), ...], len=100,
            #                       ... ], len=n

            agent_state, agent_action, agent_reward, agent_next_state, agent_done, agent_mask = \
                buffer2per_agent_list(self)

            # per_agent_list -> input list to policy network
            # states: [(b,g,g,ch*n_frames), ...], len=n=15
            # actions: [(b,), ...], len=n
            # rewards: [(b,), ...], len=n
            # next_states: [(b,g,g,ch*n_frames), ...], len=n
            # dones: [(b,), ...], len=n, bool
            # masks: [(b,1,n), ...], len=n, bool

            states, actions, rewards, next_states, dones, masks = \
                per_agent_list2input_list(self, agent_state, agent_action, agent_reward,
                                          agent_next_state, agent_done, agent_mask)

            training = False
            next_q_values, _ = self.policy(next_states, masks, training)
            # [(b,action_dim), ...], len=n

            next_q_values = tf.stack(next_q_values, axis=1)  # (b,n,action_dim)=(100,15,5)

            next_actions = tf.argmax(next_q_values, axis=-1)  # (100,15)
            next_actions = tf.cast(next_actions, dtype=tf.int32)
            next_actions_onehot = \
                tf.one_hot(next_actions, depth=self.env.config.action_dim)  # (100,15,5)

            next_maxQ = tf.reduce_sum(
                next_q_values * next_actions_onehot, axis=-1
            )  # (100,15)

            t_rewards = tf.stack(rewards, axis=1)  # (b,n)=(100,15)
            t_rewards = tf.cast(t_rewards, dtype=tf.float32)  # (100,15)

            t_dones = tf.stack(dones, axis=1)  # (b,n)=(100,15), bool
            t_dones = tf.cast(t_dones, dtype=tf.float32)  # (100,15)

            TQ = t_rewards + self.gamma * (1 - t_dones) * next_maxQ  # (b,n)=(100,15)

            q_values, _ = self.policy(states, masks, training)  # [(b,action_dim), ...], len=n
            q_values = tf.stack(q_values, axis=1)  # (b,n,action_dim)=(100,15,5)

            q_actions = tf.stack(actions, axis=1)  # (b,n)=(100,15)
            q_actions = tf.cast(q_actions, dtype=tf.int32)
            q_actions_onehot = tf.one_hot(q_actions, depth=self.env.config.action_dim)
            # (b,n,action_dim)=(100,15,5)

            Q = tf.reduce_sum(q_values * q_actions_onehot, axis=-1)  # (b,n)=(100,15)

            td_errors = TQ - Q  # (b,n)=(100,15)

            float_mask = get_td_mask(self.env.config, masks)  # (b,n), float32

            masked_td_errors = td_errors * float_mask  # (b,n)=(100,15)
            masked_td_errors = \
                np.sum(masked_td_errors, axis=-1) / np.sum(float_mask, axis=-1)  # (b,)=(100,)

        else:
            masked_td_errors = np.ones((self.env.config.actor_rollout_steps,),
                                       dtype=np.float32)  # (b,)=(100,)

        transitions = self.buffer  # list, len=100
        self.buffer = []

        return masked_td_errors, transitions, self.pid
