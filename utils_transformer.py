import numpy as np
import copy
import tensorflow as tf


def make_id_mask(alive_agents_ids, max_num_agents):  # (1,1,n)
    """ masks: [(1,1,n), ...], len=n=max_num_agents """

    masks = []
    for _ in range(max_num_agents):
        masks.append(np.zeros((1, 1, max_num_agents)))

    for i in alive_agents_ids:
        for j in alive_agents_ids:
            masks[i][:, :, j] = 1

    for i in range(max_num_agents):
        masks[i] = masks[i].astype(bool)

    return masks


def make_mask(alive_agents_ids, max_num_agents):
    mask = np.zeros(max_num_agents)  # (n,)

    for i in range(len(alive_agents_ids)):
        mask[i] = 1

    mask = mask.astype(bool)  # (n,)
    mask = np.expand_dims(mask, axis=0)  # add batch_dim, (1,n)

    return mask


def buffer2per_agent_list(self):
    """
    self.buffer には、self.env.config.actor_rollout_steps = b (=100) 分の
    transition が入っている。

        transition in buffer to per_agent_list
        agent_state = [state list of agent_1=[(1,15,15,6), ...], len=100,
                       state list of agent_2=[(1,15,15,6), ...], len=100,
                              ... ], len=n
        agent_action = [action list of agent_1=[(1,), ..., len=100,
                        action list of agent_2=[(1,), ..., len=100,
                              ... ], len=n
        agent_mask = [mask list of agent_1= [(1,1,15), ...], len=100,
                      mask list of agent_2= [(1,1,15), ...], len=100,
                              ... ], len=n, bool
    """

    agent_state = []
    agent_action = []
    agent_reward = []
    agent_next_state = []
    agent_done = []
    agent_mask = []

    for _ in range(self.env.config.max_num_red_agents):
        agent_state.append([])
        agent_action.append([])
        agent_reward.append([])
        agent_next_state.append([])
        agent_done.append([])
        agent_mask.append([])

    for transition in self.buffer:
        for i in range(self.env.config.max_num_red_agents):
            # append agent_i state: (1,g,g,ch*n_frames)
            agent_state[i].append(transition[0][i])

            # append agent_i action: (1,)
            agent_action[i].append(transition[1][i])

            # append agent_i reward: (1,)
            agent_reward[i].append(transition[2][i])

            # append agent_i next_state: (1,g,g,ch*n_frames)
            agent_next_state[i].append(transition[3][i])

            # append agent_i done: (1,)
            agent_done[i].append(transition[4][i])

            # append agent_i mask: (1,1,n)
            agent_mask[i].append(transition[5][i])

    return agent_state, agent_action, agent_reward, agent_next_state, agent_done, agent_mask


def experiences2per_agent_list(self, experiences):
    """
    experiences には、batch (=32) 分の transition が入っている。

        transition in buffer to per_agent_list
        experience.states = [state of agent_1: (1,15,15,6),
                             state of agent_2: (1,15,15,6),,
                              ... ], len=n
        experience.actions = [action of agent_1: (1,),
                              action of agent_2: (1,),
                              ... ], len=n
        experience.masks: [mask of agent_1= (1,1,15),
                           mask of agent_2= (1,1,15),
                              ... ], len=n, bool
    """

    agent_state = []
    agent_action = []
    agent_reward = []
    agent_next_state = []
    agent_done = []
    agent_mask = []

    for _ in range(self.env.config.max_num_red_agents):
        agent_state.append([])
        agent_action.append([])
        agent_reward.append([])
        agent_next_state.append([])
        agent_done.append([])
        agent_mask.append([])

    for experience in experiences:
        for i in range(self.env.config.max_num_red_agents):
            # append agent_i state: (1,g,g,ch*n_frames)
            agent_state[i].append(experience.states[i])

            # append agent_i action: (1,)
            agent_action[i].append(experience.actions[i])

            # append agent_i reward: (1,)
            agent_reward[i].append(experience.rewards[i])

            # append agent_i next_state: (1,g,g,ch*n_frames)
            agent_next_state[i].append(experience.next_states[i])

            # append agent_i done: (1,)
            agent_done[i].append(experience.dones[i])

            # append agent_i mask: (1,1,n)
            agent_mask[i].append(experience.masks[i])

    return agent_state, agent_action, agent_reward, agent_next_state, agent_done, agent_mask


def per_agent_list2input_list(self, agent_state, agent_action, agent_reward,
                              agent_next_state, agent_done, agent_mask):
    """
    per_agent_list -> input list to policy network

        states: [(b,g,g,ch*n_frames), ...], len=n
        actions: [(b,), ...], len=n
        rewards: [(b,), ...], len=n
        next_states: [(b,g,g,ch*n_frames), ...], len=n
        dones: [(b,), ...], len=n, bool
        masks: [(b,1,n), ...], len=n, bool
    """
    # b=self.env.config.actor_rollout_step=100
    states = []  # [(b,g,g,ch*n_frames), ...], len=n
    actions = []  # [(b,), ...], len=n
    rewards = []  # [(b,), ...], len=n
    next_states = []  # [(b,g,g,ch*n_frames), ...], len=n
    dones = []  # [(b,), ...], len=n, bool
    masks = []  # [(b,1,n), ...], , len=n, bool

    for i in range(self.env.config.max_num_red_agents):
        states.append(np.concatenate(agent_state[i], axis=0))  # append (100,15,15,6)
        actions.append(np.concatenate(agent_action[i], axis=0))  # append (100,)
        rewards.append(np.concatenate(agent_reward[i], axis=0))  # append (100,)
        next_states.append(
            np.concatenate(agent_next_state[i], axis=0)
        )  # append (100,15,15,6)
        dones.append(np.concatenate(agent_done[i], axis=0))  # append (100,)
        masks.append(np.concatenate(agent_mask[i], axis=0))  # append (100,1,n)

    return states, actions, rewards, next_states, dones, masks


def get_td_mask(config, masks):
    # masks: [(b,1,n), ...], len=n
    # td_mask: tensor of agent alive or not, bool

    td_mask = []
    for i in range(config.max_num_red_agents):
        mask = masks[i]  # mask of agent_i, (b,1,n)
        float_mask = tf.cast(mask[:, :, i], tf.float32)  # agent_i alive or not, (b,1)
        td_mask.append(float_mask)

    # list -> tensor
    td_mask = tf.concat(td_mask, axis=1)  # (b,n)

    return td_mask


def make_padded_obs(max_num_agents, obs_shape, raw_obs):
    padded_obs = copy.deepcopy(raw_obs)  # list of raw obs
    padding = np.zeros(obs_shape)  # 0-padding of obs

    while len(padded_obs) < max_num_agents:
        padded_obs.append(padding)

    padded_obs = np.stack(padded_obs)  # stack to sequence (agent) dim  (n,g,g,ch*n_frames)

    padded_obs = np.expand_dims(padded_obs, axis=0)  # add batch_dim (1,n,g,g,ch*n_frames)

    return padded_obs


def make_next_states_for_q(alive_agents_ids, next_alive_agents_ids, raw_next_states, obs_shape):
    """
    Make the next states list to compute Q(s',a') bases on alive agents ids

    :param alive_agents_ids: list, len=a=num_alive_agents
    :param next_alive_agents_ids: list, len =a'=num_next_alive_agents
    :param raw_next_states: lsit of obs, len=a'
    :param obs_shape: (g,g,ch*n_frames)
    :return: list of padded next_states_for_q, corresponding to the alive_agents_list, len=n,
    """

    next_states_for_q = []
    padding = np.zeros(obs_shape)

    for idx in alive_agents_ids:
        if idx in next_alive_agents_ids:
            next_states_for_q.append(raw_next_states[next_alive_agents_ids.index(idx)])
        else:
            next_states_for_q.append(padding)

    return next_states_for_q


def main():
    alive_agents_ids = [0, 2, 3, 4, 5, 7]
    next_alive_agents_ids = [2, 5]
    obs_shape = (2, 2)
    raw_next_states = [np.ones(obs_shape) * 2, np.ones(obs_shape) * 5]

    next_states_for_q = \
        make_next_states_for_q(
            alive_agents_ids=alive_agents_ids,
            next_alive_agents_ids=next_alive_agents_ids,
            raw_next_states=raw_next_states,
            obs_shape=obs_shape,
        )

    next_padded_states_for_q = \
        make_padded_obs(max_num_agents=10, obs_shape=obs_shape, raw_obs=next_states_for_q)

    print(next_padded_states_for_q)

    max_num_agents = 10
    mask = make_mask(alive_agents_ids, max_num_agents)
    print(mask)

    mask = make_id_mask(alive_agents_ids, max_num_agents)
    print(mask)


if __name__ == '__main__':
    main()
