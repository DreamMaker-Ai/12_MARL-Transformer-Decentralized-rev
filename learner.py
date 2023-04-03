from pathlib import Path

import numpy as np
import ray
import tensorflow as tf

from battlefield_strategy import BattleFieldStrategy

from model import MarlTransformerDecentralized as MarlTransformerModel
from sub_models import CNNModel as CNNModel
from sub_models import MultiHeadAttentionModel as MultiHeadAttentionModel
from sub_models import QLogitModel as QLogitModel

from utils_transformer import make_id_mask as make_mask
from utils_transformer import experiences2per_agent_list, per_agent_list2input_list, get_td_mask


# @ray.remote
@ray.remote(num_cpus=1, num_gpus=1)
class Learner:
    def __init__(self):
        self.env = BattleFieldStrategy()

        self.action_space_dim = self.env.action_space.n
        self.gamma = self.env.config.gamma

        self.q_network = \
            MarlTransformerModel(config=self.env.config,
                                 cnn_model=CNNModel,
                                 multihead_attention_model=MultiHeadAttentionModel,
                                 qlogit_model=QLogitModel)

        self.target_q_network = \
            MarlTransformerModel(config=self.env.config,
                                 cnn_model=CNNModel,
                                 multihead_attention_model=MultiHeadAttentionModel,
                                 qlogit_model=QLogitModel)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.env.config.learning_rate)

        self.count = self.env.config.n0 + 1

    def define_network(self):
        """
        Q-network, Target_networkを定義し、current weightsを返す
        """
        # self.q_network.compile(optimizer=self.optimizer, loss='mse')

        # Build graph with dummy inputs
        grid_size = self.env.config.grid_size
        ch = self.env.config.observation_channels
        n_frames = self.env.config.n_frames
        max_num_agents = self.env.config.max_num_red_agents

        state_shape = (grid_size, grid_size, ch * n_frames)

        # Define alive_agents_ids
        alive_agents_ids = [0, 2]

        # States
        padded_states = []
        for i in range(self.env.config.max_num_red_agents):
            if i in alive_agents_ids:
                state = \
                    np.random.rand(state_shape[0], state_shape[1], state_shape[2]).astype(
                        np.float32)
            else:
                state = np.zeros((state_shape[0], state_shape[1], state_shape[2])).astype(
                    np.float32)

            state = np.expand_dims(state, axis=0)  # (1,15,15,6)
            padded_states.append(state)

        # Make mask
        masks = make_mask(alive_agents_ids, max_num_agents)
        # [(1,1,n), ...]=[(1,1,15), ...], len=n=15

        self.q_network(padded_states, masks, training=True)
        self.target_q_network(padded_states, masks, training=False)

        # Load weights
        if self.env.config.model_dir:
            self.q_network.load_weights(self.env.config.model_dir)

        # Q networkのCurrent weightsをget
        current_weights = self.q_network.get_weights()

        # Q networkの重みをTarget networkにコピー
        self.target_q_network.set_weights(current_weights)

        return current_weights

    def update_network(self, minibatchs):
        """
        minicatchsを使ってnetworkを更新
        minibatchs = [minibatch,...], len=30(default)

        minibatch = [sampled_indices, correction_weights, experiences], batch_size=32(default)
            - sampled_indices: [int,...], len = batch_size=32(default)
            - correction_weights: Importance samplingの補正用重み, (batch_size,), ndarray
            - experiences: [experience,...], len=batch_size
                experience =
                    (
                        (padded_)states: [(1,g,g,ch*n_frames), ...], len=n
                        (padded_)actions: [(1,), ...], len=n
                        (padded_)rewards: [(1,), ...], len=n
                        next_(padded_)states: [(1,g,g,ch*n_frames), ...], len=n
                        (padded_)dones: [(1,), ...], len=n, bool
                        masks: [(1,1,n), ...], len=n, bool
                    )

                ※ experience.states等で読み出し

        :return:
            current_weights: 最新のnetwork weights
            indices_all: ミニバッチに含まれるデータのインデクス, [int,...], len=batch*16(default)
            td_errors_all: ミニバッチに含まれるデータのTD error, [(batch,n),...], len=16(default)
        """
        indices_all = []
        td_errors_all = []
        losses = []

        for (indices, correction_weights, experiences) in minibatchs:
            # indices:list, len=32; correction_weights: ndarray, (32,1); experiences: lsit, len=32
            # experiencesをnetworkに入力するshapeに変換

            """
                agent_state = [
                    state list of agent_1=[(1,15,15,6), ...], len=32,
                    state list of agent_2=[(1,15,15,6), ...], len=32,
                                        ... ], len=n
                agent_action = [
                    action list of agent_1=[(1,), ..., len=32,
                    action list of agent_2=[(1,), ..., len=32,
                                ... ], len=n
                                
                agent_mask = [mask list of agent_1= [(1,1,15), ...], len=32,
                              mask list of agent_2= [(1,1,15), ...], len=32,
                                    ... ], len=n, bool
            """
            agent_state, agent_action, agent_reward, agent_next_state, agent_done, agent_mask = \
                experiences2per_agent_list(self, experiences)

            """
            per_agent_list -> input list to policy network
                states: [(b,g,g,ch*n_frames), ...], len=n=15
                actions: [(b,), ...], len=n
                rewards: [(b,), ...], len=n
                next_states: [(b,g,g,ch*n_frames), ...], len=n
                dones: [(b,), ...], len=n, bool
                masks: [(b,1,n), ...], len=n, bool
            """

            states, actions, rewards, next_states, dones, masks = \
                per_agent_list2input_list(self, agent_state, agent_action, agent_reward,
                                          agent_next_state, agent_done, agent_mask)

            # Target valueの計算
            next_q_logits, _ = \
                self.target_q_network(next_states, masks, training=False)  # [(32,5), ...], len=n=15
            next_q_logits = tf.stack(next_q_logits, axis=1)  # (32,15,5)

            next_actions = tf.argmax(next_q_logits, axis=-1)  # (32,15)
            next_actions = tf.cast(next_actions, dtype=tf.int32)
            next_actions_one_hot = \
                tf.one_hot(next_actions, depth=self.action_space_dim)  # (32,15,5)

            next_maxQ = next_q_logits * next_actions_one_hot  # (32,15,5)
            next_maxQ = tf.reduce_sum(next_maxQ, axis=-1)  # (32,15)

            rewards = tf.stack(rewards, axis=1)  # (32,15)
            rewards = tf.cast(rewards, dtype=tf.float32)  # (32,15)

            dones = tf.stack(dones, axis=1)  # (32,15), bool
            dones = tf.cast(dones, dtype=tf.float32)  # *32,15)

            TQ = rewards + self.gamma * (1 - dones) * next_maxQ  # (32,15)

            # ロス計算
            with tf.GradientTape() as tape:
                q_logits, _ = self.q_network(states, masks, training=True)  # [(32,5), ...], len=15

                q_logits = tf.stack(q_logits, axis=1)  # (b,n,action_dim)=(32,15,5)

                q_actions = tf.stack(actions, axis=1)  # (b,n)=(32,15)
                q_actions = tf.cast(q_actions, dtype=tf.int32)
                q_actions_one_hot = tf.one_hot(q_actions, depth=self.action_space_dim)  # (32,15,5)

                Q = q_logits * q_actions_one_hot  # (32,15,5)
                Q = tf.reduce_sum(Q, axis=-1)  # (32,15)

                td_errors = tf.square(TQ - Q)  # (32,15)

                float_mask = get_td_mask(self.env.config, masks)  # (b,n), float32

                masked_td_errors = td_errors * float_mask  # (32,15)
                masked_td_errors = \
                    tf.reduce_sum(masked_td_errors, axis=-1) / \
                    tf.reduce_sum(float_mask, axis=-1)  # (32,)

                loss = tf.reduce_mean(correction_weights * masked_td_errors) * \
                       self.env.config.loss_coef

                losses.append(loss.numpy())  # For tensorboard

            # 勾配計算と更新
            grads = tape.gradient(loss, self.q_network.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 40)

            self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

            # Compute priority update
            if self.env.config.prioritized_replay:
                priority_td_errors = np.abs((TQ - Q).numpy())  # (32,15)
                masked_priority_td_errors = priority_td_errors * float_mask.numpy()  # (32,15)
                masked_priority_td_errors = \
                    np.sum(masked_priority_td_errors, axis=-1) / \
                    np.sum(float_mask.numpy(), axis=-1)  # (32,)

            else:
                masked_priority_td_errors = np.ones((self.env.config.batch_size,),
                                                    dtype=np.float32)  # (32,)

            # learnerの学習に使用した経験のインデクスとTD-errorのリスト
            indices_all += indices
            td_errors_all += masked_priority_td_errors.tolist()  # len=32のリストに変換

        # 最新のネットワークweightsをget
        current_weights = self.q_network.get_weights()

        # Target networkのweights更新: Soft update
        target_weights = self.target_q_network.get_weights()

        for w in range(len(target_weights)):
            target_weights[w] = \
                self.env.config.tau * current_weights[w] + \
                (1. - self.env.config.tau) * target_weights[w]

        self.target_q_network.set_weights(target_weights)

        # Save model
        if self.count % 100 == 0:
            save_dir = Path(__file__).parent / 'models'
            save_name = '/model_' + str(self.count) + '/'

            self.q_network.save_weights(str(save_dir) + save_name)

        self.count += 1

        return current_weights, indices_all, td_errors_all, np.mean(losses)
