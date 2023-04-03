import numpy as np
import tensorflow as tf

from utils_transformer import make_id_mask
from config import Config
from sub_models import CNNModel, MultiHeadAttentionModel, QLogitModel


class MarlTransformerDecentralized(tf.keras.models.Model):

    def __init__(self, config, cnn_model, multihead_attention_model, qlogit_model):
        super(MarlTransformerDecentralized, self).__init__()

        self.config = config

        self.cnn = cnn_model(self.config)
        self.mha1 = multihead_attention_model(self.config)
        self.mha2 = multihead_attention_model(self.config)
        self.qlogit = qlogit_model(self.config)

    def mha_block(self, mha, features, masks, training=True):
        """
        :param mha: mha1, mha2
        :param features: [(b,hidden_dim),...,(b,hidden_dim)], len=n=15
        :param masks: [(b,1,n),...,(b,1,n)], len=n=15
        :return: att_features: [(b,hidden_dim),...,(b,hidden_dim)], len=n=15
                 att_scores: [(b,num_heads,n),...,(b,num_heads,n)], len=n=15
        """
        att_features = []
        att_scores = []

        for i in range(self.config.max_num_red_agents):
            query_feature = features[i]  # (b,hidden_dim)=(1,256)
            agent_inputs = [query_feature, features]
            mask = masks[i]  # (b,1,n)=(1,1,n)

            att_feature, att_score = mha(
                agent_inputs, mask, training)  # (b,hidden_dim)=(1,256), (b,num_heads,n)=(1,2,15)

            broadcast_float_mask = tf.cast(mask[:, :, i], 'float32')  # (b,1)=(1,1)
            att_feature = att_feature * broadcast_float_mask  # (b,hidden_dim)=(1,256)

            broadcast_float_mask = tf.cast(
                tf.expand_dims(mask[:, :, i], axis=1),
                'float32')  # (b,1,1)=(1,1,1), add head_dim
            att_score = att_score * broadcast_float_mask  # (b,num_heads,n)=(1,2,15)

            att_features.append(att_feature)
            att_scores.append(att_score)

        return att_features, att_scores

    @tf.function
    def call(self, inputs, masks, training=True):
        """
        :param inputs: [s1,s2,...,sn], len=n=15, si=[b,g,g,ch*n_frames)=(1,15,15,6)
        :param masks: [(b,1,n),...]=[(1,1,15),...], len=n
        :return: qlogits [(b,action_dim),...,(b,action_dim)], len=n
                 attscores = [attscores_1, attscores_2],
                    attscores_i: [(b,num_heads,n),...,(b,num_heads,n)], len=n
        """

        """ CNN block """
        features = []
        for i in range(self.config.max_num_red_agents):
            feature = self.cnn(inputs[i])  # (b,hidden_dim)=(1,256)

            mask = masks[i]  # (b,1,n)=(1,1,n)

            broadcast_float_mask = tf.cast(mask[:, :, i], 'float32')  # (b,1)=(1,1)
            feature = feature * broadcast_float_mask  # (b,hidden_dim)=(1,256)

            features.append(feature)

        """ Transformer block """
        # att_features_1, _2: [(b,hidden_dim),...,(b,hidden_dim)], len=n=15
        # att_scores_1, _2: [(b,num_heads,n),...,(b,num_heads,n)], len=n
        # attscores_i: [(b, num_heads, n), ..., (b, num_heads, n)], len=n

        att_features_1, att_scores_1 = self.mha_block(self.mha1, features, masks, training)
        att_features_2, att_scores_2 = self.mha_block(self.mha2, att_features_1, masks, training)

        att_scores = [att_scores_1, att_scores_2]

        """ Q logits block """
        q_logits = []
        for i in range(self.config.max_num_red_agents):
            q_logit = self.qlogit(att_features_2[i])  # (None,action_dim)=(1,5)

            mask = masks[i]  # (b,1,n)
            broadcast_float_mask = tf.cast(mask[:, :, i], 'float32')  # (b,1)=(1,1)
            q_logit = q_logit * broadcast_float_mask  # (None,action_dim)=(1,5)

            q_logits.append(q_logit)

        return q_logits, att_scores


def main():
    config = Config()

    grid_size = config.grid_size
    ch = config.observation_channels
    n_frames = config.n_frames

    obs_shape = (grid_size, grid_size, ch * n_frames)  # (15,15,6)
    max_num_agents = config.max_num_red_agents  # 15

    # Define alive_agents_ids & raw_obs
    alive_agents_ids = [0, 2]

    """ masks """
    masks = make_id_mask(alive_agents_ids, max_num_agents)  # [(b,1,n),...], len=n

    """ MARL Transformer """
    marl_transformer = MarlTransformerDecentralized(config,
                                                    CNNModel,
                                                    MultiHeadAttentionModel,
                                                    QLogitModel)

    """ States """
    states = []
    for i in range(config.max_num_red_agents):
        if i in alive_agents_ids:
            state = np.random.rand(obs_shape[0], obs_shape[1], obs_shape[2]).astype(np.float32)
        else:
            state = np.zeros((obs_shape[0], obs_shape[1], obs_shape[2])).astype(np.float32)

        state = np.expand_dims(state, axis=0)  # (1,15,15,6)
        states.append(state)

    """ Execute MARL Transformer """
    training = True
    q_logits, att_scores = marl_transformer(states, masks, training)

    print(len(q_logits))
    print(q_logits[0].shape)
    print(len(att_scores))
    print(len(att_scores[0]))
    print(att_scores[0][0].shape)


if __name__ == '__main__':
    main()
