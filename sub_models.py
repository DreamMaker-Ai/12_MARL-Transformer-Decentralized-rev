import os.path

import tensorflow as tf
import numpy as np

from config import Config
from utils_transformer import make_id_mask


class CNNModel(tf.keras.models.Model):
    """
    :param obs_shape: (15,15,6)
    :return: (None,hidden_dim)=(None,256)
    """

    def __init__(self, config, **kwargs):
        super(CNNModel, self).__init__(**kwargs)

        self.config = config

        self.conv0 = \
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=1,
                strides=1,
                activation='relu',
                kernel_initializer='Orthogonal'
            )

        self.conv1 = \
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=3,
                strides=2,
                activation='relu',
                kernel_initializer='Orthogonal'
            )

        self.conv2 = \
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=3,
                strides=1,
                activation='relu',
                kernel_initializer='Orthogonal'
            )

        self.conv3 = \
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=3,
                strides=1,
                activation='relu',
                kernel_initializer='Orthogonal'
            )

        self.flatten1 = tf.keras.layers.Flatten()

        self.dense1 = \
            tf.keras.layers.Dense(
                units=self.config.hidden_dim,
                activation=None
            )

    @tf.function
    def call(self, inputs):
        # inputs: (b,g,g,ch*n_frames)=(1,15,15,6)

        h = self.conv0(inputs)  # (1,15,15,64)
        h = self.conv1(h)  # (1,7,7,64)
        h = self.conv2(h)  # (1,5,5,64)
        h = self.conv3(h)  # (1,3,3,64)

        h1 = self.flatten1(h)  # (1,576)

        features = self.dense1(h1)  # (1,256)

        return features


class MultiHeadAttentionModel(tf.keras.models.Model):
    """
    Two layers of MultiHeadAttention (Self Attention with provided mask)

    :param mask: (None,1,n), bool
    :param max_num_agents=15=n
    :param hidden_dim = 256

    :return: features: (None,hidden_dim)=(None,256)
             score: (None,num_heads,n)=(None,2,15)
    """

    def __init__(self, config, **kwargs):
        super(MultiHeadAttentionModel, self).__init__(**kwargs)

        self.config = config

        self.query_feature = \
            tf.keras.layers.Lambda(
                lambda x: tf.cast(tf.expand_dims(x, axis=1), dtype=tf.float32)
            )

        self.features = \
            tf.keras.layers.Lambda(
                lambda x: tf.cast(tf.stack(x, axis=1), dtype=tf.float32)
            )

        self.mha1 = \
            tf.keras.layers.MultiHeadAttention(
                num_heads=self.config.num_heads,
                key_dim=self.config.key_dim,
            )

        self.add1 = \
            tf.keras.layers.Add()

        self.dense1 = \
            tf.keras.layers.Dense(
                units=config.hidden_dim * 2,
                activation='relu',
            )

        self.dense2 = \
            tf.keras.layers.Dense(
                units=config.hidden_dim,
                activation=None,
            )

        self.dropoout1 = tf.keras.layers.Dropout(rate=self.config.dropout_rate)

        self.add2 = tf.keras.layers.Add()

        self.reshape = tf.keras.layers.Reshape(target_shape=(config.hidden_dim,))

    @tf.function
    def call(self, inputs, mask=None, training=True):
        # inputs: [(None,hiddendim),[(None,hidden_dim),...,(None,hidden_dim)]]
        #           =[(1,256),[(1,256),...,(1,256)]]
        # mask: (None,1,n)=(1,1,15), bool,  n=15: max_num_agents

        attention_mask = tf.cast(mask, 'bool')  # (None,1,n)=(1,1,15)

        query_feature = self.query_feature(inputs[0])  # (None,1,hidden_dim)=(1,1,256)
        features = self.features(inputs[1])  # (Nonen,n,hidden_dim)=(1,15,256)

        x, score = \
            self.mha1(
                query=query_feature,
                key=features,
                value=features,
                attention_mask=attention_mask,
                return_attention_scores=True,
            )  # (None,1,hidden_dim),(None,num_heads,1,n)=(1,1,256),(1,2,1,15)

        x1 = self.add1([inputs[0], x])  # (None,1,hidden_dim)=(1,1,256)

        x2 = self.dense1(x1)  # (None,1,hidden_dim*2)=(1,1,512)

        x2 = self.dense2(x2)  # (None,n,hidden_dim)=(1,1,256)

        x2 = self.dropoout1(x2, training=training)

        feature = self.add2([x1, x2])  # (None,1,hidden_dim)=(1,1,256)

        feature = self.reshape(feature)  # (1,256)

        batch_dim = inputs[0].shape[0]
        score = tf.reshape(score,
                           shape=(batch_dim, self.config.num_heads, self.config.max_num_red_agents)
                           )  # (1,2,15)

        return feature, score  # (None,hidden_dim), (None,num_heads,n)


class QLogitModel(tf.keras.models.Model):
    """
    Very simple dense model, output is logits

    :param action_dim=5
    :param hidden_dim=256
    :return: (None,action_dim)=(None,5)
    """

    def __init__(self, config, **kwargs):
        super(QLogitModel, self).__init__(**kwargs)

        self.config = config

        self.dense1 = \
            tf.keras.layers.Dense(
                units=self.config.hidden_dim * 3,
                activation='relu',
            )

        self.dropoout1 = tf.keras.layers.Dropout(rate=self.config.dropout_rate)

        self.dense2 = \
            tf.keras.layers.Dense(
                units=self.config.hidden_dim,
                activation='relu',
            )

        self.dense3 = \
            tf.keras.layers.Dense(
                units=self.config.action_dim,
                activation=None,
            )

    @tf.function
    def call(self, inputs, training=True):
        # inputs: (None,hidden_dim)=(None,256)

        x1 = self.dense1(inputs)  # (None,n,hidden_dim*3)=(1,768)

        x1 = self.dropoout1(x1, training=training)

        x1 = self.dense2(x1)  # (None,hidden_dim)=(1,256)

        logit = self.dense3(x1)  # (None,action_dim)=(1,5)

        return logit  # (None,action_dim)=(1,5)


def main():
    dir_name = './models_architecture'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    config = Config()

    grid_size = config.grid_size
    ch = config.observation_channels
    n_frames = config.n_frames

    obs_shape = (grid_size, grid_size, ch * n_frames)  # (15,15,6)
    max_num_agents = config.max_num_red_agents  # 15

    # Define alive_agents_ids & raw_obs
    alive_agents_ids = [0, 2]

    """ mask """
    mask = make_id_mask(alive_agents_ids, max_num_agents)  # (1,n)

    # add batch dim
    mask = np.expand_dims(mask, axis=0)  # (1,1,n)

    """ cnn_model """
    cnn = CNNModel(config=config)

    # Get features list of all agents
    features = []
    for i in range(max_num_agents):
        if i in alive_agents_ids:
            obs = np.random.rand(obs_shape[0], obs_shape[1], obs_shape[2]).astype(np.float32)
        else:
            obs = np.zeros((obs_shape[0], obs_shape[1], obs_shape[2])).astype(np.float32)

        obs = np.expand_dims(obs, axis=0)  # (1,15,15,6)
        feat = cnn(obs)  # (1,256)
        features.append(feat)  # [(1,256),...,(1,256)], len=15

    """ mha model """
    mha = MultiHeadAttentionModel(config=config)

    # Get output list of attention of all agents
    att_features = []
    att_scores = []
    for i in range(max_num_agents):
        query_feature = features[i]  # (1,256)

        inputs = [query_feature, features]  # [(1,256),[(1,256),...,(1,256)]]

        att_feature, att_score = \
            mha(inputs,
                mask,
                training=True
                )  # (None,hidden_dim),(None,num_heads,n)

        att_features.append(att_feature)
        att_scores.append(att_score)

    """ q_model """
    q_net = QLogitModel(config=config)

    # Get q_logits list of all agents
    q_logits = []
    for i in range(max_num_agents):
        q_logit = q_net(att_features[i])  # (None,5)
        q_logits.append(q_logit)


if __name__ == '__main__':
    main()
