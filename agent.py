from __future__ import print_function

import gc
import math
import os
import random
import time

import keras.backend as K
import scipy.spatial
from operator import mul
from keras import initializers
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import LearningRateScheduler
from keras.layers import *
from keras.layers.pooling import *
from keras.losses import mean_absolute_error, mean_squared_error
from keras.models import Model, load_model
from keras.optimizers import *
from keras.utils import np_utils

from config_agent import *
from diverse_mem import DiverseMemory
from history import *
from utils import *

from pprint import pprint


def LEAKY_RELU(): return LeakyReLU(0.01)


try:
    from PIL import Image
except:
    import Image

DEBUG = 1


def masked_error(args):
    """
        Masked asolute error function

        Args:
            y_true: Target output
            y_pred: Actual output
            mask: Scales the loss, should be compatible with the shape of 
                    y_true, if an element of the mask is set to zero, the
                    corresponding loss is ignored
    """
    y_true, y_pred, mask = args
    loss = K.abs(y_true - y_pred)
    loss *= mask
    return K.sum(loss, axis=-2)


class DeepAgent():
    def __init__(self,
                 actions,
                 objective_cnt,
                 memory_size=100000,
                 sample_size=64,
                 weights=None,
                 start_discount=0.98,
                 discount=0.98,
                 end_discount=0.98,
                 start_e=1.,
                 end_e=0.05,
                 learning_rate=1e-1,
                 target_update_interval=150,
                 frame_skip=4,
                 alg="scal",
                 memory_type="STD",
                 mem_a=1,
                 mem_e=0.02,
                 update_interval=4,
                 reuse='full',
                 extra="",
                 clipnorm=1,
                 clipvalue=1,
                 nesterov=True,
                 momentum=0.9,
                 dupe=False,
                 start_annealing=0.01,
                 min_buf_size=.001,
                 scale=1,
                 im_size=(IM_SIZE, IM_SIZE),
                 grayscale=BLACK_AND_WHITE,
                 frames_per_state=2,
                 max_episode_length=1000,
                 start_impsam=1,
                 end_impsam=1):
        """Agent implementing both Multi-Network, Multi-Head and Single-head 
            algorithms

        Arguments:
            actions {list} -- List of possible actions
            objective_cnt {int} -- Number of objectives

        Keyword Arguments:
            memory_size {int} -- Size of the replay buffer (default: {100000})
            sample_size {int} -- Number of samples used for updates (default: {64})
            weights {list} -- Default weights (default: {None})
            discount {float} -- Discount facctor (default: {0.98})
            start_e {float} -- Starting exploration rate (default: {1.})
            end_e {float} -- Final exploration rate (default: {0.05})
            learning_rate {float} -- Learning rate (default: {1e-1})
            target_update_interval {int} -- Interval between target network synchronizations (default: {600})
            frame_skip {int} -- Number of times each action is repeated (default: {4})
            alg {str} -- Algorithm to use, either 'scal','mo','cond' or 'meta' (default: {"scal"})
            memory_type {str} -- Either 'std' for a prioritized experience replay or 'crowd' for a diversity-enforced experience replay (default: {"STD"})
            update_interval {int} -- Interval between network updates (default: {4})
            reuse {str} -- 'full' for full reuse, 'proportional' for proportional reuse or 'sectioned' for no reuse (default: {'full'})
            extra {str} -- Additional information string identifying the learner (default: {""})
            clipnorm {float} -- clipnorm value (losses are normalized to this value) (default: {1})
            clipvalue {float} -- clipvalue (losses are clipped to this value) (default: {1})
            nesterov {bool} -- Wether to use nesterov momentum (default: {True})
            momentum {float} -- Momentum parameter for the SGD optimizer (default: {0.9})
            dupe {bool} -- If True, samples are trained both on the current and on a past weight (default: {False})
            min_buf_size {float} -- Minimum size of the buffer needed to start updating, as a fraction of the number of learning steps (default: {0.1})
            scale {float} -- Scales the network's layer sizes (default: {1})
            im_size {tuple} -- Size the frames should be scaled to (default: {(48, 48)})
            grayscale {bool} -- Color mode for the frames (default: {True})
            frames_per_state {int} -- How many frames the state should consist of (default: {2})
            max_episode_length {int} -- Interrupt episodes if they last longer than this (default: {5000})
        """
        self.max_error = .0
        self.direct_update = True
        self.normalize_weights = False
        self.recent_experiences = []
        self.epsilon = self.start_e = start_e
        self.max_episode_length = max_episode_length
        self.nesterov = nesterov
        self.actions = actions
        self.action_count = len(actions)
        self.frames_per_state = frames_per_state
        self.im_size = im_size
        self.grayscale = grayscale
        self.history = History(frames_per_state, im_size, grayscale)
        self.mem_a = mem_a
        self.mem_e = mem_e
        self.scale = scale
        self.clipnorm = clipnorm
        self.clipvalue = clipvalue
        self.momentum = momentum
        self.frame_skip = frame_skip
        self.reuse = reuse
        self.start_e = start_e
        self.end_e = end_e
        self.start_annealing = start_annealing
        self.start_impsam = start_impsam
        self.end_impsam = end_impsam
        self.is_first_update = True
        self.trace_values={}
        self.alg = alg
        self.dupe = dupe
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.min_buf_size = min_buf_size
        self.actions = actions

        self.memory_size = memory_size

        self.action_count = len(self.actions)
        self.obj_cnt = objective_cnt

        self.sample_size = sample_size

        self.weight_history = []
        if weights is not None:
            self.set_weights(np.array(weights))

        self.lr = learning_rate
        self.memory_type = memory_type

        self.build_models()
        self.initialize_memory()

        self.start_discount = start_discount
        self.discount = start_discount
        self.end_discount = end_discount

        self.extra = extra
        print(vars(self))

    def build_models(self):
        """Builds the networks then synchronizes the model and target networks
        """

        self.model, self.target_model, self.trainable_model = self.build_networks(
        )

        self.model.summary()
        self.update_target()

    def build_base(self):
        """Builds the feature extraction component of the network

        Returns:
            tuple -- A tuple (Input, feature layer, weight layer) 
        """

        state_input = Input((self.frames_per_state, ) + self.history.im_shape)

        # We always build the weight input, but don't connect it to the network
        # when it is not required.
        weight_input = Input((self.obj_cnt, ), name="weight_input")
        
        x = Lambda(lambda x: x / 255., name="input_normalizer")(state_input)

        # Convolutional layers
        for c, (
                filters, kernel_size, strides
        ) in enumerate(zip(CONV_FILTERS, CONV_KERNEL_SIZES, CONV_STRIDES)):
            x = TimeDistributed(
                Conv2D(
                    filters=int(filters / self.scale),
                    kernel_size=kernel_size,
                    strides=strides,
                    name="conv{}".format(c)))(x)
            x = LEAKY_RELU()(x)
            x = TimeDistributed(MaxPooling2D())(x)

        # Per dimension dense layer
        x = Dense(
            int(POST_CONV_DENSE_SIZE / self.scale),
            kernel_initializer=DENSE_INIT,
            name="post_conv_dense")(x)
        x = LEAKY_RELU()(x)
        feature_layer = Flatten()(x)

        self.shared_length = 0

        return state_input, feature_layer, weight_input

    def build_dueling_head(self, feature_layer, weight_input, obj_cnt,
                           per_stream_dense_size):
        """Given a feature layer and weight input, this method builds the
            Q-value outputs using a dueling architecture

        Returns:
            list -- List of outputs, one per action
        """

        # Connect the weight input only if in conditionned network mode
        features = Concatenate(name="features")(
            [weight_input,
             feature_layer]) if ("cond" in self.alg or "uvfa" in self.alg) else feature_layer

        # Build a dueling head with the required amount of outputs
        head_dense = [
            LEAKY_RELU()(Dense(
                per_stream_dense_size,
                name='dueling_0_{}'.format(a),
                kernel_initializer=DENSE_INIT)(features))
            for a in range(2)
        ]

        for depth in range(1, DUELING_DEPTH):
            head_dense = [
                LEAKY_RELU()(Dense(
                    per_stream_dense_size,
                    name='dueling_{}_{}'.format(depth, a),
                    kernel_initializer=DENSE_INIT)(head_dense[a]))
                for a in range(2)
            ]

        head_out = [
            Dense(
                obj_cnt,
                name='dueling_out_{}'.format(a),
                activation='linear',
                kernel_initializer=DENSE_INIT)(head_dense[0] if a == 0 else head_dense[1])
            for a in range(self.action_count + 1)
        ]

        x = Concatenate(name="concat_heads")(head_out)

        x = Reshape((self.action_count + 1, obj_cnt))(x)

        # Dueling merge function
        outputs = [
            Lambda(lambda a: a[:, 0, :] + a[:, b + 1, :] -
                   K.mean(a[:, 1:, :], axis=1, keepdims=False),
                   output_shape=(obj_cnt, ))(x)
            for b in range(self.action_count)
        ]

        return Concatenate(name="concat_outputs")(outputs)

    def build_head(self, feature_layer, inp, weight_input):
        """Builds the Q-value head on top of the feature layer

        Arguments:
            feature_layer {Keras layer} -- The feature layer
            inp {Keras Input} -- The model's image input
            weight_features {Keras Layer} -- The model's weight features

        Returns:
            tuple -- Consisting of the main model, and a trainable model that
                        accepts a masked input
        """

        head_pred = self.build_dueling_head(feature_layer, weight_input,
                                            self.qvalue_dim(), DUELING_LAYER_SIZE)

        y_pred = Reshape((self.action_count,  self.qvalue_dim()))(head_pred)

        # We mask the losses such that only losses of the relevant action
        # are taken into account for the network update, based on:
        # https://github.com/keras-rl/keras-rl/blob/master/rl/agents/sarsa.py
        y_true = Input(name='y_true', shape=(
            self.action_count,  self.qvalue_dim(), ))

        mask = Input(name='mask', shape=(
            self.action_count,  self.qvalue_dim(), ))

        loss_out = Lambda(
            masked_error, output_shape=(1, ),
            name='loss')([y_true, y_pred, mask])

        trainable_model = Model([weight_input, inp, y_true, mask],
                                [loss_out, y_pred])
        main_model = Model([weight_input, inp], y_pred)

        return main_model, trainable_model

    def build_networks(self):
        """Builds the required networks, main and target q-value networks,
            a trainable (masked) main network and a predictive network

        Returns:
            tuple -- consisting of the main model, the target model, the 
                    trainable model and a predictive model
        """

        state_input, feature_layer, weight_input = self.build_base()

        # Build dueling Q-value heads on top of the base
        main_model, trainable_model = self.build_head(
            feature_layer, state_input, weight_input)

        state_input, feature_layer, weight_input = self.build_base()
        target_model, _ = self.build_head(
            feature_layer, state_input, weight_input)

        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            # we only include this for the metrics
            lambda y_true, y_pred: K.zeros_like(y_pred),
        ]

        trainable_model.compile(
            loss=losses,
            optimizer=SGD(
                lr=self.lr,
                clipnorm=self.clipnorm,
                clipvalue=self.clipvalue,
                momentum=self.momentum,
                nesterov=self.nesterov))

        return main_model, target_model, trainable_model

    def qvalue_dim(self):
        return 1 if self.has_scalar_qvalues() else self.obj_cnt

    def policy_update(self):
        """Update the policy (Q-values)

        Returns:
            float -- The update's loss
        """

        np.random.seed(self.steps)
        ids, batch, _ = self.buffer.sample(self.sample_size)

        if self.direct_update:
            # Add recent experiences to the priority update batch
            batch = np.concatenate(
                (batch, self.buffer.get(self.recent_experiences)), axis=0)
            ids = np.concatenate((ids, self.recent_experiences)).astype(int)

        # if self.duplicate is True, we train each sample in the batch on two
        # weight vectors, hence we duplicate the batch data
        if self.duplicate():
            batch = np.repeat(batch, 2, axis=0)
            ids = np.repeat(ids, 2, axis=0)

        model_Qs, target_Qs, w_batch, states = self.get_training_data(batch)

        # Every other value in model_QS, target_Qs, w_batch and states concerns
        # the subsequent state. In other words, states=(s_i,s_{i+1},s_j,s_{j+1},...)
        x_batch = states[::2]
        y_batch = np.copy(model_Qs[::2])

        # Mask the output to ignore losses of non-concerned actions
        masks = np.zeros(
            (len(batch), self.action_count, self.qvalue_dim()),
            dtype=float)
        for i, (_, action, reward, _, terminal, _) in enumerate(batch):
            if self.has_scalar_qvalues():
                reward = np.dot(reward, w_batch[i])
            y_batch[i][action] = np.copy(reward)
            masks[i][action] = 1

            if not terminal:
                if self.alg == "naive":
                    for o in range(self.obj_cnt):
                        max_a = np.argmax(model_Qs[1::2][i][:, o])
                        y_batch[i][action][o] += self.discount * \
                            target_Qs[1::2][i][max_a][o]
                else:
                    if not self.has_scalar_qvalues():
                        max_a = np.argmax(
                            np.dot(model_Qs[1::2][i], w_batch[1::2][i]))
                    else:
                        max_a = np.argmax(model_Qs[1::2][i])
                    y_batch[i][action] += self.discount * \
                        target_Qs[1::2][i][max_a]

        inp = [w_batch[::2], x_batch, y_batch, masks]
        dummy = y_batch[:, 0, :]

        self.trainable_model.train_on_batch(inp, [dummy, y_batch])

        loss = self.update_priorities(batch, ids)
        self.recent_experiences = []

        return loss

    def train(self, environment, log_file, learning_steps, weights,
              per_weight_steps, total_steps):
        """Train agent on a series of weights for the given environment

        Arguments:
            environment {Object} -- Environment, should have reset() and step(action, frame_skip) functions
            log_file {string} -- Name of the log file
            learning_steps {int} -- Over how many steps epsilon is annealed
            weights {list} -- List of weight vectors
            per_weight_steps {int} -- How many steps each weight is active for, if 1, weights change at the end of each episode
            total_steps {int} -- Total training steps
        """

        self.epsilon = self.start_e
        per_weight_steps = per_weight_steps

        weight_index = 0
        self.steps = 0
        self.total_steps = total_steps

        # the amount of learning steps determines how quickly we start updating and how quickly epsilon is annealed
        self.learning_steps = max(1, learning_steps)

        self.env = environment
        self.log = Log(log_file, self)

        if self.alg == "mn":
            self.mn = MultiNetwork(self, reuse=self.reuse)

        self.set_weights(weights[weight_index])

        model_pol_index = None
        episodes = 1
        episode_steps = 0
        pred_idx = None

        current_state_raw = self.env.reset()
        current_state = self.history.fill_raw_frame(
            current_state_raw["pixels"])

        for i in range(int(self.total_steps)):

            self.steps = i
            episode_steps += 1

            # pick an action following an epsilon-greedy strategy
            action = self.pick_action(current_state)

            # perform the action
            next_state_raw, reward, terminal = self.env.step(
                action, self.frame_skip)

            next_state = self.history.add_raw_frame(next_state_raw["pixels"])

            # memorize the experienced transition
            pred_idx = self.memorize(
                current_state,
                action,
                reward,
                next_state,
                terminal,
                self.steps,
                trace_id=episodes if self.memory_type == "DER" else self.steps,
                pred_idx=pred_idx)

            # update the networks and exploration rate
            loss = self.perform_updates(i)
            self.update_epsilon(i)

            if terminal or episode_steps > self.max_episode_length:
                start_state_raw = self.env.reset()
                next_state = self.history.fill_raw_frame(
                    start_state_raw["pixels"])
                pred_idx = None

            self.log.log_step(self.env, i, loss, reward,
                              terminal or episode_steps > self.max_episode_length, current_state, next_state,
                              self.weights, self.end_discount, episode_steps,
                              self.epsilon, self.frame_skip,
                              current_state_raw["content"], action)

            current_state = next_state
            current_state_raw = next_state_raw

            if terminal or episode_steps > self.max_episode_length:

                is_weight_change = int(
                    (i + 1) / per_weight_steps) != weight_index

                if self.alg == "mn" and is_weight_change:
                    # Compute the trained policy's value
                    cur_value = max(
                        self.predict(next_state),
                        key=lambda q: np.dot(q, self.weights))

                    # Add the trained policy
                    self.mn.add_new_policy(
                        cur_value, weights[:weight_index + 1], model_pol_index, override=weight_index<=1)

                if per_weight_steps == 1:
                    weight_index += 1
                else:
                    weight_index = int((i + 1) / per_weight_steps)

                if per_weight_steps == 1 or is_weight_change:
                    self.set_weights(weights[weight_index])

                # Load the past policy which is optimal for the new weight vector
                if self.alg == "mn" and is_weight_change:
                    model_pol_index = self.mn.load_optimal_policy(
                        self.weights, weight_index)

                episodes += 1
                episode_steps = 0

    def set_weights(self, weights):
        """Set current weight vector

        Arguments:
            weights {np.array} -- Weight vector of size N
        """

        self.weights = np.array(weights)
        self.weight_history.append(self.weights)

    def name(self):
        return self.extra

    def get_network_weights(self, start_layer=0, end_layer=-1):
        """Extracts weights of a subset of the model and target network's layer.
            Can for example be used to extract only the policy layer's parameters

        Keyword Arguments:
            start_layer {int} -- Start of the layers subset (default: {0})
            end_layer {int} -- End of the layers subset (default: {-1})

        Returns:
            tuple -- consisting of (model network subweights, target network subweights)
        """

        return ([
            layer.get_weights()
            for layer in self.model.layers[start_layer:end_layer]
        ], [
            layer.get_weights()
            for layer in self.target_model.layers[start_layer:end_layer]
        ])

    def initialize_memory(self):
        """Initialize the replay buffer, with a secondary diverse buffer and/or
            a secondary tree to store prediction errors
        """
        if self.memory_type == "DER" or self.memory_type == "SEL" or self.memory_type == "EXP":
            main_capacity = sec_capacity = self.memory_size // 2
        else:
            main_capacity, sec_capacity = self.memory_size, 0

        def der_trace_value(trace, trace_id, memory_indices):
            """Computes a trace's value as its return

            Arguments:
                trace {list} -- list of transitions
                trace_id {object} -- the trace's id
                memory_indices {list} -- list of the trace's indexes in memory

            Returns:
                np.array -- the trace's value
            """

            if trace_id in self.trace_values:
                return self.trace_values[trace_id]
            I_REWARD = 2
            value = np.copy(trace[0][I_REWARD])
            for i, v in enumerate(trace[1:]):
                value += v[I_REWARD] * self.end_discount**(i + 1)
            if type(value) == float:
                value = np.array([value])
            self.trace_values[trace_id] = value
            return value

        def sel_trace_value(trace, trace_id, memory_indices):
            assert len(trace) == 1
            if trace_id not in self.trace_values:
                self.trace_values[trace_id] = np.random.random()

            return self.trace_values[trace_id]

        def exp_trace_value(trace, trace_id, memory_indices):

            assert len(trace) == 1
            if trace_id not in self.trace_values:
                state, action, reward, next_state, terminal, extra = trace[0]
                q_values = self.predict(state)
                best_q = np.max(np.dot(q_values, self.weights))

                self.trace_values[trace_id] = best_q - \
                    np.dot(q_values[action], self.weights)

            return self.trace_values[trace_id]

        value_function = der_trace_value if self.memory_type == "DER" else sel_trace_value if self.memory_type == "SEL" else exp_trace_value
        trace_diversity = not(self.memory_type ==
                              "SEL" or self.memory_type == "EXP")
        self.buffer = DiverseMemory(
            main_capacity=main_capacity,
            sec_capacity=sec_capacity,
            value_function=value_function,
            trace_diversity=trace_diversity,
            a=self.mem_a,
            e=self.mem_e)

    def predict(self, state, model=None, weights=None):
        """Predict values for the given state

        Arguments:
            state {np.array} -- [The input state]

        Keyword Arguments:
            weights {np.array} -- [The weights input] (default: {active weights})
            model {Keras Model} -- [The model to predict with] (default: {self.model})

        Returns:
            np.array -- [Predicted values]
        """
        model = model or self.model
        weights = self.weights if weights is None else weights
        return model.predict([weights[np.newaxis, ], state[np.newaxis, ]])[0]

    def pick_action(self, state, weights=None):
        """Given a state and weights, compute the next action, following an
            epsilon-greedy strategy

        Arguments:
            state {np.array} -- The state in which to act

        Keyword Arguments:
            weights {np.array} -- The weights on which to act (default: self.weights)

        Returns:
            int -- The selected action's index
        """
        np.random.seed(self.steps)
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)

        weights = self.weights if weights is None else weights
        self.q_values = self.predict(state, self.model, weights)
        if not self.has_scalar_qvalues():
            scalarized_qs = np.dot(self.q_values, weights)
        else:
            scalarized_qs = self.q_values
        return np.argmax(scalarized_qs)

    def perform_updates(self, steps):
        """Perform the necessary updates for the current number of steps, 
            The policy and the feature extraction (either implicitly or 
            explicitly depending on the architecture) are updated
        Arguments:
            steps {int} -- Number of steps

        Returns:
            loss -- The update loss
        """

        loss = 0
        if steps > self.min_buf_size * self.learning_steps:

            if steps % self.update_interval == 0:

                if self.is_first_update:
                    # Compute buffer's priorities all at once before the first
                    # update
                    self.recent_experiences = []
                    self.update_all_priorities()
                    self.is_first_update = False

                loss = self.policy_update()
            if steps % self.target_update_interval == 0:
                self.update_target()

        return loss
        
    def update_epsilon(self, steps):
        """Update exploration rate

        Arguments:
            steps {int} -- Elapsed number of steps
        """

        start_steps = self.learning_steps * self.start_annealing
        annealing_steps = self.learning_steps * \
            (1 - self.start_annealing)

        self.epsilon = linear_anneal(steps, annealing_steps, self.start_e,
                                     self.end_e, start_steps)
        self.discount = self.end_discount - linear_anneal(steps, annealing_steps, self.end_discount - self.start_discount,
                                                          0, start_steps)

    def memorize(self,
                 state,
                 action,
                 reward,
                 next_state,
                 terminal,
                 steps,
                 initial_error=0,
                 trace_id=None,
                 pred_idx=None):
        """Memorizes a transition into the replay, if no error is provided, the 
        transition is saved with the lowest priority possible, and should be
        updated accordingly later on.

        Arguments:
            state {Object} -- s_t
            action {int} -- a_t
            reward {np.array} -- r_t
            next_state {Object} -- s_{t+1}
            terminal {bool} -- wether s_{t+1} is terminal
            steps {int} -- t

        Keyword Arguments:
            initial_error {float} -- The initial error of the transition (default: {0})
            trace_id {object} -- The trace's identifier, if None, the transition is treated as
                                an individual trace. (default: {None})
        """
        if initial_error == 0 and not self.direct_update:
            initial_error = self.max_error
        extra = np.array(
            [steps, steps, self.epsilon, self.weights],
            dtype=object)

        transition = np.array((state, action, reward, next_state[-1], terminal,
                               extra))

        # Add transition to replay buffer

        idx = self.buffer.add(
            initial_error, transition, pred_idx=pred_idx, trace_id=trace_id)
        self.recent_experiences.append(idx)

        return idx

    def update_target(self):
        """Synchronize the target network
        """
        self.target_model.set_weights(self.model.get_weights())

    def get_full_states(self, batch):
        """To save space, we only store the new part of the next state (i.e., 
        a single frame), this method reconstructs the next states from that 
        additional frame and from the previous state.

        Arguments:
            batch {list} -- Batch of transitions

        Returns:
            np.array -- Batch of reconstructed states, with every s_i stored in
                        position i*2 and s_{i+1} stored in position i*2+1
        """

        states = np.zeros(
            (len(batch) * 2,) + self.history.shape, dtype=np.uint8)
        for i, b in enumerate(batch):
            states[i * 2] = batch[i][0]
            states[i * 2 + 1][:-1] = batch[i][0][1:]
            states[i * 2 + 1][-1] = batch[i][3]
        return states

    def get_training_weights(self, batch):
        """Given a batch of transitions, this method generates a batch of
        weights to train on

        Arguments:
            batch {list} -- batch of transitions

        Returns:
            list -- batch of weights
        """

        w_batch = np.repeat([self.weights], len(batch), axis=0)
        if self.dupe in ("CN", "CN-UVFA"):
            if len(self.weight_history) > 1:
                max_index = len(
                    self.weight_history) - 1 if self.dupe in ("CN") else len(self.weight_history)

                idx = np.random.randint(
                    max_index, size=int(len(batch)))
                w_batch[::] = np.array(self.weight_history)[idx]

                if self.dupe in ("CN"):
                    w_batch[::2] = self.weights

        return w_batch

    def duplicate(self):
        return self.dupe in ("CN", "CN-UVFA", "CN-ACTIVE")

    def update_all_priorities(self):
        """Updates all priorities of the replay buffer
        """
        data = self.buffer.get_data(True)

        chunk_size = 100
        for i in range(0, len(data[0]), chunk_size):
            chunk_data = np.array(data[1][i:i + chunk_size])
            chunk_ids = data[0][i:i + chunk_size]

            if self.duplicate():
                chunk_data = np.repeat(chunk_data, 2, axis=0)
                chunk_ids = np.repeat(chunk_ids, 2, axis=0)

    def get_training_data(self, batch):
        """Given a batch of transitions, this method reconstructs the states,
        generates a batch of weights and computes the model and target predictions for both each s_i and s_{i+1}

        Arguments:
            batch {list} -- batch of transitions

        Returns:
            tuple -- (model predictions, target predictions, weight batch, states)
        """
        states = self.get_full_states(batch)

        w_batch = self.get_training_weights(batch)
        w_batch = np.repeat(w_batch, 2, axis=0)
        inp = [w_batch, states]

        # Predict both the model and target q-values
        model_q = self.model.predict(inp, batch_size=200)
        target_q = self.target_model.predict(inp, batch_size=200)

        return model_q, target_q, w_batch, states

    def update_priorities(self, batch, ids, ignore_dupe=False, pr=False):
        """Given a batch of transitions, this method computes each transition's
        error and uses that error to update its priority in the replay buffer

        Arguments:
            batch {list} -- list of transitions
            ids {list} -- list of identifiers of each transition in the replay
                        buffer

        Returns:
            float -- The batch's mean loss
        """

        model_q, target_q, weights, _ = self.get_training_data(batch)

        errors = np.zeros(len(batch))

        for i, (_, action, reward, _, terminal,
                extra) in enumerate(batch):

            target = np.copy(reward)

            if self.has_scalar_qvalues():
                target = np.dot(target, weights[i * 2])

            if not terminal:
                if self.alg == "naive":
                    for o in range(self.obj_cnt):
                        max_a = np.argmax(model_q[i * 2 + 1][:, o])
                        target[o] += self.discount * \
                            target_q[i * 2 + 1][max_a][o]
                else:
                    if self.has_scalar_qvalues():
                        next_action = np.argmax(model_q[i * 2 + 1])
                    else:
                        next_action = np.argmax(
                            np.dot(model_q[i * 2 + 1], weights[i * 2 + 1]))
                    target += self.discount * \
                        target_q[i * 2 + 1][next_action]

            error = mae(model_q[i * 2][action], target)

            errors[i] = error

            # When dupe is True, we train each sample on two weight vectors
            # Hence, there are two TD-errors per sample, we use the mean of
            # both errors to update the priorities
            if self.duplicate():
                if i % 2 == 0:
                    continue
                error = (error + errors[i - 1]) / 2
            self.buffer.update(ids[i], error)
            self.max_error = max(error, self.max_error)

        return np.mean(errors)

    def has_scalar_qvalues(self):
        return "scal" in self.alg or "uvfa" in self.alg

    def save_weights(self):
        """Saves the networks' weights to files identified by the agent's name
        and the current weight vector
        """

        self.model.save_weights(
            "output/networks/{}_{}.weights".format(self.name(), self.weights))


class MultiNetwork():
    def __init__(self, agent, shared_length=None, reuse='full'):
        """Handles both multi-network approaches by keeping track of trained
        policies and storing/loading network parameters as necessary.

        Arguments:
            agent {DeepAgent} -- The DeepAgent that is being trained

        Keyword Arguments:
            shared_length {int} -- The number of layers that are shared by all policies (i.e., the length of the feature extraction for the multi-head architecture) (default: {None})
            reuse {str} -- The type of reuse, 'full', 'proportional' or 'sectionned' (default: {'full'})
        """

        self.policies = {}
        self.weight_policies = {0: 0}
        self.agent = agent
        if shared_length is None:
            shared_length = self.agent.shared_length

        self.shared_length = shared_length
        self.reuse = reuse

        # Note that we assume MultiNetwork gets initialized before any training has
        # occured, i.e., when the weights are still random
        self.random_head = get_weights(self.agent.model, "head_out")
        self.random_head_target = get_weights(self.agent.target_model,
                                              "head_out")

    def save_policy_for_weight(self, weight_id, policy):
        """Save the given policy for the given weight index

        Arguments:
            weight_id {int} -- The weight's identifier
            policy {dict} -- The policy to save
        """

        # Decrement the old policy's optimality counter
        if weight_id in self.weight_policies:
            old_policy_id = self.weight_policies[weight_id]
            if old_policy_id in self.policies:
                self.policies[old_policy_id]["optimal_for"] -= 1

        self.weight_policies[weight_id] = policy["id"]

        # Save the new policy and increment its optimality counter
        if policy["id"] not in self.policies:
            if policy["parameters"] == None:
                policy["parameters"] = self.agent.get_network_weights(
                    start_layer=self.shared_length)
            self.policies[policy["id"]] = policy
        self.policies[policy["id"]]["optimal_for"] += 1

    def purge(self):
        """Removes policies that are optimal for none of the weights
        """
        # Remove all policies that aren't optimal for any weights
        for k in list(self.policies.keys()):
            if self.policies[k]["optimal_for"] <= 0:
                del self.policies[k]

    def add_new_policy(self,
                       new_policy_value,
                       weights,
                       weight_history,
                       model_policy_id=None,
                       override=False):
        """Stores a new policy, this involves checking if the new policy is
        optimal for any of the past weights and then deleting policies made 
        redundant by this new policy

        Arguments:
            new_policy_value {np.array} -- The policy's value
            weights {np.array} -- The weights the policy was trained on

        Keyword Arguments:
            model_policy_id {int} -- The model policy's id (default: {None})
            override {bool} -- If True, this policy is optimal for all weights] (default: {False})
        """

        new_policy = {
            "id": len(weights),
            "updates": 0,
            "value": new_policy_value + 0,
            "optimal_for": 0,
            "parameters": None,
            "weights": np.copy(weights[-1])
        }

        # The new policy has been trained during one more weight than its model
        if model_policy_id is not None:
            new_policy[
                "updates"] = self.policies[model_policy_id]["updates"] + 1
        # Loop over all weights to find out if the newly trained policy is
        # optimal for any of them
        for j, weight in enumerate(weights):
            if j not in self.weight_policies:
                continue

            optimal_policy = self.weight_policies[j]
            if optimal_policy in self.policies:
                opt_best_value = np.dot(self.policies[optimal_policy]["value"],
                                        weight)

                new_best_value = np.dot(new_policy_value, weight)

                if new_best_value > opt_best_value + KAPPA or override:
                    self.save_policy_for_weight(j, new_policy)
            else:
                self.save_policy_for_weight(j, new_policy)

        self.purge()


    def load_parameters(self, model_pol_index, weights):
        """Load the parameter's of the given model policy
        With proportional reuse, a similarity is computed between the given 
        weights and the weights the policy was trained on to determine how much
        of the parameters is copied.

        Arguments:
            model_pol_index {int} -- The model policy's identifier
            weights {np.array} -- The new weight vector
        """

        model_weights = self.policies[model_pol_index]["weights"]

        # Determine the copy rate, only applies to the head's parameters
        if self.reuse == 'sectionned':
            copy_rate = 0
        elif self.reuse == 'proportional':
            copy_rate = 1 - \
                scipy.spatial.distance.cosine(model_weights, weights)
        else:
            copy_rate = 1

        model_layers, target_layers = self.policies[model_pol_index][
            "parameters"]

        # Copy the model's parameters to the agent's networks
        for dest, src in zip(self.agent.model.layers[self.shared_length:],
                             model_layers):
            dest.set_weights(src)
        for dest, src in zip(
                self.agent.target_model.layers[self.shared_length:],
                target_layers):
            dest.set_weights(src)

        # Shuffle random heads as a way of "reinitializing" them
        for i in range(len(self.random_head)):
            np.random.shuffle(self.random_head[i])
            np.random.shuffle(self.random_head_target[i])

        # (Partially) override the copied parameters by random ones
        merge_weights(self.agent.model, self.random_head, copy_rate)
        merge_weights(self.agent.target_model, self.random_head_target,
                      copy_rate)

    def load_optimal_policy(self, weights, weights_id):
        """Given a weight vector, this method determines the optimal existing 
        policy then loads that policy's parameters into the agent's networks

        Arguments:
            weights {np.array} -- The new weight vector
        """

        model_pol_index = 0
        max_scal_value = -INF

        # Loop over all policies to determine the optimal one for the given weights
        for k in sorted(self.policies):
            policy = self.policies[k]
            scal_value = np.max(np.dot(policy["value"], weights))
            if scal_value > max_scal_value:
                model_pol_index = k
                max_scal_value = scal_value

        # Load the model policy's parameters
        self.load_parameters(model_pol_index, weights)

        # Set the model policy as optimal for the new weights
        self.save_policy_for_weight(weights_id, self.policies[model_pol_index])
