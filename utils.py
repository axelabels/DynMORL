from __future__ import print_function
import os
import random
import re
import time
import numpy as np
import errno
import os

INF = float("Inf")


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def linear_anneal(t, anneal_steps, start_e, end_e, start_steps):
    """
        Linearly anneals epsilon

        Args:
            t: Current time
            anneal_steps: Number of steps to anneal over
            start_e: Initial epsilon
            end_e: Final epsilon
            start_steps: Number of initial steps without annealing
    """
    assert end_e <= start_e
    t = max(0, t - start_steps)
    return max(end_e,
               (anneal_steps - t) * (start_e - end_e) / anneal_steps + end_e)


def random_weights(w_cnt, op=lambda l: l):
    """ Generate random normalized weights 

        Args:
            w_cnt: size of the weight vector
            op: additional operation to perform on the generated weight vector
    """
    weights = np.random.random(w_cnt)
    weights = op(weights)
    weights /= np.sum(weights)

    return weights


def crowd_dist(datas):
    """Given a list of vectors, this method computes the crowding distance of each vector, i.e. the sum of distances between neighbors for each dimension

    Arguments:
        datas {list} -- list of vectors

    Returns:
        list -- list of crowding distances
    """

    points = np.array([Object() for _ in datas])
    dimensions = len(datas[0])
    for i, d in enumerate(datas):
        points[i].data = d
        points[i].i = i
        points[i].distance = 0.

    # Compute the distance between neighbors for each dimension and add it to
    # each point's global distance
    for d in range(dimensions):
        points = sorted(points, key=lambda p: p.data[d])
        spread = points[-1].data[d] - points[0].data[d]
        for i, p in enumerate(points):
            if i == 0 or i == len(points) - 1:
                p.distance += INF
            else:
                p.distance += (
                    points[i + 1].data[d] - points[i - 1].data[d]) / spread

    # Sort points back to their original order
    points = sorted(points, key=lambda p: p.i)
    distances = np.array([p.distance for p in points])

    return distances


def arr2str(array):
    """Converts an array into a one line string

    Arguments:
        array {array} -- Array to convert

    Returns:
        str -- The string representation
    """

    return re.sub(r'\s+', ' ',
                  str(array).replace('\r', '').replace('\n', '').replace(
                      "array", "").replace("\t", " "))


def get_weights(model, key=None):
    """
        Copy the model's weights into a dictionary, filtering on the name if a key is provided

        Arguments:
            model: The model to save weights from
            key: The key on which to filter the saved weights, only layers containing the key in their name are saved
    """
    weight_dic = {}
    for layer in model.layers:
        if key and key in layer.name:
            weight_dic[layer.name] = layer.get_weights()
    return weight_dic


def merge_weights(model, weight_dic, p=0):
    """
        Merge weight_dic by name into model.

        Arguments:
            model: The model into which the weights are merged
            weight_dic: A dictionary of weights keyed by layer names to merge into the model
            p: Relative importance of the model's weights when merging (if p=0, weight_dic is copied entirely, if p=1, the model's weights are preserved completely)
    """
    if p == 1:
        return
    for layer in model.layers:
        if layer.name in weight_dic or hasattr(layer, 'kernel_initializer'):
            if hasattr(layer, 'kernel_initializer'):
                import keras.backend as K
                new_weights = glorot_init(layer.get_weights()[0].shape)
                bias = K.get_value(K.zeros(layer.get_weights()[1].shape))
                layer.set_weights([new_weights, bias])
                weights = [new_weights, bias]
            else:
                weights = weight_dic[layer.name]
            trg = layer.get_weights()
            src = weights
            merged = []
            for t, s in zip(trg, src):
                merged.append(np.array(t) * p + np.array(s) * (p - 1))
            layer.set_weights(merged)


def glorot_init(shape, seed=0):
    assert len(shape) == 2
    from keras.initializers import _compute_fans
    fan_in, fan_out = _compute_fans(shape)
    scale = 1 / max(1., float(fan_in + fan_out) / 2)

    limit = np.sqrt(3. * scale)
    return np.random.uniform(-limit, limit, shape)


def mae(truth, prediction):
    """Computes the Mean Absolute Error between two arrays

    Arguments:
        truth {array} -- Truth
        prediction {array} -- Prediction

    Returns:
        float -- The Mean Absolute Error
    """

    return np.abs(truth - prediction).mean()


class Log(object):
    """Logs the training progress

    """

    def __init__(self, log_file, agent):
        """Initializes a log object

        Arguments:
            log_file {str} -- name of the file to which log should be written
            agent {DeepAgent} -- Agent being trained
        """

        self.accumulated_rewards = []
        self.total_steps = 0
        self.agent = agent
        self.episodes = 1
        self.episode_steps = 0
        self.log_file = log_file
        self.start_time = time.time()
        self.episode_rewards = []
        self.losses = []
        self.max_qs = []
        self.scal_acc_rewards = []
        self.opt_rewards = []
        self.straight_q = 0

    def log_step(self, env, total_steps, loss, reward, terminal, state,
                 next_state, weights, discount, episode_steps, epsilon,
                 frame_skip, content, action):

        q_values = self.agent.predict(next_state)

        if self.agent.has_scalar_qvalues():
            self.max_qs.append(np.max(q_values))
        else:
            self.max_qs.append(np.max(np.dot(q_values, weights)))

        self.losses.append(loss)

        self.episode_rewards.append(reward)
        self.accumulated_rewards.append(reward)
        self.scal_acc_rewards.append(np.dot(reward, weights))

        if terminal:
            mean_q = np.mean(q_values, axis=0)

            episode_log_length = int(episode_steps)
            elapsed = time.time() - self.start_time
            losses = self.losses[-episode_log_length:] or [0]
            prefix = "episode"
            rewards_to_log = self.scal_acc_rewards[-episode_log_length:]

            disc_actual = np.sum(
                np.array(
                    [r * discount**i for i, r in enumerate(self.accumulated_rewards[-episode_log_length:])]),
                axis=0)
            actual = np.sum(
                np.array((self.accumulated_rewards[-episode_log_length:])),
                axis=0)
            episode_line = ";".join(map(str, [
                prefix, total_steps, episode_steps, epsilon, elapsed,
                np.dot(disc_actual, weights),
                disc_actual,
                actual,
                np.nanmean(losses),
                np.mean(self.max_qs[-episode_log_length:]),
                weights, mean_q]))
            print(episode_line)
            print(episode_line,
                  file=self.log_file)

        LOG_INTERVAL = 100

        if total_steps > 0 and total_steps % int(
                LOG_INTERVAL) == 0:
            elapsed = time.time() - self.start_time
            losses = self.losses[-LOG_INTERVAL:] or [0]
            prefix = "logs"
            rewards_to_log = self.scal_acc_rewards[-LOG_INTERVAL:]
            log_line = ";".join(map(str, [
                prefix, total_steps, episode_steps, epsilon, elapsed,
                np.sum(rewards_to_log),
                np.sum(
                    np.array(self.accumulated_rewards[-LOG_INTERVAL:]), axis=0),
                np.nanmean(losses),
                np.mean(self.max_qs[-LOG_INTERVAL:]),
                np.mean(self.opt_rewards[-LOG_INTERVAL:]), weights, self.straight_q]))
            print(log_line)
            print(log_line,
                  file=self.log_file)

        if terminal or total_steps % LOG_INTERVAL == 0:
            rng_max = -max(total_steps % LOG_INTERVAL, episode_steps)
            self.episode_rewards = self.episode_rewards[rng_max:]
            self.losses = self.losses[rng_max:]
            self.max_qs = self.max_qs[rng_max:]
            self.scal_acc_rewards = self.scal_acc_rewards[rng_max:]
            self.opt_rewards = self.opt_rewards[rng_max:]

        if terminal:
            self.episode_rewards = []
            self.losses = []
            self.max_qs = []
            self.scal_acc_rewards = []
            self.opt_rewards = []


def generate_weights(count=1, n=3, m=1):
    all_weights = []

    target = np.random.dirichlet(np.ones(n), 1)[0]
    prev_t = target
    for _ in range(count // m):
        target = np.random.dirichlet(np.ones(n), 1)[0]
        if m == 1:
            all_weights.append(target)
        else:
            for i in range(m):
                i_w = target * (i + 1) / float(m) + prev_t * \
                    (m - i - 1) / float(m)
                all_weights.append(i_w)
        prev_t = target + 0.

    return all_weights


def mag(vector2d):
    return np.sqrt(np.dot(vector2d, vector2d))


def clip(val, lo, hi):
    return lo if val <= lo else hi if val >= hi else val


def scl(c):
    return (c[0] / 255., c[1] / 255., c[2] / 255.)


def truncated_mean(mean, std, a, b):
    if std == 0:
        return mean
    from scipy.stats import norm
    a = (a - mean) / std
    b = (b - mean) / std
    PHIB = norm.cdf(b)
    PHIA = norm.cdf(a)
    phib = norm.pdf(b)
    phia = norm.pdf(a)

    trunc_mean = (mean + ((phia - phib) / (PHIB - PHIA)) * std)
    return trunc_mean


def compute_angle(p0, p1, p2):
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    return np.degrees(angle)


def pareto_filter(costs, minimize=True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    from https://stackoverflow.com/a/40239615
    """
    costs_copy = np.copy(costs) if minimize else -np.copy(costs)
    is_efficient = np.arange(costs_copy.shape[0])
    n_points = costs_copy.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs_copy):
        nondominated_point_mask = np.any(
            costs_copy < costs_copy[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        # Remove dominated points
        is_efficient = is_efficient[nondominated_point_mask]
        costs_copy = costs_copy[nondominated_point_mask]
        next_point_index = np.sum(
            nondominated_point_mask[:next_point_index]) + 1
    return [costs[i] for i in is_efficient]


class Object(object):
    """
        Generic object
    """
    pass
