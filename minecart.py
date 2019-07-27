from __future__ import print_function

from scipy.spatial import ConvexHull
import math
import random
from utils import truncated_mean, compute_angle, pareto_filter

import itertools
import numpy as np
import scipy.stats
import sys
from scipy.stats import norm
from math import ceil

try:
    import cairocffi as cairo
    CAIRO = True
    print("Using cairocffi backend", file=sys.stderr)
except:
    print("Failed to import cairocffi, trying cairo...", file=sys.stderr)
    try:
        import cairo
        CAIRO = True
        print("Using cairo backend", file=sys.stderr)
    except:
        print("Failed to import cairo, trying pygame...", file=sys.stderr)
        import pygame
        CAIRO = False
        print("Using pygame backend", file=sys.stderr)


EPS_SPEED = 0.001  # Minimum speed to be considered in motion
HOME_X = .0
HOME_Y = .0
HOME_POS = (HOME_X, HOME_Y)

ROTATION = 10
MAX_SPEED = 1.

FUEL_MINE = -.05
FUEL_ACC = -.025
FUEL_IDLE = -0.005

CAPACITY = 1

ACT_MINE = 0
ACT_LEFT = 1
ACT_RIGHT = 2
ACT_ACCEL = 3
ACT_BRAKE = 4
ACT_NONE = 5
FUEL_LIST = [FUEL_MINE + FUEL_IDLE, FUEL_IDLE, FUEL_IDLE,
             FUEL_IDLE + FUEL_ACC, FUEL_IDLE, FUEL_IDLE]
FUEL_DICT = {ACT_MINE: FUEL_MINE + FUEL_IDLE, ACT_LEFT: FUEL_IDLE, ACT_RIGHT: FUEL_IDLE,
             ACT_ACCEL: FUEL_IDLE + FUEL_ACC, ACT_BRAKE: FUEL_IDLE, ACT_NONE: FUEL_IDLE}
ACTIONS = ["Mine", "Left", "Right", "Accelerate", "Brake", "None"]
ACTION_COUNT = len(ACTIONS)


MINE_RADIUS = 0.14
BASE_RADIUS = 0.15

WIDTH = 480
HEIGHT = 480

# Color definitions
WHITE = (255, 255, 255)
GRAY = (150, 150, 150)
C_GRAY = (150 / 255., 150 / 255., 150 / 255.)
DARK_GRAY = (100, 100, 100)
BLACK = (0, 0, 0)
RED = (255, 70, 70)
C_RED = (1., 70 / 255., 70 / 255.)

FPS = 180

MINE_LOCATION_TRIES = 100

MINE_SCALE = 1.
BASE_SCALE = 1.
CART_SCALE = 1.

MARGIN = 0.16 * CART_SCALE

ACCELERATION = 0.0075 * CART_SCALE
DECELERATION = 1

CART_IMG = 'images/cart.png'
MINE_IMG = 'images/mine.png'


class Mine():
    """Class representing an individual Mine
    """

    def __init__(self, ore_cnt, x, y):
        self.distributions = [
            scipy.stats.norm(np.random.random(), np.random.random())
            for _ in range(ore_cnt)
        ]
        self.pos = np.array((x, y))

    def distance(self, cart):
        return mag(cart.pos - self.pos)

    def mineable(self, cart):
        return self.distance(cart) <= MINE_RADIUS * MINE_SCALE * CART_SCALE

    def mine(self):
        """Generates collected resources according to the mine's random
        distribution

        Returns:
            list -- list of collected ressources
        """

        return [max(0., dist.rvs()) for dist in self.distributions]

    def distribution_means(self):
        """
            Computes the mean of the truncated normal distributions
        """
        means = np.zeros(len(self.distributions))

        for i, dist in enumerate(self.distributions):
            mean, std = dist.mean(), dist.std()
            means[i] = truncated_mean(mean, std, 0, float("inf"))
            if np.isnan(means[i]):
                means[i] = 0
        return means


class Cart():
    """Class representing the actual minecart
    """

    def __init__(self, ore_cnt):
        self.ore_cnt = ore_cnt
        self.pos = np.array([HOME_X, HOME_Y])
        self.speed = 0
        self.angle = 45
        self.content = np.zeros(self.ore_cnt)
        self.departed = False  # Keep track of whether the agent has left the base

    def accelerate(self, acceleration):
        self.speed = clip(self.speed + acceleration, 0, MAX_SPEED)

    def rotate(self, rotation):
        self.angle = (self.angle + rotation) % 360

    def step(self):
        """
            Update cart's position, taking the current speed into account
            Colliding with a border at anything but a straight angle will cause
            cart to "slide" along the wall.
        """

        pre = np.copy(self.pos)
        if self.speed < EPS_SPEED:
            return False
        x_velocity = self.speed * math.cos(self.angle * math.pi / 180)
        y_velocity = self.speed * math.sin(self.angle * math.pi / 180)
        x, y = self.pos
        if y != 0 and y != 1 and (y_velocity > 0 + EPS_SPEED or
                                  y_velocity < 0 - EPS_SPEED):
            if x == 1 and x_velocity > 0:
                self.angle += math.copysign(ROTATION, y_velocity)
            if x == 0 and x_velocity < 0:
                self.angle -= math.copysign(ROTATION, y_velocity)
        if x != 0 and x != 1 and (x_velocity > 0 + EPS_SPEED or
                                  x_velocity < 0 - EPS_SPEED):
            if y == 1 and y_velocity > 0:
                self.angle -= math.copysign(ROTATION, x_velocity)

            if y == 0 and y_velocity < 0:
                self.angle += math.copysign(ROTATION, x_velocity)

        self.pos[0] = clip(x + x_velocity, 0, 1)
        self.pos[1] = clip(y + y_velocity, 0, 1)
        self.speed = mag(pre - self.pos)

        return True


class Minecart:
    """Minecart environment
    """

    a_space = ACTION_COUNT

    def __init__(self,
                 mine_cnt=3,
                 ore_cnt=2,
                 capacity=CAPACITY,
                 mine_distributions=None,
                 ore_colors=None):

        # Initialize graphics backend
        if CAIRO:
            self.surface = cairo.ImageSurface(cairo.FORMAT_RGB24, WIDTH,
                                              HEIGHT)
            self.context = cairo.Context(self.surface)
            self.initialized = True
        else:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()

            self.cart_sprite = pygame.sprite.Sprite()
            self.cart_sprites = pygame.sprite.Group()
            self.cart_sprites.add(self.cart_sprite)
            self.cart_image = pygame.transform.rotozoom(
                pygame.image.load(CART_IMG).convert_alpha(), 0,
                CART_SCALE)

        self.capacity = capacity
        self.ore_cnt = ore_cnt
        self.ore_colors = ore_colors or [(np.random.randint(
            40, 255), np.random.randint(40, 255), np.random.randint(40, 255))
            for i in range(self.ore_cnt)]

        self.mine_cnt = mine_cnt
        self.generate_mines(mine_distributions)
        self.cart = Cart(self.ore_cnt)

        self.end = False

    def obj_cnt(self):
        return self.ore_cnt + 1

    def convex_coverage_set(self, frame_skip=1, discount=0.98, incremental_frame_skip=True, symmetric=True):
        """
            Computes an approximate convex coverage set

            Keyword Arguments:
                frame_skip {int} -- How many times each action is repeated (default: {1})
                discount {float} -- Discount factor to apply to rewards (default: {1})
                incremental_frame_skip {bool} -- Wether actions are repeated incrementally (default: {1})
                symmetric {bool} -- If true, we assume the pattern of accelerations from the base to the mine is the same as from the mine to the base (default: {True})

            Returns:
                The convex coverage set 
        """
        policies = self.pareto_coverage_set(
            frame_skip, discount, incremental_frame_skip, symmetric)
        origin = np.min(policies, axis=0)
        extended_policies = [origin] + policies
        return [policies[idx - 1] for idx in ConvexHull(extended_policies).vertices if idx != 0]

    def pareto_coverage_set(self, frame_skip=1, discount=0.98, incremental_frame_skip=True, symmetric=True):
        """
            Computes an approximate pareto coverage set

            Keyword Arguments:
                frame_skip {int} -- How many times each action is repeated (default: {1})
                discount {float} -- Discount factor to apply to rewards (default: {1})
                incremental_frame_skip {bool} -- Wether actions are repeated incrementally (default: {1})
                symmetric {bool} -- If true, we assume the pattern of accelerations from the base to the mine is the same as from the mine to the base (default: {True})

            Returns:
                The pareto coverage set
        """
        all_rewards = []
        base_perimeter = BASE_RADIUS * BASE_SCALE

        # Empty mine just outside the base
        virtual_mine = Mine(self.ore_cnt, (base_perimeter**2 / 2)
                            ** (1 / 2), (base_perimeter**2 / 2)**(1 / 2))
        virtual_mine.distributions = [
            scipy.stats.norm(0, 0)
            for _ in range(self.ore_cnt)
        ]
        for mine in (self.mines + [virtual_mine]):
            mine_distance = mag(mine.pos - HOME_POS) - \
                MINE_RADIUS * MINE_SCALE - BASE_RADIUS * BASE_SCALE / 2

            # Number of rotations required to face the mine
            angle = compute_angle(mine.pos, HOME_POS, [1, 1])
            rotations = int(ceil(abs(angle) / (ROTATION * frame_skip)))

            # Build pattern of accelerations/nops to reach the mine
            # initialize with single acceleration
            queue = [{"speed": ACCELERATION * frame_skip, "dist": mine_distance - frame_skip *
                      (frame_skip + 1) / 2 * ACCELERATION if incremental_frame_skip else mine_distance - ACCELERATION * frame_skip * frame_skip, "seq": [ACT_ACCEL]}]
            trimmed_sequences = []

            while len(queue) > 0:
                seq = queue.pop()
                # accelerate
                new_speed = seq["speed"] + ACCELERATION * frame_skip
                accelerations = new_speed / ACCELERATION
                movement = accelerations * (accelerations + 1) / 2 * ACCELERATION - (
                    accelerations - frame_skip) * ((accelerations - frame_skip) + 1) / 2 * ACCELERATION
                dist = seq["dist"] - movement
                speed = new_speed
                if dist <= 0:
                    trimmed_sequences.append(seq["seq"] + [ACT_ACCEL])
                else:
                    queue.append({"speed": speed, "dist": dist,
                                  "seq": seq["seq"] + [ACT_ACCEL]})
                # idle
                dist = seq["dist"] - seq["speed"] * frame_skip

                if dist <= 0:
                    trimmed_sequences.append(seq["seq"] + [ACT_NONE])
                else:
                    queue.append(
                        {"speed": seq["speed"], "dist": dist, "seq": seq["seq"] + [ACT_NONE]})

            # Build rational mining sequences
            mine_means = mine.distribution_means() * frame_skip
            mn_sum = np.sum(mine_means)
            # on average it takes up to this many actions to fill cart
            max_mine_actions = 0 if mn_sum == 0 else int(
                ceil(self.capacity / mn_sum))

            # all possible mining sequences (i.e. how many times we mine)
            mine_sequences = [[ACT_MINE] *
                              i for i in range(1, max_mine_actions + 1)]

            # All possible combinations of actions before, during and after mining
            if len(mine_sequences) > 0:
                if not symmetric:
                    all_sequences = map(
                        lambda sequences: list(sequences[0]) + list(sequences[1]) + list(
                            sequences[2]) + list(
                            sequences[3]) + list(
                            sequences[4]),
                        itertools.product([[ACT_LEFT] * rotations],
                                          trimmed_sequences,
                                          [[ACT_BRAKE] + [ACT_LEFT] *
                                              (180 // (ROTATION * frame_skip))],
                                          mine_sequences,
                                          trimmed_sequences)
                    )

                else:
                    all_sequences = map(
                        lambda sequences: list(sequences[0]) + list(sequences[1]) + list(
                            sequences[2]) + list(
                            sequences[3]) + list(
                            sequences[1]),
                        itertools.product([[ACT_LEFT] * rotations],
                                          trimmed_sequences,
                                          [[ACT_BRAKE] + [ACT_LEFT] *
                                              (180 // (ROTATION * frame_skip))],
                                          mine_sequences)
                    )
            else:
                if not symmetric:
                    print([ACT_NONE] + trimmed_sequences[1:],
                          trimmed_sequences[1:], trimmed_sequences)
                    all_sequences = map(
                        lambda sequences: list(sequences[0]) + list(sequences[1]) + list(
                            sequences[2]) + [ACT_NONE] + list(
                            sequences[3])[1:],
                        itertools.product([[ACT_LEFT] * rotations],
                                          trimmed_sequences,
                                          [[ACT_LEFT] *
                                              (180 // (ROTATION * frame_skip))],
                                          trimmed_sequences)
                    )

                else:
                    all_sequences = map(
                        lambda sequences: list(sequences[0]) + list(sequences[1]) + list(
                            sequences[2]) + [ACT_NONE] + list(
                            sequences[1][1:]),
                        itertools.product([[ACT_LEFT] * rotations],
                                          trimmed_sequences,
                                          [[ACT_LEFT] * (180 // (ROTATION * frame_skip))])
                    )

            # Compute rewards for each sequence
            fuel_costs = np.array([f * frame_skip for f in FUEL_LIST])

            def maxlen(l):
                if len(l) == 0:
                    return 0
                return max([len(s) for s in l])

            longest_pattern = maxlen(trimmed_sequences)
            max_len = rotations + longest_pattern + 1 + \
                (180 // (ROTATION * frame_skip)) + \
                maxlen(mine_sequences) + longest_pattern
            discount_map = discount**np.arange(max_len)
            for s in all_sequences:
                reward = np.zeros((len(s), self.obj_cnt()))
                reward[:, -1] = fuel_costs[s]
                mine_actions = s.count(ACT_MINE)
                reward[-1, :-1] = mine_means * mine_actions / \
                    max(1, (mn_sum * mine_actions) / self.capacity)

                reward = np.dot(discount_map[:len(s)], reward)
                all_rewards.append(reward)

            all_rewards = pareto_filter(all_rewards, minimize=False)

        return all_rewards

    @staticmethod
    def from_json(filename):
        """
            Generate a Minecart instance from a json configuration file
            Args:
                filename: JSON configuration filename
        """
        import json
        data = json.load(open(filename))
        ore_colors = None if "ore_colors" not in data else data["ore_colors"]
        minecart = Minecart(
            ore_cnt=data["ore_cnt"],
            mine_cnt=data["mine_cnt"],
            capacity=data["capacity"],
            ore_colors=ore_colors)

        if "mines" in data:
            for mine_data, mine in zip(data["mines"], minecart.mines):
                mine.pos = np.array([mine_data["x"], mine_data["y"]])
                if "distributions" in mine_data:
                    mine.distributions = [
                        scipy.stats.norm(dist[0], dist[1])
                        for dist in mine_data["distributions"]
                    ]
            minecart.initialize_mines()
        return minecart

    def generate_mines(self, mine_distributions=None):
        """
            Randomly generate mines that don't overlap the base
            TODO: propose some default formations
        """
        self.mines = []
        for i in range(self.mine_cnt):
            pos = np.array((np.random.random(), np.random.random()))

            tries = 0
            while (mag(pos - HOME_POS) < BASE_RADIUS * BASE_SCALE + MARGIN) and (tries < MINE_LOCATION_TRIES):
                pos[0] = np.random.random()
                pos[1] = np.random.random()
                tries += 1
            assert tries < MINE_LOCATION_TRIES
            self.mines.append(Mine(self.ore_cnt, *pos))
            if mine_distributions:
                self.mines[i].distributions = mine_distributions[i]

        self.initialize_mines()

    def initialize_mines(self):
        """Assign a random rotation to each mine, and initialize the necessary sprites
        for the Pygame backend
        """

        for mine in self.mines:
            mine.rotation = np.random.randint(0, 360)

        if not CAIRO:
            self.mine_sprites = pygame.sprite.Group()
            self.mine_rects = []
            for mine in self.mines:
                mine_sprite = pygame.sprite.Sprite()
                mine_sprite.image = pygame.transform.rotozoom(
                    pygame.image.load(MINE_IMG), mine.rotation,
                    MINE_SCALE).convert_alpha()
                self.mine_sprites.add(mine_sprite)
                mine_sprite.rect = mine_sprite.image.get_rect()
                mine_sprite.rect.centerx = (
                    mine.pos[0] * (1 - 2 * MARGIN)) * WIDTH + MARGIN * WIDTH
                mine_sprite.rect.centery = (
                    mine.pos[1] * (1 - 2 * MARGIN)) * HEIGHT + MARGIN * HEIGHT
                self.mine_rects.append(mine_sprite.rect)

    def step(self, action, frame_skip=1, incremental_frame_skip=True):
        """Perform the given action `frame_skip` times
         ["Mine", "Left", "Right", "Accelerate", "Brake", "None"]
        Arguments:
            action {int} -- Action to perform, ACT_MINE (0), ACT_LEFT (1), ACT_RIGHT (2), ACT_ACCEL (3), ACT_BRAKE (4) or ACT_NONE (5)

        Keyword Arguments:
            frame_skip {int} -- Repeat the action this many times (default: {1})
            incremental_frame_skip {int} -- If True, frame_skip actions are performed in succession, otherwise the repeated actions are performed simultaneously (e.g., 4 accelerations are performed and then the cart moves).

        Returns:
            tuple -- (state, reward, terminal) tuple
        """
        change = False  # Keep track of whether the state has changed

        if action < 0 or action >= ACTION_COUNT:
            action = ACT_NONE

        reward = np.zeros(self.ore_cnt + 1)
        if frame_skip < 1:
            frame_skip = 1

        reward[-1] = FUEL_IDLE * frame_skip

        if action == ACT_ACCEL:
            reward[-1] += FUEL_ACC * frame_skip
        elif action == ACT_MINE:
            reward[-1] += FUEL_MINE * frame_skip

        for _ in range(frame_skip if incremental_frame_skip else 1):

            if action == ACT_LEFT:
                self.cart.rotate(-ROTATION *
                                 (1 if incremental_frame_skip else frame_skip))
                change = True
            elif action == ACT_RIGHT:
                self.cart.rotate(
                    ROTATION * (1 if incremental_frame_skip else frame_skip))
                change = True
            elif action == ACT_ACCEL:
                self.cart.accelerate(
                    ACCELERATION * (1 if incremental_frame_skip else frame_skip))
            elif action == ACT_BRAKE:
                self.cart.accelerate(-DECELERATION *
                                     (1 if incremental_frame_skip else frame_skip))
            elif action == ACT_MINE:
                for _ in range(1 if incremental_frame_skip else frame_skip):
                    change = self.mine() or change

            if self.end:
                break

            for _ in range(1 if incremental_frame_skip else frame_skip):
                change = self.cart.step() or change

            distanceFromBase = mag(self.cart.pos - HOME_POS)
            if distanceFromBase < BASE_RADIUS * BASE_SCALE:
                if self.cart.departed:
                    # Cart left base then came back, ending the episode
                    self.end = True
                    # Sell resources
                    reward[:self.ore_cnt] += self.cart.content
                    self.cart.content = np.zeros(self.ore_cnt)
            else:
                # Cart left base
                self.cart.departed = True

        if not self.end and change:
            self.render()

        return self.get_state(change), reward, self.end

    def mine(self):
        """Perform the MINE action

        Returns:
            bool -- True if something was mined
        """

        if self.cart.speed < EPS_SPEED:
            # Get closest mine
            mine = min(self.mines, key=lambda mine: mine.distance(self.cart))

            if mine.mineable(self.cart):
                cart_free = self.capacity - np.sum(self.cart.content)
                mined = mine.mine()
                total_mined = np.sum(mined)
                if total_mined > cart_free:
                    # Scale mined content to remaining capacity
                    scale = cart_free / total_mined
                    mined = np.array(mined) * scale

                self.cart.content += mined

                if np.sum(mined) > 0:
                    return True
        return False

    def get_pixels(self, update=True):
        """Get the environment's image representation

        Keyword Arguments:
            update {bool} -- Whether to redraw the environment (default: {True})

        Returns:
            np.array -- array of pixels, with shape (width, height, channels)
        """

        if update:
            if CAIRO:
                self.pixels = np.array(self.surface.get_data()).reshape(
                    WIDTH, HEIGHT, 4)[:, :, [2, 1, 0]]
            else:
                self.pixels = pygame.surfarray.array3d(self.screen)

        return self.pixels

    def get_state(self, update=True):
        """Returns the environment's full state, including the cart's position,
        its speed, its orientation and its content, as well as the environment's
        pixels

        Keyword Arguments:
            update {bool} -- Whether to update the representation (default: {True})

        Returns:
            dict -- dict containing the aforementioned elements
        """

        return {
            "position": self.cart.pos,
            "speed": self.cart.speed,
            "orientation": self.cart.angle,
            "content": self.cart.content,
            "pixels": self.get_pixels(update)
        }

    def reset(self):
        """Resets the environment to the start state

        Returns:
            [type] -- [description]
        """

        self.cart.content = np.zeros(self.ore_cnt)
        self.cart.pos = np.array(HOME_POS)
        self.cart.speed = 0
        self.cart.angle = 45
        self.cart.departed = False
        self.end = False
        self.render()
        return self.get_state()

    def __str__(self):
        string = "Completed: {} ".format(self.end)
        string += "Departed: {} ".format(self.cart.departed)
        string += "Content: {} ".format(self.cart.content)
        string += "Speed: {} ".format(self.cart.speed)
        string += "Direction: {} ({}) ".format(self.cart.angle,
                                               self.cart.angle * math.pi / 180)
        string += "Position: {} ".format(self.cart.pos)
        return string

    def render(self):
        """Update the environment's representation
        """

        if CAIRO:
            self.render_cairo()
        else:
            self.render_pygame()

    def render_cairo(self):

        # Clear canvas
        self.context.set_source_rgba(*C_GRAY)
        self.context.rectangle(0, 0, WIDTH, HEIGHT)
        self.context.fill()

        # Draw home
        self.context.set_source_rgba(*C_RED)
        self.context.arc(HOME_X, HOME_Y,
                         int(WIDTH / 3 * BASE_SCALE), 0, 2 * math.pi)
        self.context.fill()

        # Draw Mines
        for mine in self.mines:
            draw_image(self.context, MINE_IMG,
                       (mine.pos[0] * (1 - 2 * MARGIN) + MARGIN) * WIDTH,
                       (mine.pos[1] * (1 - 2 * MARGIN) + MARGIN) * HEIGHT,
                       MINE_SCALE, -mine.rotation)

        # Draw cart
        cart_x = (self.cart.pos[0] * (1 - 2 * MARGIN) + MARGIN) * WIDTH
        cart_y = (self.cart.pos[1] * (1 - 2 * MARGIN) + MARGIN) * HEIGHT
        cart_surface = draw_image(self.context, CART_IMG, cart_x,
                                  cart_y, CART_SCALE, -self.cart.angle + 90)

        # Draw cart content
        width = cart_surface.get_width() / (2 * self.ore_cnt)
        height = cart_surface.get_height() / 3
        content_width = (width + 1) * self.ore_cnt
        offset = (cart_surface.get_width() - content_width) / 2
        for i in range(self.ore_cnt):

            rect_height = height * self.cart.content[i] / self.capacity

            if rect_height >= 1:

                self.context.set_source_rgba(*scl(self.ore_colors[i]))
                self.context.rectangle(cart_y - offset / 1.5,
                                       cart_x - offset + i * (width + 1),
                                       rect_height, width)
                self.context.fill()

    def render_pygame(self):

        pygame.event.get()
        self.clock.tick(FPS)

        self.mine_sprites.update()

        # Clear canvas
        self.screen.fill(GRAY)

        # Draw Home
        pygame.draw.circle(self.screen, RED, (int(WIDTH * HOME_X), int(
            HEIGHT * HOME_Y)), int(WIDTH / 3 * BASE_SCALE))

        # Draw Mines
        self.mine_sprites.draw(self.screen)

        # Draw cart
        self.cart_sprite.image = rot_center(self.cart_image,
                                            -self.cart.angle).copy()

        self.cart_sprite.rect = self.cart_sprite.image.get_rect(
            center=(200, 200))

        self.cart_sprite.rect.centerx = self.cart.pos[0] * (
            1 - 2 * MARGIN) * WIDTH + MARGIN * WIDTH
        self.cart_sprite.rect.centery = self.cart.pos[1] * (
            1 - 2 * MARGIN) * HEIGHT + MARGIN * HEIGHT

        self.cart_sprites.update()

        self.cart_sprites.draw(self.screen)

        # Draw cart content
        width = self.cart_sprite.rect.width / (2 * self.ore_cnt)
        height = self.cart_sprite.rect.height / 3
        content_width = (width + 1) * self.ore_cnt
        offset = (self.cart_sprite.rect.width - content_width) / 2
        for i in range(self.ore_cnt):

            rect_height = height * self.cart.content[i] / self.capacity

            if rect_height >= 1:
                pygame.draw.rect(self.screen, self.ore_colors[i], (
                    self.cart_sprite.rect.left + offset + i * (width + 1),
                    self.cart_sprite.rect.top + offset * 1.5, width,
                    rect_height))

        pygame.display.update()

    @staticmethod
    def action_space():
        return range(ACTION_COUNT)


images = {}


def draw_image(ctx, image, top, left, scale, rotation):
    """Rotate, scale and draw an image on a cairo context
    """
    if image not in images:
        images[image] = cairo.ImageSurface.create_from_png(image)
    image_surface = images[image]
    img_height = image_surface.get_height()
    img_width = image_surface.get_width()
    ctx.save()
    w = img_height / 2
    h = img_width / 2

    left -= w
    top -= h

    ctx.translate(left + w, top + h)
    ctx.rotate(rotation * math.pi / 180.0)
    ctx.translate(-w, -h)

    ctx.set_source_surface(image_surface)

    ctx.scale(scale, scale)
    ctx.paint()
    ctx.restore()
    return image_surface


def rot_center(image, angle):
    """Rotate an image while preserving its center and size"""
    orig_rect = image.get_rect()
    rot_image = pygame.transform.rotate(image, angle)
    rot_rect = orig_rect.copy()
    rot_rect.center = rot_image.get_rect().center
    rot_image = rot_image.subsurface(rot_rect).copy()
    return rot_image


def mag(vector2d):
    return np.sqrt(np.dot(vector2d, vector2d))


def clip(val, lo, hi):
    return lo if val <= lo else hi if val >= hi else val


def scl(c):
    return (c[0] / 255., c[1] / 255., c[2] / 255.)
