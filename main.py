from __future__ import print_function

import os
import random
import sys
import time
from optparse import OptionParser

import numpy as np
import tensorflow as tf
from scipy import spatial

from agent import DeepAgent
from minecart import *
from utils import *
mkdir_p("output")
mkdir_p("output/logs")
mkdir_p("output/networks")
mkdir_p("output/pred")
mkdir_p("output/img")
parser = OptionParser()
parser.add_option(
    "-l",
    "--algorithm",
    dest="alg",
    choices=["scal", "mo", "mn", "cond", "uvfa", "random","naive"],
    default="scal",
    help="Architecture type, one of 'scal','mo','meta','cond'")
parser.add_option(
    "-m",
    "--memory",
    dest="mem",
    default="STD",
    choices=["STD", "DER", "SEL", "EXP"],
    help="Memory type, one of 'std','crowd','exp','sel'")
parser.add_option(
    "-d",
    "--dupe",
    dest="dupe",
    default="none",
    choices=["none", "CN", "CN-UVFA", "CN-ACTIVE"],
    help="Extra training")
parser.add_option(
    "-e",
    "--end_e",
    dest="end_e",
    default="0.05",
    help="Final epsilon value",
    type=float)
parser.add_option(
    "-r", "--lr", dest="lr", default="0.02", help="learning rate", type=float)
parser.add_option(
    "--clipnorm", dest="clipnorm", default="0", help="clipnorm", type=float)
parser.add_option(
    "--mem-a", dest="mem_a", default="2.", help="memory error exponent", type=float)
parser.add_option(
    "--mem-e", dest="mem_e", default="0.01", help="error offset", type=float)
parser.add_option(
    "--clipvalue",
    dest="clipvalue",
    default="0",
    help="clipvalue",
    type=float)
parser.add_option(
    "--momentum", dest="momentum", default=".9", help="momentum", type=float)
parser.add_option(
    "-u",
    "--update_period",
    dest="updates",
    default="4",
    help="Update interval",
    type=int)
parser.add_option(
    "--target-update",
    dest="target_update_interval",
    default="150",
    help="Target update interval",
    type=int)
parser.add_option(
    "-f",
    "--frame-skip",
    dest="frame_skip",
    default="4",
    help="Frame skip",
    type=int)
parser.add_option(
    "--sample-size",
    dest="sample_size",
    default="64",
    help="Sample batch size",
    type=int)
parser.add_option(
    "-g",
    "--discount",
    dest="discount",
    default="0.98",
    help="Discount factor",
    type=float)
parser.add_option("--scale", dest="scale", default=1,
                  help="Scaling", type=float)
parser.add_option("--anneal-steps",
                  dest="steps", default=100000, help="steps",  type=int)
parser.add_option("-x", "--extra", dest="extra", default="")
parser.add_option("-p", "--reuse", dest="reuse",
                  choices=["full", "sectionned", "proportional"], default="full")
parser.add_option(
    "-c", "--mode", dest="mode", choices=["regular", "sparse"], default="sparse")
parser.add_option(
    "-s", "--seed", dest="seed", default=None, help="Random Seed", type=int)

(options, args) = parser.parse_args()

extra = "a-{} m-{} s-{}  e-{} d-{} x-{} {} p-{} fs-{} d-{} up-{} lr-{} e-{} p-{} m-{}-{}".format(
    options.alg, options.mem, options.seed,
    options.end_e, options.dupe, options.extra, options.mode, options.reuse,
    options.frame_skip,
    np.round(options.discount, 4), options.updates,
    np.round(options.lr, 4),
    np.round(options.scale, 2), np.round(options.steps, 2), np.round(options.mem_a, 2), np.round(options.mem_e, 2))

random.seed(options.seed)
np.random.seed(options.seed)

json_file = "mine_config.json"
minecart = Minecart.from_json(json_file)
obj_cnt = minecart.obj_cnt()
all_weights = generate_weights(
    count=options.steps, n=minecart.obj_cnt(), m=1 if options.mode == "sparse" else 10)

agent = DeepAgent(
    Minecart.action_space(),
    minecart.obj_cnt(),
    options.steps,
    sample_size=options.sample_size,
    weights=None,
    discount=options.discount,
    learning_rate=options.lr,
    target_update_interval=options.target_update_interval,
    alg=options.alg,
    frame_skip=options.frame_skip,
    end_e=options.end_e,
    memory_type=options.mem,
    update_interval=options.updates,
    reuse=options.reuse,
    mem_a=options.mem_a,
    mem_e=options.mem_e,
    extra=extra,
    clipnorm=options.clipnorm,
    clipvalue=options.clipvalue,
    momentum=options.momentum,
    scale=options.scale,
    dupe=None if options.dupe == "none" else options.dupe)

steps_per_weight = 50000 if options.mode == "sparse" else 1
log_file = open('output/logs/rewards_{}'.format(extra), 'w', 1)
agent.train(minecart, log_file,
            options.steps, all_weights, steps_per_weight, options.steps*10)
