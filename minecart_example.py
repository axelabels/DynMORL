import numpy as np
from minecart import Minecart

# Generate minecart from configuration file (2 ores + 1 fuel objective, 5 mines)
json_file = "mine_config_det.json"
env = Minecart.from_json(json_file)

# # Or alternatively, generate a random instance
# env = Minecart(mine_cnt=5,ore_cnt=2,capacity=1)

# Initial State
s_t = env.reset()

# Note that s_t is a dictionary containing among others the state's pixels but also the cart's position, velocity, etc...
s_t = s_t["pixels"]

# flag indicates termination
terminal = False

while not terminal:
  # randomly pick an action
  a_t = np.random.randint(env.a_space)

  # apply picked action in the environment
  s_t1, r_t, terminal = env.step(a_t)
  s_t1 = s_t1["pixels"]

  # update state
  s_t = s_t1
    
  print("Taking action", a_t, "with reward", r_t)
  
env.reset()
