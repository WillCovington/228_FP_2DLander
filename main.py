import os
import time
import sys
from datetime import datetime

from rocket_simulator import RocketSimulator
from visualizer import visualize_landing
from pid_policy import simple_guidance_policy, refined_guidance_policy, rollout_guidance_policy
from kf_wrapper import kf_rollout_guidance_policy

#redirect terminal output to a file and start timer for analysis later on
os.makedirs("trials", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join("trials", f"run_log_{timestamp}.txt")
sys.stdout = open(log_path, "w")
start_time = time.time()

# hi! here's the main script for actually running everything
# the setup here is pretty easy -- first, just run the simulator

# first you initialize the simulation itself
sim = RocketSimulator(seed=123)

# then you define whatever your policy is (probably easiest just to leave it named as 'policy')
policy = refined_guidance_policy

# then everything else down here is probably just best left the same 
# it runs the simulation with the given policy n_episodes many times
# the '.act' part is just included because there's a method which actually does the finding-best-action stuff
results = sim.run_episodes(policy, n_episodes=10)

#stop timer and record runtime
end_time = time.time()
elapsed = end_time - start_time
print(f"\nTotal runtime: {elapsed:.2f} seconds")

#close output file
sys.stdout.close()

# and then we make a nice little graph of it (the speedup part is needed, trust me)
visualize_landing(sim, policy, speedup=100.0)
