from rocket_simulator import RocketSimulator
from visualizer import visualize_landing
from pid_policy import simple_guidance_policy

# hi! here's the main script for actually running everything
# the setup here is pretty easy -- first, just run the simulator

# first you initialize the simulation itself
sim = RocketSimulator(seed=123)

# then you define whatever your policy is (probably easiest just to leave it named as 'policy')
policy = simple_guidance_policy

# then everything else down here is probably just best left the same 
# it runs the simulation with the given policy n_episodes many times
# the '.act' part is just included because there's a method which actually does the finding-best-action stuff
results = sim.run_episodes(policy, n_episodes=10)

# and then we make a nice little graph of it (the speedup part is needed, trust me)
visualize_landing(sim, policy, speedup=100.0)
