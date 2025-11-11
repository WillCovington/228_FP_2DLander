# AA228 Final Project: 2D Rocket Autonomously Guided Landing
The codebase for our 228 Final Project which involves developing a policy for having a rocket land itself

# Upcoming To-Do's
- Implement a Kalman Filter for state estimation that sees through the sensor noise issue
- Implement some policies that successfully land the rocket (feel free to fiddle with settings and reward values in the rocket_simulator script) ((there's an example PID policy that I got from Chat that should be uploaded))

# A quick rundown on each of the core scripts
- rocket_simulator.py: Exactly what it sounds like. This script is what generates the dynamics of the rocket as well as the noise distribution data coming from our sensors (noisy observations). The main class in this script, RocketSimulator, allows you to run however many episodes you want with your policy and then compare those runs. Upon running all those runs, the data is exported to a .npz file titled 'rocket_episodes.npz'.
- visualizer.py: This just plots the last simulation which was run in a way that lets you actually see what was going on with your policy.
- main.py: Again, this is exactly what it sounds like. First the simulator is initialized, then a policy is defined (currently it's listed as 'simple_guidance_policy'. This policy was just ripped from Chat for testing). The policy is run for n-many episodes in the simulator, and the data is presented in the form of individual performances and then finally an averaged performance. Then, the visualizer is run and shows you the last run performed.

# Overleaf Docs
- Project Status Update: https://www.overleaf.com/5524244129bmmtxmrmnrsj#020416
- Final Report: 
