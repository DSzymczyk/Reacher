# Reacher
 - Purpose of this project is to solve Multi Agent Reacher Unity Environment using Dueling Deep Deterministic Policy 
   Gradient(DDPG) algorithm.
 - The state space has 20x33 dimensions: 20 arms x 33 variables corresponding to 
   position, rotation, velocity, and angular velocities of the arm.
 - The action space contains 20x4 actions: 20 arms x 4 numbers, corresponding to torque applicable to two joints. Every 
   entry in the action vector should be a number between -1 and 1.
 - Environment will be considered solved when average score in 100 episodes is equal or greater than 30.
 - Project requires 64-bit Windows to run.
 - Application has 3 parameters:
     - `n-episodes` - number of episodes to run
     - `checkpoint-prefix` - prefix of checkpoint filename
     - `load-checkpoint` - boolean parameter, if true loading checkpoint is enabled
  - Application is saving checkpoint after each episode. Checkpoint is stored in 
     `weights/<checkpoint-prefix>policy_checkpoint.pth` and `weights/<checkpoint-prefix>value_checkpoint.pth`
  - Training progress is displayed as tqdm progressbar

## Getting started:
1. Install required packages: `pip install -r requirements.txt`
2. Launch training: `python Reacher.py`:
3. TO BE DONE
   
## Accreditation
Dueling Double DQN algorithm was written based on `Grokking Deep Reinforcement Learning` by Miguel Morales. 
   
## Trained model playthrough
####   TO BE DONE