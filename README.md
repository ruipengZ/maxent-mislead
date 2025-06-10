<div align="center">

  
![ICML 2025](https://img.shields.io/badge/ICML-2025-blue) 
# [When Maximum Entropy Misleads Policy Optimization](https://www.arxiv.org/abs/2506.05615)

</div>

This repo is for ICML 2025 paper **_When Maximum Entropy Misleads Policy Optimization_**.  

In this work, we analyze how maximum entropy objectives, while beneficial for exploration, can sometimes misguide policy learning in environments requiring high-precision control. 

We provide RL environments and PPO/SAC/SAC-AdaEnt algorithms scripts in this repo.

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/ruipengz/maxent-mislead.git
   cd maxent-mislead

2. **Create & activate a Python environment**
   ```bash
   conda create -n maxent python=3.9
   conda activate maxent

3. **Install dependencies with:**
   ```bash
   pip install -r requirements.txt

Our code is based on RL benchmark [tianshou](https://github.com/thu-ml/tianshou/tree/v0.5.0).

## Environments
We provide a set of RL environments that are designed to test the effects of maximum entropy objectives on policy learning. The environments include:
- **Acrobot**(`Acrobot-v0`): is a two-link planar robot arm with one end fixed at the shoulder and the only actuated joint at the elbow. The action is changed to continuous torque values instead of simpler discrete values in OpenAI Gym.
- **Vehicle**(`Vehicle-v0`): the task is to control a wheeled vehicle under the nonlinear dynamic bicycle model to move at a constant high speed along a path. Effective control is critical for steering the vehicle onto the path.
- **Quadrotor**(`QuadRotor-v0`): the task is to control a quadrotor to track a simple path under small perturbations. The actions are the independent speeds of its four rotors, which makes the learning task harder than simpler models. 
- **Obstacle2D**(`Obstacle2D-v0`): the goal is to navigate an agent to the goal behind a wall, which creates a clear suboptimal local policy that the agent should learn to avoid.
- **Opencat**(`OpenCat-v0`):  environment simulates an open-source Arduino-based quadruped robot (PetoiCamp). The action space is the 8 joint angles of the robot.
- **Hopper**(`Hopper-v4`): is the standard MuJoCo environment where SAC typically learns faster and more stably than PPO.

## Training
To train and evaluate the RL algorithms, you run sh scripts `run_experiments`. 
```bash
sh run_experiments.sh
```

For convenience, we provide individual python files for each environment and algorithm, but you can also run the python script to train all algorithms on all environments.

1. **PPO**
```
python run_ppo.py --task TASK --logdir LOG_DIR 
                --hidden-size HIDDEN_SIZE
                --wandb-project WANDB_PROJECT
                --lr LEARNING_RATE
                --seed SEED
                --epoch EPOCH
                --step-per-epoch STEP_PER_EPOCH
```

- `task` - environment
- `logdir` - log directory path
- `hidden-size` - hidden sizes of the MLP network
- `wandb-project` - wandb project to log to
- `lr` - learning rate
- `seed` - running seed
- `epoch` - total training epochs
- `step-per-epoch` - environment steps per epoch


2. **SAC**
```
python run_sac.py --task TASK --logdir LOG_DIR 
                --hidden-size HIDDEN_SIZE
                --wandb-project WANDB_PROJECT
                --actor-lr ACTOR_LEARNING_RATE
                --critic-lr CRITIC_LEARNING_RATE
                --alpha ALPHA
                --seed SEED
                --epoch EPOCH
                --step-per-epoch STEP_PER_EPOCH
```

- `task` - environment
- `logdir` - log directory path
- `hidden-size` - hidden sizes of the MLP network
- `wandb-project` - wandb project to log to
- `actor-lr` - actor's learning rate
- `critic-lr` - critic's learning rate
- `alpha` - entropy alpha
- `seed` - running seed
- `epoch` - total training epochs
- `step-per-epoch` - environment steps per epoch

3. **SAC-auto-alpha**
```
python run_sac.py --algo-name 'SAC_autoalpha' --task TASK --logdir LOG_DIR 
                --hidden-size HIDDEN_SIZE
                --wandb-project WANDB_PROJECT
                --actor-lr ACTOR_LEARNING_RATE
                --critic-lr CRITIC_LEARNING_RATE
                --auto-alpha
                --alpha-lr ALPHA_LEARNING_RATE
                --seed SEED
                --epoch EPOCH
                --step-per-epoch STEP_PER_EPOCH
```

- `task` - environment
- `logdir` - log directory path
- `hidden-size` - hidden sizes of the MLP network
- `wandb-project` - wandb project to log to
- `actor-lr` - actor's learning rate
- `critic-lr` - critic's learning rate
- `auto-alpha` - use auto-tuning alpha
- `alpha-lr` - auto-tuning alpha learning rate
- `seed` - running seed
- `epoch` - total training epochs
- `step-per-epoch` - environment steps per epoch


4. **SAC-AdaEnt**
```
python run_sac_adaent.py --task TASK --logdir LOG_DIR 
                --hidden-size HIDDEN_SIZE
                --wandb-project WANDB_PROJECT
                --actor-lr ACTOR_LEARNING_RATE
                --critic-lr CRITIC_LEARNING_RATE
                --alpha ALPHA
                --entropy-n-sample ENT_N_SAMPLE
                --entropy-dist-threshold ENT_DIST_THRESHOLD
                --seed SEED
                --epoch EPOCH
                --step-per-epoch STEP_PER_EPOCH
```

- `task` - environment
- `logdir` - log directory path
- `hidden-size` - hidden sizes of the MLP network
- `wandb-project` - wandb project to log to
- `actor-lr` - actor's learning rate
- `critic-lr` - critic's learning rate
- `alpha` - entropy alpha
- `entropy-n-sample` - number of samples per state to estimate plain and soft Q
- `entropy-dist-threshold` - similarity threshold to select using plain or soft Q values
- `seed` - running seed
- `epoch` - total training epochs
- `step-per-epoch` - environment steps per epoch
