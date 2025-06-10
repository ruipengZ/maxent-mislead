# Environments
# Toy Example
python examples/ppo_toyexample.py --epoch 100 --step-per-epoch 1000
python examples/sac_toyexample.py --epoch 100 --step-per-epoch 1000
python examples/sac_toyexample_adaent.py --epoch 100 --step-per-epoch 1000

# Acrobot
python examples/ppo_acrobot.py --epoch 100 --step-per-epoch 10000
python examples/sac_acrobot.py --epoch 100 --step-per-epoch 10000
python examples/sac_acrobot.py --epoch 100 --step-per-epoch 10000 --auto-alpha
python examples/sac_acrobot_adaent.py --epoch 100 --step-per-epoch 10000

# Vehicle
python examples/ppo_vehicle.py --epoch 200 --step-per-epoch 10000
python examples/sac_vehicle.py --epoch 200 --step-per-epoch 10000
python examples/sac_vehicle.py --epoch 200 --step-per-epoch 10000 --auto-alpha
python examples/sac_vehicle_adaent.py --epoch 200 --step-per-epoch 10000

# Quadrotor
python examples/ppo_quadrotor.py --epoch 200 --step-per-epoch 10000
python examples/sac_quadrotor.py --epoch 200 --step-per-epoch 10000
python examples/sac_quadrotor.py --epoch 200 --step-per-epoch 10000 --auto-alpha
python examples/sac_quadrotor_adaent.py --epoch 200 --step-per-epoch 10000

# Hopper
python examples/ppo_hopper.py --epoch 200 --step-per-epoch 10000
python examples/sac_hopper.py --epoch 200 --step-per-epoch 10000
python examples/sac_hopper.py --epoch 200 --step-per-epoch 10000 --auto-alpha
python examples/sac_hopper_adaent.py --epoch 200 --step-per-epoch 10000

# Obstacle
python examples/ppo_obstacle2d.py --epoch 50 --step-per-epoch 1000
python examples/sac_obstacle2d.py --epoch 50 --step-per-epoch 1000
python examples/sac_obstacle2d.py --epoch 50 --step-per-epoch 1000 --auto-alpha

# OpenCat
python examples/ppo_obstacle2d.py --epoch 200 --step-per-epoch 10000
python examples/sac_obstacle2d.py --epoch 200 --step-per-epoch 10000
python examples/sac_obstacle2d.py --epoch 200 --step-per-epoch 10000 --auto-alpha