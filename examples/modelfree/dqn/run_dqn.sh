#!/bin/bash

# classic
echo "---------Run DQN in CartPole-v1---------"
python classic_dqn.py --env=CartPole-v1 --learn_id=modelfree/dqn/classic/CartPole-v1/seed10 --seed=10
python classic_dqn.py --env=CartPole-v1 --learn_id=modelfree/dqn/classic/CartPole-v1/seed20 --seed=20
python classic_dqn.py --env=CartPole-v1 --learn_id=modelfree/dqn/classic/CartPole-v1/seed30 --seed=30

# box2d
echo "---------Run DQN in LunarLander-v2---------"
python box2d_dqn.py --env=LunarLander-v2 --learn_id=modelfree/dqn/box2d/LunarLander-v2/seed10 --seed=10
python box2d_dqn.py --env=LunarLander-v2 --learn_id=modelfree/dqn/box2d/LunarLander-v2/seed20 --seed=20
python box2d_dqn.py --env=LunarLander-v2 --learn_id=modelfree/dqn/box2d/LunarLander-v2/seed30 --seed=30

# atari
echo "--------- Run DQN in PongNoFrameskip-v4 ---------" 
python atari_dqn.py --env=PongNoFrameskip-v4 --learn_id=modelfree/dqn/atari/Pong-v4/seed10 --train_num=8 --eval_num=8 --max_train_step=1000000 --seed=10
python atari_dqn.py --env=PongNoFrameskip-v4 --learn_id=modelfree/dqn/atari/Pong-v4/seed20 --train_num=8 --eval_num=8 --max_train_step=1000000 --seed=20
python atari_dqn.py --env=PongNoFrameskip-v4 --learn_id=modelfree/dqn/atari/Pong-v4/seed30 --train_num=8 --eval_num=8 --max_train_step=1000000 --seed=30
echo "--------- Run DQN in BreakoutNoFrameskip-v4 ---------" 
python atari_dqn.py --env=BreakoutNoFrameskip-v4 --learn_id=modelfree/dqn/atari/Breakout-v4/seed10 --train_num=8 --eval_num=8 --max_train_step=3000000 --seed=10
python atari_dqn.py --env=BreakoutNoFrameskip-v4 --learn_id=modelfree/dqn/atari/Breakout-v4/seed20 --train_num=8 --eval_num=8 --max_train_step=3000000 --seed=20
python atari_dqn.py --env=BreakoutNoFrameskip-v4 --learn_id=modelfree/dqn/atari/Breakout-v4/seed30 --train_num=8 --eval_num=8 --max_train_step=3000000 --seed=30
echo "--------- Run DQN in EnduroNoFrameskip-v4 ---------" 
python atari_dqn.py --env=EnduroNoFrameskip-v4 --learn_id=modelfree/dqn/atari/Enduro-v4/seed10 --train_num=8 --eval_num=8 --max_train_step=3000000 --seed=10
python atari_dqn.py --env=EnduroNoFrameskip-v4 --learn_id=modelfree/dqn/atari/Enduro-v4/seed20 --train_num=8 --eval_num=8 --max_train_step=3000000 --seed=20
python atari_dqn.py --env=EnduroNoFrameskip-v4 --learn_id=modelfree/dqn/atari/Enduro-v4/seed30 --train_num=8 --eval_num=8 --max_train_step=3000000 --seed=30