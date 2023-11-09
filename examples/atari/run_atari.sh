#!/bin/bash

# ========================= DQN =========================
echo "--------- Run DQN in PongNoFrameskip-v4 ---------" 
python atari_dqn.py --env=PongNoFrameskip-v4 --learn_id=atari/dqn/Pong-v4 --train_num=8 --eval_num=8 --max_train_step=1000000
echo "--------- Run DQN in BreakoutNoFrameskip-v4 ---------" 
python atari_dqn.py --env=BreakoutNoFrameskip-v4 --learn_id=atari/dqn/Breakout-v4 --train_num=8 --eval_num=8 --max_train_step=3000000
echo "--------- Run DQN in EnduroNoFrameskip-v4 ---------" 
python atari_dqn.py --env=EnduroNoFrameskip-v4 --learn_id=atari/dqn/Enduro-v4 --train_num=8 --eval_num=8 --max_train_step=3000000

# ========================= PPO =========================
echo "--------- Run PPO in PongNoFrameskip-v4 ---------"  # the result is bad.
python box2d_ppo.py --env=PongNoFrameskip-v42 --learn_id=atari/ppo/Pong-v4