import numpy as np


def get_d4rl_dataset(env, get_num=None) -> dict:
    """
    d4rl dataset: https://github.com/rail-berkeley/d4rl
    install: pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
    :param get_num: how many data get form dataset
    """
    dataset = env.get_dataset()
    data_num = dataset['actions'].shape[0]

    obs = dataset['observations'].astype(np.float32)
    acts = dataset['actions'].astype(np.float32)
    rews = dataset['rewards'].astype(np.float32)
    term = dataset['terminals']
    if 'timeouts' in dataset:
        trun = dataset['timeouts']
    else:
        trun = np.zeros_like(term, dtype=bool)

    done = np.logical_or(term, trun)
    
    if get_num is None:
        data = {'obs': obs,
                'acts': acts,
                'rews': rews,
                'term': term,
                'trun': trun,
                'done': done}
    else:
        ind = np.random.choice(data_num, size=get_num, replace=False)
        data = {'obs': obs[ind],
                'acts': acts[ind],
                'rews': rews[ind],
                'term': term[ind],
                'trun': trun[ind],
                'done': done[ind]}


    return data, data_num
