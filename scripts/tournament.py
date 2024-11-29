import math
from random import randint
import copy
import heapq
from itertools import combinations
import gymnasium
from gymnasium.wrappers import RecordVideo
import torch
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from bomberman_rl import settings as s, Actions, Bomberman
from bomberman_rl.wrappers import *

from argparsing import parse

log_path = "scripts/logs/stable_baselines"


def makeEnvs(args, n_envs = 5, vec_env_cls=SubprocVecEnv, name_prefix=None, wrapper=[], wrapperKwargs=[], obsWrapper=None, obsWrapperKwargs={}, demo=False):
    train_render = {"no_gui": True, "render_mode": None}
    args.__dict__.update(train_render)
    env_eval = makeSingleEnv(args, wrapper=wrapper, wrapperKwargs=wrapperKwargs, obsWrapper=obsWrapper, obsWrapperKwargs=obsWrapperKwargs)
    env_train = make_vec_env(lambda: makeSingleEnv(args, wrapper=wrapper, wrapperKwargs=wrapperKwargs, obsWrapper=obsWrapper, obsWrapperKwargs=obsWrapperKwargs), n_envs=n_envs, vec_env_cls=vec_env_cls)

    env_demo = None
    if demo:
        demo_render = {"no_gui": False, "video": f"{log_path}/replays" if args.video else None, "render_mode": "rgb_array" if args.video else "human"}
        demo_args = copy.copy(args)
        demo_args.__dict__.update(demo_render)
        env_demo = makeSingleEnv(demo_args, wrapper=wrapper, wrapperKwargs=wrapperKwargs, obsWrapper=obsWrapper, obsWrapperKwargs=obsWrapperKwargs)
        if demo_args.video:
            env_demo = RecordVideo(env_demo, video_folder=args.video, name_prefix=name_prefix if name_prefix else args.match_name)
        
    return env_train, env_eval, env_demo

def makeSingleEnv(args, wrapper=[], wrapperKwargs=[], obsWrapper=None, obsWrapperKwargs={}):
    env = gymnasium.make("bomberman_rl/bomberman-v0", disable_env_checker=True, args=args) # TODO enable env checker
    env = wrapEnv(env, wrapper=wrapper, wrapperKwargs=wrapperKwargs, obsWrapper=obsWrapper, obsWrapperKwargs=obsWrapperKwargs)
    return env

def wrapEnv(env, **kwargs):
    print("TODO: wrapEnv()")
    return env # TODO


def train(model, env, total_timesteps=100_000, n_model_saves=1, id=""):
    logger = configure(log_path, ["stdout", "csv"])
    model.set_logger(logger)
    save_freq = math.floor(total_timesteps / n_model_saves / env.num_envs)
    name_prefix = f"{model.__class__.__name__}.{model.policy.__class__.__name__}.{id}"
    callback = CheckpointCallback(save_freq=save_freq, save_path=log_path, name_prefix=name_prefix, verbose=2)
    # log_interval = math.floor(total_timesteps / 20 / 200)
    # print(log_interval)
    model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=500)


def evaluate(model, env, n_episodes=10, deterministic=False):
    mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=n_episodes, deterministic=deterministic)
    return mean_r, std_r


def demo(model, env, n_steps=100, deterministic=True):
    obs, _ = env.reset()
    terminated, truncated = False, False
    reward = 0
    for i in range(n_steps):
        if not (terminated or truncated):
            action, _state = model.predict(obs, deterministic=deterministic)
            action = action.squeeze()
            obs, r, terminated, truncated, _ = env.step(action)
            reward += r
    print(f"Demo reward: {reward}")
    env.close()

def collectLeaders(dones, infos):
    return [info["leaderboard"] for done, info in zip(dones, infos) if done]

def aggregateLeaders(leaderboards):
    print("TODO aggregateLeaders") # TODO
    return leaderboards[0]

def match(agents, n_episodes=10, n_envs=28):
    """
    Compete agents by running multiple episodes.
    """
    # TODO: use agents
    args = parse()
    env_train, _, env_demo = makeEnvs(args=args, n_envs=n_envs, demo=True)
    env_train.reset()
    leaderboards = []
    while len(leaderboards) < n_episodes:
        _, _, dones, infos = env_train.step([None] * n_envs)
        leaderboards.extend(collectLeaders(dones, infos))
    env_train.close()
    aggregatedLeaders = aggregateLeaders(leaderboards)
    print(leaderboards)
    print(aggregatedLeaders)


def pairings(competitors: list[str], pairing_cardinality, n_competitor_pairings, exhaustive):
    def pairing_hash(pairing):
        return hash(tuple([c for _, c in sorted(pairing)]))
    
    if exhaustive:
        for pairing in combinations(competitors, pairing_cardinality):
            yield pairing
    else:
        pairing_hashes = set()
        c_counts = []
        for c in competitors:
            heapq.heappush(c_counts, (0, c))
        while c_counts[0][0] < n_competitor_pairings:
            pairing, dropped = [], []
            for i in range(pairing_cardinality):
                pairing.append(heapq.heappop(c_counts))
            while pairing_hash(pairing) in pairing_hashes:
                idx = randint(0, pairing_cardinality - 1)
                dropped.append(pairing[idx])
                del pairing[idx]
                try:
                    pairing.append(heapq.heappop(c_counts))
                except IndexError:
                    print("index error")
                    for p, c in dropped:
                        heapq.heappush(c_counts, (p, c))
                    pairing.append(heapq.heappop(c_counts))
                    continue
            yield [c for _, c in pairing]
            pairing_hashes.add(pairing_hash(pairing))
            for p, c in pairing:
                heapq.heappush(c_counts, (p + 1, c))
                

def qualification_phase(competitors, n_max_competitor_pairings=5, pairing_cardinality=4):
    n_possible_competitor_pairings = math.comb(len(competitors) - 1, pairing_cardinality - 1)
    exhaustive = n_possible_competitor_pairings <= n_max_competitor_pairings
    n_competitor_pairings = min(n_possible_competitor_pairings, n_max_competitor_pairings)
    for pairing in pairings(competitors, pairing_cardinality=pairing_cardinality, n_competitor_pairings=n_competitor_pairings, exhaustive=exhaustive):
        yield match(pairing) 
    

def tournament(competitors):
    assert(len(competitors) != len(set(competitors))), "Duplicate competitors"

    # Qualification

    # Playoffs # TODO

    # Final

    #args = parse(argv)
    #match(args)

if __name__ == "__main__":
    l = list(pairings(list("qwertzuiopasdfghjkl"), 4, 20, False))
    print(l)
    #tournament()