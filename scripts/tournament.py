import math
from random import randint
import copy
import heapq
from sympy import primerange
from itertools import combinations
from collections import defaultdict
import gymnasium
from gymnasium.wrappers import RecordVideo
import torch
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    DummyVecEnv,
    VecVideoRecorder,
)
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from bomberman_rl import settings as s, Actions, Bomberman
from bomberman_rl.wrappers import *

from argparsing import parse

log_path = "scripts/logs/stable_baselines"


def makeEnvs(
    args,
    n_envs=5,
    vec_env_cls=SubprocVecEnv,
    name_prefix=None,
    wrapper=[],
    wrapperKwargs=[],
    obsWrapper=None,
    obsWrapperKwargs={},
    demo=False,
):
    train_render = {"no_gui": True, "render_mode": None}
    train_args = copy.copy(args)
    train_args.__dict__.update(train_render)
    env_eval = makeSingleEnv(
        train_args,
        wrapper=wrapper,
        wrapperKwargs=wrapperKwargs,
        obsWrapper=obsWrapper,
        obsWrapperKwargs=obsWrapperKwargs,
    )
    env_train = make_vec_env(
        lambda: makeSingleEnv(
            train_args,
            wrapper=wrapper,
            wrapperKwargs=wrapperKwargs,
            obsWrapper=obsWrapper,
            obsWrapperKwargs=obsWrapperKwargs,
        ),
        n_envs=n_envs,
        vec_env_cls=vec_env_cls,
    )
    env_demo = None
    if demo:
        demo_render = {
            "no_gui": False,
            "video": f"{log_path}/replays" if args.video else None,
            "render_mode": "rgb_array" if args.video else "human",
        }
        demo_args = copy.copy(args)
        demo_args.__dict__.update(demo_render)
        env_demo = makeSingleEnv(
            demo_args,
            wrapper=wrapper,
            wrapperKwargs=wrapperKwargs,
            obsWrapper=obsWrapper,
            obsWrapperKwargs=obsWrapperKwargs,
        )
        if demo_args.video:
            env_demo = RecordVideo(
                env_demo,
                video_folder=args.video,
                episode_trigger=lambda _: True,
                name_prefix=name_prefix if name_prefix else args.match_name,
            )
    return env_train, env_eval, env_demo


def makeSingleEnv(
    args,
    wrapper=[],
    wrapperKwargs=[],
    obsWrapper=None,
    obsWrapperKwargs={},
    name_prefix="",
):
    env = gymnasium.make(
        "bomberman_rl/bomberman-v0", disable_env_checker=False, args=args
    )
    env = wrapEnv(
        env,
        wrapper=wrapper,
        wrapperKwargs=wrapperKwargs,
        obsWrapper=obsWrapper,
        obsWrapperKwargs=obsWrapperKwargs,
    )
    if args.video:
        env = RecordVideo(env, video_folder=args.video, name_prefix=name_prefix)
    return env


def wrapEnv(env, **kwargs):
    print("TODO: wrapEnv()")
    return env  # TODO


def train(model, env, total_timesteps=100_000, n_model_saves=1, id=""):
    logger = configure(log_path, ["stdout", "csv"])
    model.set_logger(logger)
    save_freq = math.floor(total_timesteps / n_model_saves / env.num_envs)
    name_prefix = f"{model.__class__.__name__}.{model.policy.__class__.__name__}.{id}"
    callback = CheckpointCallback(
        save_freq=save_freq, save_path=log_path, name_prefix=name_prefix, verbose=2
    )
    # log_interval = math.floor(total_timesteps / 20 / 200)
    # print(log_interval)
    model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=500)


def evaluate(model, env, n_episodes=10, deterministic=False):
    mean_r, std_r = evaluate_policy(
        model, env, n_eval_episodes=n_episodes, deterministic=deterministic
    )
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


def collectEpisodeResults(dones, infos):
    """Return leaderboards from environments with just finished episode"""
    return [info["leaderboard"] for done, info in zip(dones, infos) if done]


def aggregateScoreboards(scoreboards):
    """Average scores per entry over entry appearances in multiple scoreboards"""
    avg_scores, avg_counters = defaultdict(float), defaultdict(float)
    for board in scoreboards:
        for competitor, score in board.items():
            avg_counters[competitor] = avg_counters[competitor] + 1
            avg_scores[competitor] = avg_scores[competitor] + 1 / avg_counters[
                competitor
            ] * (score - avg_scores[competitor])
    return dict(avg_scores)


def episodeResult2scoreboard(match_result):
    """Transform raw match scores to ranking based scores"""
    scores = {
        0: 3,  # 3 points for 1st
        1: 1,  # 1 point for 2nd
    }
    return {
        competitor_result[0]: scores.get(i, 0)
        for i, competitor_result in enumerate(
            sorted(match_result.items(), key=lambda x: x[1], reverse=True)
        )
    }


def recordMatchingDemo(args, scoreboard):
    """Records episodes until it finds one which result matches the aggregated result"""
    demo_render = {
        "no_gui": False,
        "video": f"{log_path}/replays/matchingDemo",
        "render_mode": "rgb_array",
    }
    demo_args = copy.copy(args)
    demo_args.__dict__.update(demo_render)
    # env_demo = make_vec_env(lambda: makeSingleEnv(demo_args, [], {}, [], {}), n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    env_demo = makeSingleEnv(demo_args, [], {}, [], {}, name_prefix="test")
    env_demo.reset()
    finished = False
    while not finished:
        # _, _, dones, infos = env_demo.step([None] * n_envs)
        _, _, terminated, truncated, info = env_demo.step(None)
        # new_episode_results = collectEpisodeResults(dones, infos)
        new_episode_results = [info["leaderboard"]] if terminated or truncated else []
        for episode_result in new_episode_results:
            comparison = list(
                zip(
                    sorted(
                        episodeResult2scoreboard(episode_result).items(),
                        key=lambda x: x[1],
                    ),
                    sorted(scoreboard.items(), key=lambda x: x[1]),
                )
            )
            if all([c[0][0] == c[1][0] for c in comparison]):
                finished = True
            else:
                env_demo.reset()
    env_demo.close()


def match(agents, n_episodes=10, n_envs=28, demo=False):
    """Compete agents by running multiple episodes"""
    # return dict(zip(agents, range(1, len(agents) + 1)))
    args = parse()
    args.learners = []
    args.players = agents
    # args.players = ["rule_based_agent"] * 4 for testing
    env_train, _, _ = makeEnvs(args=args, n_envs=n_envs, demo=False)

    # Aggregated match results TODO consider seeds?
    env_train.reset()
    episode_results = []
    while len(episode_results) < n_episodes:
        _, _, dones, infos = env_train.step([None] * n_envs)
        episode_results.extend(collectEpisodeResults(dones, infos))
    env_train.close()
    episode_scoreboards = [episodeResult2scoreboard(r) for r in episode_results]
    aggregated_episode_scoreboard = aggregateScoreboards(episode_scoreboards)
    if demo:
        recordMatchingDemo(args, scoreboard=aggregated_episode_scoreboard)
    #mapping = {
    #    f"rule_based_agent_{i}": agents[i] for i in range(4)
    #}
    return aggregated_episode_scoreboard
    # return {mapping[k]: v for k, v in aggregated_episode_scoreboard.items()} for testing


def generatePairings(
    competitors: list[str], pairing_cardinality, n_competitor_pairings, exhaustive
):
    def pairing_hash(pairing):
        return hash(tuple([c for _, c in sorted(pairing, key=lambda x: x[1])]))

    if exhaustive:
        for pairing in combinations(competitors, pairing_cardinality):
            yield pairing
    else:
        pairing_hashes = set()
        c_counts = []
        for c in competitors:
            heapq.heappush(c_counts, (0, c))
        while c_counts[0][0] < n_competitor_pairings:
            pairing, dropped, done = [], [], False
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
                    for p, c in pairing:
                        heapq.heappush(c_counts, (p, c))
                    for pairing in combinations(competitors, pairing_cardinality):
                        if not pairing_hash(zip(pairing, pairing)) in pairing_hashes:
                            c_counts = [
                                (p + 1, c) if c in pairing else (p, c)
                                for p, c in c_counts
                            ]
                            heapq.heapify(c_counts)
                            pairing_hashes.add(pairing_hash(zip(pairing, pairing)))
                            yield pairing
                            done = True
                            break
                    break
            if not done:
                pairing_hashes.add(pairing_hash(pairing))
                for p, c in dropped:
                    heapq.heappush(c_counts, (p, c))
                for p, c in pairing:
                    heapq.heappush(c_counts, (p + 1, c))
                yield [c for _, c in pairing]


# def test(pairings):
#     #print(sorted(pairings, key=lambda p: math.prod(p)))
#     duplicates = []
#     result = defaultdict(int)
#     for p in pairings:
#         p_hash = hash("".join(sorted(p)))
#         if p_hash in duplicates:
#             raise AssertionError(p)
#         else:
#             duplicates.append(p_hash)
#             for k in p:
#                 result[k] = result[k] + 1
#     return result
# l = list(pairings(list("asdfölkjqwermnbv"), 4, 200, False))


def play_qualification(competitors, n_max_competitor_pairings, pairing_cardinality):
    n_possible_competitor_pairings = math.comb(
        len(competitors) - 1, pairing_cardinality - 1
    )
    exhaustive = n_possible_competitor_pairings <= n_max_competitor_pairings
    n_competitor_pairings = min(
        n_possible_competitor_pairings, n_max_competitor_pairings
    )
    pairings = generatePairings(
        competitors,
        pairing_cardinality=pairing_cardinality,
        n_competitor_pairings=n_competitor_pairings,
        exhaustive=exhaustive,
    )
    match_scoreboards = [match(pairing) for pairing in pairings]
    qualification_scoreboard = aggregateScoreboards(
        match_scoreboards
    )  # reuse scoreboard aggregation also across different pairings
    return qualification_scoreboard


def play_final(competitors, demo):
    scoreboard = match(agents=competitors, demo=demo)
    return scoreboard


def tournament(competitors, pairing_cardinality=4, demo=False):
    assert len(competitors) == len(set(competitors)), "Duplicate competitors"

    # Qualification
    qualification_results = play_qualification(
        competitors,
        n_max_competitor_pairings=2,
        pairing_cardinality=pairing_cardinality,
    )
    qualification_results = dict(
        sorted(qualification_results.items(), key=lambda x: x[1], reverse=True)
    )
    print(f"Qualification results: {qualification_results}")

    # Playoffs
    # # TODO

    # Final
    final_competitors = list(qualification_results.keys())[:pairing_cardinality]
    final_results = play_final(competitors=final_competitors, demo=demo)
    print(f"Final results: {final_results}")


if __name__ == "__main__":
    tournament(competitors=list("asdfqwertzui"), demo=True)
