import os
from configargparse import ArgParser
import time
import gymnasium
from gymnasium.wrappers import RecordVideo

from bomberman_rl import settings as s, Actions, Bomberman

from random_agent.agent import RandomAgent


def parse(argv=None):
    parser = ArgParser(default_config_files=[])

    parser.add(
        "--seed",
        type=int,
        help="Seed the env's random number generator for the sake of reproducibility",
    )
    parser.add(
        "--no-gui",
        default=False,
        action="store_true",
        help="Disable GUI rendering to increase speed",
    )
    parser.add(
        "--players",
        nargs='+',
        help="Set agents that participate playing",
    )
    parser.add(
        "--learners",
        nargs='+',
        help="Set agents that participate playing and learning",
    )
    parser.add(
        "--competition",
        action="store_true",
        default=False,
        help="Whether you want an environment without single user interface",
    )
    parser.add(
        "--match-name",
        help="Match name (used for e.g. displaying, separating recordings, etc.)",
    )
    parser.add(
        "--silence-errors",
        default=False,
        action="store_true",
        help="Ignore errors from agents",
    )
    parser.add(
        "--user-play",
        default=False,
        action="store_true",
        help="Wait for key press until next movement",
    )
    parser.add(
        "--train",
        default=False,
        action="store_true",
        help="Call the agent's training endpoints",
    )
    parser.add(
        "--log-dir", default=os.path.dirname(os.path.abspath(__file__)) + "/logs"
    )
    parser.add(
        "--video",
        nargs="?",
        const=os.path.dirname(os.path.abspath(__file__)) + "/replays",
        help="Record the session",
    )
    parser.add("--scenario", default="classic", choices=s.SCENARIOS)

    args = parser.parse_args(argv)
    if args.learners is None and args.players is None:
        args.players = ["rule_based_agent"] * 3
    return args


def main(argv=None):
    args = parse(argv)
    print(args)
    print("dummy")

if __name__ == "__main__":
    main()
