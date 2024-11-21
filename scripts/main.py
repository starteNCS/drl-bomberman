import os
from configargparse import ArgParser
import time
import gymnasium
from gymnasium.wrappers import RecordVideo

from bomberman_rl import settings as s, Actions, Bomberman

from random_agent.agent import Agent as RandomAgent


def parse(argv=None):
    parser = ArgParser(
        default_config_files=[
            #    os.path.dirname(os.path.abspath(__file__)) + "/config.ini"
        ]
    )

    # Play arguments
    # parser.add("-c", "--my-config", is_config_file=True, help="config file path")
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
        "--opponents",
        type=str,
        nargs="+",
        default=["rule_based_agent"] * 3,
        help="Set opponent agents",
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

    # TODO
    # parser.add(
    #     "--save-stats",
    #     const=True,
    #     default=False,
    #     action="store",
    #     nargs="?",
    #     help="Store the game results as .json for evaluation",
    # )
    # parser.add("--multi-process", default=False, action="store_true")
    # -----------------
    #
    # Compete arguments
    #
    # agent_group = compete_parser.add_mutually_exclusive_group()
    # agent_group.add(
    #     "--my-agent",
    #     type=str,
    #     help="Compete agent of name ... against three rule_based_agents",
    # )
    # agent_group.add(
    #     "--agents",
    #     type=str,
    #     nargs="+",
    #     default=["rule_based_agent"] * 4,
    #     help="Compete these agents",
    # )
    # compete_parser.add(
    #     "--train",
    #     default=0,
    #     type=int,
    #     choices=[0, 1, 2, 3, 4],
    #     help="First … agents should be set to training mode",
    # )
    # compete_parser.add_argument(
    #     "--n-rounds", type=int, default=10, help="How many rounds to play"
    # )

    args = parser.parse_args(argv)
    if args.video:
        args.render_mode = "rgb_array"
    elif not args.no_gui:
        args.render_mode = "human"
    else:
        args.render_mode = None

    return args


def loop(env, agent, args):
    state, info = env.reset()
    terminated, truncated, quit = False, False, False

    while not (terminated or truncated):
        if args.user_play:
            action, quit = env.unwrapped.get_user_action()
            while action is None and not quit:
                time.sleep(0.5) # wait until user closes GUI
                action, quit = env.unwrapped.get_user_action()
        else:
            action, quit = agent.act(state), env.unwrapped.get_user_quit()
            action = Actions(action)._name_

        if quit:
            env.close()
            return None
        else:
            new_state, _, terminated, truncated, info = env.step(action)
            if args.train:
                agent.game_events_occurred(state, action, new_state, info["events"])
            state = new_state

    if args.train:
        agent.end_of_round()

    if not args.no_gui:
        quit = env.unwrapped.get_user_quit()
        while not quit:
            time.sleep(0.5)
            quit = env.unwrapped.get_user_quit()

    env.close()


def main(argv=None):
    args = parse(argv)
    env = gymnasium.make('bomberman_rl/bomberman-v0', args=args)
    # env = Bomberman(args=args)
    if args.video:
        env = RecordVideo(env, video_folder=args.video, name_prefix=args.match_name)

    # Agent setup    
    agent = RandomAgent()
    agent.setup()
    if args.train:
        agent.setup_training()

    loop(env, agent, args)


if __name__ == "__main__":
    main()
