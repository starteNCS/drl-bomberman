import time
import gymnasium
from gymnasium.wrappers import RecordVideo

from bomberman_rl import settings as s, Actions, Bomberman

from argparsing import parse
from random_agent.agent import RandomAgent
from dummy_agent.agent import DummyAgent


def loop(env, agent, args):
    state, info = env.reset()
    terminated, truncated, quit = False, False, False

    while not (terminated or truncated):
        if args.user_play:
            action, quit = env.unwrapped.get_user_action()
            while action is None and not quit:
                time.sleep(0.1)  # wait for user action or quit
                action, quit = env.unwrapped.get_user_action()
        else:
            action, quit = agent.act(state), env.unwrapped.get_user_quit()
            action = Actions(action)._name_ if action is not None else None

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
            time.sleep(0.5) # wait for quit
            quit = env.unwrapped.get_user_quit()

    env.close()


def provideAgent(env, tournament: bool):
    if tournament:
        return DummyAgent()
    else:
        agent = RandomAgent()
        agent.setup()
        return agent

def main(argv=None):
    args = parse(argv)
    env = gymnasium.make("bomberman_rl/bomberman-v0", args=args)
    if args.video:
        env = RecordVideo(env, video_folder=args.video, name_prefix=args.match_name)

    agent = provideAgent(env, tournament=args.tournament)
    if agent is None and not args.competition:
        raise AssertionError("Either provide an agent or run a tournament")
    if args.train:
        agent.setup_training()

    loop(env, agent, args)


if __name__ == "__main__":
    main()