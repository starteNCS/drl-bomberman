import time
import gymnasium
from gymnasium.wrappers import RecordVideo

from bomberman_rl import ScoreRewardWrapper, TimePenaltyRewardWrapper

from argparsing import parse
from q_learning.agent import QLearningAgent


class DummyAgent:
    def setup(self):
        pass

    def setup_training(self, *args, **kwargs):
        pass

    def act(self, *args, **kwargs):
        return None
    
    def game_events_occurred(self, *args, **kwargs):
        pass

    def end_of_round(self, *args, **kwargs):
        pass

def loop(env, agent, args, n_episodes=10_000):
    for episode in range(n_episodes):
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
                # time.sleep(5)

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


def provideAgent(passive: bool):
    if passive:
        return DummyAgent()
    else:
        # agent = Agent()
        agent = QLearningAgent()
        return agent

def main(argv=None):
    args = parse(argv)
    env = gymnasium.make("bomberman_rl/bomberman-v0", args=args)

    # Notice that you can not use wrappers in the tournament!
    # However, you might wanna use this example interface to kickstart your experiments
    env = ScoreRewardWrapper(env)
    env = TimePenaltyRewardWrapper(env, penalty=.1)
    #env = RestrictedKeysWrapper(env, keys=["self_pos"])
    # env = FlattenWrapper(env)
    if args.video:
        env = RecordVideo(env, video_folder=args.video, name_prefix=args.match_name)

    agent = provideAgent(passive=args.passive)
    if agent is None and not args.passive and not args.user_play:
        raise AssertionError("Either provide an agent or run in passive mode by providing the command line argument --passive")
    if args.train:
        agent.setup_training()
    else:
        agent.setup()

    loop(env, agent, args)


if __name__ == "__main__":
    main()