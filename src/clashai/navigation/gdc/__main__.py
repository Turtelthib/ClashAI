# clashai/navigation/gdc/__main__.py
# CLI: uv run python -m clashai.navigation.gdc --attack N | --navigate N | --monitor

import argparse

from clashai.navigation.gdc.navigator import GdCNavigator
from clashai.navigation.gdc.orchestrator import GdCOrchestrator


def main():
    parser = argparse.ArgumentParser(description="ClashAI GdC Navigator")
    parser.add_argument('--attack', type=int,
                        help="Attack target n°X in CW")
    parser.add_argument('--navigate', type=int,
                        help="Navigate to target without attacking")
    parser.add_argument('--monitor', action='store_true',
                        help="Start chat monitoring")
    parser.add_argument('--bot-name', type=str, default='mini_pekka')
    parser.add_argument('--interval', type=int, default=30)

    args = parser.parse_args()

    # Load models
    from clashai.navigation import game_loop
    models = game_loop.load_models()

    if args.attack:
        nav = GdCNavigator(models)
        success = nav.attack_target(args.attack)
        if success:
            print("Attack phase reached — V3 agent can take over")
        else:
            print("ERROR: Navigation failed")

    elif args.navigate:
        nav = GdCNavigator(models)
        if nav.navigate_to_war_map():
            success = nav.select_target(args.navigate)
            if success:
                print(f"Target #{args.navigate} selected")
            else:
                print(f"ERROR: Target #{args.navigate} not found")

    elif args.monitor:
        orchestrator = GdCOrchestrator(models, bot_name=args.bot_name)
        orchestrator.run(monitor_interval=args.interval)

    else:
        print("Usage:")
        print(" --attack 3 Attack target #3 in CW")
        print(" --navigate 5 Navigate to target #5 (without attacking)")
        print(" --monitor Monitor chat and execute commands")
        print(" --bot-name X Bot name (default: mini_pekka)")
        print(" --interval N Monitoring interval in seconds")


if __name__ == "__main__":
    main()
