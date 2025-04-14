import argparse

from utils.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Run the training.")
    parser.add_argument("--curriculum", action="store_true", help="Use curriculum learning")
    return parser.parse_args()


def main():
    args = parse_args()
    trainer = Trainer(config_path='config.yaml')
    if args.curriculum:
        trainer.curriculum_learn()
    else:
        trainer.fit()

if __name__ == "__main__":
    main()