import argparse
import json
from utils.params import FT_Configer
from finetune.trainer import Trainer


def load(json_file):

    with open(json_file, 'r') as f:
        return json.load(f)


def main():

    parser = argparse.ArgumentParser(description="Argparser")
    parser.add_argument("--params",
                        default={},
                        help="JSON dict of model hyperparameters.")
    args = parser.parse_args()

    model_args = FT_Configer(load(args.params))

    trainer = Trainer(model_args)
    trainer.run_finetune()


if __name__ == '__main__':

    main()
