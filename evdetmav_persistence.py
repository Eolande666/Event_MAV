"""Optional post-detector persistence; disabled mode delegates unchanged baseline CLI."""
import argparse
import sys
from persistence.config import Config


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config',required=True)
    known,args=parser.parse_known_args()
    config=Config.load(known.config)
    if config.enabled:
        from persistence.runner import run
        run(config,args)
    else:
        from evdetmav.cli import main as baseline_main
        sys.argv=[sys.argv[0],*args]
        baseline_main()


if __name__=='__main__':main()
