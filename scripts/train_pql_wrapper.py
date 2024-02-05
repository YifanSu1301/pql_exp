import argparse
import subprocess
from time import sleep

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, choices=["AllegroHand", "ShadowHand", "AllegroKuka", "Trifinger", "FrankaCubeStack", "FrankaCabinet", "Humanoid"])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--task', type=str, default="regrasping", choices=["regrasping", "throw", "reorientation"])
    parser.add_argument('--num_envs', type=int, default=24576)
    parser.add_argument('--batch_size', type=int, default=49152)
    parser.add_argument('--wandb-entity', type=str, default="jayeshs999")
    parser.add_argument('--artifact', type=str, default=None)
    
    return parser

def main(args):
    cmd = f"python scripts/train_pql.py task={args.env} seed={args.seed} num_envs={args.num_envs} algo.batch_size={args.batch_size} logging.wandb.entity={args.wandb_entity}"
    if args.env == 'AllegroKuka':
        cmd += ' task/env=' + args.task
    if args.artifact is not None:
        cmd += f" artifact={args.artifact}"
    
    print(cmd)
    process = subprocess.Popen(cmd.split(), stdout=None, stderr=None)
    while process.poll() is None:
        sleep(1)
    
    print("Process finished with exit code: ", process.returncode)


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    main(args)