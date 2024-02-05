import glob
import wandb
import torch
from loguru import logger
from pathlib import Path
import pql
from pql.utils.common import load_class_from_path
import os


def load_model(model, model_type, cfg):
    if cfg.local_artifact_path is not None:
        # find the local artifact path using id
        id = cfg.artifact.split(":")[0].split("/")[-1]
        subdirectories = glob.glob(f"{cfg.local_artifact_path}/**/*{id}*", recursive=True)
        subdirectory = [path for path in subdirectories if os.path.isdir(path)][0]
        
        all_weights = []
        for subdirectory in subdirectories:
            model_path = os.path.join(subdirectory, 'files', "model.pth")
            if not os.path.exists(model_path):
                continue
            weights = torch.load(model_path, map_location=None if torch.cuda.device_count() > 1 else 'cuda:0')
            all_weights.append(weights)
        weights = sorted(all_weights, key=lambda x: x['step'], reverse=True)[0]
        weights["step"] -= 10000000
    else:
        artifact = wandb.Api().artifact(cfg.artifact)
        artifact.download(pql.LIB_PATH)
        logger.warning(f'Load {model_type}')
        weights = torch.load(Path(pql.LIB_PATH, "model.pth"))

    if model_type in ["actor", "critic", "obs_rms"]:
        if model_type == "obs_rms" and weights[model_type] is None:
            logger.warning(f'Observation normalization is enabled, but loaded weight contains no normalization info.')
            return
        model.load_state_dict(weights[model_type])
    elif model_type in ["step", "train_env_state", "eval_env_state"]:
        return weights[model_type]
    else:
        logger.warning(f'Invalid model type:{model_type}')


def save_model(path, actor, critic, rms, wandb_run, ret_max, step, train_env_state, eval_env_state):
    checkpoint = {'obs_rms': rms,
            'actor': actor,
            'critic': critic,
            'step': step,
            'train_env_state': train_env_state,
            'eval_env_state': eval_env_state,
            }
    torch.save(checkpoint, path)  # save policy network in *.pth

    model_artifact = wandb.Artifact(wandb_run.id, type="model", description=f"return: {int(ret_max)}")
    model_artifact.add_file(path)
    wandb.save(path, base_path=wandb_run.dir)
    wandb_run.log_artifact(model_artifact)