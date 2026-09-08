import logging

logger = logging.getLogger(__name__)
import os

import numpy as np
from src.methods import setup_model
from src.utils.utils import get_accuracy, merge_cfg_from_args, get_args
from src.utils.conf import cfg, load_cfg_fom_args
from src.data.data import load_ood_dataset_test
import wandb

def validation(cfg):
    model = setup_model(cfg)
    
    dom_names_all = cfg.CORRUPTION.TYPE
    logger.info(f"Using the following domain sequence: {dom_names_all}")
    wandb.log({"domain_sequence": dom_names_all})


    severities = [cfg.CORRUPTION.SEVERITY[0]]

    accs = []
    aucs = []
    h_scores = []

    
    for i_dom, domain_name in enumerate(dom_names_all):
        if cfg.MODEL.CONTINUAL == 'Fully':
            try:
                model.reset()
                logger.info("resetting model")
                wandb.log({"resetting_model": True})
            except:
                logger.warning("not resetting model")
        elif cfg.MODEL.CONTINUAL == 'Continual':
            logger.warning("not resetting model")

        

        for severity in severities:
            testset, test_loader = load_ood_dataset_test(cfg.DATA_DIR, cfg.CORRUPTION.ID_DATASET,
                                                         cfg.CORRUPTION.OOD_DATASET, cfg.CORRUPTION.NUM_OOD_SAMPLES,
                                                         batch_size=cfg.TEST.BATCH_SIZE,
                                                         domain=domain_name, level=severity,
                                                         adaptation=cfg.MODEL.ADAPTATION,
                                                         workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count()),
                                                         ckpt=os.path.join(cfg.CKPT_DIR, 'Datasets'),
                                                         num_aug=cfg.TEST.N_AUGMENTATIONS if cfg.MODEL.ADAPTATION != 'stamp' else cfg.STAMP.NUM_AUG)
            for epoch in range(cfg.TEST.EPOCH):
                acc, auc = get_accuracy(
                    model, data_loader=test_loader, cfg=cfg)
            h_score = 2 * acc * auc / (acc + auc)
            accs.append(acc)
            aucs.append(auc)
            h_scores.append(h_score)
            logger.info(
                f"{cfg.CORRUPTION.ID_DATASET} with {cfg.CORRUPTION.OOD_DATASET} [
                f":acc: {acc:.2%}, auc: {auc:.2%}, h-score: {h_score:.2%}")
            wandb.log({f"{cfg.CORRUPTION.ID_DATASET} with {cfg.CORRUPTION.OOD_DATASET} [

        logger.info(f"mean acc: {np.mean(accs):.2%}, "
                    f"mean auc: {np.mean(aucs):.2%}, "
                    f"mean h-score: {np.mean(h_scores):.2%}")
        wandb.log({"mean_acc": np.mean(accs), "mean_auc": np.mean(aucs), "mean_h_score": np.mean(h_scores)})
        wandb.run.summary["mean_acc"] = np.mean(accs)
        wandb.run.summary["mean_auc"] = np.mean(aucs)
        wandb.run.summary["mean_h_score"] = np.mean(h_scores)
        
        


if __name__ == "__main__":
    args = get_args()

    args.output_dir = args.output_dir if args.output_dir else 'evaluation_os'
    wandb.init(project='sta-tta_adaptation', entity='exps')
    wandb.config.update(args)
    
    
    wandb_run_name = wandb.run.name
    
    
    wandb_run_name = os.path.join('./output_logs', wandb_run_name)
    args.output_dir = wandb_run_name

    os.makedirs(args.output_dir, exist_ok=True)
    
    cfg.OUTPUT_DIR = args.output_dir
    wandb.config.update(cfg)
    load_cfg_fom_args(args.cfg, args.output_dir)
    merge_cfg_from_args(cfg, args)
    cfg.CORRUPTION.SOURCE_DOMAIN = cfg.CORRUPTION.SOURCE_DOMAINS[0]
    logger.info(cfg)
    validation(cfg)
