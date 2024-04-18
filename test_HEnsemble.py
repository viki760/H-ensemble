import gc
import glob
import os
from pprint import pformat
from typing import Any

from ignite.engine import create_supervised_evaluator
import yaml
from data import setup_data
from ignite.engine import Events
from ignite.metrics import Accuracy, Loss
from ignite.utils import manual_seed
from models import *
from torch import nn, optim
import torch.functional as F
from utils import *

from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

def get_model(config, domain, logger, return_feat=False):
    if config.if_use_shot_model:
        return load_SHOT_model(config, domain, return_feat)

    checkpoint_root = os.path.join(config.checkpoint_dir, config.dataset.name)

    model = setup_model(config, return_feat=return_feat)

    candidate_checkpoint = glob.glob(
        os.path.join(checkpoint_root, domain, "*.pt")
    )
    assert len(candidate_checkpoint) == 1
    checkpoint_path = candidate_checkpoint[0]

    to_save_eval = {"model": model}
    resume_from(to_save_eval, checkpoint_path, logger)

    if logger is not None:
        logger.info(f"checkpoint loaded from {checkpoint_path}")
    return model


def normalize(features):
    # return (features - features.mean(axis=0))
    if features.shape[0] == 1:
        return features
    return (features - features.mean(axis=0)) / features.std(axis=0)

   
def get_target_feature(alpha, all_features):
    # # target_feature: sum of all features weighted by alpha
    # target_feature = torch.zeros_like(all_features[0, :, :])
    # for i in range(len(domains)):
    #     target_feature += alpha[i] * all_features[i, :, :]
    # return target_feature

    # (3, 1, 1) * (3, 2, 4) = (3, 3, 4) ; then sum
    return torch.sum(alpha.view(-1, 1, 1) * all_features, dim=0)

def get_target_feature_train(alpha, all_features):
    # target_feature = torch.zeros_like(all_features[0, :, :])
    # for i in range(len(domains) - 1):
    #     target_feature += alpha[i] * all_features[i, :, :]
    # i = len(domains) - 1
    # target_feature += (1 - alpha.sum()) * all_features[i, :, :]
    # return target_feature
    return torch.sum(alpha.view(-1, 1, 1) * all_features[:-1], dim=0) \
        + (1 - alpha.sum()) * all_features[-1, :, :]


# for calc G
def get_conditional_exp(feature, label, num_classes):
    # "calculate conditional expectation of fx"
    ce_f = torch.zeros((num_classes, feature.shape[1]))

    for i in range(num_classes):
        fx_i = feature[torch.where(label==i)] - feature.mean(0)
        ce_f[i] = fx_i.mean(axis=0)
    
    return ce_f

def get_hscore(features, labels):
    Covf = torch.cov(features.T)  # (hidden_dim, hidden_dim)
    label_choice = torch.unique(labels)
    g = torch.zeros_like(features)
    for z in label_choice:
        fl = features[labels == z, :]
        Ef_z = torch.mean(fl, dim=0) # (hidden_dim)
        g[labels == z] = Ef_z
    Covg = torch.cov(g.T)

    dif = torch.trace(Covg) / torch.trace(Covf)
    # orignal hscore
    # print("original H-score: dif = torch.trace(torch.pinverse(Covf, rcond=1e-15) @ Covg)")
    # dif = torch.trace(torch.pinverse(Covf, rcond=1e-15) @ Covg)

    return dif


def optimize_alpha(config, domains, all_features_train, all_label_train, leave_one_out):
    # # insert 0 to alpha at target_domain_index
    # alpha = torch.cat([alpha[:target_domain_index], torch.zeros(1), alpha[target_domain_index:]], dim=0)

    if leave_one_out:
        alpha = torch.ones(len(domains) - 1) / len(domains)
    else:
        alpha = torch.ones(len(domains)) / len(domains)

    alpha = alpha.cuda()
    alpha.requires_grad = True
    all_features_train.requires_grad = False
    all_label_train.requires_grad = False

    # optimizer = optim.SGD([alpha], lr=0.5)

    optimizer_class = optim.__dict__[config.alpha_opt.optimizer]
    optimizer = optimizer_class([alpha], lr=config.alpha_opt.lr)
    # weight_decay=config.alpha_opt.weight_decay
    
    for epoch in tqdm(range(config.alpha_opt.epoch)):
        optimizer.zero_grad()

        if leave_one_out:
            target_feature = get_target_feature_train(alpha, all_features_train)
        else:
            target_feature = get_target_feature(alpha, all_features_train)
        
        h_score = -get_hscore(target_feature, all_label_train)

        h_score.backward()
        optimizer.step()

        # print(f"epoch {epoch}: h_score {h_score.item()}")

        # alpha.requires_grad = False
        # alpha[alpha < 0] = 0
        # alpha = alpha / alpha.sum() if alpha.sum() > 1 else alpha
        # alpha.requires_grad = True
    
    alpha.requires_grad = False
    if leave_one_out:
        alpha = torch.cat([alpha, (1 - alpha.sum()).view(-1)], dim=0)
    # else:
    # alpha = alpha / alpha.sum() if alpha.sum() > 1 else alpha
    return alpha


def run(config: Any):
    # make a certain seed
    manual_seed(config.seed)
    logger = logging.getLogger()

    dataloader_train, dataloader_test = setup_data(
        config, is_test=True, few_shot_num=config.few_shot_num
    )

    num_classes = config.dataset.num_classes

    domains = config.dataset.domains
    target_domain = config.dataset.domain
    target_domain_index = -1

    for i, d in enumerate(domains):
        if d == target_domain:
            target_domain_index = i
            break
    if target_domain_index != -1:
        del domains[target_domain_index]

    # domains = ["v_task2", "v_task1"]
    # source_domains = [i for i in config.dataset.domains if i != config.dataset.domain]

    model_list = []

    all_features_train = torch.zeros((len(domains), len(dataloader_train.dataset), config.model.hidden_dim)).cuda()
    all_label_train = None

    print(f"target domain: {config.dataset.domain}")

    with torch.no_grad():

        # save labels & features
        for i, d in enumerate(domains):
            print(f"domain: {d}")
            
            model = get_model(config, d, logger, return_feat=True).cuda().eval()
            model_list.append(model)

            print(f"extract features by {d}-trained model")
            features_list = []
            labels_list = []
            res = []
            for data in tqdm(dataloader_train):
                # get the inputs
                inputs, labels = data

                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs, features = model(inputs)

                res.append(torch.argmax(outputs, dim=1).detach() == labels.detach())

                # features_list.append(normalize(features.detach()))
                features_list.append(features.detach())
                if all_label_train is None:
                    labels_list.append(labels.detach())

            # all_features_train[i, :, :] = torch.cat(features_list, dim=0).cuda()
            all_features_train[i, :, :] = normalize(torch.cat(features_list, dim=0)).cuda()
            if all_label_train is None:
                all_label_train = torch.cat(labels_list, dim=0).cuda()

            print(f"source {d} model on {target_domain} accuracy: {torch.cat(res, dim=0).sum() / len(all_label_train)}")
    # exit()

    # stop with torch.no_grad()

    del inputs, labels, data
    del dataloader_train
    del model
    del features_list, labels_list


    # compute H-score and validation accuracy
    print(f"\n\n*******start calc alpha by H-Score*******")
    
    # random, average, hscore, opt_all, opt_leave_one_out
    if config.alpha_type == "opt_leave_one_out":
        alpha = optimize_alpha(config, domains, all_features_train, all_label_train, leave_one_out=True)
    elif config.alpha_type == "opt_all":
        alpha = optimize_alpha(config, domains, all_features_train, all_label_train, leave_one_out=False)
    elif config.alpha_type == "random":
        alpha = torch.rand(len(domains)).cuda()
        alpha = alpha / alpha.sum()
    elif config.alpha_type == "average":
        alpha = torch.ones(len(domains)).cuda() / len(domains)
    elif config.alpha_type == "hscore":
        hscore_list = [
            get_hscore(all_features_train[i], all_label_train) for i in range(len(domains))
        ]
        alpha = torch.tensor(hscore_list).cuda()
        alpha = alpha / alpha.sum()
    else:
        raise NotImplementedError
    
    alpha = alpha.cuda()
    print(f"alpha strategy: {config.alpha_type}")
    print(f"final alpha: {alpha}")
    

    print(f"\n\n*******get G*******")
    with torch.no_grad():
        target_feature = get_target_feature(alpha, all_features_train)
        # target_feature = normalize(target_feature) # already done in feature_list
        # target_feature = normalize(target_feature)

        ce_f = get_conditional_exp( label=all_label_train, feature=target_feature, num_classes=num_classes )  # (num_classes, hidden_dim)

        # torch.permute = np.transpose ; torch.transpose = np.swapaxes; torch.mm = np.dot ; torch.inverse = np.linalg.inv; 
        # gamma_f = target_feature.T@target_feature / target_feature.shape[0] # (hidden_dim, hidden_dim)
        # g = (torch.inverse(gamma_f) @ (ce_f_s.permute((1,2,0))@alpha).T).T
        # g = ( torch.inverse(gamma_f) @ ce_f.T ).T # (hidden_dim, num_classes).T
        g = ce_f.cuda()


    del all_features_train,
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\n\n*******start test on {config.dataset.domain}*******")
    with torch.no_grad():
        score = target_feature @ g.T
        acc_test = (torch.argmax(score, dim=1) == all_label_train).sum().item() / len(all_label_train)
        print("Target(Train) accuracy: ", acc_test)

        del score, target_feature, all_label_train

        # test_features = torch.zeros(len(dataloader_test.dataset), config.model.hidden_dim).cuda()

        acc_test = 0

        # model_list = []
        # for i, d in tqdm(enumerate(domains)):    
        #     model = get_model(config, d, None, return_feat=True).cuda().eval()
        #     model_list.append(model)
        test_features = torch.zeros(len(dataloader_test.dataset), config.model.hidden_dim).cuda()
        test_label = None

        for i, d in enumerate(domains):
            features_list = []
            labels_list = []
            model = model_list[i]
            for data in tqdm(dataloader_test):
                inputs, labels = data
                inputs = inputs.cuda()

                _, features = model(inputs)

                features_list.append(features.detach())
                if test_label is None:
                    labels_list.append(labels.detach())

            test_features += normalize(torch.cat(features_list, dim=0)).cuda() * alpha[i]
            if test_label is None:
                test_label = torch.cat(labels_list, dim=0).cuda()

        del model, features_list, labels_list, features, labels, data, inputs
        # g_norm = normalize(g.cuda())
        g_norm = g.cuda()
        # f_norm = normalize(test_features.cuda())
        f_norm = test_features.cuda()
        del test_features,
        score = f_norm @ g_norm.T
        print("Correct num: ", (torch.argmax(score, dim=1) == test_label.cuda()).sum().cpu().item())
        print("Incorrect num: ", (torch.argmax(score, dim=1) != test_label.cuda()).sum().cpu().item())
        acc_test = (torch.argmax(score, dim=1) == test_label.cuda()).sum().cpu().item() / len(dataloader_test.dataset)
        print(f"Target(Test) accuracy: {acc_test} ; test sample: {len(dataloader_test.dataset)}")
    print(f"*******done - target: {config.dataset.domain} ; source: {domains}*******")
    print("#########################################################\n\n\n")


# main entrypoint
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # with idist.Parallel() as p:
    #     # with idist.Parallel("gloo") as p:
    #     p.run(run, config=cfg)


    run(cfg)


if __name__ == "__main__":
    # CUBLAS_WORKSPACE_CONFIG=:4096:8
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    main()
