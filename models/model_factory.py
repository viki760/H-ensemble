import os
from models.classifier import Classifier
from torchvision import models
import torch.nn as nn

def setup_model(config, return_feat=False):
    
    name = config.model.name

    backbone = models.get_model(
        name,
        weights="DEFAULT", # if enable this, it won't allow to change num_classes directly
        # num_classes=config.model.hidden_dim
    )
    backbone.fc = nn.Linear(backbone.fc.in_features, config.model.hidden_dim)


    return Classifier(
        backbone, 
        hidden_dim=config.model.hidden_dim,
        num_classes=config.dataset.num_classes,
        return_feat=return_feat,
    )


def load_SHOT_model(config, domain, return_feat=False):
    import importlib
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    import network
    sys.path.pop()

    name = config.model.name

    import omegaconf
    args = omegaconf.OmegaConf.create({
        "net":name,
        "src": config.dataset.domains,
        "output_dir_src": os.path.join(config.checkpoint_dir, config.dataset.name),
        "class_num": config.dataset.num_classes,
        "bottleneck": config.model.hidden_dim,
        "classifier": "bn",
        "layer": "wn",
    })

    netF_list = [network.ResBase(res_name=args.net) for i in range(len(args.src))]

    netB_list = [network.feat_bottleneck(type=args.classifier, feature_dim=netF_list[i].in_features, bottleneck_dim=args.bottleneck) for i in range(len(args.src))] 
    netC_list = [network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck) for i in range(len(args.src))]

    # for i in range(len(args.src)):
    i = args.src.index(domain)
    modelpath = os.path.join(args.output_dir_src, args.src[i], 'source_F.pt') 
    # print(modelpath)
    netF_list[i].load_state_dict(torch.load(modelpath))
    netF_list[i].eval()

    modelpath = os.path.join(args.output_dir_src, args.src[i], 'source_B.pt') 
    # print(modelpath)
    netB_list[i].load_state_dict(torch.load(modelpath))
    netB_list[i].eval()

    modelpath = os.path.join(args.output_dir_src, args.src[i], 'source_C.pt') 
    # print(modelpath)
    netC_list[i].load_state_dict(torch.load(modelpath))
    netC_list[i].eval()

    class Classifier(nn.Module):
        def __init__(self, netF, netB, netC, return_feat=False):
            super(Classifier, self).__init__()
            self.netF = netF
            self.netB = netB
            self.netC = netC
            self.forward = self.forward_with_hidden if return_feat else self.forward_without_hidden

        
        def forward_with_hidden(self, x):
            x = self.netF(x)
            feas = self.netB(x)
            feas_uniform = F.normalize(feas)
            x = self.netC(feas)

            return x, feas_uniform
        
        def forward_without_hidden(self, x):
            x = self.netF(x)
            x = self.netB(x)
            x = self.netC(x)

            return x


    return Classifier(netF_list[i], netB_list[i], netC_list[i], return_feat=return_feat)

