Official Code for AAAI-24 [H-ensemble: An Information Theoretic Approach to Reliable Few-Shot Multi-Source-Free Transfer](https://ojs.aaai.org/index.php/AAAI/article/view/29528)

## Abstract

Multi-source transfer learning is an effective solution to data scarcity by utilizing multiple source tasks for the learning of the target task. However, access to source data and model details is limited in the era of commercial models, giving rise to the setting of multi-source-free (MSF) transfer learning that aims to leverage source domain knowledge without such access. As a newly defined problem paradigm, MSF transfer learning remains largely underexplored and not clearly formulated. In this work, we adopt an information theoretic perspective on it and propose a framework named H-ensemble, which dynamically learns the optimal linear combination, or ensemble, of source models for the target task, using a generalization of maximal correlation regression. The ensemble weights are optimized by maximizing an information theoretic metric for transferability. Compared to previous works, H-ensemble is characterized by: 1) its adaptability to a novel and realistic MSF setting for few-shot target tasks, 2) theoretical reliability, 3) a lightweight structure easy to interpret and adapt. Our method is empirically validated by ablation studies, along with extensive comparative analysis with other task ensemble and transfer learning methods. We show that the H-ensemble can successfully learn the optimal task ensemble, as well as outperform prior arts.

## Requirements
Pytorch&Torchvision and other dependencies can be installed using the following command:
```bash
pip install hydra-core tqdm pytorch-ignite
pip install opencv-python numpy pandas scipy scikit-learn
```

## Usage

We train source models using the [SHOT Repo](https://github.com/tim-learn/SHOT), i.e., "train_source.py" and "data_list.py". For SHOT splits a network into several parts, we add a flag `if_use_shot_model` for our script to load it correctly. See example command in "run.sh".

To build up the task-split datasets, use the `_dataset/readfile.py` first to convert the original datasets that are in [ImageFolder type](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) to a image-label annotation. Then use `_dataset/tasksplit.py` to perform actually task splitting operation ( manual dir renaming and other step must be done due to incomplete automation 😣 ) We ship an example dataset folder with these scripts.

Modify config files under `./conf` to specify dataset, path, etc..