# train source

#python train_source.py --dset office-caltech --s $i --max_epoch 100 --trte val --gpu_id 0 --output ckps/source/uda
# $i is the i'th task

# run
python test_HEnsemble.py dataset=office-caltech model.name=resnet18 model.hidden_dim=256 seed=2023 output_dir=./logs dataset.domain=amazon_1 eval_batch_size=64 max_epochs=8 few_shot_num=8 checkpoint_dir=./checkpoints val_sample_num=2000 if_use_shot_model=True 