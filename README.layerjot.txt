MetaFormer instructions

Basic instructions:

Branch is layerjot in MetaFormer repository.

    cd MetaFormer-repo-directory

To build the container:

In the MetaFormer-repo-directory, run

    docker build -f Dockerfile -t metaformer.forest:latest .

This container is already built, but if you need to make a modification then you may have to rebuild the container.

To run the container:

In the MetaFormer-repo-directory, run

    . ./run_container.sh

The MetaFormer repository is

    cd /layerjot/MetaFormer

in the container.

To train:

In the MetaFormer-repo-directory, run

    python3 -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --master_port 12345  main.py  --cfg ./configs/MetaFG_2_224.yaml --batch-size 32 --tag n1_multimodal_data.0 --lr 5e-5 --min-lr 5e-7 --warmup-lr 5e-8 --epochs 5 --warmup-epochs 20 --layerjot --dataset n1_multimodal_data --pretrained /data/MetaFormer/pretrained/metafg_2_1k_224.pth --accumulation-steps 2 --opts DATA.IMG_SIZE 384 --data-path /data/MetaFormer/datasets --output /data/output --use_attr --amp-opt-level O0

where n1_multimodal_data is the dataset name.  The input data will be consumed from the datalake and written to /data/MetaFormer/datasets/<dataset name> which in this case will be /data/MetaFormer/datasets/n1_multimodal_data.

The full path to the output will be /data/output/MetaFG_2/<tag> which in this cas would be /data/output/MetaFG_2/n1_multimodal_data.0.  Training will periodically write out a checkpoint file to the output directory.  The name of the file is ckpt_epoch_XXX.pth where XXX is a integer representing the epoch.

Most of the output data can be deleted after archiving the relavent files.  I typically throwaway all of the checkpoint files except the final file.  I keep the log file, and the configuration json file, and the final checkpoint file.

To eval:

Specify the output module as the pretrained input for the evaluation step.  If we trained for 300 steps, then the last checkpoint file will be ckpt_epoch_299.pth:

    python3 -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --master_port 12345  main.py  --cfg ./configs/MetaFG_2_224.yaml --batch-size 32 --tag n1_multimodal_data.0 --lr 5e-5 --min-lr 5e-7 --warmup-lr 5e-8 --epochs 300 --warmup-epochs 20 --layerjot --dataset n1_multimodal_data --pretrained /data/output/MetaFG_2/n1_multimodal_data.0/ckpt_epoch_299.pth --accumulation-steps 2 --opts DATA.IMG_SIZE 384 --data-path /data/MetaFormer/datasets --output /data/output --eval --use_attr --amp-opt-level O0

One note.  The training/validation datasets description is put into the train and val directories respectively.  It is a json file named imageinfo.json and it describes the attributes of the files in the training or validation datasets.

The evaluation dataset current goes into the eval directory (as opposed to val) as a test directory and potentially it should just be renamed to test.  The members of the eval dataset are 10% of the overall dataset and hence the model will have seen all of the images as part of training or validation.

Right now we, we are making a mistake by not partitioning the overall dataset into training, validation, and test.  But I don't think this is the highest priority right now.

Background Information:

I had to modify data/build.py in order for the dataset name to be understood by MetaFormer main.py.  The configuration object class can be found in config.py.

All parameters in MetaFormer are handled by a config file/object which causes lots of problems.  I added a new configuration option that forces the dataset name to be interpreted as a layerjot dataset name IS_LAYERJOT.  That is why --layerjot is added as a command line argument.
