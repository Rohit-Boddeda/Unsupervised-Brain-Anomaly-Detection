# command to run on single gpu
CUDA_VISIBLE_DEVICES=0 python train.py --n-gpu-per-node 1 --name IXI --beta-max 0.3 --clip-denoise --image-size 256 --cond-x1 --batch-size 8 --microbatch 4 --num-itr 50000 --space 'latent' --log-dir ./IXI_logs_latent/ --log-writer tensorboard 

# python sample.py --ckpt mvtec --n-gpu-per-node 4 --batch-size 4 --savename iter1
