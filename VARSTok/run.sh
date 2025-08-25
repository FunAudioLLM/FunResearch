torchrun --nproc_per_node=4 --rdzv_endpoint=localhost:23456 train.py fit \
    --config configs/varstok_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml

