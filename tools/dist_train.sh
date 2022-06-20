HOST=127.0.0.1
PORT=9999
RANK=0
NODES=1
USER_LOGS_DIR=/tmp
config_file=$1
config_func=utt
(torchrun    --nproc_per_node=8 \
            --master_port=$PORT \
            --master_addr=$HOST \
            --node_rank=$RANK \
            --nnodes=$NODES \
            tools/train_dist.py --config-file ${config_file} \
            --config-func ${config_func} | tee -a $USER_LOGS_DIR/stdout) 3>&1 1>&2 2>&3 | tee -a $USER_LOGS_DIR/stderr
