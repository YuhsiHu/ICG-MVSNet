source scripts/data_path.sh

THISNAME="release"
BESTEPOCH="14"
DTU_TRAINLIST="lists/dtu/train.txt"
DTU_TESTLIST="lists/dtu/test.txt"
DTU_CKPT_FILE="./checkpoints/dtu/"$THISNAME"/finalmodel_"$BESTEPOCH".ckpt"  # dtu pretrained model

exp=$1
PY_ARGS=${@:2}

DTU_LOG_DIR="./checkpoints/dtu/"$exp 
if [ ! -d $DTU_LOG_DIR ]; then
    mkdir -p $DTU_LOG_DIR
fi

python -m torch.distributed.launch --nproc_per_node=1 train_dtu.py --logdir $DTU_LOG_DIR --dataset=dtu_yao4 --batch_size=1 --trainpath=$DTU_TRAIN_ROOT --summary_freq 100 --lr=0.000025 \
        --ndepths 8,8,4,4 --depth_inter_r 0.5,0.5,0.5,0.5 --group_cor --inverse_depth --rt --loadckpt $DTU_CKPT_FILE --epochs=10 --use_raw_train --attn_temp 2 --trainlist $DTU_TRAINLIST --testlist $DTU_TESTLIST $PY_ARGS | tee -a $DTU_LOG_DIR/log.txt
