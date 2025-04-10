source scripts/data_path.sh

THISNAME="release"
BESTEPOCH="14"
DTU_TESTLIST="lists/dtu/test.txt"
DTU_CKPT_FILE="./checkpoints/dtu/"$THISNAME"/finalmodel_"$BESTEPOCH".ckpt" # dtu pretrained model

exp=$1
PY_ARGS=${@:2}

DTU_LOG_DIR="./checkpoints/dtu/"$exp 
if [ ! -d $DTU_LOG_DIR ]; then
    mkdir -p $DTU_LOG_DIR
fi

DTU_OUT_DIR="./outputs/dtu/"$exp
if [ ! -d $DTU_OUT_DIR ]; then
    mkdir -p $DTU_OUT_DIR
fi

PLY_OUT_DIR="./outputs/dtu/"$exp"/pcd_fusion_plys/"
if [ ! -d $PLY_OUT_DIR ]; then
    mkdir -p $PLY_OUT_DIR
fi

python test_dtu_dypcd.py --dataset=general_eval4 --batch_size=1 --testpath=$DTU_TEST_ROOT  --testlist=$DTU_TESTLIST --loadckpt $DTU_CKPT_FILE --interval_scale 1.06 --outdir $DTU_OUT_DIR \
            --ndepths 8,8,4,4 --depth_inter_r 0.5,0.5,0.5,0.5 --conf 0.55 --group_cor --attn_temp 2 --inverse_depth $PY_ARGS | tee -a $DTU_LOG_DIR/log_test.txt

