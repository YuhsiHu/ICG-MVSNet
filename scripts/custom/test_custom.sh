source scripts/data_path.sh

THISNAME="release"
BESTEPOCH="9"
CUSTOM_TESTLIST="lists/custom/test.txt"
DTU_CKPT_FILE="./checkpoints/bld/"$THISNAME"/finalmodel_"$BESTEPOCH".ckpt" # bld pretrained model

exp=$1
PY_ARGS=${@:2}

CUSTOM_LOG_DIR="./checkpoints/custom/"$exp 
if [ ! -d $CUSTOM_LOG_DIR ]; then
    mkdir -p $CUSTOM_LOG_DIR
fi

CUSTOM_OUT_DIR="./outputs/custom/"$exp
if [ ! -d $CUSTOM_OUT_DIR ]; then
    mkdir -p $CUSTOM_OUT_DIR
fi

PLY_OUT_DIR="./outputs/custom/"$exp"/pcd_fusion_plys/"
if [ ! -d $PLY_OUT_DIR ]; then
    mkdir -p $PLY_OUT_DIR
fi

python test_dtu_dypcd.py --dataset=general_eval4 --batch_size=1 --testpath=$CUSTOM_TEST_ROOT  --testlist=$CUSTOM_TESTLIST --loadckpt $DTU_CKPT_FILE --interval_scale 1.06 --outdir $CUSTOM_OUT_DIR \
            --max_h=1280 --max_w=1920 \
            --ndepths 8,8,4,4 --depth_inter_r 0.5,0.5,0.5,0.5 --conf 0.55 --group_cor --attn_temp 2 --inverse_depth $PY_ARGS | tee -a $CUSTOM_LOG_DIR/log_test.txt

