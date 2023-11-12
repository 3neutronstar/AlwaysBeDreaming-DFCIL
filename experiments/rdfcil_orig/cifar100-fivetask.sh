# CIFAR-100 five task shell script
# process inputs
# command : bash experiments/cifar100-fivetask.sh --gpuid $GPUID
DEFAULTGPU=0
GPUID=0 # ${1:-$DEFAULTDR}
# GPUID=${1:-$DEFAULTDR}

# benchmark settings
DATE=AAAI2023
SPLIT=20
OUTDIR=outputs/${DATE}/DFCIL-fivetask/CIFAR100

###############################################################

# make save directory
mkdir -p $OUTDIR

# load saved models
OVERWRITE=0

# number of tasks to run
MAXTASK=-1

# hard coded inputs
REPEAT=1
SCHEDULE="80 160 210"
# SCHEDULE="100 150 200 250"
PI=5000
MODELNAME=resnet32
BS=128
WD=0.0005
MOM=0.9
OPT="SGD"
LR=0.1
 
#########################
#         OURS          #
#########################

# Full Method
python -u run_dfcil.py --dataset CIFAR100 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
    --first_split_size $SPLIT --other_split_size $SPLIT --schedule $SCHEDULE --schedule_type decay --batch_size $BS \
    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
    --mu 0.15 --memory 0 --model_name $MODELNAME --model_type resnet \
    --learner_type datafree --learner_name RDFCIL \
    --gen_model_name CIFAR_GEN --gen_model_type generator \
    --beta 1 --power_iters $PI --deep_inv_params 1e-3 5e1 1e-3 1e3 1 \
    --overwrite $OVERWRITE --max_task $MAXTASK --log_dir ${OUTDIR}/rdfcil_orig \
    --ce_mu 0.5 --rkd_mu 0.15 --mu 0.15 --finetuning_epoch 40 --finetuning_lr 0.005
