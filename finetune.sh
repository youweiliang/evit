now=$(date +"%Y%m%d_%H%M%S")
logdir=/train_log/exp_$now
datapath=""
ckpt=deit_small_patch16_224-cd65a155.pth

echo "output dir: $logdir"

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env \
	main.py \
	--model deit_small_patch16_shrink_base \
	--fuse_token \
	--base_keep_rate 0.7 \
	--input-size 224 \
	--sched cosine \
	--lr 2e-5 \
	--min-lr 2e-6 \
	--weight-decay 1e-6 \
	--batch-size 256 \
	--shrink_start_epoch 0 \
	--warmup-epochs 0 \
	--shrink_epochs 0 \
	--epochs 30 \
	--dist-eval \
	--finetune $ckpt \
	--data-path $datapath \
	--output_dir $logdir

echo "output dir for the last exp: $logdir"\
