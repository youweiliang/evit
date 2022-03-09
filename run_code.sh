now=$(date +"%Y%m%d_%H%M%S")
logdir=/train_log/exp_$now
datapath=""

echo "output dir: $logdir"

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env \
	main.py \
	--model deit_small_patch16_shrink_base \
	--fuse_token \
	--base_keep_rate 0.7 \
	--input-size 224 \
	--batch-size 256 \
	--warmup-epochs 5 \
	--shrink_start_epoch 10 \
	--shrink_epochs 100 \
	--epochs 300 \
	--dist-eval \
	--data-path $datapath \
	--output_dir $logdir

echo "output dir for the last exp: $logdir"\
