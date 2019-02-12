# HIGH-QUALITY-MONOCULAR-DEPTH-ESTIMATION
Code for paper "HIGH QUALITY MONOCULAR DEPTH ESTIMATION VIA MULTI-SCALE NETWORK AND DETAIL-PRESERVING OBJECTIVE"

1. To trian the model with the full loss of the paper on the NYU Depth V2 dataset, \
python train.py \
		--dataset nyu_depth_v2 \
		--train_file "../filenames/nyu_depth_v2_train_even.txt" \
		--val_file "../filenames/nyu_depth_v2_val_2k.txt" \
		--cnn_model "resnet_v2_50" \
		--decoding_at_image_size \
		--output_stride 16 \
		--multi_grid 1 2 4 \
		--aspp_rates 4 8 12 \
		--loss_depth_norm berhu \
		--loss_gradient_magnitude_norm l1 \
		--loss_gradient_magnitude_weight 1.0 \
		--loss_gradient_direction_norm l1 \
		--loss_gradient_direction_weight 1.0 \
		--loss_normal_weight 0.0 \
		--batch_size 8 \
		--num_epochs 20 \
		--learning_rate 1e-4 \
		--num_gpus 1 \
		--num_threads 4 \
		--batch_norm_epsilon 1e-5 \
		--batch_norm_decay 0.9997 \
		--l2_regularizer 1e-4
 
  2. To evaluate on the model, \
  python test_nyu_depth_v2.py --process_id_for_evaluation pid(12582) \
  You can download the trained model and results of the testset from https://mega.nz/#F!7ypiyIoa!vKgyTXSgxwieY0o8u9pksQ.
  
  Test performance: \
  |abs_rel: 0.127 |sq_rel: 0.088 |rmse: 0.468 |rmse_log: 0.165 |log10: 0.054 |acc1: 0.841 |acc2: 0.967 |acc3: 0.993
