dataset_name=charades
visual_feature=i3d
exp_id=debug

root_path=.
device_ids=0

text_feat_path=./dataset/charades_i3d/TextData/roberta_charades_query_feat.hdf5
caption_train_txt=./dataset/charades_i3d/TextData/charadestrain.caption.txt
caption_test_txt=./dataset/charades_i3d/TextData/charadesval.caption.txt

neg_query_loss='both' #neg只取action
neg_query_loss_branch=both #两个分支



neg_loss_label_smoothing=0.2

seed=666
deformable_heads=8
deformable_offset_group=8
deformable_offset_num=8
deformable_offset_scale=96 
neg_action_num=8
neg_object_num=8
neg_query_loss_weight=0.005
description="charades"

CUDA_VISIBLE_DEVICES=0 python method/train.py  --dataset_name $dataset_name --visual_feature $visual_feature \
                --root_path $root_path  --dset_name $dataset_name --exp_id $exp_id \
                --device_ids $device_ids \
                --topk 0.2 --caption_train_txt $caption_train_txt --caption_test_txt $caption_test_txt\
                --best_frame_loss   \
                --phrase_action_branch   \
                --description $description --flow_feat --visual_flow_feat_dim 512   --bsz 128 \
                --text_feat_path $text_feat_path --seed $seed\
                --cross_branch_fusion --deformable_attn --deformable_heads $deformable_heads --deformable_offset_groups $deformable_offset_group\
                --deformable_offset_num $deformable_offset_num --deformable_offset_scale $deformable_offset_scale \
                --only_eval --eval_ckpt 'checkpoints/charades.ckpt'
                # --neg_loss_label_smoothing $neg_loss_label_smoothing\
                # --neg_query_loss $neg_query_loss --neg_query_loss_weight $neg_query_loss_weight \
                # --neg_action_num $neg_action_num --neg_object_num $neg_object_num --neg_query_loss_branch $neg_query_loss_branch \
                # --only_eval --eval_ckpt 'checkpoints/charades.ckpt'
