dataset_name=activitynet
visual_feature=i3d
exp_id=aaai26_bestresult_forbetter_173.2 #roberta_add_neg_loss_final #roberta_add_neg_loss_final #roberta_add_neg_loss_final # #roberta_add_neg_loss_final #without_cross_attn #_add_kiv_loss
root_path=.
device_ids=0
###上面的内容不动，没动过，是mssl的参数

#这里的参数也不需要改动
text_feat_path=./dataset/activitynet_i3d/TextData/roberta_activitynet_query_feat.hdf5
caption_train_txt=./dataset/activitynet_i3d/TextData/activitynettrain.caption.txt
caption_test_txt=./dataset/activitynet_i3d/TextData/activitynetval.caption.txt

#### loss的配置
neg_query_loss='both' #neg只取action
neg_query_loss_branch=both #两个分支
neg_action_num=12
neg_object_num=12
neg_query_loss_weight=0.005
max_type=max_pos
neg_loss_label_smoothing=0.2

### qdta的参数配置
deformable_offset_groups=8
deformable_heads=8
deformable_offset_num=8
deformable_offset_scale=96
seed=520
description="debug_remake_tvr_173.2"
CUDA_VISIBLE_DEVICES=0 python method/train.py  --dataset_name $dataset_name --visual_feature $visual_feature \
                    --root_path $root_path  --dset_name $dataset_name --exp_id $exp_id \
                    --device_ids $device_ids \
                    --topk 0.2 --caption_train_txt $caption_train_txt --caption_test_txt $caption_test_txt\
                    --best_frame_loss   \
                    --phrase_action_branch   \
                    --description $description --flow_feat --visual_flow_feat_dim 512  --bsz 128 \
                    --text_feat_path $text_feat_path --seed $seed\
                    --cross_branch_fusion --deformable_attn --deformable_heads $deformable_heads --deformable_offset_groups $deformable_offset_groups\
                    --deformable_offset_num $deformable_offset_num --deformable_offset_scale $deformable_offset_scale \
                    --max_type $max_type  --neg_loss_label_smoothing $neg_loss_label_smoothing
                    # --neg_query_loss $neg_query_loss --neg_query_loss_weight $neg_query_loss_weight \
                    # --neg_action_num $neg_action_num --neg_object_num $neg_object_num --neg_query_loss_branch $neg_query_loss_branch
