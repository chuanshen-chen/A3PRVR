dataset_name=tvr
visual_feature=i3d
q_feat_size=768
margin=0.1
exp_id=aaai26_bestresult
root_path=.
device_ids=0

caption_train_txt=./dataset/tvr_i3d/TextData/tvrtrain.caption.txt
caption_test_txt=./dataset/tvr_i3d/TextData/tvrval.caption.txt
text_feat_path=./dataset/tvr_i3d/TextData/roberta_tvr_query_feat.hdf5

deformable_heads=8
deformable_offset_groups=8
deformable_offset_num=8
deformable_offset_scale=64
num_workers=4
seed=808
description="debug_eval_tvr_result"
CUDA_VISIBLE_DEVICES=0 python method/train.py  --dataset_name $dataset_name --visual_feature $visual_feature \
                    --root_path $root_path  --dset_name $dataset_name --exp_id $exp_id \
                    --device_ids $device_ids --q_feat_size $q_feat_size --margin $margin \
                    --topk 0.2 --caption_train_txt $caption_train_txt --caption_test_txt $caption_test_txt\
                    --best_frame_loss  \
                    --phrase_action_branch   \
                    --description $description --flow_feat --visual_flow_feat_dim 512 --visual_feat_dim 3072  --bsz 128 \
                    --text_feat_path $text_feat_path --cross_branch_fusion\
                    --deformable_attn --deformable_heads $deformable_heads --deformable_offset_groups $deformable_offset_groups\
                    --deformable_offset_num $deformable_offset_num --deformable_offset_scale $deformable_offset_scale --seed $seed --num_workers $num_workers \
                    --only_eval --eval_ckpt 'checkpoints/tvr.ckpt'
