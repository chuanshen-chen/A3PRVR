import json
import h5py
from tqdm import tqdm


# 求一个数组中topK元素的下标
def top_k_indices(lst, k: int):
    # 使用 sorted() 对列表进行排序，并获取前三个元素的下标
    indices = sorted(range(len(lst)), key=lambda i: lst[i], reverse=True)[:k]
    return indices

def get_max_min_frame_num_object():
    input_features_path = "/mnt/cephfs/home/shenqingwei/zhuweikun/clip_image_encoder/clip_features_64frame.h5"
    frame_max_num_object = 25
    frame_min_num_object = 3
       
    exception_box = []
    exception_path = "/mnt/cephfs/home/shenqingwei/models/AME-Net/64frames_exception_box.txt"
    
    # box_info_path = "/mnt/cephfs/home/shenqingwei/zhuweikun/clip_image_encoder/boxes.json"
    box_info_path = "/mnt/cephfs/home/shenqingwei/zhuweikun/clip_image_encoder/boxes_64frame.json"

    with open(box_info_path, 'r', encoding='utf-8') as file:
        box_info = json.load(file)
    seq_len = 32

    for vid in tqdm(box_info.keys(), desc='Reading json file'):
        # if vid == 'ZSREG':            
        #     continue
        
        box_num = [len(item['box']) for item in box_info[vid]]
        indices = top_k_indices(box_num, seq_len) # 选取top48
        box_num = [box_num[i] for i in indices]

        for i, num_obj in enumerate(box_num):
            if num_obj > frame_max_num_object:
                frame_max_num_object = num_obj
            if num_obj < frame_min_num_object:
                frame_min_num_object = num_obj 
            if num_obj < 4:
                # print(f'index:{i}')
                exception_box.append(f'{vid}\n')
                break              
       
    with open(exception_path, 'w', encoding='utf-8') as file:
        file.writelines(exception_box)

    print("帧内最大box数量:", frame_max_num_object)
    print("帧内最小box数量:", frame_min_num_object)   
    print(f"故障视频数量:{len(exception_box)}")

    # with h5py.File(input_features_path, 'r') as hf:
    #     for vid in tqdm(hf.keys(), desc="Reading videos in h5 file..."): # 每一个video取若干帧，每一帧有N个region, 每个region的特征是512维
    #         group = hf[vid]  

    #         box_num = [len(item['box']) for item in box_info[vid]]
    #         indices = top_k_indices(box_num, seq_len) # 选取top48
    #         frame_name = [box_info[vid][idx]['image_name'] for idx in indices]

    #         for frame in frame_name:
    #             if frame == 'mnt':
    #                 print(f'Error:{frame}, 图片名称错误')
    #                 continue

    #             temp = group[frame]
    #             num_obj = temp.shape[0] - 1 # 减去原始图片的特征
    #             if num_obj == 1:
    #                 one_box.append(f"{vid}/{frame}\n")
            
    #             if num_obj > frame_max_num_object:
    #                 frame_max_num_object = num_obj
    #             if num_obj < frame_min_num_object:
    #                 frame_min_num_object = num_obj     

    #     with open(exception_path, 'w', encoding='utf-8') as file:
    #         file.writelines(one_box)

    # print("帧内最大box数量:", frame_max_num_object)
    # print("帧内最小box数量:", frame_min_num_object)

if __name__ == "__main__":
    get_max_min_frame_num_object()