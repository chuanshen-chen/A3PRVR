# A3PRVR

## Environment Setup
Strict version replication is not required. If version conflicts occur, install the versions supported by your local machine.

```bash
# Create and activate conda environment
conda create -n A3PRVR python=3.9
conda activate A3PRVR

# Install dependencies
cd root_path  # Replace with your project root directory path
pip install -r requirements.txt
```

## Data Dependencies
Only the **Charades-STA** dataset needs to be downloaded for reproduction (smaller in size than ActivityNet and TVR). Complete dependency data download:  
[Data Download Link](link)

## Pretrained Model Preparation (Optional)
Only when you need to enable the **Hard negative sample generation module** (the loss function corresponding to Figure 4 in the paper), you must first download the RoBERTa pretrained weights:

```bash
cd root_path  # Project root directory
python3 utils/download_roberta.py
```

## Training
Each dataset has a separate training script. Execute the following commands to start training:

### Charades-STA
```bash
cd root_path
bash do_charades.sh
```

### ActivityNet
```bash
cd root_path
bash do_activitynet.sh
```

### TVR
```bash
cd root_path
bash do_tvr.sh
```

### Training Parameter Explanation
- If **not enabling Hard negative sample generation**: Do not pass the `neg_query_loss` parameter, and you can delete all related parameters in the script:
  - `neg_query_loss`
  - `neg_query_loss_weight`
  - `neg_action_num`
  - `neg_object_num`
  - `neg_query_loss_branch`
- If **enabling**: Retain the above parameters and adjust their values as needed.

## Testing
The testing process reuses the training scripts, and only two additional parameters need to be added:

### Testing Steps
1. Download the pretrained weights (ckpt):  
   [Model Checkpoint Download Link](download_link)
2. Modify the training script and add the following parameters:
   - `only_eval`: Enable evaluation-only mode
   - `eval_ckpt="your_ckpt_path"`: Replace with the storage path of the downloaded ckpt file

### Testing Example (Taking Charades-STA as an Example)
```bash
cd root_path
# Assume the modified script already includes the only_eval and eval_ckpt parameters
bash do_charades.sh
```
