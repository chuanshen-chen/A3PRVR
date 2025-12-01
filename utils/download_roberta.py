from transformers import RobertaModel, RobertaTokenizer

# 指定要下载的模型名称
model_name = "roberta-large"

# 下载并保存模型权重和配置
model = RobertaModel.from_pretrained(model_name)
model.save_pretrained(f"./{model_name}_saved")  # 保存到当前目录下的roberta-large_saved文件夹

# 下载并保存分词器
tokenizer = RobertaTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(f"./{model_name}_saved")

print(f"模型和分词器已保存到 ./{model_name}_saved 文件夹")