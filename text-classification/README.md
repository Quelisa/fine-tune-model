### 文本分类任务fine-tune

针对文本分类任务进行的fine-tune，fine-tune前需要按照数据格式整理好需要数据，数据文件包括train文件、dev文件以及test文件。
fine-tune参数可以在params.json文件中对照修改，然后执行命令如下命令进行fine-tune
```shell
CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --model_name_or_path "xxx" \
    --train_file "xx.txt" \
    --dev_file "xx.txt" \
    --test_file "xx.txt" \
    --num_labels 5 \
    --max_seq_length 512 \
    --pad_to_max_length true
```


