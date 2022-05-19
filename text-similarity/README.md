### 检索类任务fine-tune

针对检索类任务进行的fine-tune，fine-tune前需要按照数据格式整理好需要数据，数据文件包括train文件、dev文件以及test文件。
fine-tune参数可以在params.json文件中对照修改，然后执行如下命令进行fine-tune
```shell
CUDA_VISIBLE_DEVICES=0 python run_finetune.py --params params.json
```


#### (1) 模型选择：BERT、RoBerta
目前支持Bert、Robert模型的训练，在配置文件params.json中修改模型模型字段"model_type"为"bert"或者"roberta"即可对选择的模型进行fine-tune。

#### (2) 训练方式：监督和无监督
fine-tune主要分为无监督fine-tune和有监督fine-tune，无监督fine-tune采用对比学的方法，本例中为simcse中方法。有监督方法可视为分类问题或者排序问题。选择有监督或无监督方法需要在配置文件params.json中修改"unsupervised"为true或者false。


#### (3) 测试任务：检索和相似度
检索任务主要是将测试集合中的一部分数据提前建立索引，测试的时候用另一部分句子进行检索，索引比例可以自己确定。检索任务的文件名中有"index"字样。相似度测试则将句子全部进行encode后，计算相似度，评估top1命中的准确率。选择哪一种测试任务需要在配置文件params.json中修改"evalution_task"为"retrieve"或者"similarity"。