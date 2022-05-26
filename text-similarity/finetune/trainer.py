import sys

sys.path.append("..")
from utils.params import FT_Configer
import torch
from torch import nn
import numpy as np
import time
import logging
from tqdm import tqdm
from transformers import BertTokenizer, XLMRobertaTokenizer, get_cosine_schedule_with_warmup, AdamW
from finetune.model import BertForTS, XLMRobertaForTS
from torch.utils.data import TensorDataset, DataLoader
from finetune.retrieve import Retrieve

logging.basicConfig(filename='../log/ts_finetune.log',
                    filemode="w",
                    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                    datefmt="%d-%m-%Y %H:%M:%S",
                    level=logging.DEBUG)
logger = logging.getLogger("Text Similarity Finetune Logger")


class Similarity(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Trainer():
    def __init__(self, params: FT_Configer):
        self.params = params
        self.sim = Similarity(temp=params.cl_temperature)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if params.model_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(
                self.params.model_path)

        elif params.model_type == 'xmlroberta':
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(
                self.params.model_path)

    def dataset_fakelabel(self, data_path):

        logger.info("load unsupervised dataset begin:")
        input_ids, attention_masks, labels = [], [], []
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in tqdm(enumerate(f)):
                text = line.strip()
                encode_dict = self.tokenizer.encode_plus(
                    text=text,
                    max_length=self.params.max_len,
                    padding='max_length',
                    truncation=True)
                input_ids.append(encode_dict['input_ids'])
                attention_masks.append(encode_dict['attention_mask'])
                labels.append(i)
        logger.info("load unsupervised dataset finish!")
        return torch.tensor(input_ids), torch.tensor(
            attention_masks), torch.tensor(labels)

    def dataset(self, data_path):

        logger.info("load supervised dataset begin:")
        input_ids, attention_masks, labels = [], [], []
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in tqdm(enumerate(f)):
                text, label = line.strip().split('\t')
                encode_dict = self.tokenizer.encode_plus(
                    text=text,
                    max_length=self.params.max_len,
                    padding='max_length',
                    truncation=True)
                input_ids.append(encode_dict['input_ids'])
                attention_masks.append(encode_dict['attention_mask'])
                labels.append(int(label))
        logger.info("load supervised dataset finish!")
        return torch.tensor(input_ids), torch.tensor(
            attention_masks), torch.tensor(labels)

    def data_loader(self, input_ids, attention_masks, labels):

        data = TensorDataset(input_ids, attention_masks, labels)
        loader = DataLoader(data,
                            batch_size=self.params.batch_size,
                            shuffle=True)
        return loader

    def accuracy(self, sim_matrix, labels):
        top1_acc = 0
        mask_matrix = torch.eye(sim_matrix.size(0)).to(self.device)
        p_red = (sim_matrix - mask_matrix).argmax(dim=0)

        for i in range(len(p_red)):
            if labels[p_red[i]] == labels[i]:
                top1_acc += 1

        return top1_acc / sim_matrix.size(0)

    def evalute(self, model, data_loader):
        top1_acc = []
        model.eval()
        model.to(self.device)
        with torch.no_grad():
            for step, (inputs_ids, attention_mask,
                       labels) in tqdm(enumerate(data_loader)):
                ids, att, label = inputs_ids.to(
                    self.device), attention_mask.to(self.device), labels.to(
                        self.device)
                outputs = model(ids, att)
                cos_sim = self.sim(outputs.unsqueeze(1), outputs.unsqueeze(0))
                top1_acc.append(self.accuracy(cos_sim, label))
        return np.sum(top1_acc) / len(top1_acc)

    def train(self, model, train_loader, dev_loader, index_loader, optimizer,
              schedule):
        best_top1_acc = 0.0

        model.train()
        model.to(self.device)
        for i in range(self.params.epoch):
            start_time = time.time()
            train_loss_sum = 0.0

            logger.info("beign to train model: ")
            for step, (inputs_ids, attention_mask,
                       labels) in tqdm(enumerate(train_loader)):
                ids, att = inputs_ids.to(self.device), attention_mask.to(
                    self.device)
                outputs = model(ids, att)
                cos_sim = self.sim(outputs.unsqueeze(1), outputs.unsqueeze(0))
                if self.params.unsupervised:
                    labels = torch.arange(cos_sim.size(0)).long().to(
                        self.device)

                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(cos_sim, labels)
                loss /= self.params.gradient_acc

                optimizer.zero_grad()
                loss.backward()

                if step % self.params.gradient_acc == 0:
                    optimizer.step()
                    schedule.step()
                train_loss_sum += loss.item()

                if (step + 1) % (len(train_loader) // 10) == 0:
                    logger.info(
                        "Epoch {:02d} | Step {:03d}/{:03d} | Loss {:.4f} | Time {:.2f}"
                        .format(i + 1, step + 1, len(train_loader),
                                train_loss_sum / (step + 1),
                                time.time() - start_time))

            if self.params.evalution_task == 'retrieve' and index_loader is not None:
                """
                采用检索方式测试，检索集合与测试集合分离
                首先用检索集合建立索引，然后用测试集合检索测试
                """
                retrieve = Retrieve(model, self.device)
                retrieve.build_index(index_loader)
                logger.info("begin to evalute dev for retrieve:")
                top1_acc = retrieve.evalute(dev_loader)
                logger.info(
                    "dev for retrieve top1_acc: {:.2f}".format(top1_acc))

            elif self.params.evalution_task == 'similarity':
                logger.info("begin to evalute dev for similarity:")
                top1_acc = self.evalute(model, dev_loader)
                logger.info(
                    "dev for similarity top1_acc: {:.2f}".format(top1_acc))

            if top1_acc > best_top1_acc:
                best_top1_acc = top1_acc
                if self.params.model_type == 'bert':
                    model.bert.save_pretrained(self.params.model_save_dir)
                elif self.params.model_type == 'xmlroberta':
                    model.roberta.save_pretrained(self.params.model_save_dir)
                else:
                    raise NotImplementedError
                self.tokenizer.save_pretrained(self.params.model_save_dir)
                model.config.save_pretrained(self.params.model_save_dir)

            logger.info(
                "Current dev top1_acc is {:.4f}, best top1_acc is {:.4f}".
                format(top1_acc, best_top1_acc))
            logger.info("Time costed : {}s \n".format(
                round(time.time() - start_time, 3)))

    def run_finetune(self):
        if self.params.unsupervised:
            train_loader = self.data_loader(*self.dataset_fakelabel(
                f'{self.params.data_dir}unsup_train.txt'))
        else:
            train_loader = self.datalabel_loader(
                *self.dataset(f'{self.params.data_dir}train.txt'))

        dev_loader = self.data_loader(
            *self.dataset(f'{self.params.data_dir}dev.txt'))
        test_loader = self.data_loader(
            *self.dataset(f'{self.params.data_dir}test.txt'))

        if self.params.evalution_task == 'retrieve':
            dev_index_loader = self.data_loader(
                *self.dataset(f'{self.params.data_dir}index_dev.txt'))
            test_index_loader = self.data_loader(
                *self.dataset(f'{self.params.data_dir}index_test.txt'))
        elif self.params.evalution_task == 'similarity':
            dev_index_loader = None
            test_index_loader = None
        else:
            raise NotImplementedError

        if self.params.model_type == 'bert':
            model = BertForTS(self.params).to(self.device)
        elif self.params.model_type == 'xmlroberta':
            model = XLMRobertaForTS(self.params).to(self.device)
        else:
            raise NotImplementedError

        total_steps = len(train_loader) * self.params.epoch
        optimizer = AdamW(params=model.parameters(),
                          lr=self.params.learning_rate,
                          weight_decay=self.params.weight_decay)
        schedule = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.params.warmup_rate * total_steps,
            num_training_steps=total_steps)

        self.train(model, train_loader, dev_loader, dev_index_loader,
                   optimizer, schedule)

        if self.params.evalution_task == 'retrieve':
            retrieve = Retrieve(model, self.device)
            retrieve.build_index(test_index_loader)
            top1_acc = retrieve.evalute(test_loader)
        elif self.params.evalution_task == 'similarity':
            top1_acc = self.evalute(model, test_loader)
        else:
            raise NotImplementedError

        logger.info("test top1_acc is {:.4f}".format(top1_acc))
