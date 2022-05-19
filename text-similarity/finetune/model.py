from torch import nn
from transformers import BertConfig, BertModel, XLMRobertaConfig, XLMRobertaModel


class Pooler(nn.Module):
    """
    'cls': [CLS] + MLP
    'cls_before_pooler': [CLS]
    'avg': 最后一层每个节点的平均
    'avg_top2': 最后2层每个节点的平均
    'avg_first_last': 第一层和最后一层的平均
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in [
            "cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"
        ], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsuqeeze(-1)).sum(1) /
                    attention_mask.sum(-1).unsuqeeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 *
                             attention_mask.unsuqeeze(-1).sum(1) /
                             attention_mask.sum(-1).unsuqeeze(-1))
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((second_last_hidden + last_hidden) / 2.0 *
                             attention_mask.unsuqeeze(-1).sum(1) /
                             attention_mask.sum(-1).unsuqeeze(-1))
            return pooled_result
        else:
            raise NotImplementedError


class MLPLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features):
        x = self.dense(features)
        x = self.activation(x)

        return x


def ts_init(cls):
    """
    初始化模型的cls，进行文本将相似度计算
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.pooler_type)
    if cls.pooler_type == "cls":
        cls.mlp = MLPLayer(cls.model_args.hidden_size)


def ts_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    input_ids = input_ids.view(
        (-1, input_ids.size(-1)))  # (bs * num_sent, len)
    attention_mask = attention_mask.view(
        (-1, attention_mask.size(-1)))  # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(
            (-1, token_type_ids.size(-1)))  # (bs * num_sent, len)

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type
        in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    batch_size = input_ids.size(0)
    num_sent = input_ids.size(1)
    pooler_output = cls.pooler(attention_mask, outputs)

    if cls.pooler_type == "cls":
        pooler_output = pooler_output.view((batch_size, pooler_output.size(-1)))  # (bs, hidden)
        pooler_output = cls.mlp(pooler_output)

    return pooler_output


class BertForTS(nn.Module):
    def __init__(self, model_args):

        super().__init__()
        self.config = BertConfig.from_pretrained(model_args.model_path)
        self.bert = BertModel.from_pretrained(model_args.model_path, config=self.config)
        self.model_args = model_args
        for param in self.bert.parameters():
            param.requires_grad = True

        ts_init(self)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return ts_forward(
            self,
            self.bert,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class XLMRobertaForTS(nn.Module):
    def __init__(self, model_args):

        super().__init__()
        self.model_args = model_args
        self.config = XLMRobertaConfig.from_pretrained(model_args.model_path)
        self.roberta = XLMRobertaModel.from_pretrained(model_args.model_path, config=self.config)
        for param in self.roberta.parameters():
            param.requires_grad = True

        ts_init(self)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return ts_forward(
            self,
            self.roberta,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
