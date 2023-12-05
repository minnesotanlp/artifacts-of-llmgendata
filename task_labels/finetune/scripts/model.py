import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel,PreTrainedModel, RobertaConfig
import utils
accelerator = utils.get_accelerator()

class MultiTaskRobertaModel(PreTrainedModel):
    def __init__(self, roberta_name, num_annots, num_classes):
        super(MultiTaskRobertaModel, self).__init__(RobertaConfig())
        self.roberta = RobertaModel.from_pretrained(roberta_name,).to(accelerator.device)
        self.config = self.roberta.config
        self.classifiers = nn.ModuleList([nn.Linear(self.roberta.config.hidden_size, num_classes).to(accelerator.device) for _ in range(num_annots)])

    def forward(self, input_ids, attention_mask, labels=[]):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state[:, 0, :]  # Use the [CLS] token representation
        outputs = [classifier(last_hidden_state) for classifier in self.classifiers]
        return outputs

