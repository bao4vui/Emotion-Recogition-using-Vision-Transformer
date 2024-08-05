from datasets import *
from transformers import ViTFeatureExtractor, ViTImageProcessor
from transformers import ViTModel, ViTPreTrainedModel, ViTConfig
from torch import nn
from transformers.modeling_outputs import ImageClassifierOutput


string_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
class ViTForImageClassification2(ViTPreTrainedModel):
    #define architecture
    def __init__(self, config: ViTConfig, num_labels=len(string_labels)):
        super().__init__(config)
        self.vit = ViTModel(config, add_pooling_layer=False)
        self.config = config
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.num_labels = num_labels

    #define a forward pass through that architecture + loss computation
    def forward(self, pixel_values, labels=None):
        outputs = self.vit(pixel_values=pixel_values)
        output = self.dropout(outputs.last_hidden_state[:,0])
        logits = self.classifier(output)

        loss = None
        if labels is not None:
          loss_fct = nn.CrossEntropyLoss()
          loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

