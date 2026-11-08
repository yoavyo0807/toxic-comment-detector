import os
import torch
import lightning as L
# from transformers import DistilBertTokenizer, DistilBertModel
from torchmetrics.classification import MultilabelF1Score, MultilabelAUROC
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer
from pickle import load

user_name_hugging_face = "borodache"
model_name_hugging_face = "distilBERT_toxic_detector"
LABEL_COLUMNS = ["nsfw", "hate_speech", "bullying"]
n_labels = len(LABEL_COLUMNS)
learning_rate = 1e-5
token_huggingface = os.environ.get("HF_TOKEN")  # None is fine if the model repo is public


class MultiLabelDetectionModel(L.LightningModule):
    def __init__(self, model_name=model_name_hugging_face, n_labels = n_labels):
        # print("I am in MultiLabelDetectionModel.__init__")
        super().__init__()
        self.save_hyperparameters()
        self.n_labels = n_labels
        self.tokenizer = AutoTokenizer.from_pretrained(f"{user_name_hugging_face}/{model_name}",
                                                       token=token_huggingface)
        self.model = AutoModel.from_pretrained(f"{user_name_hugging_face}/{model_name}",
                                               token=token_huggingface)
        # self.classifier = nn.Linear(self.model.config.hidden_size, n_labels)
        with open("top_layer_classifier.pkl", 'rb') as pickle_file:
            self.classifier = load(pickle_file)
        self.criterion = nn.BCEWithLogitsLoss()
        self.auroc = MultilabelAUROC(n_labels)
        self.multi_label_f1 = MultilabelF1Score(n_labels)

        self.train_loss = []
        self.train_predictions = []
        self.train_labels = []
        self.val_loss = []
        self.val_predictions = []
        self.val_labels = []
        self.test_loss = []
        self.test_predictions = []
        self.test_labels = []

    def forward(self, input_ids, attention_mask):
        # print("I am in forward")
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        pooler = last_hidden_state[:, 0]
        logits = self.classifier(pooler)

        return logits

    def training_step(self, batch, batch_idx):
        # print("I am in training_step")
        input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.sigmoid(logits)  # Apply sigmoid to convert logits to probabilities
        loss = self.criterion(predictions, labels)
        self.train_loss.append(loss)
        # if self.train_loss:
        self.log("training/loss_epoch:", torch.stack(self.train_loss).mean(), on_step=False, on_epoch=True, logger=True)
        self.train_predictions.extend(predictions)
        self.train_labels.extend(labels)

        return loss

    def on_training_epoch_end(self):
        print("I am in on_training_epoch_end")
        if self.train_loss:
            avg_loss = torch.stack(self.train_loss).mean()
            self.log("training/loss_epoch:", avg_loss, prog_bar=True, logger=True)

        self.train_predictions = torch.stack(self.train_predictions)
        self.train_labels = torch.stack(self.train_labels).long()
        multi_label_f1_score = self.multi_label_f1(self.train_predictions, self.train_labels)
        print("training/multi label f1: ", multi_label_f1_score)
        self.log("training/multi label f1: ", multi_label_f1_score, prog_bar=True, logger=True)
        multi_label_auroc_score = self.auroc(self.train_predictions, self.train_labels)
        print("training/multi label auroc: ", multi_label_auroc_score)
        self.log("training/multi label auroc: ", multi_label_auroc_score, prog_bar=True, logger=True)

        self.train_loss = []
        self.train_predictions = []
        self.train_labels = []

    def validation_step(self, batch, batch_idx):
        # print("I am in validation_step")
        input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.sigmoid(logits)  # Apply sigmoid to convert logits to probabilities
        loss = self.criterion(predictions, labels)
        self.val_loss.append(loss)
        avg_loss = torch.stack(self.val_loss).mean()
        self.log("validation/loss_epoch:", avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.val_predictions.extend(predictions)
        self.val_labels.extend(labels)

        return loss

    def on_validation_epoch_end(self):
        print("I am in on_validation_epoch_end")
        if self.val_loss:
            avg_loss = torch.stack(self.val_loss).mean()
            print("validation/loss_epoch:", avg_loss)
            self.log("validation/loss_epoch:", avg_loss, prog_bar=True, logger=True)

        self.val_predictions = torch.stack(self.val_predictions)
        self.val_labels = torch.stack(self.val_labels).long()
        multi_label_f1_score = self.multi_label_f1(self.val_predictions, self.val_labels)
        print("validation/multi label f1: ", multi_label_f1_score)
        self.log("validation/multi label f1: ", multi_label_f1_score, prog_bar=True, logger=True)
        multi_label_auroc_score = self.auroc(self.val_predictions, self.val_labels)
        print("validation/multi label auroc: ", multi_label_auroc_score)
        self.log("validation/multi label auroc: ", multi_label_auroc_score, prog_bar=True, logger=True)

        self.val_loss = []
        self.val_predictions = []
        self.val_labels = []

    def test_step(self, batch, batch_idx):
        # print("I am in test_step")
        input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.sigmoid(logits)  # Apply sigmoid to convert logits to probabilities
        loss = self.criterion(predictions, labels)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.test_predictions.extend(predictions)
        self.test_labels.extend(labels)

        return loss

    def on_test_epoch_end(self):
        print("I am in on_test_epoch_end")
        if self.test_loss:
            avg_loss = torch.stack(self.test_loss).mean()
            print("test/loss_epoch:", avg_loss)
            self.log("test/loss_epoch:", avg_loss, prog_bar=True, logger=True)

        self.test_predictions = torch.stack(self.test_predictions)
        self.test_labels = torch.stack(self.test_labels).long()
        multi_label_f1_score = self.multi_label_f1(self.test_predictions, self.test_labels)
        print("test/multi label f1: ", multi_label_f1_score)
        self.log("test/multi label f1: ", multi_label_f1_score, prog_bar=True, logger=True)
        multi_label_auroc_score = self.auroc(self.test_predictions, self.test_labels)
        print("test/multi label auroc: ", multi_label_auroc_score)
        self.log("test/multi label auroc: ", multi_label_auroc_score, prog_bar=True, logger=True)

        self.test_loss = []
        self.test_predictions = []
        self.test_labels = []

    def predict(self, text):
        # print("I am in predict")
        inputs = self.tokenizer(text, return_tensors='pt')

        rets = []
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs[0]
            pooler = last_hidden_state[:, 0]
            logits = self.classifier(pooler)
            predictions = torch.sigmoid(logits)

            for i, prediction in enumerate(np.array(predictions[0])):
                if prediction >= 0.5:
                    rets.append(LABEL_COLUMNS[i].replace("_", " "))

            if len(rets) == 0:
                ret = "This message was approved"
            elif 1 == len(rets):
                ret = f"This message was {rets[0]}"
            elif 2 == len(rets):
                ret = f"This message was {rets[0]} and {rets[1]}"
            else:
                ret = f"This message was {rets[0]}, {rets[1]}, and {rets[2]}"

        return ret

    def configure_optimizers(self):
        print(f"I am in configure_optimizers - learning_rate: {learning_rate}")  # self.hparams.lr: {self.hparams.lr} -
        optimizer = torch.optim.Adam(params = self.parameters(), lr=learning_rate)

        return optimizer


def load_model_wrapper():
    model_wrapper = MultiLabelDetectionModel()

    return model_wrapper
