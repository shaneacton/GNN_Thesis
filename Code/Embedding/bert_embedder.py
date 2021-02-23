import torch
from transformers import AutoTokenizer, AutoModel

from Code.Config.config import config
from Code.Embedding.string_embedder import StringEmbedder
from Code.Training import device


class BertEmbedder(StringEmbedder):

    """
    sizes available:
    tiny (L=2, H=128)
    mini (L=4, H=256)
    small (L=4, H=512)
    medium (L=8, H=512)
    """

    def __init__(self, fine_tune=False):
        super().__init__()
        self.size = config.bert_size
        self.fine_tune = fine_tune
        model_name = "prajjwal1/bert-" + self.size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.dims = self.model.config.hidden_size
        from Code.Training.Utils.training_utils import num_params
        print("Loaded bert model with", self.dims, "dims and ", num_params(self), ("trainable" if fine_tune else "static"), "params")
        if self.dims != config.embedded_dims:
            raise Exception("config embedded dims wrong. bert embedder=" + str(self.dims) + " conf=" + str(config.embedded_dims))

        for param in self.model.parameters():
            param.requires_grad = fine_tune

    def embed(self, string):
        encoding = self.tokenizer(string, return_tensors="pt")
        input_ids = encoding["input_ids"].to(device)
        if input_ids.size(-1) > 512:
            raise Exception("too many tokens:", input_ids.size(-1))
        attention_mask = encoding["attention_mask"].to(device)
        # print("input ids:", input_ids.size())
        if self.fine_tune:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = out["last_hidden_state"]
        # print("last:", last_hidden_state.size())
        return last_hidden_state


if __name__ == "__main__":
    embedder = BertEmbedder()
    embedder.embed("hello world . I am Groot!")
    # embedder.embed(["hello world . I am Groot!", "yoyoyo"])