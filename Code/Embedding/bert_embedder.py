import torch
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizerBase, LongformerSelfAttention, RobertaForMaskedLM

from Code.Embedding.string_embedder import StringEmbedder
from Code.Training import dev
from Config.config import get_config


class BertEmbedder(StringEmbedder):

    """
        Model: prajjwal1/bert-
            sizes available:
                        tiny (L=2, H=128)
                        mini (L=4, H=256)
                        small (L=4, H=512)
                        medium (L=8, H=512)

        Model: google/bigbird-roberta-base
        Model: roberta-base
        Model: emilyalsentzer/Bio_ClinicalBERT
        Model: simonlevine/bioclinical-roberta-long
    """

    def __init__(self):
        super().__init__()
        self.size = get_config().bert_size
        self.fine_tune = get_config().fine_tune_embedder
        model_name = get_config().bert_name

        if "prajjwal1" in model_name:
            model_name += self.size
            self.model = AutoModel.from_pretrained(model_name)
            self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)

        elif "bigbird" in model_name:
            from transformers import BigBirdModel, BigBirdTokenizerFast
            self.model = BigBirdModel.from_pretrained(model_name, block_size=16, num_random_blocks=2)
            self.tokenizer: BigBirdTokenizerFast = BigBirdTokenizerFast.from_pretrained(model_name)

        elif "roberta" in model_name:
            from transformers import RobertaModel, RobertaTokenizerFast
            self.model = RobertaModel.from_pretrained(model_name)
            self.tokenizer: RobertaTokenizerFast = RobertaTokenizerFast.from_pretrained(model_name)
        elif "simonlevine/bioclinical-roberta-long" in model_name:
            from transformers import RobertaTokenizerFast
            self.model = RobertaLongForMaskedLM.from_pretrained('simonlevine/bioclinical-roberta-long')
            self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        self.dims = self.model.config.hidden_size
        # self.set_trainable_params()
        self.set_all_params_trainable(False)
        from Code.Utils.model_utils import num_params
        print("Loaded bert model with", self.dims, "dims and ", num_params(self),
              ("trainable" if self.fine_tune else "static"), "params")

    def set_all_params_trainable(self, trainable: bool):
        for param in self.model.parameters():
            param.requires_grad = trainable

    def set_trainable_params(self):
        def is_in_fine_tune_list(name):
            if "all" in bert_fine_tune_layers:
                return True
            if name == "":  # full model is off by default
                return False

            for l in bert_fine_tune_layers:
                if l in name:
                    return True
            return False

        self.set_all_params_trainable(False)
        """all params are turned off. then we selectively reactivate grads"""

        if self.fine_tune:
            bert_fine_tune_layers = get_config().bert_fine_tune_layers
            for n, m in self.model.named_modules():
                if not is_in_fine_tune_list(n):
                    continue
                for param in m.parameters():
                    param.requires_grad = True

    def embed(self, string):
        encoding = self.tokenizer(string, return_tensors="pt")
        input_ids = encoding["input_ids"].to(dev())
        is_long = "long" in get_config().bert_name or "bigbird" in get_config().bert_name
        if input_ids.size(-1) > 512 and not is_long:
            # print("string too long for bert:", string)
            raise TooManyTokens("too many tokens:", input_ids.size(-1))
        # attention_mask = encoding["attention_mask"].to(dev())
        if self.fine_tune:
            out = self.model(input_ids=input_ids)#, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                out = self.model(input_ids=input_ids)#, attention_mask=attention_mask)

        last_hidden_state = out["last_hidden_state"]
        return last_hidden_state


class TooManyTokens(Exception):
    pass


class RobertaLongSelfAttention(LongformerSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        return super().forward(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)


class RobertaLongForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.roberta.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = RobertaLongSelfAttention(config, layer_id=i)

if __name__ == "__main__":
    embedder = BertEmbedder()
    embedder.embed("hello world . I am Groot!")
    # embedder.embed(["hello world . I am Groot!", "yoyoyo"])