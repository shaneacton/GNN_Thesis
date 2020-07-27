from torch.nn import LSTM

from Code.Data.Text.Answers.extracted_answer import ExtractedAnswer
from Code.Data.Text.Answers.candidate_answer import CandidateAnswer
from Code.Data import embedded_size
from Code.Models.Layers.attention_flow import AttentionFlow
from Code.Models.Layers.seq2cand import Seq2Cand
from Code.Models.Layers.seq2span_flow import Seq2SpanFlow
from Code.Models.qa_model import QAModel

"""
    source: https://github.com/galsang/BiDAF-pytorch
    author: Taeuk Kim, Ph.D. student, Seoul National University
    
    code has been modified to use projects main text encoder instead of the char + word
    embedding and contextualisation used by the original implementation
    thus parts 1-3 of the BiDaf model have been removed
"""


class BiDAF(QAModel):

    def __init__(self, hidden_size, dropout=0.2):
        super(BiDAF, self).__init__()
        # 4. Attention Flow Layer
        self.att_flow_layer = AttentionFlow(embedded_size, embedded_size)

        # 5. Modeling Layer
        self.modeling_LSTM1 = LSTM(input_size=embedded_size * 4,
                                   hidden_size=hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=dropout)

        self.modeling_LSTM2 = LSTM(input_size=hidden_size * 2,
                                   hidden_size=hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=dropout)

        # 6. Output Layer
        self.span_output = Seq2SpanFlow(embedded_size * 4, hidden_size, dropout)
        self.candidate_output = Seq2Cand(hidden_size)

    def forward(self, *args):
        """
            context and query are vecs of (batch, seq, embedding_dim)
        """
        context, query, candidates = QAModel.get_context_query_candidates_vecs(*args)
        # 1. Attention Flow Layer
        attended_vec = self.att_flow_layer(context, query)
        # attended_vec ~ (batch, context_seq_len, embedding_dim * 4)
        # 2. Modeling Layer
        modeled_vec = self.modeling_LSTM2(self.modeling_LSTM1(attended_vec)[0])[0]
        # modeled_vec ~  (batch, context_seq_len, hidden_size * 2)
        # 3. Output Layer
        return self.get_output(attended_vec, modeled_vec, candidates, *args)

    def get_output(self, attended_vec, modeled_vec, candidates, *args):
        answer_type = QAModel.get_answer_type(*args)
        if answer_type == ExtractedAnswer:
            # (batch, c_len), (batch, c_len)
            return self.span_output(attended_vec, modeled_vec)

        if answer_type == CandidateAnswer:
            return self.candidate_output(modeled_vec, candidates)

        raise Exception("unrecognised answer type: " + repr(answer_type))

