from typing import List, Dict

import en_core_web_sm
import neuralcoref
from neuralcoref.neuralcoref import Cluster
from spacy.tokens.doc import Doc

from Code.Data.Text.Tokenisation.document_extract import DocumentExtract
from Code.Data.Text.Tokenisation.entity import Entity
from Code.Data.Text.Tokenisation.token_span import TokenSpan
from Code.Models import tokeniser, subtoken_mapper, basic_tokeniser

nlp = en_core_web_sm.load()
neuralcoref.add_to_pipe(nlp)


class TokenSequence:

    """
        contains an ordered collection of word tokens and subword subtokens
        as well as the mapping between them
    """

    def __init__(self, text):
        self.text = text
        self.tokens = []  # all unsplit tokens
        self.sub_tokens = []  # all split tokens
        self.token_subtoken_map = []  # value at index i is [start,end) subtoken id's for token i
        subtoken_map = subtoken_mapper(text.text)
        for stm in subtoken_map:
            self.add_token_and_subtokens(stm[0], stm[1])

        self._entities = None
        self._corefs = None
        self._sentences = None
        self._passages = None
        self._spacy_processed_doc = None

    @property
    def spacy_processed_doc(self):
        if self._spacy_processed_doc is None:
            self._spacy_processed_doc = nlp(self.text.text)
        return self._spacy_processed_doc

    @property
    def entities(self) -> List[Entity]:  # ordered, non unique
        if self._entities is None:
            self._entities = self.get_spacy_entities(spacy_processed_doc=self.spacy_processed_doc)
        return self._entities

    @property
    def corefs(self) -> Dict[Entity, List[Entity]]:  # ordered, non unique
        if self._corefs is None:
            self._corefs = self.get_spacy_coreferences(spacy_processed_doc=self.spacy_processed_doc)
        return self._corefs

    @property
    def entities_and_corefs(self) -> List[Entity]:
        ents_and_corefs: List[Entity] = []  # inserts corefs right after their mains resulting in an imperfect by nearly sorted array
        [ents_and_corefs.extend([ent] + (self.corefs[ent] if ent in self.corefs else [])) for ent in self.entities]
        #todo sort which exploits nearly sortedness
        ents_and_corefs = sorted(ents_and_corefs, key=lambda ent: 0.5 * (ent.token_span[0] + ent.token_span[1]))
        return ents_and_corefs

    @property
    def sentences(self) -> List[DocumentExtract]:
        if self._sentences is None:
            self._sentences = self.get_spacy_sentences(self.spacy_processed_doc)
        return self._sentences

    @property
    def full_document(self):
        return DocumentExtract(self, (0, len(self.tokens)), DocumentExtract.DOC)

    @property
    def passages(self):
        if self._passages is None:
            self._passages = self.get_passages()
        return self._passages

    def get_text_from_span(self, span):
        return " ".join(self.tokens[span[0]:span[1]])

    def add_token_and_subtokens(self, token, subtokens):
        self.tokens.append(token)
        num_existing_subtokens = len(self.sub_tokens)
        self.token_subtoken_map.append((num_existing_subtokens, num_existing_subtokens + len(subtokens)))
        self.sub_tokens.extend(subtokens)

    def convert_token_span_to_subtoken_span(self, token_span):
        t_start, t_end = token_span
        sub_start = self.token_subtoken_map[t_start][0]
        sub_end = self.token_subtoken_map[t_end-1][1] # todo test ensure -1 is correct
        return (sub_start, sub_end)

    def get_token_span_from_chars(self, start_char_id, end_char_id, subtokens=False):
        sub_string = self.text.text[start_char_id: end_char_id]
        matches = self.find_seq_in_seq(basic_tokeniser(self.text.text[:end_char_id]), basic_tokeniser(sub_string))
        # since the text was sheered at the end_char_id, final match must be accurate
        if not matches:
            raise Exception("Possible substring does not align with chars")
        if subtokens:
            return self.convert_token_span_to_subtoken_span(matches[-1])
        return matches[-1]

    @staticmethod
    def find_seq_in_seq(seq, query):
        seq_index = 0
        num_seq_tokens = len(seq)
        num_query_tokens = len(query)
        matches = []  # list of tuples [start,end) ids of the matching subtokens

        def does_match_from(start_id):
            for i in range(1, num_query_tokens):
                if query[i] != seq[start_id + i]:
                    return False
            return True

        while seq_index < num_seq_tokens:
            try:
                next_match_id = seq.index(query[0], seq_index, num_seq_tokens + 1)
                if does_match_from(next_match_id):
                    matches.append((next_match_id, next_match_id + num_query_tokens))
                seq_index = next_match_id + 1
            except:  # no more matches
                break

        return matches

    def find_string_in_tokens(self, string):
        return self.find_seq_in_seq(self.tokens, basic_tokeniser(string))

    def find_string_in_subtokens(self, string):
        return self.find_seq_in_seq(self.sub_tokens, tokeniser(string))

    def match_heirarchical_span_seqs(self, key_spans: List[TokenSpan], value_spans: List[TokenSpan]) -> Dict[
        TokenSpan, List[TokenSpan]]:
        """
        inputs 2 sorted span sequences such as a list of entities and sentences or sentences and passages
        each list should be in order of appearance in text
        the there should be a one to many mapping of these seq
        each key span may contain many value span

        :return: dict{span_key, List[span_values]}
        """

        mapping = {}
        v = 0
        val = lambda v: value_spans[v]

        for key in key_spans:
            mapping[key] = []

            while  v < len(value_spans) and key.contains(val(v)):
                mapping[key] += [val(v)]
                v += 1

        if v != len(value_spans):
            raise Exception("matching failed to place all value spans - " + str(v) + "/" + str(len(key_spans)))

        return mapping

    def get_spacy_entities(self, spacy_processed_doc=None):
        """
        uses the unprocessed text to perform NER using Spacy.
        """
        processed: Doc = nlp(self.text.text) if not spacy_processed_doc else spacy_processed_doc
        entities = []

        for ent in processed.ents:
            exact_match = self.get_token_span_from_chars(ent.start_char, ent.end_char)
            entity = Entity(self, exact_match, ent)
            entities.append(entity)

        return entities

    def get_spacy_coreferences(self, spacy_processed_doc=None):
        processed: Doc = nlp(self.text.text) if not spacy_processed_doc else spacy_processed_doc
        clusters: List[Cluster] = processed._.coref_clusters

        corefs = {}
        for cluster in clusters:
            token_matches = [self.get_token_span_from_chars(mention.start_char, mention.end_char)
                             for mention in cluster.mentions]
            entities = [Entity(self, token_matches[i], cluster.mentions[i], is_coref=i>0)
                        for i in range(len(token_matches))]
            corefs[entities[0]] = entities[1:]  # ent[0] is main, rest are corefs

        corefs = {ent: corefs[ent] for ent in self.entities if ent in corefs}
        # replaces the main ents produced by the ents produced by NER as these have more metadata
        return corefs

    def get_spacy_sentences(self, spacy_processed_doc=None):
        processed: Doc = nlp(self.text.text) if not spacy_processed_doc else spacy_processed_doc
        spacy_sents = processed.sents
        sentences = []

        for sent in spacy_sents:
            exact_match = self.get_token_span_from_chars(sent.start_char, sent.end_char)
            sentence = DocumentExtract(self, exact_match, DocumentExtract.SENTENCE, sent)
            sentences.append(sentence)

        return sentences

    def get_passages(self):
        from Code.Data.Text.context import Context
        text_passages = self.text.text.split(Context.PASSAGE_BREAK_STRING)
        passages = []
        for text in text_passages:
            matches = self.find_string_in_tokens(text)
            if len(matches) > 1:
                raise Exception("duplicate passage in token seq")

            passage = DocumentExtract(self,matches[0], DocumentExtract.PASSAGE)
            passages.append(passage)

        return passages

    def __eq__(self, other):
        return self.text == other.text

    def __hash__(self):
        return self.text.__hash__()




if __name__ == "__main__":
    text = """Weimar Republic is an unofficial, historical designation for the German state between 1919 and 1933. The name derives from the city of Weimar, where
its constitutional assembly first took place. The official name of the state was still "Deutsches Reich"; it had remained unchanged since 1871. In
English the country was usually known simply as Germany. A national assembly was convened in Weimar, where a new constitution for the "Deutsches
Reich" was written, and adopted on 11 August 1919. In its fourteen years, the Weimar Republic faced numerous problems, including hyperinflation,
political extremism (with paramilitaries  both left- and right-wing); and contentious relationships with the victors of the First World War. The
people of Germany blamed the Weimar Republic rather than their wartime leaders for the country's defeat and for the humiliating terms of the Treaty of
Versailles. However, the Weimar Republic government successfully reformed the currency, unified tax policies, and organized the railway system. Weimar
Germany eliminated most of the requirements of the Treaty of Versailles; it never completely met its disarmament requirements, and eventually paid
only a small portion of the war reparations (by twice restructuring its debt through the Dawes Plan and the Young Plan). Under the Locarno Treaties,
Germany accepted the western borders of the republic, but continued to dispute the Eastern border.""" * 1

    from Code.Data.Text.text import Text
    ts = TokenSequence(Text(text))

    print("entities:",ts.entities)
    # spans = [ent.token_span for ent in ts.entities]
    # sub_spans = [ts.convert_token_span_to_subtoken_span(span) for span in spans]
    # subs = [ts.sub_tokens[ss[0]:ss[1]] for ss in sub_spans]
    # # print("ent sub toks:", subs)
    print("\ncorefs:", ts.corefs)
    mapping = ts.match_heirarchical_span_seqs(ts.sentences, ts.entities_and_corefs)

    print("\nents and corefs:", ts.entities_and_corefs)

    sentences = ts.sentences
    print(type(sentences[0]))
    sentences = [repr(sent) + ":" + repr([ent for ent in mapping[sent]]) for sent in sentences]
    print("\n\nsentences:\n", "\nS:\t".join(sentences))

