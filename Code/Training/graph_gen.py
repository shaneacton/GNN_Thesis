import random
import time
from multiprocessing import Process, Queue

from nlp import tqdm

from Code.HDE.Graph.graph_utils import create_graph

BUF_SIZE = 10
graph_queue = Queue(BUF_SIZE)


def produce_graphs(q, dataset=None, glove_embedder=None, tokeniser=None, support_encodings=None):
    """produce 1 epoch worth of graphs"""
    for i, example in tqdm(enumerate(dataset)):
        # if q is full, spin until one is consumed
        while q.full():
            time.sleep(0.05)

        graph = create_graph(example, glove_embedder=glove_embedder, tokeniser=tokeniser, support_encodings=support_encodings)
        q.put(graph)


class GraphGenerator:
    def __init__(self, dataset, model=None, glove_embedder=None, tokeniser=None, support_encodings=None):
        super(GraphGenerator, self).__init__()
        if model is not None:
            if model.embedder_name == "bert":
                tokeniser = model.embedder.tokenizer
            else:
                glove_embedder = model.embedder

        self.num_examples = len(dataset)
        print("num ex:", self.num_examples)
        self.kwargs = {"dataset": dataset, "glove_embedder": glove_embedder, "tokeniser": tokeniser, "support_encodings": support_encodings}
        self.dataset = dataset
        self.p = None

    def start(self):
        self.p = Process(target=produce_graphs, args=(graph_queue,), kwargs=self.kwargs)
        self.p.start()  # begins generation

    def graphs(self):
        """the consume method"""
        self.start()  # begins generation

        for i in range(self.num_examples):  # consume full epoch
            yield graph_queue.get()  # will hang until graph is ready

        self.p.join()
        self.p.close()

    def shuffle(self, epoch):
        random.seed(epoch)
        random.shuffle(self.dataset)




