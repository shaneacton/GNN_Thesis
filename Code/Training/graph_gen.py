import copy
import random
import time
from multiprocessing import Process, Queue

from nlp import tqdm

from Code.HDE.Graph.graph_utils import create_graph

BUF_SIZE = 1500
graph_queue = Queue(BUF_SIZE)
SKIP = "skip"


def produce_graphs(q, start_at, dataset=None, glove_embedder=None, tokeniser=None, support_encodings=None):
    """produce 1 epoch worth of graphs"""
    for i, example in tqdm(enumerate(dataset)):
        # if q is full, spin until one is consumed
        while q.full():
            time.sleep(0.01)
            # print("q full")

        if start_at != -1 and i < start_at:  # fast forward
            q.put(SKIP)
            continue

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
        self.kwargs = {"dataset": dataset, "glove_embedder": glove_embedder, "tokeniser": tokeniser,
                       "support_encodings": support_encodings}
        self.dataset = dataset
        self.p = None

    def start(self, start_at):
        kwargs = copy.deepcopy(self.kwargs)
        kwargs.update({"start_at": start_at})
        self.p = Process(target=produce_graphs, args=(graph_queue,), kwargs=kwargs)
        self.p.start()  # begins generation

    def graphs(self, start_at=0):
        """the consume method"""
        self.start(start_at)  # begins generation

        for i in range(self.num_examples):  # consume full epoch
            # if graph_queue.empty():
                # print("empty graph queue")
            yield graph_queue.get()  # will hang until next graph is ready

        self.p.join()
        self.p.close()

    def shuffle(self, epoch):
        random.seed(epoch)
        random.shuffle(self.dataset)




