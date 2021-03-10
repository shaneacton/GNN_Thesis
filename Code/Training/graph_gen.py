import copy
import os
import random
import time
import torch.multiprocessing as mp

from Code.HDE.Graph.graph_utils import create_graph

BUF_SIZE = 1500
graph_queue = None
SKIP = "skip"

def new_queue(ctx):
    global graph_queue
    graph_queue = ctx.Queue(BUF_SIZE)


def produce_graphs(q, start_at, dataset=None, glove_embedder=None, tokeniser=None, support_encodings=None):
    """produce 1 epoch worth of graphs"""
    print("producing graphs in process", os.getpid())
    for i, example in enumerate(dataset):
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
        ctx = mp.get_context('fork')
        new_queue(ctx)
        self.p = ctx.Process(target=produce_graphs, args=(graph_queue,), kwargs=kwargs)
        print("starting gen process in process", os.getpid())

        self.p.start()  # begins generation

    def graphs(self, start_at=0):
        """the consume method"""
        print("getting graphs in process", os.getpid())
        self.start(start_at)  # begins generation
        print("waiting for graphs in process", os.getpid())

        for i in range(self.num_examples):  # consume full epoch
            # if graph_queue.empty():
                # print("empty graph queue")
            yield graph_queue.get()  # will hang until next graph is ready

        self.p.join()
        self.p.close()

    def shuffle(self, epoch):
        random.seed(epoch)
        random.shuffle(self.dataset)




