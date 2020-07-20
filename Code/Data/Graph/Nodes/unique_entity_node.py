from Code.Data.Graph.Nodes.node import Node


class UniqueEntityNode(Node):
    def __init__(self, mention_ids, name):
        super().__init__()
        self.mention_ids = mention_ids
        self.name = name

    def get_node_viz_text(self):
        return "UNIQUE: " + self.name