import Code.constants


class NodePosition:

    def __init__(self, source, sequence_level, sequence_id, window_size=-1):
        self.window_size = window_size
        self.source = source
        from Code.Data.Text.Tokenisation.token_span_hierarchy import TokenSpanHierarchy
        self.sequence_level = TokenSpanHierarchy.strip_query(sequence_level)  # an identifier for which sequence this node is in (token,sentence, etc)
        self.sequence_id = sequence_id  # the abs/rel position of this node in the relevant sequence

        if sequence_level == Code.constants.DOCUMENT:
            raise Exception("document nodes cannot have positions")

    def __sub__(self, other):
        """the difference between two node positions is a relative node positions"""
        if self.window_size == -1:
            raise Exception("must set the windowsize for source=" + self.source + " lev=" + self.sequence_level)

        if self.source == other.source and self.sequence_level == other.sequence_level:
            # compatible
            diff = abs(self.sequence_id - other.sequence_id)
            # clamp the rel dist in the window
            diff = min(max(-self.window_size, diff), self.window_size - 1)
            return NodePosition(self.source, self.sequence_level, diff, self.window_size)

        # incompatible
        return incompatible

    def __hash__(self):
        return self.source.__hash__() + 3 * self.sequence_level.__hash__() + 5 * self.sequence_id.__hash__()

    def __eq__(self, other):
        return self.source == other.source and self.sequence_level == other.sequence_level \
               and self.sequence_id == other.sequence_id

    def __repr__(self):
        return "Node Pos: {source=" + repr(self.source) + ", sequence_level=" + repr(self.sequence_level) + ", sequence_id=" \
               + repr(self.sequence_id)


incompatible = NodePosition(None, None, -1, -1)
