from Code.Models.GNNs.LayerModules.layer_module import LayerModule


class UpdateModule(LayerModule):

    def __init__(self):
        super().__init__()

    def update(self, aggr_out, x, **kwargs):
        """

        :param aggr_out:
        :param x:
        :param kwargs:
        :return:
        """

