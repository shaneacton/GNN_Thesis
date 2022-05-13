import copy
import inspect

from torch import nn


class WrapGNN(nn.Module):

    def __init__(self, gnn_layer):
        super().__init__()
        self.gnn_layer = gnn_layer
        self.backup_gnn_layer = copy.deepcopy(gnn_layer)
        for param in self.backup_gnn_layer.parameters():
            param.requires_grad = False

        # store originals
        self.base_forward = gnn_layer.forward
        self.base_message = gnn_layer.message
        self.forward_needed_args = inspect.getfullargspec(self.base_forward)[0]
        self.message_needed_args = inspect.getfullargspec(self.base_message)[0]

        # replace original with wrapper, so base layer will call out message func
        gnn_layer.message = self.message

        self.last_custom_kwargs = None
        # temporarily stores forward kwargs which cannot be passed through the gnn layers forward
        # these kwargs can then be accessed in the custom forward method

    def message(self, *args, **kwargs):
        """
            called by the message passing forward method.
            calls our wrap message, then calls the original base message
        """
        kwargs = self.wrap_message(*args, **kwargs)
        if self.base_message == self.message:
            """
                python saves bound messages as an object reference + method name. 
                Thus when saving and loading a wrap gnn, the original gnn's message method will be lost.
                We recover from this by keeping a clone of the layer, copying our trained weights into the clone, 
                and switching to the clone, which still retains its original message function
            """
            print("lost reference to base gnn message func. now has: " + repr(self.base_message) +
                            " gnn has: " + repr(self.gnn_layer.message), " backup has: " + repr(self.backup_gnn_layer.message))
            self.backup_gnn_layer.load_state_dict(self.gnn_layer.state_dict())  # load trained weights into backup
            self.gnn_layer = self.backup_gnn_layer  # retorre from backup
            self.backup_gnn_layer = copy.deepcopy(self.gnn_layer)  # create new backup
            # set trainable params
            for param in self.backup_gnn_layer.parameters():
                param.requires_grad = False
            for param in self.gnn_layer.parameters():
                param.requires_grad = True
            self.base_message= self.gnn_layer.message  # perform the wrap message switch again
            self.gnn_layer.message = self.message
            # recovery complete!

        return self.base_message(*args, **kwargs)

    def wrap_message(self, *args, **kwargs):
        """called before the base layers forward. used for pretransforms"""
        pass

    def forward(self, x, edge_index, *args, **kwargs):
        """caller triggers Wrap's forward, which then triggers the wrap_forward, and finally the base layers forward"""
        base_gnn_kwargs = {k: v for k, v in kwargs.items() if k in self.forward_needed_args}
        custom_kwargs = {k: v for k, v in kwargs.items() if k not in self.forward_needed_args}
        self.last_custom_kwargs = custom_kwargs
        self.wrap_forward(x, edge_index, *args, **kwargs)
        if "previous_attention_scores" in kwargs:
            base_gnn_kwargs.update({"previous_attention_scores": kwargs["previous_attention_scores"]})

        output = self.base_forward(x, edge_index, *args, **base_gnn_kwargs)
        self.last_custom_kwargs = None
        return output

    def wrap_forward(self, x, edge_index, *args, **kwargs):
        """called before the base layers forward. used for pretransforms"""
        pass


