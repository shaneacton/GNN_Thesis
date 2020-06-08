from typing import Union, Tuple, Dict


class TypeMap:

    def __init__(self):
        self.type_to_id: Dict[Tuple[type, Union[None, str]], int] = {}
        self.id_to_type: Dict[int, Tuple[type, Union[None, str]]] = {}
        self.next_id = 0

    def register_node_type(self, type):
        """
        stores newly encountered node/edge types and maps to type id
        this is used to encode node/edge types
        """

        if type in self.type_to_id.keys():
            return
        self.type_to_id[type] = self.next_id
        self.id_to_type[self.next_id] = type
        self.next_id += 1

    def get_id_from_type(self, type):
        self.register_node_type(type)
        return self.type_to_id[type]
