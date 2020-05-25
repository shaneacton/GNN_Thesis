from typing import Union, Tuple, Dict

type_to_id: Dict[Tuple[type, Union[None, str]], int] = {}
id_to_type: Dict[int, Tuple[type, Union[None, str]]] = {}
next_id = 0


def register_node_type(type):
    """
    stores newly encountered node/edge types and maps to type id
    this is used to encode node/edge types
    """

    global next_id

    if type in type_to_id.keys():
        return
    type_to_id[type] = next_id
    id_to_type[next_id] = type
    next_id += 1
