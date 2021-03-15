NUM_LAYERS = "num_layers"
NUM_FEATURES = "num_features"
SAME_WEIGHT_REPEATS = "same_weight_repeats"
DISTINCT_WEIGHT_REPEATS = "distinct_weight_repeats"
LAYER_ARGS = "layer_args"
MODULE_TYPE = "module_type"
MODULES = "modules"
PREPARATION_MODULES = "preparation_" + MODULES
MESSAGE_MODULES = "message_" + MODULES
UPDATE_MODULES = "update_" + MODULES
NUM_BASES = "num_bases"
HEADS = "heads"
ACTIVATION_TYPE = "activation_type"
ACTIVATION_ARGS = "activation_args"
NEGATIVE_SLOPE = "negative_slope"  # for Leaky_Relu
NUM_LINEAR_LAYERS = "num_linear_layers"
PROPAGATION_TYPE = "prop_type"
POOL_TYPE = "pool_type"
POOL_ARGS = "pool_args"
POOL_RATIO = "ratio"
ENTITY = "entity"
UNIQUE_ENTITY = "unique_entity"
COREF = "coref"
TOKEN = "token"
WORD = "word"
NOUN = "noun"
SENTENCE = "sentence"
PARAGRAPH = "paragraph"
DOCUMENT = "document"
CODOCUMENT = "codocument"
LEVELS = [TOKEN, WORD, SENTENCE, PARAGRAPH, DOCUMENT]
CONNECTION_TYPE = "connection_type"
WINDOW = "window"  # fully connects nodes which are within a max distance of each other
SEQUENTIAL = "sequential"  # connects nodes to the next node at its structure level, within optional max distance
WINDOW_SIZE = "window_size"  # an optional arg for both window and seq
CONTEXT = "context"
QUERY = "query"
CANDIDATE = "candidate"
SUMMARISER_NAME = "summariser_name"
HEAD_AND_TAIL_CAT = "head_and_tail_cat"
SELF_ATTENTIVE_POOLING = "self_attentive_pooling"
NODE_TYPES = "node_types"
EDGE_TYPES = "edge_types"
COMENTION = "comention"
COREFERENCE = "coreference"
UNIQUE_REFERENCE = "unique_ref"