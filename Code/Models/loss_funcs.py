from torch.nn import CrossEntropyLoss

from Code.Training import device


def get_span_loss(start_positions, end_positions, start_logits, end_logits):
    # If we are on multi-GPU, split add a dimension
    if len(start_positions.size()) > 1:
        start_positions = start_positions.squeeze(-1)
    if len(end_positions.size()) > 1:
        end_positions = end_positions.squeeze(-1)
    # sometimes the start/end positions are outside our model inputs, we ignore these terms
    ignored_index = start_logits.size(1)
    start_positions.clamp_(0, ignored_index)
    end_positions.clamp_(0, ignored_index)

    loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
    start_loss = loss_fct(start_logits, start_positions.to(device))
    end_loss = loss_fct(end_logits, end_positions.to(device))
    total_loss = (start_loss + end_loss) / 2
    return total_loss


def get_span_element_loss(positions, logits):
    if len(positions.size()) > 1:
        positions = positions.squeeze(-1)

    ignored_index = logits.size(1)
    positions.clamp_(0, ignored_index)
    loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
    loss = loss_fct(logits, positions.to(device))
    return loss
