import torch
import torch.nn.functional as F

# for the absolute fairest comparison ever
# with adaptations for ignoring
def steve_cross_entropy(pred, target_idx, *, num_classes, ignore_index, BT):
    assert torch.all(target_idx < num_classes)
    # prepare to put the ignore_index outside of the tensor
    target_idx = torch.where(target_idx == ignore_index, num_classes, target_idx)
    # do the one-hot (+1 b/c we have the extra fake class for ignored indices)
    target = F.one_hot(target_idx, num_classes=num_classes+1)
    # cut the final "class", where the ignored indices have been mapped to
    target = target[..., :-1]
    # now ignored indices have zero everywhere, so they don't contribute to the loss
    # now, finally, we can do the loss
    return -(target * torch.log_softmax(pred, dim=-1)).sum() / (BT)


def _get_pred_classes(probs):
    if probs.shape[1] == 1:  # binary
        probs = torch.cat((1-probs, probs), dim=1)
    pred_classes = probs.argmax(dim=1, keepdim=True)
    return pred_classes

# more of a metric than a loss, but w/e
@torch.no_grad()
def accuracy(probs, targets):
    pred_classes = _get_pred_classes(probs)
    correct = torch.count_nonzero(pred_classes == targets)
    return correct / probs.shape[0]



# indices are [target, pred]
class ConfusionMatrix:
    def __init__(self, input, target, *, num_classes, input_is_probs):
        if input_is_probs:
            input = _get_pred_classes(input)
        self.pred_classes = input
        self.matrix = torch.zeros((num_classes, num_classes), dtype=torch.int)
        assert target.shape == input.shape
        assert target.shape[1] == 1, f"Wrong shape: {target.shape}"
        for i in range(target.shape[0]):
            tg = target[i].to(int)
            pr = input[i].to(int)
            self.matrix[tg, pr] += 1

    def __getitem__(self, idx):
        return self.matrix[idx]

    def __setitem__(self, idx, data):
        self.matrix[idx] = data

# indices are [target, pred]
class BinaryConfusionMatrix(ConfusionMatrix):
    def __init__(self, input, target, *, input_is_probs):
        super().__init__(input, target, num_classes=2, input_is_probs=input_is_probs)

    # this is not to be called w/ "self."
    def _propertyfactory(idx):
        def _get(self):
            return self[idx]
        def _set(self, data):
            self[idx] = data
        return property(_get, _set)

    true_positives = _propertyfactory((1,1))
    true_negatives = _propertyfactory((0,0))
    false_positives = _propertyfactory((0,1))
    false_negatives = _propertyfactory((1,0))

    @property
    def accuracy(self):
        return (self.true_positives + self.true_negatives) / (self.true_positives+self.true_negatives+self.false_positives+self.false_negatives)

    @property
    def precision(self):
        return (self.true_positives) / (self.true_positives + self.false_positives)

    @property
    def recall(self):
        return (self.true_positives) / (self.true_positives + self.false_negatives)

    @property
    def f1_score(self):
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)
    
    # not really a metric, just a sanity check
    @property
    def yes_rate(self):
        return (self.true_positives + self.false_positives) / (self.true_positives+self.true_negatives+self.false_positives+self.false_negatives)
