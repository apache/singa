from collections import Counter, deque
from .tensor import Tensor, Dummy


def infer_dependency(op):
    '''
    Infer the dependency of all operations with the
    given op as the last operation.

    Operation A is depending on B is A uses the output(s) of B.

    Args:
        op: an Operation instance, e.g. the loss operation.

    Return:
        a Counter instance with the operation as the key,
        and the number of operations that are depending on it as the value
    '''
    # dependency = {}
    dependency_count = Counter()
    queue = deque([op])
    while len(queue) > 0:
        cur_op = queue.pop()
        for src_op, _, _, _ in cur_op.src:
            if src_op not in dependency_count and \
                    (not isinstance(src_op, Dummy)):
                # dependency[src_op] = [Counter() for _ in src_op.y_id2idx]
                dependency_count[src_op] = 0
                queue.append(src_op)
            # y_idx = src_op.y_id2idx[x_id]
            # dependency[src_op][y_idx][cur_op] += 1
            dependency_count[src_op] += 1
    return dependency_count


def backward(y, dy=None):
    '''
    Run the backward propagation starting at y.

    Args:
        y: a Tensor instance, usually the loss
        dy: a number or a Tensor instance, for the gradient of the
            objective/loss w.r.t y, usually 1.0

    Return:
        a dictionary storing the gradient tensors of all tensors
        whose stores_grad is true (e.g. parameter tensors)
    '''
    dependency = infer_dependency(y.creator)
    assert y.size() == 1, 'y must be a Tensor with a single value;'\
        'size of y is % d' % y.size()

    # by default the dy is a tensor with 1.0 for each sample;
    if dy is None:
        dy = float(1.0)
    elif isinstance(dy, Tensor):
        dy = dy.data
    else:
        dy = float(dy)

    # ready is a queue of (operation, dy list)
    ready = deque([(y.creator, (dy,))])
    not_ready = {}  # mapping: op->[dy]
    gradients = {}  # mapping: x->dx if x.stores_grad
    if y.stores_grad:
        gradients[y] = dy

    while len(ready) > 0:
        op, dys = ready.pop()
        if not op.requires_grad or isinstance(op, Dummy):
            continue
        # if not isinstance(op, tensor.Dummy):
        dxs = op._do_backward(*dys)
        # TODO src and dx must match
        assert len(op.src) == len(dxs), \
            'the number of src ops (=%d) and dx (=%d) not match' \
            % (len(op.src), len(dxs))
        for (src_op, x_id, y, y_stores_grad), dx in zip(op.src, dxs):
            # prefix x is w.r.t op; prefix y is w.r.t src_op.
            # x_id is the python id of one input arg of src_op, denoted as x.
            # y_idx (below) is the index of x among the outputs of src_op.
            # not_ready[src_op][y_idx] records the intermediate gradient
            # of the y_idx'th output of src_op. 'intermediate gradient'
            # indicates that if this output is used in multiple children
            # operations, then we have to add the graident (dx) from all these
            # children operations. When src_op is ready, it means that
            # the gradient of all its outputs are available, i.e. all children
            # operations have been backwarded.
            # y is None if y.stores_grad is false; otherwise it is a Tensor
            y_idx = src_op.y_id2idx[x_id]
            if src_op not in not_ready:
                # src_op may have mulitple outputs
                not_ready[src_op] = [None for _ in src_op.y_id2idx]
                not_ready[src_op][y_idx] = dx
            else:
                dxs = not_ready[src_op]
                if dxs[y_idx] is None:
                    dxs[y_idx] = dx
                else:
                    # add the gradient from another children operation that
                    # uses y_idx'th output of src_op as input arg
                    dxs[y_idx] += dx
            if y_stores_grad:
                # store the gradient for final return, e.g. if x is parameter
                g = not_ready[src_op][y_idx]
                gradients[y] = Tensor(device=g.device, data=g)
            dependency[src_op] -= 1
            if src_op.requires_grad is True:
                if dependency[src_op] == 0:
                    if not isinstance(src_op, Dummy):
                        ready.append((src_op, not_ready[src_op]))
                    del not_ready[src_op]

    return gradients
