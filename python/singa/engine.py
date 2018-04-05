from collections import Counter
from singa import singa_wrap as singa
from singa import tensor


class GradientFlowController(object):
    '''
    Control backward gradients flow by running the method, run_backward()

    '''
    def __init__(self):
        pass

    def dependency_check(self, function):
        '''
        Compute how many times each 'previous_function'(Operation object) influent its next_functions

        though which outputs

        Arg:
            function: a Operation object which is the termination

        Return:
            dependencies: a dictionary recording dependencies among functions(Operations)
            seen: a set recording all functions(Operations) observed

        '''
        dependencies = {}
        seen = {function}
        queue = [function]
        while len(queue) > 0:
            f = queue.pop()
            for previous_function, Arg_ID in f.previous_functions:
                if previous_function not in dependencies:
                    dependencies[previous_function] = [Counter() for _ in previous_function.output_ids]
                output_idx = previous_function.output_ids[Arg_ID]
                dependencies[previous_function][output_idx][f] += 1
                if previous_function not in seen:
                    queue.append(previous_function)
                    seen.add(previous_function)
        return dependencies, seen

    def dependency_release(self, dependencies, previous_function, function, Arg_ID):
        '''
        To release dependency: if previous_function receive one gradient though its

        output(can be found by Arg_ID) from function, the corresponding dependency counter

        minus one.

        '''
        deps = dependencies[previous_function]
        output_idx = previous_function.output_ids[Arg_ID]
        output_deps = deps[output_idx]
        output_deps[function] -= 1
        if output_deps[function] == 0:
            del output_deps[function]
        return output_idx

    def is_ready_for_backward(self, dependencies, function):
        '''
        Check if a function(Operation) is ready for backward.

        Return: Trur or Flase

        '''
        for deps in dependencies[function]:
            if len(deps) > 0:
                return False
        return True

    def run_backward(self, Tensor, grad):
        '''
        Run the autograd process.

        Args:
            Tensor: the object tensor to optimize, usually the loss
            grad: received gradients

        Return:
            gradients: a dictionary recording the gradients

        '''
        ready = [(Tensor.creator, (grad,))]
        not_ready = {}

        dependencies, seen = self.dependency_check(Tensor.creator)

        while len(ready) > 0:
            function, grad = ready.pop()
            gradient_inputs = function._do_backward(*grad)
            for (previous_function, Arg_ID), gradient_input in zip(function.previous_functions, gradient_inputs):
                if not previous_function.requires_grad:
                    continue
                
                output_index = self.dependency_release(dependencies, previous_function, function, Arg_ID)
                is_ready = self.is_ready_for_backward(dependencies, previous_function)
                
                if is_ready:
                    if previous_function in not_ready:
                        previous_functions_gradients = not_ready[previous_function]
                        if not previous_functions_gradients[output_index]:
                            previous_functions_gradients[output_index] = gradient_input
                        else:
                            previous_functions_gradients[output_index] = \
                                singa.__add__(previous_functions_gradients[output_index], gradient_input)
                        del not_ready[previous_function]
                    else:
                        assert output_index == 0
                        previous_functions_gradients = (gradient_input,)
                    ready.append((previous_function, previous_functions_gradients))
                else:
                    if previous_function in not_ready:
                        previous_functions_gradients = not_ready[previous_function]
                    else:
                        previous_functions_gradients = [None for _ in previous_function.output_ids]

                    if not previous_functions_gradients[output_index]:
                        previous_functions_gradients[output_index] = gradient_input
                    else:
                        previous_functions_gradients[output_index] = \
                            singa.__add__(previous_functions_gradients[output_index], gradient_input)

                    not_ready[previous_function] = previous_functions_gradients

        gradients = {}
        for f in seen:
            if isinstance(f, tensor.Initializer):
                if f.Tensor.grad_outlet is True:
                    gradients[f.Tensor] = f.grads
                    f.grads = f.init.Clone()
        return gradients


def gradients(Tensor, out_gradient):
    '''
    Compute gradients of Tensor.

    '''
    Controller = GradientFlowController()
    return Controller.run_backward(Tensor, out_gradient)
