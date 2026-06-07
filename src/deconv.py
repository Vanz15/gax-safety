"""
DeconvNet Attribution Method Implementation.

This module provides the `DeconvNet` class, which is used for model interpretability (Explainable AI).
It works by modifying the backward pass of a PyTorch neural network—specifically attaching hooks to 
ReLU layers to clamp negative gradients to zero. When calculating the attribution for a target class, 
it generates a heatmap that highlights the input pixels most responsible for the model's prediction.
"""
import torch
import torch.nn as nn

class DeconvNet:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            # DeconvNet logic:
            # Standard Backprop: grad_input = grad_output * (input > 0)
            # DeconvNet:         grad_input = grad_output * (grad_output > 0)
            # This effectively means passing the gradient through a ReLU.
            
            # grad_output is a tuple (tensor,), we want to modify the gradient 
            # flowing backwards (which becomes grad_input for the previous layer).
            if grad_output[0] is not None:
                # Clamp negative gradients to 0
                new_grad = torch.clamp(grad_output[0], min=0.0)
                return (new_grad,)
            return None

        # Traverse the model and register the hook on all ReLU layers
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                # We must set inplace=False, otherwise the hook cannot modify gradients safely
                module.inplace = False
                # register_full_backward_hook is the modern PyTorch way
                self.hooks.append(module.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor, target_class):
        """
        Generate the DeconvNet map.
        input_tensor: [1, C, H, W]
        target_class: int
        """
        self.model.zero_grad()
        
        # Ensure input requires grad so we can retrieve it later
        input_tensor.requires_grad = True
        
        output = self.model(input_tensor)
        
        # Create a one-hot vector for the target class
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        
        # Backward pass with the one-hot vector
        output.backward(gradient=one_hot)
        
        # The gradient at the input image is the DeconvNet visualization
        gradient = input_tensor.grad.detach().cpu().numpy() # [1, C, H, W]
        return gradient[0] # [C, H, W]

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()