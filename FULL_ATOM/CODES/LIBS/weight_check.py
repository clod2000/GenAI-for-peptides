import torch
import torch.nn as nn
import numpy as np

# Dictionary to store activation statistics from hooks
activation_stats = {}

def get_activation_stats_hook(name):
    """Factory function to create a hook that logs activation stats."""
    def hook(model, input, output):
        # Detach the output tensor to prevent holding onto the computation graph
        output_data = output.detach()
        stats = {
            'shape': output_data.shape,
            'min': torch.min(output_data).item(),
            'max': torch.max(output_data).item(),
            'mean': torch.mean(output_data).item(),
            'std': torch.std(output_data).item(),
            'has_nan': torch.isnan(output_data).any().item(),
            'has_inf': torch.isinf(output_data).any().item()
        }
        activation_stats[name] = stats
    return hook

def inspect_model_state(model, data_batch, loss=None, level='full'):
    """
    Provides a detailed health report of a PyTorch model.

    Args:
        model (nn.Module): The model to inspect.
        data_batch (torch_geometric.data.Data): A single batch of data for the forward pass.
        loss (torch.Tensor, optional): The computed loss tensor *after* .backward() has been
                                       called. Required for inspecting gradients. Defaults to None.
        level (str, optional): The level of inspection.
                               'basic': Inspects weights and gradients.
                               'full': Inspects weights, gradients, and activations.
                               Defaults to 'full'.
    """
    print("\n" + "="*50)
    print(f"MODEL HEALTH REPORT (Level: {level})")
    print("="*50 + "\n")

    # --- 1. Inspect Activations (requires a forward pass) ---
    if level == 'full':
        print("\n--- 1. Activation Statistics (from forward pass) ---")
        global activation_stats
        activation_stats = {}
        hooks = []

        # Register a forward hook on all leaf modules
        for name, module in model.named_modules():
            if not list(module.children()): # Check if it's a leaf module
                hooks.append(module.register_forward_hook(get_activation_stats_hook(name)))

        # Perform a forward pass to trigger the hooks
        try:
            with torch.no_grad():
                _ = model(data_batch)
            print(f"{'Layer Name':<40} | {'Shape':<20} | {'Min':<10} | {'Max':<10} | {'Mean':<10} | {'Std':<10} | {'Has NaN?':<10} | {'Has Inf?':<10}")
            print("-"*140)

            for name, stats in activation_stats.items():
                is_problem = stats['has_nan'] or stats['has_inf']
                # Highlight problematic layers
                prefix = "🔴 " if is_problem else "🟢 "
                print(f"{prefix}{name:<38} | {str(stats['shape']):<20} | {stats['min']:.2e} | {stats['max']:.2e} | {stats['mean']:.2e} | {stats['std']:.2e} | {str(stats['has_nan']):<10} | {str(stats['has_inf']):<10}")

        except Exception as e:
            print(f"🔴 ERROR during forward pass for activation inspection: {e}")
        finally:
            # IMPORTANT: Remove hooks to avoid memory leaks and slowdowns
            for h in hooks:
                h.remove()
        print("\n")


    # --- 2. Inspect Weights and Gradients ---
    print("\n--- 2. Parameter Statistics (Weights & Gradients) ---")
    print(f"{'Parameter Name':<60} | {'Weight Min':<12} | {'Weight Max':<12} | {'Weight Std':<12} | {'Grad Min':<12} | {'Grad Max':<12} | {'Grad Std':<12}")
    print("-"*150)

    for name, param in model.named_parameters():
        if param.requires_grad:
            p_data = param.data.detach()
            
            # Weight stats
            w_min = p_data.min().item()
            w_max = p_data.max().item()
            w_std = p_data.std().item()

            # Gradient stats
            g_min, g_max, g_std = "N/A", "N/A", "N/A"
            is_grad_problem = False
            if param.grad is not None:
                g_data = param.grad.detach()
                if torch.isnan(g_data).any() or torch.isinf(g_data).any():
                    is_grad_problem = True
                    g_min, g_max, g_std = "NaN/Inf!", "NaN/Inf!", "NaN/Inf!"
                else:
                    g_min = g_data.min().item()
                    g_max = g_data.max().item()
                    g_std = g_data.std().item()
            elif loss is not None:
                g_min, g_max, g_std = "None", "None", "None" # Grad is None even after backward pass

            # Highlight problematic parameters
            prefix = "🔴 " if is_grad_problem else "🟢 "
            try:
                print(f"{prefix}{name:<58} | {w_min:<12.2e} | {w_max:<12.2e} | {w_std:<12.2e} | {g_min:<12.2e} | {g_max:<12.2e} | {g_std:<12.2e}")
            except Exception as e:
                print(f"🔴 ERROR during parameter inspection: {e}")

    print("\n" + "="*50)
    print("END OF REPORT")
    print("="*50 + "\n")