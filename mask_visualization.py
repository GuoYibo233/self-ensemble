"""
Attention Mask Visualization Utilities

This module provides functions to visualize 2D attention masks as heatmaps.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def visualize_mask_heatmap(mask_2d, segment_positions=None, title="Attention Mask", save_path=None, show=True, save=False):
    """
    Visualize a 2D attention mask as a heatmap and display it.
    
    Args:
        mask_2d: 2D tensor or numpy array [Q, K] 
                 Values should be 0 (allowed) or -inf (blocked), or boolean
        segment_positions: Optional list of (start, end) tuples for segment boundaries
        title: Title for the plot
        save_path: Optional path to save the figure
        show: Whether to display the figure (default: True)
        save: Whether to save the figure (default: False, only saves if save_path is provided and save=True)
    
    Returns:
        matplotlib figure object
    
    Example:
        >>> mask_2d = struct_mask[0, 0]  # Extract [Q, K] from [1, 1, Q, K]
        >>> segment_positions = [(0, 50), (50, 70), (70, 90)]
        >>> visualize_mask_heatmap(mask_2d, segment_positions, title="Custom Mask")
    """
    # Convert to numpy if needed
    if torch.is_tensor(mask_2d):
        # Convert to float32 first if needed (handles bfloat16, float16, etc.)
        if mask_2d.dtype in [torch.bfloat16, torch.float16]:
            mask_np = mask_2d.float().cpu().detach().numpy()
        else:
            mask_np = mask_2d.cpu().detach().numpy()
    else:
        mask_np = np.array(mask_2d)
    
    # Squeeze if needed (remove singleton dimensions)
    while mask_np.ndim > 2:
        mask_np = mask_np.squeeze()
    
    # Convert mask to binary for visualization:
    # - For additive masks: 0 = allowed, -inf (or very large negative) = blocked
    # - For boolean masks: False = allowed, True = blocked
    # Output: 1 = allowed (green), 0 = blocked (white)
    if mask_np.dtype == bool:
        # Boolean mask: True = blocked, False = allowed
        binary_mask = (~mask_np).astype(float)  # Invert: False->1 (allowed), True->0 (blocked)
    else:
        # Additive mask: 0 = allowed, -inf or very large negative = blocked
        # Use a threshold to catch both -inf and very large negative values (like -3.3895e+38)
        threshold = -1e10  # Values below this are considered "blocked"
        binary_mask = (mask_np > threshold).astype(float)
    
    Q, K = binary_mask.shape
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create custom colormap: white = blocked (0), green = allowed (1)
    colors = ['white', 'lightgreen']
    n_bins = 2
    cmap = mcolors.LinearSegmentedColormap.from_list('mask', colors, N=n_bins)
    
    # Plot heatmap
    im = ax.imshow(binary_mask, cmap=cmap, aspect='auto', interpolation='nearest', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Blocked', 'Allowed'])
    
    # Add segment boundaries if provided
    if segment_positions is not None:
        for i, (start, end) in enumerate(segment_positions):
            # Vertical lines (key/column boundaries)
            ax.axvline(x=start - 0.5, color='red', linewidth=2, linestyle='--', alpha=0.7)
            ax.axvline(x=end - 0.5, color='red', linewidth=2, linestyle='--', alpha=0.7)
            
            # Horizontal lines (query/row boundaries)
            ax.axhline(y=start - 0.5, color='blue', linewidth=2, linestyle='--', alpha=0.7)
            ax.axhline(y=end - 0.5, color='blue', linewidth=2, linestyle='--', alpha=0.7)
            
            # Add labels
            mid = (start + end) // 2
            label = "Shared" if i == 0 else f"Para{i}"
            ax.text(mid, -Q*0.02, label, ha='center', va='top', fontsize=10, fontweight='bold', color='red')
            ax.text(-K*0.02, mid, label, ha='right', va='center', fontsize=10, fontweight='bold', color='blue', rotation=90)
    
    # Set labels and title
    ax.set_xlabel('Key Position (K)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Query Position (Q)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add grid for better readability
    ax.set_xticks(np.arange(0, K, max(1, K//20)))
    ax.set_yticks(np.arange(0, Q, max(1, Q//20)))
    ax.grid(True, which='both', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested and path provided
    if save and save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Mask visualization saved to: {save_path}")
    
    # Show the plot
    if show:
        plt.show(block=False)  # Non-blocking show
        plt.pause(0.1)  # Allow GUI to update
    
    return fig


def visualize_causal_mask(seq_len, title="Causal Mask", save_path=None, show=True, save=False):
    """
    Visualize a standard causal attention mask.
    
    Args:
        seq_len: Sequence length
        title: Title for the plot
        save_path: Optional path to save the figure
        show: Whether to display the figure
        save: Whether to save the figure (default: False)
    
    Returns:
        matplotlib figure object
    
    Example:
        >>> visualize_causal_mask(50, title="Standard Causal Attention")
    """
    # Create causal mask (lower triangular)
    causal_mask = np.tril(np.ones((seq_len, seq_len)))
    
    return visualize_mask_heatmap(
        causal_mask.astype(bool),
        title=title,
        save_path=save_path,
        show=show,
        save=save
    )
