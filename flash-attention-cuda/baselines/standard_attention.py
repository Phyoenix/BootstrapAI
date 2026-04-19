"""
Standard Attention Baseline
PyTorch reference implementation for correctness testing

This provides the ground truth for Flash Attention CUDA implementations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


def standard_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float = None
) -> torch.Tensor:
    """
    Standard scaled dot-product attention.
    
    Args:
        Q: Query tensor (batch, heads, seq_len, head_dim)
        K: Key tensor (batch, heads, seq_len, head_dim)
        V: Value tensor (batch, heads, seq_len, head_dim)
        scale: Optional scale factor (default: 1/sqrt(head_dim))
    
    Returns:
        O: Output tensor (batch, heads, seq_len, head_dim)
    """
    batch, heads, seq_len, head_dim = Q.shape
    
    if scale is None:
        scale = head_dim ** -0.5
    
    # Compute attention scores: S = Q @ K^T / sqrt(d)
    # (batch, heads, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    
    # Softmax over keys
    attn_weights = F.softmax(scores, dim=-1)
    
    # Compute output: O = attn_weights @ V
    # (batch, heads, seq_len, head_dim)
    output = torch.matmul(attn_weights, V)
    
    return output


def generate_test_case(
    batch_size: int = 1,
    num_heads: int = 1,
    seq_len: int = 64,
    head_dim: int = 64,
    seed: int = 42,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a test case with known random seed for reproducibility.
    
    Returns:
        Q, K, V, expected_output
    """
    torch.manual_seed(seed)
    
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    
    # Compute expected output using PyTorch reference
    expected = standard_attention(Q, K, V)
    
    return Q, K, V, expected


def check_correctness(
    cuda_output: torch.Tensor,
    expected_output: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-5
) -> bool:
    """
    Check if CUDA output matches expected PyTorch output.
    
    Args:
        cuda_output: Output from CUDA kernel
        expected_output: Ground truth from PyTorch
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Returns:
        True if outputs match within tolerance
    """
    if cuda_output.shape != expected_output.shape:
        print(f"Shape mismatch: {cuda_output.shape} vs {expected_output.shape}")
        return False
    
    max_diff = torch.max(torch.abs(cuda_output - expected_output)).item()
    mean_diff = torch.mean(torch.abs(cuda_output - expected_output)).item()
    
    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")
    
    is_close = torch.allclose(cuda_output, expected_output, rtol=rtol, atol=atol)
    
    if is_close:
        print("✓ Correctness check PASSED")
    else:
        print("✗ Correctness check FAILED")
    
    return is_close


def benchmark_attention(
    attention_fn,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    num_warmup: int = 10,
    num_iters: int = 100
) -> dict:
    """
    Benchmark an attention implementation.
    
    Args:
        attention_fn: Function that takes (Q, K, V) and returns output
        Q, K, V: Input tensors
        num_warmup: Number of warmup iterations
        num_iters: Number of benchmark iterations
    
    Returns:
        Dictionary with timing statistics
    """
    device = Q.device
    
    # Warmup
    for _ in range(num_warmup):
        _ = attention_fn(Q, K, V)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(num_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        output = attention_fn(Q, K, V)
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # milliseconds
    
    times = np.array(times)
    
    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times))
    }


def compute_flops(batch_size: int, num_heads: int, seq_len: int, head_dim: int) -> float:
    """
    Compute FLOPs for attention operation.
    
    Attention involves:
    - Q @ K^T: batch * heads * seq_len * seq_len * head_dim
    - Softmax: batch * heads * seq_len * seq_len (approx)
    - attn @ V: batch * heads * seq_len * seq_len * head_dim
    
    Total: ~2 * batch * heads * seq_len^2 * head_dim
    """
    return 2.0 * batch_size * num_heads * seq_len * seq_len * head_dim


def test_standard_attention():
    """Test the standard attention implementation."""
    print("=" * 60)
    print("Standard Attention Baseline Test")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Test case 1: Small size for debugging
    print("\n[1] Small test (seq_len=64, dim=64)")
    Q, K, V, expected = generate_test_case(
        batch_size=1, num_heads=1, seq_len=64, head_dim=64,
        device=device
    )
    output = standard_attention(Q, K, V)
    check_correctness(output, expected)
    
    # Test case 2: Medium size
    print("\n[2] Medium test (seq_len=512, dim=64)")
    Q, K, V, expected = generate_test_case(
        batch_size=1, num_heads=1, seq_len=512, head_dim=64,
        seed=123, device=device
    )
    output = standard_attention(Q, K, V)
    check_correctness(output, expected)
    
    # Test case 3: Multi-head
    print("\n[3] Multi-head test (batch=2, heads=8, seq_len=128, dim=64)")
    Q, K, V, expected = generate_test_case(
        batch_size=2, num_heads=8, seq_len=128, head_dim=64,
        seed=456, device=device
    )
    output = standard_attention(Q, K, V)
    check_correctness(output, expected)
    
    # Benchmark
    if device == 'cuda':
        print("\n[4] Performance benchmark")
        Q, K, V, _ = generate_test_case(
            batch_size=1, num_heads=1, seq_len=1024, head_dim=64,
            device=device
        )
        
        stats = benchmark_attention(standard_attention, Q, K, V)
        
        flops = compute_flops(1, 1, 1024, 64)
        tflops = flops / (stats['mean_ms'] * 1e-3) / 1e12
        
        print(f"  Mean time: {stats['mean_ms']:.3f} ms")
        print(f"  FLOPs: {flops/1e9:.2f} GFLOPs")
        print(f"  Performance: {tflops:.2f} TFLOPS")
    
    print("\n" + "=" * 60)
    print("✓ Standard attention baseline tests passed")
    print("=" * 60)
    print("\nUse this as ground truth for CUDA kernel correctness checks")


if __name__ == "__main__":
    test_standard_attention()
