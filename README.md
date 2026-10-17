Trainable Mixture-of-Experts in CUDA

A CUDA/C++ implementation of a trainable Mixture-of-Experts (MoE) layer, built from low-level GPU kernels for matrix operations, sparse routing, token dispatch, expert computation, and backpropagation.

Architecture

Input Tokens
     |
Router + Top-K Gating
     |
Token Dispatch
     |
Expert MLPs
     |
Weighted Output Combination
     |
MoE Output

Implementation

The project covers:

CUDA Primitives: Matrix multiplication, bias operations, ReLU, GELU, and softmax.

Sparse Routing: Top-K expert selection and gating.

Token Dispatch: Gather, scatter, and expert-output combination.

Expert Networks: MLP forward and backward passes.

Training: Gradient computation, load-balancing loss, and SGD updates.