# Trainable Mixture of Experts in CUDA

Build a trainable Mixture-of-Experts layer from scratch in CUDA, starting from low-level matmul and activation kernels and culminating in a full forward, backward, and training loop with load-balancing auxiliary loss. The project gives you hands-on experience with sparse routing, token dispatch, and end-to-end MoE training on the GPU.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** matmul_naive_kernel
- [x] **7.** elementwise_add_kernel
- [x] **8.** relu_forward_kernel
- [x] **9.** relu_backward_kernel

---

Built on Deep-ML.
