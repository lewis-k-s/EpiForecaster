# Plan: Compile Backward Step with torch.compile

## Current State

✅ Completed optimizations:
1. Removed gradient accumulation complexity (accum_steps=1)
2. Updated gradient clipping to use `foreach=True` for multi-tensor parallelism
3. Simplified step counting (removed effective_step calculation)

## Goal
Extend `torch.compile` to capture both forward AND backward passes for additional 10-30% speedup.

## Challenge

`torch.compile` by default only compiles the forward pass. The backward pass is executed via PyTorch's autograd engine in eager mode. To compile the backward pass, we need to either:

### Option A: Compile Full Training Step (Recommended)

Create a compiled function that includes forward + backward + optimizer step:

```python
class EpiForecasterTrainer:
    def __init__(self, ...):
        # ... existing init ...
        
        # Compile the full training step
        if self.config.training.compile:
            self.compiled_training_step = torch.compile(
                self._training_step_impl,
                mode="reduce-overhead",
                fullgraph=False  # Allows data-dependent control flow
            )
    
    def _training_step_impl(self, batch_data, region_embeddings):
        """Pure function containing forward + backward + optimizer"""
        # Forward
        model_outputs, targets_dict = self.model.forward_batch(
            batch_data=batch_data,
            region_embeddings=region_embeddings,
        )
        loss = self.criterion(model_outputs, targets_dict, batch_data)
        
        # Backward
        loss.backward()
        
        # Clip and step
        grad_norm, _ = self._compute_gradient_norms_and_clip(step=self.global_step)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        
        return loss.detach(), grad_norm
    
    def _train_epoch(self):
        # ... in training loop ...
        if self.compiled_training_step:
            loss, grad_norm = self.compiled_training_step(batch_data, self.region_embeddings)
        else:
            # Fallback to eager mode
            loss, grad_norm = self._training_step_impl(batch_data, self.region_embeddings)
```

### Option B: torch.func.grad (Experimental)

Use functional transforms to compile gradients:

```python
from torch.func import grad

# Define forward function
def forward_fn(params, batch_data, region_embeddings):
    # Set model params
    # Forward pass
    # Return loss

# Get compiled gradient function
compiled_grad = torch.compile(grad(forward_fn))
```

**Pros:** More control over backward
**Cons:** Complex integration with existing model structure

## Implementation Plan

### Phase 1: Extract Training Step (1-2 hours)

1. Create `_training_step_impl()` method
   - Move forward pass logic
   - Move backward pass logic
   - Move optimizer step logic
   - Return: `(loss, grad_norm)`

2. Update `_train_epoch()` to call the new method
   - Handle NaN checks outside compiled function
   - Handle curriculum/warmup logic outside
   - Keep logging outside

### Phase 2: Add Compilation Wrapper (1 hour)

1. Add compiled version in `__init__`:
   ```python
   if config.training.compile:
       self._compiled_step = torch.compile(
           self._training_step_impl,
           mode=config.training.compile_mode
       )
   ```

2. Add config option:
   ```yaml
   training:
     compile_backward: false  # New option, default false for safety
   ```

### Phase 3: Testing (2-3 hours)

1. Verify correctness:
   - Loss values match eager mode
   - Gradients match eager mode
   - Checkpoints save/load correctly

2. Performance validation:
   - Measure step time vs eager
   - Verify no graph breaks in hot path
   - Check for memory leaks

### Phase 4: Edge Cases (1 hour)

Handle:
- Curriculum transitions (recompilation)
- Dynamic shapes (if any)
- Mixed precision compatibility
- Gradient debugging mode

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Graph breaks in backward | High | Use `fullgraph=False`, monitor with `TORCH_COMPILE_DEBUG=1` |
| Increased memory usage | Medium | Test with profiling, fallback to eager if OOM |
| Recompilation on curriculum transitions | Medium | Cache compiled function, handle transitions gracefully |
| Debugging difficulty | Low | Add `compile_backward: false` config toggle |

## Expected Performance

Based on PyTorch 2.x benchmarks:
- **Forward pass:** Already optimized (~35ms)
- **Backward pass (eager):** ~45ms (estimated from trace)
- **Backward pass (compiled):** ~30-35ms (15-25% improvement)
- **Total step:** ~80ms → ~65ms (20% improvement)

## Next Steps

1. **Immediate:** Test the foreach gradient clipping change
2. **This week:** Implement Phase 1 (extract training step)
3. **Next week:** Add compilation wrapper
4. **Validation:** Run full training comparison

## Questions to Resolve

1. Should we compile the entire epoch loop or just per-step?
2. How to handle NaN loss detection without breaking graph?
3. Should we use `torch.compile` or `torch.jit.script` for backward?

## References

- PyTorch 2.0 compile tutorial: https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- Backward compilation discussion: https://github.com/pytorch/pytorch/issues/93754
- CUDA graphs for training: https://pytorch.org/docs/stable/cuda.html#cuda-graphs