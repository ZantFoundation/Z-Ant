# Heuristics for static memory planning

## Current approach (v0)

The current algorithm to determine a number of buffers to statically allocate
for tensors and their size uses a very simple, greedy approach:

 - Start with an empty list of usable, free buffers
 - For each operator in the fused graph:
   - For each output tensor, look for a free buffer of the same type with a size
   that is greater or equal to the size of this tensor. If one can not be found,
   create one with the size of the tensor
   - For each input tensor, decrease by one the count of references on the
   corresponding buffer. If the counter reaches 0, give it back to the free
   list

With some variations specific for the Zant codebase, this basically corresponds
to the greedy allocation strategy described here:
https://apxml.com/courses/compiler-runtime-optimization-ml/chapter-3-advanced-graph-level-optimizations/static-memory-planning.

Given the NP-hardness of finding the absolute optimal memory allocation, it's a
good enough strategy to get started, but far from optimal in terms of memory
usage and fragmentation (for beer for example it allocates a total of 377856
bytes ahead of time, versus a peak memory usage of tensors that need to be
alive at the same time of 221184).

The following strategies remain to be experimented with:

  - Introducing a form of "lookahead" in the graph to find a buffer that has a higher likelihood of
  being re-used before allocating a new one, instead of allocating a buffer for the current tensor
  - Pre-populating the free pool with the peak memory buffers (i.e. n-buffers
  large enough for the operator with the largest input and output combined)
  - Best-fit or worst-fit search for a free buffer for the current tensor
  - Buddy memory allocation (https://en.wikipedia.org/wiki/Buddy_memory_allocation)

The best metrics to evaluate these strategies include:

  - number of buffers in relation to their recycle rate (i.e. few buffers used often are better)
  - total size of all buffers and comparison with the peak memory usage
  - complexity of computation and speed of running them
  - total size of the generated binary, and especially of the `.bss` section if
  an ELF file is produced (as show with, e.g., `size -A <binary>`)

Fragmentation per se is not a good metric, neither in the positive nor in the
negative (e.g. a graph with a high peak memory usage in the middle and
progressively smaller tensors afterwards will inevitably have a high, yet
unavoidable, fragmentation).

It's also worth noting that different strategies might fit better a certain
graph and badly another. Therefore, a user configurable toggle to increase the
effort the Zant compiler will do to find a more optimal solution might be
desirable, with higher effort corresponding to a longer compilation time that
might involve trying different strategies, comparing them for total size of the
generated binary, and using the best.

Correctness remains paramount. Remember to always test each strategy for
correctness via the lib-test and other tests on the output of the generated
Zant code.

## WIP: New approach (v1)

The v1 heuristic changes the planning model from an online greedy allocator to
an offline interval-packing allocator. Instead of deciding buffer reuse while
walking the graph and maintaining a live free-list, it first builds a global
view of every produced tensor lifetime in the already linearized graph.

The algorithm currently works in execution steps, where each node in the
linearized graph has a monotonically increasing step index:

  - For every output tensor of every node, record a `TensorInfo` entry with its
  name, element count, type, production step, last-use step, and liveness.
  - The production step is the index of the node that creates the tensor.
  - The last-use step is computed by scanning the remaining nodes in the
  linearized graph and checking which later nodes list that tensor as an input.
  - If a tensor has no later internal consumer, it currently gets a conservative
  fallback interval ending at `producer_step + 1`.

After this analysis phase, tensors are sorted before allocation. The main sort
key is `size * liveness`, descending, so tensors that are both large and alive
for a long time are placed first. Ties are broken by larger `size`, then longer
`liveness`, then earlier production step. The intent is to reserve space for the
most constraining tensors before smaller or shorter-lived tensors fill the
available gaps.
- **Note:** the sorting criteria is still being experimented with:
depending on the graph structure, it might be better to prioritize some things 
over others. For exaple with the graph structure of beer, the best strategy was
to prioritize `size`, then `liveness`, then later production step, without using the
`size * liveness` combined metric.

Buffers are planned separately per tensor type. For each tensor, v1 searches
the already planned buffers of the same type and chooses the smallest buffer
that is large enough and has no reserved interval overlapping the tensor's
lifetime interval. If such a buffer exists, the tensor is assigned to it and
its interval is added to that buffer's reserved list. If no compatible buffer
exists, a new backing buffer is created with exactly the tensor's size.

v1 is still a heuristic, but it uses global lifetime information
instead of only the current free-list state. This should reduce unnecessary
buffer creation when two tensors are produced far apart in the execution order,
and it should also reduce total statically allocated memory by using a
best-fit choice among compatible buffers.

Some important details and open points:

  - Interval overlap currently uses inclusive bounds, so two tensors whose
  intervals touch at the same step are considered overlapping.
  - Reuse is only allowed across tensors with the same `TensorType`.
  - The current last-use computation checks actual node input tensors, which is
  more precise than assuming every child consumes every output.
  - Graph outputs are not yet handled as a distinct case; tensors with no
  internal consumer use the `producer_step + 1` fallback.
  - This approach still does not try multiple alternative orderings or solve
  the optimal packing problem. It is meant to be deterministic, simple enough
  to reason about, and better informed than v0.

### Results

| Model tested | v0 backing buffers | v1 backing buffers | v0 total statically allocated buffer size | v1 total statically allocated buffer size | Peak live tensor memory | v0 percentile extra (%) | v1 percentile extra (%) | Percentile decrease (%) | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| beer | 9 | 6 | 1410048 | 1161216 | 884736 | 159.4 | 131.3 | -17.6 | base-v1 with `size * liveness` as main sort key, then `size`, then `liveness`, then first production step |
| beer | 9 | 6 | 1410048 | 1059840 | 884736 | 159.4 | 119.8 | -24.8 | v1 with `size` as main sort key, then `liveness`, then later production step |
| new2 | 9 | 7 | 254080 | 138880 | 110592 | 229.7 | 125.6 | -45.3 | base-v1 with `size * liveness` as main sort key, then `size`, then `liveness`, then first production step |
