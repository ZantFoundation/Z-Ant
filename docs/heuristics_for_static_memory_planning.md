# Heuristics for static memory planning

## Exact planner for small graphs: branch and bound

For small graphs, Zant currently uses an exact branch-and-bound planner instead
of the heuristic planners. The current cutoff is graphs with at most 25
nodes.

The branch-and-bound planner uses the same tensor lifetime model as the v1
heuristic: each produced tensor has a type, a size, a production step, and a
last-use step. Two tensors may share the same backing buffer only when:

  - they have the same tensor type
  - the backing buffer is large enough for both tensors
  - their lifetime intervals do not overlap

The goal is to minimize the total reserved memory across all unique backing
buffers.

The algorithm works by trying every valid assignment, but pruning branches that
cannot beat the best solution found so far:

  - Sort tensors using the selected `tensorInfoLessThan` ordering 
    - the build flag indicates which comparator to use: no comparator is better in terms of correctness, only performance–not so relevant in practice for such small graphs
  - Start with no backing buffers.
  - For the current tensor, first try assigning it to each existing compatible
  buffer.
  - Then try creating a new backing buffer exactly large enough for that tensor.
  - Recurse to assign the next tensor.
  - When all tensors are assigned, keep the solution if its total reserved
  memory is smaller than the current best.
  - If the current partial solution is already at least as expensive as the
  best solution, stop exploring that branch.

Because this search is exact, it can be much slower than the heuristic planners
on large graphs. That is why it is only used for small graphs. For larger
graphs, Zant falls back to the heuristic planner.

## Old approach (v0)

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

## Current approach (v1)

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
order is selected by the `-Dstatic_planning` flag and implemented by
`tensorInfoLessThan`. The default enabled ordering uses `size * liveness`
descending, then `size` descending, then production step, so tensors that are
both large and alive for a long time are placed early. Other orderings can
prioritize `liveness`, `size`, or reverse the production-step tie-breaker.

The intent is to reserve space for the most constraining tensors before smaller
or shorter-lived tensors fill the available gaps. The sorting criteria is still
being experimented with: depending on the graph structure, it might be better to
prioritize some things over others. For example, with the graph structure of
beer, a strong strategy was to prioritize `size`, then `liveness`, then **later**
production step, without using the `size * liveness` combined metric.

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
  - This approach still does not automatically try multiple orderings and pick
  the best result, nor does it solve the optimal packing problem. It is meant
  to be deterministic, simple enough to reason about, and better informed than
  v0.

### `tensorInfoLessThan` ordering options

The v1 planner sorts `TensorInfo` entries through `tensorInfoLessThan` before
assigning tensors to backing buffers. The sort order is selected with the
`-Dstatic_planning` build flag when static allocation is enabled:

```sh
zig build lib-gen -Ddynamic=false -Dstatic_planning=enabled
```

The valid base options are:

  - `disabled`: disables static memory planning.
  - `enabled`: enables static memory planning and uses the same tensor ordering
  as `size_first`.
  - `pressure_then_size`: sort by `size * liveness` descending, then `size`
  descending, then production step.
  - `pressure_then_liveness`: sort by `size * liveness` descending, then `liveness`
  descending, then production step.
  - `liveness_first`: sort by `liveness` descending, then `size` descending,
  then production step.
  - `size_first`: sort by `size` descending, then `liveness` descending, then
  production step.
  - `first_step`: sort by production step only.

By default, the final production-step tie-breaker places earlier-produced
tensors first (`first_step` ascending). Any **explicit** ordering can append the
`_inverse_first_step` suffix to flip only that final tie-breaker:

```sh
zig build lib-gen -Ddynamic=false -Dstatic_planning=size_first_inverse_first_step
```

For example, `size_first_inverse_first_step` still sorts by `size` first and
`liveness` second, but ties are resolved by placing later-produced tensors
first. The suffix is a modifier, not a standalone planner mode, and it is not
valid with `disabled` or the `enabled` convenience alias.

### Results from the new heuristic (v1) compared to the old one (v0)

| Model tested | v0 backing buffers | v1 backing buffers | v0 total statically allocated buffer size | v1 total statically allocated buffer size | Peak live tensor memory | v0 percentile extra (%) | v1 percentile extra (%) | Percentile decrease (%) | Build flag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| beer | 9 | 6 | 1410048 | 1161216 | 884736 | 159.4 | 131.3 | -17.6 | `pressure_then_size` |
| beer | 9 | 6 | 1410048 | 1059840 | 884736 | 159.4 | 119.8 | -24.8 | `size_first_inverse_first_step` |
| new2 | 9 | 7 | 254080 | 138880 | 110592 | 229.7 | 125.6 | -45.3 | `pressure_then_size` |
| mobilenet_v2 | 7 | 4 | 323584 | 235520 | 196608 | 164.6 | 119.8 | -27.2 | `pressure_then_size` |
| mobilenet_v2 | 7 | 4 | 323584 | 208896 | 196608 | 164.6 | 106.3 | -35.4 | `size_first`/`size_first_inverse_first_step` |
| mobilenet_v2 | 7 | 5 | 323584 | 210944 | 196608 | 164.6 | 107.3 | -34.8 | `liveness_first`/`liveness_first_inverse_first_step` |
| resnet50 | 3 | 4 | 9633792 | 9633792 | 6422528 | 150 | 150 | -0.0 | `pressure_then_size_inverse_first_step` |
| resnet50 | 3 | 3 | 9633792 | 10035200 | 6422528 | 150 | 156.3 | +4.2 | `pressure_then_size`/`size_first` |
| r18_net | 6 | 3 | 221184 | 147456 | 131072 | 168.8 | 112.5 | -33.3 | `size_first`/`pressure_then_size` |
| r18_net | 6 | 4 | 221184 | 155648 | 131072 | 168.8 | 118.8 | -29.6 | `pressure_then_liveness` |
| r18_net | 6 | 4 | 221184 | 163840 | 131072 | 168.8 | 125 | -25.9 | `size_first_inverse_first_step`/`liveness_first` |
| r18_net | 6 | 6 | 221184 | 221184 | 131072 | 168.8 | 168.8 | -0.0 | `first_step` |
