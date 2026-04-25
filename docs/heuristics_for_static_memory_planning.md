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

The upcoming v1 heuristic moves the greedy decision from "what free buffer is
available while walking the graph" to "which full tensor lifetime interval can
fit into an already planned buffer". 

The compiler first computes the epoch in which each tensor is produced and the 
last epoch in which it must remain alive, then sorts tensors by decreasing size 
and liveness. Each tensor is placed into the best existing buffer of the same type
whose reserved intervals do not overlap with its own interval; if none exists,
a new backing buffer is created.

This keeps the algorithm simple and deterministic, while giving it enough
global information to reuse buffers across non-overlapping tensor lifetimes and
reduce both the number of backing buffers and their total statically allocated
size.
