# Development plan

## Working approach

Develop a generic automatic-differentiation library supporting numerical structures
such as scalars or tensors through concrete implementations. Deliver small,
reviewable increments in this standalone Java 25 Maven repository.
Agree and document interface contracts in [SPEC.md](SPEC.md) before implementing
changes, then add meaningful tests and validate the result. Established contracts
require explicit user agreement before modification. Add dependencies only when needed.

SPEC.md is the current contract reference. This plan describes milestones and future
work; detailed implementation history and earlier validation results remain in Git.
Future items are discussion tracks, not authorization or a prescribed implementation order.

## Completed foundations

- Standalone build, Maven Wrapper, documented contracts and scalar test/demo fixtures.
- Stable numerical values, lazy data access, optional caching, computation nodes and
  independent gradient-state storage.
- Shared unary/binary graph construction, local derivative rules and graph-wide
  backward traversal, including duplicate/shared inputs, explicit/implicit seeds,
  repeated passes, receiver-local reset and higher derivatives.
- Raw and wrapped scalar interfaces, implementations and leaf factories; shared
  arithmetic/comparison, ReLU/sigmoid and exp/log/pow capabilities.
- Stable parameter handles with replacement and wrapper extraction; gradient access,
  reset and tracking through AutogradOperations without requiring node access.
- Extensible/supplied gradient nodes and a successful-backward completion hook.
- Construction-origin diagnostics identifying forward and unary/left/right gradient
  failures while preserving lazy evaluation and original exception identity.

## Current status

Conditional unary/binary backward graph construction is complete.
Untracked results preserve lazy forward evaluation and reject tracking changes;
tracked and specialized graph construction retain existing differentiation behaviour.
Documentation consolidation is complete:
SPEC.md describes current contracts and this plan focuses on remaining decisions.
The existing differentiable arithmetic, activation and math hierarchy is retained.

Core supports retained derivative expressions and reusable graphs, with lifetime-stable
numerical contents and parameter replacement for updates. It has no configurable
recording/retention, native-resource lifecycle or tensor copying/detachment policy.
Numerical boundary limitations and always-enabled diagnostic capture remain as specified.

## Future decisions

Choose one topic and agree the smallest useful increment before implementation.
Detailed questions and constraints are in [SPEC.md: Open decisions](SPEC.md#open-decisions).

- [ ] Agree reduction/context contracts, then softmax placement and dependencies.
- [ ] Decide power zero/one special cases and mathematical boundary policies,
      preserving supported higher derivatives.
- [ ] Decide whether native-gradient integration is needed and define ownership,
      selection, accumulation/reset and derivative capabilities before adding it.
- [ ] Define tensor storage copying separately from graph retention/detachment.
- [ ] Agree native-resource ownership and validate sustained training lifetimes.
- [ ] Revisit derivative-recording and graph-retention configuration when workload
      needs justify it; agree defaults and release/reuse semantics first.
- [ ] Decide whether operation-origin diagnostics need optional capture or different
      reporting policies, considering construction cost and retained memory.

Extension hooks enable backend work but do not settle these policies. Keep core
independent of tensor types and backend libraries. Integration work is scoped
separately when requested.

## Validation expectations

- Run `./mvnw verify` for implementation and build changes. Cover relevant laziness,
  state, failure, repeated-pass and higher-derivative behaviour with meaningful tests.
- For documentation-only changes, review consistency with current contracts and
  implementation and run whitespace/link checks; a build is unnecessary.
- Keep validation within this repository unless the agreed task includes external
  integration. For such tasks, establish relevant consumers and checks from their
  current code and instructions; do not assume a particular backend or application suite.
- Native ownership work additionally needs sustained memory/lifetime validation and
  checks that retained graphs remain usable. Short correctness tests do not prove cleanup.
- Report checks performed and limitations. Record the outcome of the active increment
  concisely; fold completed work into milestones rather than appending a permanent log.
