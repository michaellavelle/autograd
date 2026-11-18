# Specification

## Scope and change control

Generic Java 25 Maven automatic-differentiation library supporting numerical
structures such as scalars, vectors, matrices and tensors, depending on the
implementation. Implementations supply numerical operations and compatible context
(such as shape); core graph and gradient contracts are shared across structures.
The repository builds independently of parent projects, sibling repositories and
tensor backends, and includes scalar implementations to demonstrate these contracts.
Build and test with `./mvnw verify`; documentation-only edits need consistency review.
`PLAN.md` records implementation history; this specification describes current contracts.

Record each new interface's purpose, signatures, behaviour and constraints here
before implementation. Established contracts, including identity, nullability,
lifecycle and failure behaviour, require explicit user agreement before changing.
Future work listed below is not authorization to implement it.

Public Javadoc starts with a simple explanation and useful examples, then precise
guarantees and parameter/return/exception tags. Include scalar and tensor examples
where helpful, without introducing backend dependencies. Name accessors `x()` and
actions with verbs. Keep examples consistent with the detailed contract.

## Agreed interface: `org.ml4j.autograd.DataSupplier<D>`

Purpose: expose data access independently of context, arithmetic, and autograd state.
For example, supply Float scalar data or a backend tensor object.

```java
Supplier<D> data();
```

- Returns a non-null supplier without evaluating data; evaluation is deferred to get().
- Caching, supplier/result identity, data stability, and result nullability are
  implementation-defined. Evaluation failures may propagate from the supplier.
- Adds no context, graph, gradient, or mutation methods.
- Value<D, C> extends DataSupplier<D> and redeclares data() with its existing stronger
  guarantees: stable numerical contents and non-null successful evaluations.
  AutogradValue inherits this capability through Value. Existing implementations
  and the Value data-access contract are unchanged.

## Agreed interface: `org.ml4j.autograd.Value<D, C>`

Purpose: provide numerical data access and the metadata needed to interpret that data.
Extends DataSupplier<D> while preserving the stronger data-access guarantees below.
`D` describes the numerical data type; `C` describes its context type.
For example, `D` can be a tensor implementation type and `C` a shape or size type
describing its dimensions. These examples introduce no tensor-library dependency.

```java
Supplier<D> data();
C context();
```

- `data()` returns a non-null supplier without evaluating numerical data. Successful
  supplier evaluations return non-null data; null does not represent absence.
- Numerical contents and context remain stable throughout the value's lifetime.
  A lazy supplier must represent the same contents on repeated evaluations, rather
  than a changing source. Context must remain compatible with the numerical data.
- Stability is semantic, not an object-identity guarantee. Implementations and callers
  must not mutate exposed data or context objects in ways that change the represented value.
- Caching is an implementation choice. Neither supplier nor result/context object
  identity is guaranteed. Evaluation failures may propagate from the supplier;
  this interface defines no recovery policy or particular evaluation exception type.
- `context()` does not evaluate numerical data. It may return null only when no
  additional metadata is needed; `Void` is the intended context type for such values.
- The contract provides no mutation, cache invalidation, gradient, graph, or naming operations.
- Numerical in-place operations are not supported on autograd values. Gradient accumulation
  and reset remain supported mutations of bookkeeping, without changing numerical contents.
  Direct backing-storage changes and changing external data suppliers are unsupported.

## Implementation: `org.ml4j.autograd.scalar.FloatValue`

- A final class implementing `Value<Float, Void>`, constructed with a primitive float.
- Stores fixed scalar data; its supplier returns that data without a mutation API.
- Uses null context because the scalar needs no additional metadata.
- Accepts all primitive float values, including signed zero, infinities, and NaN.

## Agreed interface: `org.ml4j.autograd.node.ValueNode<V>`

Purpose: represent a forward value, its immediate computation dependencies, and its
local backward callback. The type bound is `V extends Value<?, ?>`; no gradient
interface or implementation is required by this contract.

```java
V value();
List<? extends ValueNode<?>> inputs();
void setBackwardFunction(Consumer<V> resultValueBackwardFunction);
Consumer<V> backwardFunction();
void backward();
```

- `value()` returns a non-null represented value without evaluating numerical data.
- `inputs()` returns an unmodifiable, non-null list of non-null nodes in operand order.
  Leaves have no inputs. Duplicate operand references are preserved. Inputs may use
  different numerical data and context types.
- The callback consumes the represented forward/result value, not its gradient.
  Gradient lookup and contribution calculation are responsibilities of the callback.
- Installing a callback replaces the previous one without executing either callback.
  Passing null removes it. `backwardFunction()` returns the installed callback or null,
  without executing it or evaluating data.
- `backward()` invokes the installed callback once with `value()`, or does nothing
  when no callback exists. It does not traverse inputs, seed gradients, or extract
  gradients. A separate coordinator ensures contributions are ready before invocation.
- Callback exceptions propagate; this local operation offers no rollback guarantee.
- Scalar and tensor operations use the same node contract; context such as tensor
  shape remains on the represented value.

## Agreed interface: `org.ml4j.autograd.Accumulatable<V>`

Purpose: describe how values combine into a sum, independently of the state that
stores accumulated gradients. For example, scalar contributions of 2 and 3 sum
to 5; tensor contributions add compatible numerical contents.

```java
V add(V other);
```

- Accepts a non-null compatible operand and returns a non-null sum.
- Addition does not mutate either operand's numerical data, computation graph,
  or gradient state. Storing the sum in an accumulator is a separate responsibility.
- Differentiable implementations combine both operands' gradient requirements.
  They retain operand dependencies and a backward rule only when the result requires gradients.
- Numerical data and context must be compatible. Tensor shape compatibility and
  broadcasting rules are defined by concrete numerical implementations, not this interface.
- No result object-identity, evaluation, or caching strategy is prescribed.
- The exception policy for invalid operands is not specified by this interface.

## Agreed interface: `org.ml4j.autograd.node.GradNode<V>`

Purpose: hold the current gradient and combine contributions using the gradient
value's own addition. The type bound is `V extends Value<?, ?> & Accumulatable<V>`.
This supports scalar and tensor gradient values without requiring a complete
autograd-value interface.

```java
Optional<V> gradient();
void accumulate(V contribution);
void reset();
```

- `gradient()` returns a non-null optional without evaluating data. Absence differs
  from a present gradient whose contents are zero. Gradient data/context must be
  compatible with the forward value it differentiates.
- The first contribution is stored directly. Subsequent contributions replace the
  held reference with `current.add(contribution)`, without mutating either operand.
- Contributions and sums are non-null and compatible with the represented gradient.
  Null contributions or sums cause `NullPointerException`. If addition throws or
  returns null, the previously held reference remains unchanged. Compatibility rules
  and their exception types belong to the numerical implementation.
- The node does not directly evaluate data; addition determines any evaluation and
  whether differentiable dependencies are retained.
- Accumulation and reset preserve the node as a stable handle. Held gradient value
  identity is not guaranteed; earlier expressions are not refreshed or mutated.
  Their data stability continues to follow the `Value` contract.
- `reset()` clears only the held reference and is idempotent. It does not release
  forward data or mutate earlier gradient values or their computation graphs.
- The accumulator has no graph-input or backward-execution responsibilities. A
  differentiable gradient value can have its own computation node.

## Agreed interface: `org.ml4j.autograd.AutogradOperations<V>`

Purpose: expose gradient access, tracking and graph-wide backward operations independently
of numerical data access, arithmetic, and node access. The gradient type `V` is
unbounded; this capability introduces no self-type or numerical-type requirement.

```java
Optional<V> gradient();
boolean requiresGrad();
void setRequiresGrad(boolean enabled);
void backward();
void backward(V gradient);
void resetGradient();
```

This capability requires no numerical, arithmetic or node API; V can be a tensor
wrapper type. Its tracking/backward policy is specified below for AutogradValue.

- gradient() returns a non-null Optional without numerical evaluation or mutation.
  Absence differs from a present zero. Gradients have compatible data/context and
  retain derivative dependencies. Later reads may return replacement expressions;
  previously returned values and graphs remain unchanged.
- resetGradient() is lazy, receiver-local and idempotent. It clears the held gradient
  to absence, preserving data, context, tracking, nodes, graph state and previously
  returned expressions. It does not release graph or native resources.
- AutogradValue supplies defaults delegating to gradNode().gradient()/reset(). Direct
  capability implementations supply both methods; wrappers delegate to their value.

## Agreed interface: `org.ml4j.autograd.AutogradValue<V, D, C>`

Purpose: connect numerical values to their computation and gradient nodes and provide
graph-wide backward entry points. Extends `Value<D, C>`, `Accumulatable<V>`, and `AutogradOperations<V>`, with
the self-type bound `V extends AutogradValue<V, D, C>`. Gradients use the same concrete
value type, so gradient expressions can have their own computation graphs.

```java
ValueNode<V> valueNode();
GradNode<V> gradNode();
```

- `valueNode()` returns a non-null node representing this value and its inputs.
- `gradNode()` returns the same non-null gradient node throughout the value's lifetime,
  including reset and backward passes. Stored gradient values may change; previously
  returned expressions are neither refreshed nor mutated by replacement.
- `requiresGrad()` requests gradient calculation, independently of gradient availability.
  Constants may have computation nodes, but receive no contributions from a backward pass.
- `backward(gradient)` takes a non-null seed compatible with the result's data/context.
  Tensor seed compatibility follows the numerical implementation's rules.
- Traversal visits reachable nodes once by identity, running local callbacks after all
  downstream contributions arrive. Repeated operand positions contribute separately.
- Each pass propagates only new contributions. Leaves are nodes with no inputs; their
  gradients accumulate across passes. Intermediate gradients are replaced. Callers reset
  leaf gradient nodes explicitly to discard previous contributions.
- Gradient computations retain differentiable dependencies for higher derivatives.
  A constant derivative expression need not itself require gradients.
- A null explicit seed causes `NullPointerException`. A root not requiring gradients,
  or a cyclic graph, causes `IllegalStateException`. Callback failures can leave partial
  gradient updates; no graph-wide rollback is guaranteed.
- Passes execute sequentially; concurrent execution over shared values is unsupported.
- `backward()` uses the numerical type's implicit seed (one for float scalars).
  Tensor seed policies are type-specific. Types with no implicit seed throw
  `UnsupportedOperationException` and require explicit seeds.
- Inherited addition follows the existing `Accumulatable` contract; inherited data
  access and context stability follow the existing `Value` contract.

## Current backward policy: derivative recording and graph retention

Backward produces differentiable gradient expressions and keeps computation graphs
available for reuse. For example, differentiating `x*x*x` produces an expression for
`3*x*x` that can be differentiated again. Tensor applications can use this behaviour
for Hessian-vector products, provided their operation rules support higher derivatives.

- `createGraph` means recording derivative computations so their results retain
  differentiable connections to the original operands. The current policy is broadly
  equivalent to `createGraph = true`; there is no flag or detached-gradient mode.
  Constant derivatives need not require gradients, and each numerical operation must
  supply suitable differentiable rules for the derivative orders it supports.
- `keepGraph` (also called `retainGraph`) means preserving the original graph's
  saved backward state for another pass. The current implementation does not release
  inputs, callbacks, or numerical suppliers after backward, so the graph remains reusable
  while reachable. There is no retention flag or explicit saved-state release policy;
  unreachable objects remain eligible for normal Java garbage collection.
- These are separate concepts: retaining the original graph alone does not make
  computed gradients differentiable. Recording derivatives can also retain references
  to original operands and their graphs, increasing memory use.
- Repeated passes follow the existing accumulation contract: leaf gradients accumulate
  and intermediate gradients are replaced. `gradNode().reset()` clears the stored
  gradient reference; it does not detach previously returned expressions or release
  forward graph state. Neither concept changes the lazy forward-evaluation policy.

Potential future configuration is deferred, not agreed or implemented. Revisit it when:

- First-order training needs numerical gradients without building derivative graphs.
- Larger tensor workloads or long-lived gradient expressions reveal memory pressure
  from retained dependencies, callbacks, suppliers, or future saved tensors.
- Operations introduce saved backward buffers that need an explicit release lifecycle,
  requiring a decision about repeated backward passes after release.
- A new numerical backend or operation supports only first derivatives, requiring
  explicit capability documentation and a policy for unsupported higher derivatives.

Before adding flags, agree their defaults, detachment semantics, release behaviour,
and failure rules. Update interface contracts, Javadoc, and tests together, including
first derivatives, higher derivatives, repeated passes, and reset behaviour.

## Agreed interface: `org.ml4j.autograd.operators.BinaryGradient<V>`

Purpose: calculate one operand's contribution from a result's incoming gradient
and its forward operands. Scalar addition returns the incoming gradient unchanged;
tensor rules can adapt contributions to operand shapes.

```java
V apply(V incomingGradient, V leftOperand, V rightOperand);
```

- A functional interface with explicit non-null incoming gradient and forward operands
  in their original order, including duplicate references. The forward result itself
  is not passed to this rule.
- Returns a non-null contribution compatible with the differentiated operand's data/context.
- Does not mutate arguments or accumulate state. The caller performs accumulation.
- Differentiable rules retain dependencies for higher derivatives; a constant derivative
  expression need not require gradients itself.
- Numerical compatibility and invalid-argument exception policies belong to concrete rules.

`UnaryGradient<V>` is the corresponding functional interface for one operand:
`V apply(V incomingGradient, V operand)`. It has the same non-null, compatible,
non-mutating contribution contract and leaves accumulation to its caller.


## Shared graph implementation

`node.impl.ValueNodeImpl` is final, accepts a deferred represented-value supplier
and copies the input list immutably. Its leaf constructor uses empty inputs.
Construction rejects null suppliers/lists/nodes without resolving the value;
value access rejects null results. The supplier consistently represents the owner.
Callbacks start absent. Execution and use are sequential.

`node.impl.GradNodeImpl` is extensible, initially empty, and implements the GradNode
contract without constructor-time evaluation. Subclasses may expose equivalent lazy
gradient views, preserving absence, stable data/context and derivative dependencies.
Inherited accumulation/reset use the private stored expression without invoking the
overridden accessor. Cached views must follow replacement/reset and preserve earlier views.

`impl.AutogradValueImpl<V,D,C>` owns data, context, tracking and stable nodes.
Its deferred self() hook consistently returns the concrete owner and is not called
during construction. The four-argument constructor creates an independent GradNodeImpl;
the five-argument overload accepts a non-null gradient node after inputs and retains
that exact handle. Supply an independent, initially empty node; construction does not
inspect, reset or invoke it, evaluate data, or call subclass hooks. Data evaluation
rejects null results and adds no caching.

Ordinary unary/binary operations snapshot the combined gradient requirement at
construction. When false, their nodes have empty inputs and no backward callback;
when true, ordered inputs (including constants and duplicates) and derivative rules
are preserved. Forward suppliers still capture operands lazily. Optional
CachingDataSupplier wrappers release those closures after success; uncached values
continue to retain forward dependencies for repeated evaluation. No eager evaluation
or numerical copying is introduced. Non-differentiable operations have no backward graph.
Direct constructors/result factories preserve explicitly supplied inputs, including
specialized differentiation graphs; they are not filtered by requiresGrad.

Tracking changes are leaf-only, including unchanged-flag requests: ordinary operation results
throw IllegalStateException even when their untracked nodes have no inputs.
Non-differentiable-operation results retain their existing leaf behaviour. Leaf changes are idempotent and lazy, preserving data,
context, nodes and stored gradients. Toggle before constructing a new graph and finish
backward/higher derivatives on previous graphs first. Existing results retain their
flags; callbacks read current operand flags. Reuse after leaf toggling is unsupported;
there is no recursive update or outstanding-graph detection. The setter is overridable
subject to this contract.

### Operator construction hooks

DifferentiableArithmeticOperations declares these public hooks, implemented by the base:

```java
V applyUnaryOperator(UnaryOperator<D> forward, UnaryGradient<V> gradient,
                     UnaryOperator<C> contextMapper);
V applyBinaryOperator(V other, BinaryOperator<D> forward,
                     BinaryGradient<V> leftGradient, BinaryGradient<V> rightGradient,
                     BinaryOperator<C> contextMapper);
V applyNonDifferentiableOperator(UnaryOperator<D> forwardOperator, UnaryOperator<C> contextMapper);
V applyNonDifferentiableOperator(V other, BinaryOperator<D> forwardOperator,
                                 BinaryOperator<C> contextMapper);
C mapContext(C leftContext, C rightContext);
```

Non-null functions/operands are required. Context maps once without numerical reads;
scalar context may be null, tensor context describes compatible output shape.
Unary results inherit tracking; binary results require gradients if either operand
does. Tracked results record ordered inputs including duplicates and local callbacks;
untracked results record neither inputs nor callbacks. Local callbacks
skip constants and absent gradients and accumulate separate contributions for each
tracked operand position. They do not traverse the graph or mutate operand contents.
Non-differentiable operations create untracked leaves with empty inputs, absent gradients and
no callback; captured data suppliers do not establish graph edges. Unary operations
retain the receiver in their lazy numerical supplier; binary operations retain both
operands in left/right order, regardless of operand tracking. Context mapping runs
once at construction without numerical evaluation. Results may depend numerically
on their operands; non-differentiability does not mean numerical constancy. Optional
caching retains/releases these closures under its existing policy. Null operands or
functions cause NullPointerException. The renamed unary hook has no compatibility bridge;
custom implementations must also supply the binary hook.

Hooks are overridable: implementations may wrap numerical/derivative rules for backend
recording or broadcast-gradient reduction, preserving laziness, context mapping once,
input ordering, tracking, local accumulation, constant handling and higher derivatives.
Internal createAutogradValue constructs graph results; leaf factories do not replace it.

### Successful backward completion

The protected no-op afterBackward(List<ValueNode<?>> visitedNodes) runs once on the
root after successful propagation and restoration of previous leaf totals, for either
seed path. It receives an unmodifiable copy of the existing identity-deduplicated
original traversal in root-first propagation order, including constants, excluding
new derivative nodes. It adds no traversal or numerical reads. Earlier failures skip
it; hook exceptions propagate with completed gradient changes retained, without rollback.


## Numerical and differentiable capabilities

`arithmetic.ArithmeticOperations<V extends ArithmeticOperations<V>>` declares all
methods below without implementations. Operations return non-null results and do not
mutate operand data, graphs or gradient state. Backends define compatibility,
multiplication semantics, rounding and numerical boundaries.

```java
V add(V other); V sub(V other); V mul(V other); V div(V other);
V add(float other); V sub(float other); V mul(float other); V div(float other);
V neg(); V gt(float threshold); V gte(float threshold);
float[] dataAsFloatArray();
```

Float arguments are fixed constants. Comparisons return zero/one masks. Array access
may evaluate/convert data; results are read-only by contract whether copied or shared.
Backends document flattening order and non-float conversion; callers copy before editing.

`DifferentiableArithmeticOperations<V,D,C>` extends AutogradValue and
ArithmeticOperations<V>; D supplies numerical arithmetic and V the differentiable
self-type. Defaults delegate numerical operations through the graph hooks. They
support scalars and compatible elementwise tensors; broadcasting reductions and
non-elementwise products require overrides. For incoming gradient g:

| Operation | Gradient contributions |
| --- | --- |
| left + right | g, g |
| left - right | Composed as left.add(right.neg()) |
| left * right | g * right, g * left |
| left / right | g / right, -(g * left) / (right * right) |
| -x | -g |
| x + c, x - c | g |
| x * c, x / c | g * c, g / c |

Scalar overloads use one unary input and identity context mapping, with no scalar
leaf or gradient for c. gt/gte use applyNonDifferentiableOperator and preserve context;
gradients do not flow through the masks. dataAsFloatArray() remains a required
implementation method. Derivative expressions retain differentiable dependencies;
singularities and overflow/underflow are not regularised.

`activation.ActivationOperations<V>` declares relu() and sigmoid();
`math.MathOperations<V>` declares exp(), log() and pow(float exponent).
These numerical capabilities are non-mutating, non-null, elementwise and
shape-preserving for tensors, with backend-defined domains and precision.

DifferentiableActivationOperations and DifferentiableMathOperations both extend
DifferentiableArithmeticOperations: their gradient rules need arithmetic as well as
graph construction. D supplies arithmetic and the relevant numerical capability;
V supplies differentiable arithmetic and that family's differentiable self-type.
The current hierarchy is retained; no separate DifferentiableOperations is introduced.
Defaults use one lazy unary input, preserve context/tracking and skip constants.

| Operation | Gradient contribution |
| --- | --- |
| relu(x) = max(0,x) | g * x.gt(0) |
| sigmoid(x) | g * s * (1 - s), s = x.sigmoid() |
| exp(x) | g * x.exp() |
| log(x) | g / x |
| x.pow(p) | g * x.pow(p - 1) * p |

ReLU masks are fixed: derivative zero at/below zero and one above zero. Higher
orders retain incoming-gradient dependencies but do not differentiate the mask or
promise smoothness at thresholds. Power exponents are fixed floats with no zero/one
special cases: pow(0) at zero and higher polynomial derivatives can yield NaN through
singular intermediates. No clipping or common finite-result/domain guarantee is imposed.

## Scalar implementations and factories

Interfaces and FloatValue live in org.ml4j.autograd.scalar; implementation classes
live in scalar.impl. Demos live in org.ml4j.autograd.demo: AutogradDemo,
FloatAutogradValueDemo and FloatWrapperAutogradValueDemo.

- FloatValue implements Value<Float,Void> only, storing fixed primitive data.
- FloatAutogradValue is an interface over raw Float data with add/sub/mul/div/neg;
  FloatAutogradValueImpl supplies lazy, non-cached scalar arithmetic and null context.
  It does not implement ArithmeticOperations or shared activation/math capabilities.
- FloatWrapper exposes floatValue() and numerical arithmetic, activation and math
  capabilities. FloatWrapperImpl stores fixed primitive data, uses Java float
  arithmetic/comparisons (NaN comparisons false), and returns fresh one-element arrays.
  ReLU uses Math.max(0f,x), preserving NaN and producing positive zero for signed zeros.
  exp/log/pow use Java Math with double intermediates and float rounding. Sigmoid uses
  a stable sign-dependent double calculation: signed zeros yield 0.5, infinities zero/one,
  and NaN stays NaN.
- FloatWrapperAutogradValue combines wrapper data with all shared differentiable
  capabilities; its implementation delegates array reads to wrapped data.
- Both autograd scalar implementations use constant scalar one as the implicit seed.
  Float values, including signed zero, infinities and NaN, follow Java numerical rules.
- FloatAutogradValueFactory and FloatWrapperAutogradValueFactory bind the generic
  factory types and add create(float value, boolean requiresGrad). Their Impl classes
  delegate both overloads to constructors, creating independent leaves with explicit
  tracking and null context, without supplier evaluation or additional caching.


## Agreed interface: `org.ml4j.autograd.AutogradValueFactory<V, D, C>`

Purpose: create autograd leaves with an explicit caller-selected gradient requirement.
The self-type bound is V extends AutogradValue<V, D, C>. This is a functional interface.

```java
V create(Supplier<D> data, C context, boolean requiresGrad);
```

- Returns a new non-null leaf without evaluating numerical data or modifying existing
  values/gradients. Inputs are empty, the backward callback is absent, and its gradient
  is initially absent. requiresGrad() matches the supplied boolean.
- Rejects a null supplier with NullPointerException. Supplied data/context follow
  Value's stable, compatible numerical-data contract, including non-null successful
  evaluations and null context only when metadata is unnecessary. Caching is an
  implementation choice; no supplier or numerical-result identity is promised.
- Scalar factories can accept Float data with Void context; tensor factories can
  accept backend numerical data with shape context. No backend dependency is introduced.
- There is no two-argument overload or implicit gradient requirement. Operation-result
  construction with graph inputs remains separate.

## Agreed interface: `org.ml4j.autograd.Parameter<V>`

Purpose: hold a replaceable current value through a stable parameter handle.
The type parameter V is unbounded, allowing numerical wrappers such as Tensor
that expose an AutogradValue without implementing it.

```java
V value();
void replace(V value);
```

- value() returns the non-null current value without numerical evaluation, with
  the same identity until replacement.
- replace selects the supplied value for future reads without numerical evaluation.
  Null causes NullPointerException; invalid replacement preserves the current reference.
  Replacing with the current value is permitted. Replacement changes only the held
  reference, preserving existing values, gradients and graphs.
- Callers provide values compatible with the model's numerical type and context,
  including tensor shape. Compatibility and leaf validation belong to implementations.
- Graphs capture the selected autograd leaves rather than live parameter reads.
  Wrappers must keep their represented numerical contents and leaf association stable
  when used in graphs; replacement does not refresh earlier graph expressions.
- Use sequentially. No arithmetic, optimizer, gradient storage, native-resource
  cleanup, or concurrent replacement/graph-construction contract is introduced.

## Implementation: `org.ml4j.autograd.impl.ParameterImpl<V>`

Final implementation with unbounded V and constructors:

```java
ParameterImpl(V value, Function<? super V, ? extends AutogradValue<?, ?, ?>> extractor)
ParameterImpl(V value)
```

The two-argument constructor accepts direct autograd values or wrappers, for example
scalar wrapper extraction or Tensor::autogradValue for a tensor with shape context.
The extractor must return the stable represented autograd value without numerical
evaluation or mutation. It is invoked once per non-null construction/replacement
candidate, never by value(). The extractor, candidate and extracted result must be
non-null; violations cause NullPointerException. Extracted values must have empty
graph-input lists, otherwise IllegalArgumentException is thrown. Extractor exceptions
propagate. Validation completes before assignment, preserving the current reference
on failure; extractor side effects cannot be rolled back.

The one-argument constructor uses direct extraction: candidates must themselves
implement AutogradValue, otherwise IllegalArgumentException is thrown, including on
later replacement. With unbounded V this requirement is checked at runtime.
Neither constructor evaluates numerical data. Replacement changes only the held
reference, preserving wrapped leaves, gradients and graphs. Numerical/context
compatibility and native-resource ownership remain the caller's responsibility.
Sequential use only.

## Agreed utility: `org.ml4j.autograd.cache.CachingDataSupplier<D>`

A final optional Supplier<D> implementation, constructed from Supplier<? extends D>.
Construction rejects null with NullPointerException without evaluating data. get()
evaluates lazily and caches the first successful non-null result, returning that same
object thereafter. Source exceptions propagate and null results cause
NullPointerException; both leave the cache empty so later calls retry.

`Optional<D> cachedValue()` inspects the existing cached field without evaluating
or invoking the source supplier, or modifying either cache or source. It returns
empty before successful evaluation, including after exceptions or null results;
after success it contains the exact cached object, without copying or transformation.
Repeated inspection adds no evaluation or concurrency guarantees.

After success the wrapper releases its source closure reference. It exposes no reset,
invalidation methods. Sequential use only; no thread-safety guarantee.
Callers must preserve cached numerical contents to satisfy Value stability. It performs
no native-resource cleanup. Tensor adapters can opt in to cache random generation or
native-array allocation; the core base class still leaves caching optional. This class
implements Supplier, not the DataSupplier owner capability.

## Implementation diagnostics: operation origins

AutogradValueImpl captures one construction stack per shared unary, binary or
non-differentiable operation, reused for lazy data evaluation and the entire local backward
callback (including accumulation). Leaf suppliers capture their own origin.
RuntimeExceptions retain their identity and type; the first failing boundary adds
a private origin marker as a suppressed exception. Enclosing boundaries do not add
more markers. Suppression-disabled exceptions remain unchanged. Errors are not caught.
Evaluation remains lazy and uncached. Context mapping and graph-wide coordination
are outside these boundaries. Stack capture is currently always enabled and costs
time and retained memory; configurable diagnostics are deferred. No interface changes.

Origin diagnostic messages identify forward evaluation, unary gradient, left gradient,
right gradient, or the remaining backward callback. Gradient labels include both
rule execution and accumulation. Phase-specific diagnostics reuse the captured
construction stack and do not change the first-origin policy. Lazy gradient data
failures are forward-evaluation failures of the gradient expression itself.

## Backend boundaries

Core is independent of tensor libraries. Parameter names belong to model metadata such as
Map<String,Parameter<T>> and remain associated with handles across replacement.
Numerical in-place operations and copy/mapped-value constructors are not provided.
Backward discovers dependencies through nodes, not global registration.

Tensor adapters own shape validation/broadcast-gradient reduction and numerical
backend operations. The agreed implicit tensor seed is an untracked ones leaf matching
result shape/context, supplied by defaultGradient(); generic implementations without
an implicit seed throw UnsupportedOperationException. Wrappers can expose
AutogradOperations<Tensor> without exposing nodes. Reductions and softmax are outside
this core capability set. Downstream implementation and validation history lives in
PLAN.md; this document does not assert current sibling-repository status.

Java graph reachability, gradient reset and registry removal do not establish native
resource cleanup. Cached data, old parameter leaves and retained gradient expressions
may outlive a training step. Core defines no graph-release or native ownership API.

## Open decisions

These require separate agreement before implementation:

- **Reductions and softmax:** whole-value/axis reduction signatures, shape/context
  hooks, axes, keepDimensions, empty inputs, element counts, gradient expansion,
  broadcasting, max ties/non-finite policies and stable softmax composition. Keep
  backend types out of core and preserve supported higher derivatives. Validate
  shapes, explicit seeds and both backends before downstream training integration.
- **Power and mathematical boundaries:** zero/one exponent special cases, singularity
  handling, signed zero, negative bases, fractional/non-finite exponents and higher
  derivatives. Decide separately whether log/overflow/domain behaviour remains
  backend-defined. State behaviour changes and backend feasibility first.
- **Native gradients:** recording and graph ownership, gradient selection, seed/
  accumulation/reset, leaf tracking, higher derivatives and unsupported combinations.
  Prevent double accumulation and compare deterministic results with library rules.
  Extension hooks do not by themselves define native-gradient integration.
- **Copying and detachment:** distinguish shared/copied storage from retained/detached
  graph connections. Agree names, tracking/state, laziness, caching, context and shared
  resource lifetime; validate storage independence and graph connectivity separately.
- **Native resource ownership:** owners of parameters, temporary arrays, cached data
  and retained derivative graphs; cleanup on success/failure and shared-storage rules.
  Validate live-graph usability and bounded memory over sustained training. Short
  correctness tests and Java garbage collection do not prove native cleanup.
- **Backward configuration:** recording versus retention, defaults, saved-state release,
  repeated-pass failures and limited derivative capabilities. The current always-record,
  retain-while-reachable policy remains in force.
- **Diagnostic configuration:** stack capture is currently always enabled; optional
  capture and alternative trace reporting policies remain deferred.
