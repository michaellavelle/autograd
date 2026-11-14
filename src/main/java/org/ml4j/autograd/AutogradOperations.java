package org.ml4j.autograd;

import org.ml4j.autograd.node.ValueNode;

import java.util.Optional;

/**
 * Reads gradient state, controls tracking and runs graph-wide backward passes.
 *
 * <p>For example, enable tracking on a scalar or tensor leaf before building a
 * graph, then differentiate its result with an implicit or compatible explicit
 * seed. This capability requires no numerical-data, arithmetic, or node access API.
 *
 * @param <V> the seed and returned gradient type
 */
public interface AutogradOperations<V> {

    /**
     * Returns this value's current gradient, if one is available.
     *
     * <p>An empty result means no gradient is held, including after reset. This
     * differs from a present scalar zero or tensor of zeros with compatible shape.
     * Reading does not evaluate numerical data or change state. The gradient must
     * be compatible with this value's numerical data and context.
     *
     * <p>Later reads may return replacement expressions; no gradient object-identity
     * guarantee is made. Previously returned gradient values and their computation
     * graphs remain unchanged, preserving differentiable dependencies.
     *
     * @return a non-null optional containing the current gradient, or empty when absent
     */
    Optional<V> gradient();

    /**
     * Indicates whether differentiation should calculate a gradient for this value.
     *
     * <p>A constant scalar or tensor may have a computation node while returning
     * false. Returning true requests gradient calculation; it does not imply that
     * a gradient has already been calculated.
     *
     * @return whether this value requires a gradient
     */
    boolean requiresGrad();

    /**
     * Changes whether future graphs calculate this leaf's gradient.
     *
     * <p>For example, a scalar or tensor parameter can enable tracking before
     * building a training graph and disable it before building an inference graph.
     * Only leaves can change this flag; ordinary unary/binary operation results
     * cannot, including untracked results with empty graph inputs. Set it before
     * constructing the next graph; do not change it while a graph involving this
     * leaf is still needed for backward or higher derivatives.
     *
     * <p>Existing operation results keep their original gradient requirements.
     * Numerical data, context, graph nodes, and stored gradients are unchanged;
     * clearing a gradient requires an explicit {@link #resetGradient()} call. The implementation
     * does not detect outstanding graphs or support concurrent toggling/backward.
     *
     * @param enabled whether future differentiation should calculate this leaf's gradient
     * @throws IllegalStateException if this value is an ordinary operation result or has graph inputs,
     * even if the flag is unchanged
     */
    void setRequiresGrad(boolean enabled);

    /**
     * Propagates a supplied seed gradient through this value's reachable graph.
     *
     * <p>A scalar seed of one differentiates the scalar result itself. For a tensor
     * result, a compatible tensor seed specifies the incoming gradient. Seed shape
     * and context compatibility follow the concrete numerical type's rules.
     *
     * <p>Shared nodes are visited once by node identity, after all downstream
     * contributions arrive. Duplicate operand positions still contribute separately.
     * Each pass propagates only its new contributions: gradients of leaves (nodes
     * with no inputs) accumulate across passes; intermediate gradients are replaced.
     * Constants receive no contributions from the pass. Reset leaf gradient nodes
     * explicitly to discard previous contributions.
     *
     * <p>Gradient computations retain differentiable dependencies for higher derivatives;
     * constant derivative expressions need not require gradients themselves. Gradient
     * value identity is not guaranteed, while gradient-node identity remains stable.
     * Existing gradient values and their graphs are not mutated by replacement.
     * This records derivative computations as with {@code createGraph = true};
     * no detached-gradient mode is provided.
     *
     * <p>The original graph remains available for another pass: backward does not
     * release its inputs, callbacks, or numerical suppliers. This serves the purpose
     * of {@code keepGraph}/{@code retainGraph}, independently of derivative recording.
     * Resetting a gradient node clears its held reference, not forward graph state
     * or previously returned expressions. Unreachable objects can still be garbage
     * collected. These policies do not change lazy forward evaluation.
     *
     * <p>Execution is sequential; concurrent passes over shared values are unsupported.
     * Callback failures may leave partially updated gradient state; no rollback is
     * guaranteed. Graph-wide coordination is distinct from {@link ValueNode#backward()}.
     *
     * @param gradient a non-null seed compatible with this value's numerical data and context
     * @throws NullPointerException if the seed is null
     * @throws IllegalStateException if this value does not require gradients or its graph contains a cycle
     */
    void backward(V gradient);

    /**
     * Runs a backward pass using this numerical type's implicit seed.
     *
     * <p>A float scalar uses one. Tensor types must define their own implicit seed
     * policy, or require callers to use the explicit-seed overload.
     *
     * <p>The derivative-recording, graph-retention, traversal, accumulation, and failure rules are the same as for
     * {@link #backward(Object)}.
     *
     * @throws UnsupportedOperationException if this type does not define an implicit seed
     * @throws IllegalStateException if this value does not require gradients or its graph contains a cycle
     */
    void backward();
    /**
     * Clears only this value's accumulated gradient.
     *
     * <p>For a scalar or tensor, the gradient becomes absent rather than a present
     * zero. Other values in the graph keep their gradients. Repeated resets have
     * no additional effect and do not evaluate numerical data.
     *
     * <p>Numerical contents, context, tracking, node identity, graph state, and
     * previously returned gradient expressions are preserved. This operation does
     * not release graph data or native resources and does not reset an entire graph.
     */
    void resetGradient();
}
