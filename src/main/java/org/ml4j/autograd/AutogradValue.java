package org.ml4j.autograd;

import org.ml4j.autograd.node.GradNode;
import org.ml4j.autograd.node.ValueNode;

import java.util.Optional;

/**
 * Connects numerical data to its computation node and accumulated gradient state.
 *
 * <p>A scalar addition result or tensor operation result has a value node describing
 * its inputs. Its gradient node holds contributions produced by differentiation.
 * A gradient can itself be an autograd value with its own computation graph,
 * allowing gradient expressions to be differentiated for higher derivatives.
 * For example, the gradient of a scalar cube can be differentiated again; tensor
 * operation rules can support Hessian-vector products in the same way.
 *
 * <p>Backward currently records differentiable gradient computations (broadly
 * equivalent to {@code createGraph = true}) and preserves the original graph for
 * reuse (the purpose of {@code keepGraph}, also called {@code retainGraph}). There
 * are no configuration flags for either policy. Constant derivatives need not
 * require gradients; higher-derivative support depends on the operation rules.
 *
 * <p>These policies may be revisited for first-order training without derivative
 * graphs, tensor memory pressure, explicit release of saved backward buffers, or
 * operations with limited higher-derivative support. Such configuration requires
 * a separately agreed contract; it is not provided by this interface today.
 *
 * <p>This value coordinates graph-wide backward passes. Value nodes execute local
 * rules; gradient nodes store contributions. Addition follows the non-mutating
 * {@link Accumulatable} contract, retaining backward dependencies when gradients are required.
 *
 * @param <V> the concrete autograd value type, also used to represent gradients
 * @param <D> the numerical data type, such as Float or a tensor implementation
 * @param <C> the context type, such as shape/size or Void when no metadata is needed
 */
public interface AutogradValue<V extends AutogradValue<V, D, C>, D, C>
        extends Value<D, C>, Accumulatable<V>, AutogradOperations<V> {

    /**
     * Returns the node describing this value's computation and inputs.
     *
     * <p>A directly supplied scalar or tensor is a leaf with no inputs. An operation
     * tracked result records its operands as input nodes.
     *
     * @return a non-null node representing this value
     */
    ValueNode<V> valueNode();

    /**
     * Returns a stable handle to this value's gradient state.
     *
     * <p>Retain the node and read {@code gradNode().gradient()} to obtain the current
     * scalar or tensor gradient. That gradient may be absent or replaced later.
     *
     * <p>The same node instance is returned throughout this value's lifetime,
     * including after accumulation, backward passes, and reset. Previously returned
     * gradient expressions are not refreshed or mutated when the held reference changes.
     *
     * @return the non-null gradient node owned by this value
     */
    GradNode<V> gradNode();

    /**
     * Returns the current gradient through this value's gradient node.
     *
     * <p>Delegates to {@code gradNode().gradient()} without evaluating numerical
     * data, preserving absence and gradient expressions as specified by
     * {@link AutogradOperations#gradient()}.
     *
     * @return a non-null optional containing the current gradient, or empty when absent
     */
    @Override
    default Optional<V> gradient() {
        return gradNode().gradient();
    }

    /**
     * Clears this value's accumulated gradient through its gradient node.
     *
     * <p>Delegates to {@code gradNode().reset()}, preserving the node and all
     * other state as specified by {@link AutogradOperations#resetGradient()}.
     */
    @Override
    default void resetGradient() {
        gradNode().reset();
    }
}
