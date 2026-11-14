package org.ml4j.autograd;

/**
 * A value that can combine with another value to form a sum.
 *
 * <p>Two scalar gradient contributions of 2 and 3 combine to make 5. Tensor
 * contributions can similarly combine by adding compatible numerical contents.
 *
 * <p>This contract describes value combination, not the state that holds an
 * accumulated gradient. Addition does not mutate either operand. Differentiable
 * implementations retain input dependencies and a backward rule when the result
 * requires gradients, so that the sum can itself participate in differentiation.
 *
 * @param <V> the type of value being combined and returned
 */
public interface Accumulatable<V> {

    /**
     * Returns the sum of this value and another compatible value.
     *
     * <p>Scalar values add their numerical contents. Tensor values require
     * compatible data and context, such as shapes; this interface does not
     * prescribe broadcasting or shape-conversion rules.
     *
     * <p>The result is non-null. Neither operand's data, computation graph, nor
     * gradient state is modified. Differentiable implementations combine both
     * operands' gradient requirements and retain backward dependencies when required.
     * Result object identity and evaluation/caching strategy are not prescribed.
     *
     * @param other a non-null compatible operand
     * @return the sum of the two values
     */
    V add(V other);
}
