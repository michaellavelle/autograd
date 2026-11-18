package org.ml4j.autograd.arithmetic;

import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.operators.BinaryGradient;
import org.ml4j.autograd.operators.UnaryGradient;

import java.util.function.BinaryOperator;
import java.util.function.UnaryOperator;

/**
 * Defines differentiable operations by delegating forward arithmetic to wrapped data.
 *
 * <p>A scalar wrapper supplies add and mul on its numerical values. A tensor
 * adapter can supply numerical operations and context mapping while reusing graph
 * construction supplied by the autograd implementation.
 *
 * <p>The current default addition passes gradients unchanged. Tensor implementations
 * requiring broadcast reduction must adapt those contributions explicitly. This
 * interface supplies addition, composed subtraction, ordinary product-rule multiplication,
 * quotient-rule division, and negation by default.
 * Tensor types with broadcasting or non-elementwise multiplication must override
 * the relevant defaults to provide appropriate shape and derivative rules.
 *
 * <p>Scalar overloads capture fixed constants, preserve context, and record one input.
 * Threshold comparisons produce constant masks without differentiable graph inputs.
 * Forward arithmetic and numerical boundary behaviour are supplied by wrapped data.
 * Implementations supply dataAsFloatArray() for their numerical representation.
 *
 * @param <V> the differentiable operand and result type
 * @param <D> the wrapped numerical type implementing arithmetic
 * @param <C> the context type, such as shape or Void
 */
public interface DifferentiableArithmeticOperations<V extends AutogradValue<V, D, C> & DifferentiableArithmeticOperations<V, D, C>, D extends ArithmeticOperations<D>, C> extends AutogradValue<V, D, C>, ArithmeticOperations<V> {

    /**
     * Builds a lazy addition result with both operands recorded as inputs.
     *
     * <p>Scalar arithmetic or tensor addition is supplied by the concrete type.
     * Forward construction leaves operand data and gradient state unchanged.
     * Each tracked operand receives the result's incoming gradient unchanged when
     * the result node's local backward callback is invoked.
     * This default assumes contribution shapes already match the operands; tensor
     * implementations with broadcasting must override it to reduce gradients appropriately.
     *
     * @param other the right operand
     * @return the new forward sum
     * @throws NullPointerException if other is null
     */
    @Override
    default V add(V other) {
        return applyBinaryOperator(other, D::add,
                (incomingGradient, leftOperand, rightOperand) -> incomingGradient,
                (incomingGradient, leftOperand, rightOperand) -> incomingGradient,
                this::mapContext);
    }

    /**
     * Builds a lazy difference by adding the right operand's negation.
     *
     * <p>Scalar 5 minus 3 is 2. For compatible same-shape tensors, the left operand
     * receives the incoming gradient and the right receives its negation.
     *
     * <p>This composes two operation nodes rather than calling numerical sub directly.
     * Context mapping, gradient requirements, and higher-derivative dependencies follow
     * negation and addition. Neither operand is mutated. Tensor broadcasting requires
     * an addition implementation with the appropriate gradient reductions.
     *
     * @param other the non-null compatible right operand
     * @return the lazy differentiable difference
     * @throws NullPointerException if other is null
     */
    @Override
    default V sub(V other) {
        return add(other.neg());
    }

    /**
     * Builds a lazy product using the wrapped numerical data's multiplication.
     *
     * <p>Scalar values 2 and 3 produce 6. The left contribution is incoming-gradient
     * times the right operand; the right contribution is incoming-gradient times the
     * left operand. This also describes same-shape elementwise tensor multiplication,
     * but not matrix multiplication or broadcast-gradient reductions.
     *
     * <p>Neither operand is mutated. The result records ordered inputs and combined
     * gradient requirements, and gradient operations retain higher-derivative graphs.
     * Implementations with different numerical semantics must override this default.
     *
     * @param other the non-null compatible right operand
     * @return the lazy differentiable product
     * @throws NullPointerException if other is null
     */
    @Override
    default V mul(V other) {
        return applyBinaryOperator(other, D::mul,
                (incomingGradient, leftOperand, rightOperand) -> incomingGradient.mul(rightOperand),
                (incomingGradient, leftOperand, rightOperand) -> incomingGradient.mul(leftOperand),
                this::mapContext);
    }

    /**
     * Builds a lazy quotient, for example 6 divided by 2 is 3.
     *
     * <p>For scalars or compatible same-shape elementwise tensors, the numerator
     * contribution is incoming-gradient divided by the denominator. The denominator
     * contribution is the negated incoming-gradient times numerator divided by
     * denominator squared. These operations retain higher-derivative dependencies.
     *
     * <p>The result records both inputs and combines their gradient requirements;
     * neither operand is mutated. Context follows mapContext. Numerical data defines
     * zero/non-finite behaviour, including singular derivative expressions. Tensor
     * broadcasting or non-elementwise division requires overriding this default.
     *
     * @param other the non-null compatible denominator
     * @return the lazy differentiable quotient
     * @throws NullPointerException if other is null
     */
    @Override
    default V div(V other) {
        return applyBinaryOperator(other, D::div,
                (incomingGradient, leftOperand, rightOperand) -> incomingGradient.div(rightOperand),
                (incomingGradient, leftOperand, rightOperand) ->
                        incomingGradient.mul(leftOperand).neg().div(rightOperand.mul(rightOperand)),
                this::mapContext);
    }

    /**
     * Builds a lazy negation with this operand recorded as its single input.
     *
     * <p>A scalar 2 becomes -2; a tensor negates each element and keeps its shape.
     * Each tracked operand receives the negated incoming gradient.
     *
     * <p>Context and gradient requirements are preserved. Numerical data and gradient
     * state are unchanged; the derivative uses differentiable negation so dependencies
     * in the incoming gradient remain available for higher derivatives.
     *
     * @return the lazy differentiable negation
     */
    @Override
    default V neg() {
        return applyUnaryOperator(D::neg,
                (incomingGradient, operand) -> incomingGradient.neg(),
                UnaryOperator.identity());
    }


    /**
     * Adds a fixed scalar constant through a lazy unary operation.
     *
     * <p>For scalars or elementwise tensors, preserves context and records only this
     * operand. The constant is untracked; derivative operations preserve higher-order
     * dependencies. Wrapped data defines numerical boundary behaviour.
     *
     * @param constant the fixed scalar operand
     * @return the lazy arithmetic result without mutating this operand
     */
    @Override
    default V add(float constant) {
        return applyUnaryOperator(data -> data.add(constant),
                (incoming, operand) -> incoming, UnaryOperator.identity());
    }

    /**
     * Subtracts a fixed scalar constant through a lazy unary operation.
     *
     * <p>For scalars or elementwise tensors, preserves context and records only this
     * operand. The constant is untracked; derivative operations preserve higher-order
     * dependencies. Wrapped data defines numerical boundary behaviour.
     *
     * @param constant the fixed scalar operand
     * @return the lazy arithmetic result without mutating this operand
     */
    @Override
    default V sub(float constant) {
        return applyUnaryOperator(data -> data.sub(constant),
                (incoming, operand) -> incoming, UnaryOperator.identity());
    }

    /**
     * Multiplies by a fixed scalar constant through a lazy unary operation.
     *
     * <p>For scalars or elementwise tensors, preserves context and records only this
     * operand. The constant is untracked; derivative operations preserve higher-order
     * dependencies. Wrapped data defines numerical boundary behaviour.
     *
     * @param constant the fixed scalar operand
     * @return the lazy arithmetic result without mutating this operand
     */
    @Override
    default V mul(float constant) {
        return applyUnaryOperator(data -> data.mul(constant),
                (incoming, operand) -> incoming.mul(constant), UnaryOperator.identity());
    }

    /**
     * Divides by a fixed scalar constant through a lazy unary operation.
     *
     * <p>For scalars or elementwise tensors, preserves context and records only this
     * operand. The constant is untracked; derivative operations preserve higher-order
     * dependencies. Wrapped data defines numerical boundary behaviour.
     *
     * @param constant the fixed scalar operand
     * @return the lazy arithmetic result without mutating this operand
     */
    @Override
    default V div(float constant) {
        return applyUnaryOperator(data -> data.div(constant),
                (incoming, operand) -> incoming.div(constant), UnaryOperator.identity());
    }

    /**
     * Builds a lazy constant mask for elements greater than the threshold.
     *
     * <p>Wrapped data supplies float masks (one for true, zero for false) and boundary
     * behaviour. Context is preserved. The result is an untracked leaf with no inputs
     * or backward callback; gradients do not flow through this comparison.
     *
     * @param threshold the fixed scalar threshold
     * @return the lazy constant mask without mutating this operand
     */
    @Override
    default V gt(float threshold) {
        return applyNonDifferentiableOperator(data -> data.gt(threshold), UnaryOperator.identity());
    }

    /**
     * Builds a lazy constant mask for elements greater than or equal to the threshold.
     *
     * <p>Wrapped data supplies float masks (one for true, zero for false) and boundary
     * behaviour. Context is preserved. The result is an untracked leaf with no inputs
     * or backward callback; gradients do not flow through this comparison.
     *
     * @param threshold the fixed scalar threshold
     * @return the lazy constant mask without mutating this operand
     */
    @Override
    default V gte(float threshold) {
        return applyNonDifferentiableOperator(data -> data.gte(threshold), UnaryOperator.identity());
    }

    /**
     * Builds a lazy non-differentiable result without recording graph inputs.
     *
     * <p>Used for scalar or tensor comparison masks. Maps context once, does not
     * evaluate data at construction, and installs no backward callback. The result
     * requires no gradients and starts with an absent gradient. The numerical supplier
     * retains the receiver for evaluation, regardless of tracking. The result need
     * not be numerically constant. Existing state is unchanged.
     *
     * @param forwardOperator the non-null numerical operation
     * @param contextMapper the non-null context mapper
     * @return the non-null untracked result
     * @throws NullPointerException if an argument is null
     */
    V applyNonDifferentiableOperator(UnaryOperator<D> forwardOperator, UnaryOperator<C> contextMapper);

    /**
     * Builds a lazy non-differentiable result from two operands.
     *
     * <p>For example, a scalar difference or an elementwise tensor comparison can
     * depend on both operands without propagating gradients. Context mapping runs
     * once at construction; numerical evaluation remains deferred. The numerical
     * supplier retains both operands regardless of their gradient requirements.
     * The result is untracked with empty differentiation inputs, no backward
     * callback and no stored gradient. Neither operand is changed.
     *
     * @param other the non-null right operand
     * @param forwardOperator the non-null numerical operation, in left/right order
     * @param contextMapper the non-null mapper of left/right operand contexts
     * @return the non-null untracked result
     * @throws NullPointerException if an argument is null
     */
    V applyNonDifferentiableOperator(
            V other,
            BinaryOperator<D> forwardOperator,
            BinaryOperator<C> contextMapper);

    /**
     * Builds a lazy unary result, recording backward state only when gradients are required.
     *
     * <p>Scalar or tensor rules receive the result's incoming gradient and original
     * operand. The context mapper runs once; numerical evaluation remains deferred.
     * The result preserves the operand's gradient requirement. Only tracked results
     * record an input and callback; untracked results retain only lazy forward dependencies.
     * Local propagation skips constants and absent gradients, accumulating a contribution
     * without traversing inputs. Rules must preserve higher-derivative dependencies.
     *
     * @param forward the non-null numerical operation
     * @param gradient the non-null operand contribution rule
     * @param contextMapper the non-null output context mapper
     * @return the differentiable unary result
     * @throws NullPointerException if an argument is null
     */
    V applyUnaryOperator(UnaryOperator<D> forward, UnaryGradient<V> gradient,
                         UnaryOperator<C> contextMapper);

    /**
     * Maps operand metadata to result metadata.
     *
     * <p>Scalars use null context; a tensor implementation can calculate output shape.
     * Context mapping does not evaluate numerical data.
     *
     * @param leftContext left operand context
     * @param rightContext right operand context
     * @return compatible output context, or null when no metadata is needed
     */
    C mapContext(C leftContext, C rightContext);

    /**
     * Builds a lazy binary result, recording backward state only when gradients are required.
     *
     * <p>Default arithmetic passes a numerical method reference such as D::add and
     * two derivative rules. The result callback receives the forward result value;
     * derivative rules receive its incoming gradient and the ordered operands.
     *
     * <p>Implementations map context once, defer numerical evaluation, preserve
     * ordered inputs including duplicates when tracked, and combine operand gradient requirements.
     * Untracked results have no graph inputs or callback, while forward evaluation remains lazy.
     * Local propagation skips absent gradients and constants and accumulates each
     * tracked operand's contribution without traversing the graph.
     *
     * @param other the non-null right operand
     * @param forward the non-null numerical operation
     * @param leftGradient the non-null left operand contribution rule
     * @param rightGradient the non-null right operand contribution rule
     * @param contextMapper the non-null output context mapper
     * @return the differentiable binary result
     * @throws NullPointerException if an argument is null
     */
    V applyBinaryOperator(V other, BinaryOperator<D> forward,
                          BinaryGradient<V> leftGradient, BinaryGradient<V> rightGradient,
                          BinaryOperator<C> contextMapper);
}
