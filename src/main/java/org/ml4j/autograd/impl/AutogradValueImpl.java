package org.ml4j.autograd.impl;

import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.Accumulatable;
import org.ml4j.autograd.Value;
import org.ml4j.autograd.node.GradNode;
import org.ml4j.autograd.node.ValueNode;
import org.ml4j.autograd.node.impl.GradNodeImpl;
import org.ml4j.autograd.node.impl.ValueNodeImpl;
import org.ml4j.autograd.operators.BinaryGradient;
import org.ml4j.autograd.operators.UnaryGradient;

import java.util.List;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Collections;
import java.util.Objects;
import java.util.function.BinaryOperator;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;

/**
 * Provides shared node ownership and forward-operation construction for autograd values.
 *
 * <p>Implementations supply numerical operations, result construction and context
 * mapping for structures such as scalars, vectors, matrices or tensors, either
 * directly or through differentiable operation defaults. All structures use the
 * same ordered backward graph inputs when tracked and stable gradient-state handles.
 *
 * <p>Each operation supplies its local derivative rules. Graph-wide passes order reachable
 * nodes by dependency, propagate new contributions, and retain previous leaf gradients.
 * Numerical subclasses define implicit seed policies; explicit seeds work generically.
 * Supplied data and context must satisfy {@link org.ml4j.autograd.Value}'s stability
 * contract. The base class adds no caching and is intended for sequential use.
 *
 * <p>Backward uses the supplied differentiable rules without detaching their results,
 * following {@link AutogradValue}'s current {@code createGraph}-like policy. It does
 * not release graph inputs, callbacks, or numerical suppliers after a pass, so graphs
 * remain reusable while reachable. Neither derivative recording nor graph retention
 * has a configuration flag. First-order training, tensor memory pressure, saved-buffer
 * release, or limited derivative support may motivate a future contract discussion.
 *
 * <p>Runtime evaluation and local backward failures retain their exception identity
 * and receive one suppressed construction trace at the first failing boundary.
 * Construction stacks are always captured; this adds time and memory overhead.
 *
 * @param <V> the concrete autograd value type
 * @param <D> the numerical data type
 * @param <C> the context type, such as shape or Void
 */
public abstract class AutogradValueImpl<V extends AutogradValue<V, D, C>, D, C>
        implements AutogradValue<V, D, C> {

    private final Supplier<D> data;
    private final C context;
    private boolean requiresGrad;
    private boolean operationResult;
    private final ValueNode<V> valueNode;
    private final GradNode<V> gradNode;

    /**
     * Creates the shared state without evaluating data or calling subclass hooks.
     *
     * @param data a supplier of stable, non-null numerical data
     * @param context stable metadata, or null when none is needed
     * @param requiresGrad whether differentiation should calculate a gradient
     * @param inputs ordered operands, empty for a leaf
     * @throws NullPointerException if data, inputs, or an input node is null
     */
    protected AutogradValueImpl(Supplier<D> data, C context, boolean requiresGrad,
                                List<? extends ValueNode<?>> inputs) {
        this(data, context, requiresGrad, inputs, new GradNodeImpl<>());
    }

    /**
     * Creates a value using the supplied gradient-state implementation.
     *
     * <p>A numerical backend can supply a gradient node exposing equivalent lazy
     * gradient views while keeping its own data selection outside this library.
     * The exact node is retained for this value's lifetime. Supply an independent,
     * initially empty node for each value; this constructor does not inspect, reset
     * or invoke it, evaluate data, or call subclass hooks.
     *
     * @param data a supplier of stable, non-null numerical data
     * @param context stable metadata, or null when none is needed
     * @param requiresGrad whether differentiation should calculate a gradient
     * @param inputs ordered operands, empty for a leaf
     * @param gradNode non-null gradient state preserving the GradNode contract
     * @throws NullPointerException if data, inputs, an input node or gradNode is null
     */
    protected AutogradValueImpl(Supplier<D> data, C context, boolean requiresGrad,
                                List<? extends ValueNode<?>> inputs, GradNode<V> gradNode) {
        this.gradNode = Objects.requireNonNull(gradNode, "gradNode");
        Supplier<D> source = Objects.requireNonNull(data, "data");
        this.data = source instanceof OriginSupplier<?> ? source : new OriginSupplier<>(source);
        this.context = context;
        this.requiresGrad = requiresGrad;
        // The supplier defers the subclass hook until the node's represented value is read.
        this.valueNode = new ValueNodeImpl<>(this::self, inputs);
    }

    // Carries the same origin through the result constructor without capturing twice.
    private static final class OriginSupplier<T> implements Supplier<T> {
        private final Supplier<T> source;
        private final OperationOrigin origin = new OperationOrigin();

        private OriginSupplier(Supplier<T> source) { this.source = source; }

        @Override public T get() {
            try {
                return Objects.requireNonNull(source.get(), "numerical data");
            } catch (RuntimeException failure) {
                throw origin.annotate(failure, "forward evaluation");
            }
        }

        private void gradient(String phase, Runnable action) {
            try {
                action.run();
            } catch (RuntimeException failure) {
                throw origin.annotate(failure, phase);
            }
        }

        private <R> Consumer<R> backward(Consumer<R> callback) {
            return value -> {
                try {
                    callback.accept(value);
                } catch (RuntimeException failure) {
                    throw origin.annotate(failure, "backward callback");
                }
            };
        }
    }

    private static final class OperationOrigin extends Throwable {
        private OperationOrigin() { super("Autograd value/operation constructed here"); }
        private OperationOrigin(String message) { super(message); }

        private RuntimeException annotate(RuntimeException failure, String phase) {
            for (Throwable suppressed : failure.getSuppressed()) {
                if (suppressed instanceof OperationOrigin) {
                    return failure;
                }
            }
            var diagnostic = new OperationOrigin("Autograd failure during " + phase
                    + "; operation constructed here");
            diagnostic.setStackTrace(getStackTrace());
            failure.addSuppressed(diagnostic);
            return failure;
        }
    }

    /**
     * Supplies the concrete owner when the forward node is read.
     *
     * <p>A scalar or tensor subclass normally returns this. The constructor does
     * not invoke this hook; the reference must consistently represent this instance.
     *
     * @return this value as its concrete type, never null
     */
    protected abstract V self();

    /** {@inheritDoc} */
    @Override public final Supplier<D> data() { return data; }

    /** {@inheritDoc} */
    @Override public final C context() { return context; }

    /** {@inheritDoc} */
    @Override public final boolean requiresGrad() { return requiresGrad; }

    /** {@inheritDoc} */
    @Override
    public void setRequiresGrad(boolean requiresGrad) {
        if (operationResult || !valueNode.inputs().isEmpty()) {
            throw new IllegalStateException("Only leaves can change their gradient requirement");
        }
        this.requiresGrad = requiresGrad;
    }

    /** {@inheritDoc} */
    @Override public final ValueNode<V> valueNode() { return valueNode; }

    /** {@inheritDoc} */
    @Override public final GradNode<V> gradNode() { return gradNode; }

    /**
     * Constructs a lazy binary result, recording backward state only when gradients are required.
     *
     * <p>The context mapper runs once at construction; numerical evaluation is deferred.
     * Scalar operands may have no context; tensor implementations map compatible shapes.
     * The node callback receives the forward result value and extracts its gradient
     * before invoking the separate left/right derivative rules.
     *
     * <p>Subclasses may wrap forward operations or derivative rules, for example to
     * reduce broadcast tensor gradients or record backend operations, then delegate
     * to super. Overrides must preserve lazy evaluation, context mapping once,
     * ordered inputs including duplicates when tracked, combined gradient requirements, and local
     * accumulation without mutating operands. Derivative rules must preserve
     * higher-derivative dependencies and skip constants and absent gradients.
     *
     * @param other the right operand
     * @param forward the numerical operation
     * @param leftGradient the left operand's contribution rule
     * @param rightGradient the right operand's contribution rule
     * @param contextMapper maps operand metadata to output metadata
     * @return the concrete result, requiring gradients if either operand does
     */
    public V applyBinaryOperator(V other, BinaryOperator<D> forward,
                                          BinaryGradient<V> leftGradient, BinaryGradient<V> rightGradient,
                                          BinaryOperator<C> contextMapper) {
        Objects.requireNonNull(other, "other");
        Objects.requireNonNull(forward, "forward");
        Objects.requireNonNull(leftGradient, "leftGradient");
        Objects.requireNonNull(rightGradient, "rightGradient");
        Objects.requireNonNull(contextMapper, "contextMapper");
        C outputContext = contextMapper.apply(context(), other.context());
        OriginSupplier<D> operation = new OriginSupplier<D>(() -> forward.apply(data().get(), other.data().get()));
        boolean trackGradients = requiresGrad() || other.requiresGrad();
        V result = createOperationResult(operation, outputContext, trackGradients,
                trackGradients ? List.of(valueNode(), other.valueNode()) : List.of());
        if (!trackGradients) {
            return result;
        }
        result.valueNode().setBackwardFunction(operation.backward(resultValue -> {
            if (!resultValue.requiresGrad()) {
                return;
            }
            resultValue.gradNode().gradient().ifPresent(incomingGradient -> {
                if (requiresGrad()) {
                    operation.gradient("left gradient calculation/accumulation", () ->
                            gradNode().accumulate(leftGradient.apply(incomingGradient, self(), other)));
                }
                if (other.requiresGrad()) {
                    // Duplicate operands still have two derivative rules and two contributions.
                    operation.gradient("right gradient calculation/accumulation", () ->
                            other.gradNode().accumulate(rightGradient.apply(incomingGradient, self(), other)));
                }
            });
        }));
        return result;
    }

    /**
     * Constructs a lazy unary result, recording backward state only when gradients are required.
     *
     * <p>Tracked scalar negation or a tensor operation records this value as its single input.
     * Untracked results have no inputs or callback; forward evaluation still needs the operand.
     * Context is mapped once without evaluating data. The callback extracts the
     * result's incoming gradient and passes it with the original operand to the rule.
     * Constants and absent gradients cause no propagation; accumulation is local.
     *
     * <p>Subclasses may wrap the forward operation or derivative rule for backend
     * integration, then delegate to super. Overrides must preserve lazy evaluation,
     * context mapping once, a single input only when tracked, the operand's gradient
     * requirement, and local accumulation without mutating the operand. Derivative
     * rules must preserve higher-derivative dependencies and skip constants and
     * absent gradients.
     *
     * @param forward the non-null numerical operation
     * @param gradient the non-null operand contribution rule
     * @param contextMapper the non-null output context mapper
     * @return the lazy concrete result, preserving the operand's gradient requirement
     * @throws NullPointerException if an argument is null
     */
    public V applyUnaryOperator(UnaryOperator<D> forward, UnaryGradient<V> gradient,
                                      UnaryOperator<C> contextMapper) {
        Objects.requireNonNull(forward, "forward");
        Objects.requireNonNull(gradient, "gradient");
        Objects.requireNonNull(contextMapper, "contextMapper");
        C outputContext = contextMapper.apply(context());
        OriginSupplier<D> operation = new OriginSupplier<D>(() -> forward.apply(data().get()));
        boolean trackGradients = requiresGrad();
        V result = createOperationResult(operation, outputContext, trackGradients,
                trackGradients ? List.of(valueNode()) : List.of());
        if (!trackGradients) {
            return result;
        }
        result.valueNode().setBackwardFunction(operation.backward(resultValue -> {
            if (resultValue.requiresGrad() && requiresGrad()) {
                resultValue.gradNode().gradient().ifPresent(incomingGradient ->
                        operation.gradient("unary gradient calculation/accumulation", () ->
                                gradNode().accumulate(gradient.apply(incomingGradient, self()))));
            }
        }));
        return result;
    }

    // Preserve operation-result tracking immutability without retaining operand nodes.
    private V createOperationResult(Supplier<D> supplier, C outputContext, boolean trackGradients,
                                    List<? extends ValueNode<?>> inputs) {
        V result = createAutogradValue(supplier, outputContext, trackGradients, inputs);
        if (result instanceof AutogradValueImpl<?, ?, ?> sharedResult) {
            sharedResult.operationResult = true;
        }
        return result;
    }

    /**
     * Builds a lazy untracked leaf, for example a scalar or tensor comparison mask.
     *
     * <p>Context is mapped once; the result has no graph inputs or backward callback
     * and starts with no gradient. Overrides must preserve these guarantees and
     * leave existing operand state unchanged.
     *
     * @param forwardOperator the non-null numerical operation
     * @param contextMapper the non-null output context mapper
     * @return the new untracked result
     * @throws NullPointerException if an argument is null
     */
    public V applyNonDifferentiableOperator(
            UnaryOperator<D> forwardOperator,
            UnaryOperator<C> contextMapper) {
        Objects.requireNonNull(forwardOperator, "forwardOperator");
        Objects.requireNonNull(contextMapper, "contextMapper");

        C outputContext = contextMapper.apply(context());
        return createAutogradValue(new OriginSupplier<>(() -> forwardOperator.apply(data().get())),
                outputContext, false, List.of());
    }

    /**
     * Builds a lazy untracked result whose numerical data depends on both operands.
     *
     * <p>Maps contexts once and retains both operands in the deferred computation,
     * independently of tracking. Installs no differentiation inputs or backward
     * callback. Scalar and tensor numerical semantics belong to the supplied operator.
     *
     * @param other the non-null right operand
     * @param forwardOperator the non-null numerical operation, in left/right order
     * @param contextMapper the non-null mapper of operand contexts, in left/right order
     * @return the new untracked result
     * @throws NullPointerException if an argument is null
     */
    public V applyNonDifferentiableOperator(
            V other,
            BinaryOperator<D> forwardOperator,
            BinaryOperator<C> contextMapper) {
        Objects.requireNonNull(other, "other");
        Objects.requireNonNull(forwardOperator, "forwardOperator");
        Objects.requireNonNull(contextMapper, "contextMapper");

        C outputContext = contextMapper.apply(context(), other.context());
        return createAutogradValue(
                new OriginSupplier<>(() -> forwardOperator.apply(data().get(), other.data().get())),
                outputContext, false, List.of());
    }

    /**
     * Maps operand metadata to the result context, such as tensor shape or null for scalars.
     * @param left left context
     * @param right right context
     * @return compatible output context
     */
    public abstract C mapContext(C left, C right);

    /**
     * Constructs the concrete operation result, preserving the supplied metadata and inputs.
     * @param dataSupplier lazy result data
     * @param context mapped output metadata
     * @param requiresGrad combined operand gradient requirement
     * @param inputs ordered backward dependencies, preserving duplicates; empty for
     *               ordinary untracked operations
     * @return the new result value
     */
    protected abstract V createAutogradValue(Supplier<D> dataSupplier, C context, boolean requiresGrad,
                                             List<? extends ValueNode<?>> inputs);

    /** {@inheritDoc} */
    @Override
    public void backward(V gradient) {
        Objects.requireNonNull(gradient, "seed gradient");
        if (!requiresGrad()) {
            throw new IllegalStateException("The backward root must require gradients");
        }
        // Discover and validate the original graph before derivative operations add new nodes.
        List<ValueNode<?>> order = new ArrayList<>();
        visit(valueNode(), new IdentityHashMap<>(), order);
        List<Runnable> restoreLeaves = new ArrayList<>();
        for (ValueNode<?> node : order) {
            AutogradValue<?, ?, ?> owner = (AutogradValue<?, ?, ?>) node.value();
            if (node.inputs().isEmpty()) {
                saveLeafGradient(owner.gradNode(), restoreLeaves);
            }
            owner.gradNode().reset();
        }
        gradNode().accumulate(gradient);
        Collections.reverse(order);
        for (ValueNode<?> node : order) {
            AutogradValue<?, ?, ?> owner = (AutogradValue<?, ?, ?>) node.value();
            if (owner.requiresGrad() && owner.gradNode().gradient().isPresent()) {
                node.backward();
            }
        }
        // Restore previous leaves AFTER propagation, preventing stale contributions from running again.
        restoreLeaves.forEach(Runnable::run);
        afterBackward(List.copyOf(order));
    }

    /**
     * Called on the root after propagation and restoration of previous leaf gradients.
     * For example, a backend can inspect final leaf totals using the already visited
     * nodes instead of traversing the graph again. The default implementation does nothing.
     *
     * <p>Called once for a successful implicit- or explicit-seed pass. Earlier failures
     * skip this hook. Exceptions from overrides propagate without rolling back gradients.
     * The shared implementation does not evaluate numerical data for this hook.
     *
     * @param visitedNodes unmodifiable, identity-deduplicated original graph nodes in
     *                     root-first propagation order, including constants and leaves;
     *                     newly constructed derivative nodes are excluded
     */
    protected void afterBackward(List<ValueNode<?>> visitedNodes) {
        // Default: no action.
    }

    /** Retains typed gradient references for restoration without mutating earlier expressions. */
    private static <G extends Value<?, ?> & Accumulatable<G>> void saveLeafGradient(
            GradNode<G> node, List<Runnable> restoreLeaves) {
        node.gradient().ifPresent(previous -> restoreLeaves.add(() -> node.accumulate(previous)));
    }

    /** Produces dependency-first order by identity; a false entry marks a node still being visited. */
    private static void visit(ValueNode<?> node, Map<ValueNode<?>, Boolean> visited, List<ValueNode<?>> order) {
        Boolean complete = visited.get(node);
        if (complete != null) {
            if (!complete) throw new IllegalStateException("Cycle in computation graph");
            return;
        }
        if (!(node.value() instanceof AutogradValue<?, ?, ?>)) {
            throw new IllegalArgumentException("Backward graph inputs must represent autograd values");
        }
        visited.put(node, false);
        for (ValueNode<?> input : node.inputs()) {
            visit(input, visited, order);
        }
        visited.put(node, true);
        order.add(node);
    }

    /** {@inheritDoc} */
    @Override
    public void backward() {
        if (!requiresGrad()) {
            throw new IllegalStateException("The backward root must require gradients");
        }
        backward(defaultGradient());
    }

    /**
     * Supplies this numerical type's implicit seed, if defined.
     *
     * <p>Float scalars use one. Tensor subclasses can define a compatible seed
     * policy or keep the default requirement for an explicit seed.
     *
     * @return the implicit seed
     * @throws UnsupportedOperationException unless the subclass defines an implicit seed
     */
    protected V defaultGradient() {
        throw new UnsupportedOperationException("Supply an explicit seed gradient");
    }
}
