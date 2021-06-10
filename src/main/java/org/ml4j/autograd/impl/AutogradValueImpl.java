/*
 * Copyright 2020 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

package org.ml4j.autograd.impl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.BackwardConfig;
import org.ml4j.autograd.CachingDataSupplierImpl;
import org.ml4j.autograd.node.GradNode;
import org.ml4j.autograd.node.Node;
import org.ml4j.autograd.node.ValueNode;
import org.ml4j.autograd.operators.DifferentiableBinaryOperator;
import org.ml4j.autograd.operators.DifferentiableUnaryOperator;

/**
 * Not yet implemented base class for AuotgradValues.
 *
 * @param <V> The concrete type of this AutogradValue.
 * @param <D> The type of data wrapped by this AutogradValue, eg. Float, Matrix, Tensor
 * @param <C> The type of context required for this AutogradValue, eg. Size,
 * @author Michael Lavelle
 */
public abstract class AutogradValueImpl<V extends AutogradValue<V, D, C>, D, C> implements AutogradValue<V, D, C> {

    private V currentInstance;
    private GradNode<V> gradNode;
    private ValueNode<V> valueNode;
    protected C context;
    private Supplier<D> data;
    private boolean requires_grad;
    private String name;
    private V cachedGrad;
    protected boolean create_graph;

    protected AutogradValueImpl(Supplier<D> data, C context, List<Node<?>> children, boolean requires_grad, boolean create_graph) {
        this.data = new CachingDataSupplierImpl<>(data);
        this.context = context;
        this.valueNode = new NodeImpl<>(() -> self(), children);
        this.gradNode = new GradNodeImpl<V>(() -> null, () -> Optional.empty());
        if (data == null) {
            throw new IllegalArgumentException("Data supplier can not be null");
        }
        this.currentInstance = getInitialInstance();
        this.requires_grad = requires_grad;
        this.create_graph = create_graph;
    }

    @Override
    public V self() {
        return currentInstance;
    }


    protected abstract V getInitialInstance();

    /**
     * Apply a binary operator to this AutogradValue.
     *
     * @param other         The other value participating in this binary operation.
     * @param forward       The forward propagation operator to apply to the data wrapped by this value.
     * @param backThis      The backward propagation function to apply to this AutogradValue, to be applied
     *                      to the inbound gradient, and the pair of AutogradValues to which this operation applies,
     *                      and returning the gradient to be accumulated by this AutogradValue.
     * @param backOther     The backward propagation function to apply to the other AutogradValue, to be applied
     *                      to the inbound gradient, and the pair of AutogradValues to which this operation applies,
     *                      and returning the gradient to be accumulated by the other AutogradValue.
     * @param op            The name of the operation.
     * @param contextMapper A function that specifies how to map the context of the two AutogradValues into the
     *                      context for the resultant AutogradValue (eg. to specify a size transformation).
     * @return The resultant AutogradValue.
     */
    public V applyBinaryOperator(V other, BinaryOperator<D> forward, BiFunction<V, Pair<V, V>, V> backThis,
                                 BiFunction<V, Pair<V, V>, V> backOther, String op, BinaryOperator<C> contextMapper) {
        V gradValue = createAutogradValue(() -> forward.apply(data().get(), other.data().get()), contextMapper.apply(context(), other.context()),
                Arrays.asList(getValueNode(), other.getValueNode()), false, false);

        gradValue.getValueNode().setBackwardFunction(createBinaryBackwardFunction(other, backThis, backOther, context, other.context(), contextMapper.apply(context(), other.context())));
        if (requires_grad() || other.requires_grad()) {
            gradValue.requires_grad_(true);
        }
        return gradValue;
    }

    private BiConsumer<V, BackwardConfig> createBinaryBackwardFunction(V other, BiFunction<V, Pair<V, V>, V> backThis,
                                                                       BiFunction<V, Pair<V, V>, V> backOther, C inputContext, C otherContext, C outputContext) {

        BiFunction<V, Pair<V, V>, V> backThisAdapted = (g, p) -> backThis.apply(g, p);

        BiFunction<V, Pair<V, V>, V> backOtherAdapted = (g, p) -> backOther.apply(g, p);

        if (backThis != null) {
            backThisAdapted = (g, p) -> {
                D dat = g.data().get();
                V adapted = adapt(dat, backThis.apply(dummy(g, !this.getGradNode().isDisableNativeGradient() && this.self().getGradNode().native_grad().isPresent()), p), this.self());
                return adapted;
            };
        }

        if (backOther != null) {

            backOtherAdapted = (g, p) -> {
                D dat = g.data().get();
                V adapted = adapt(dat, backOther.apply(dummy(g, !other.getGradNode().isDisableNativeGradient() && other.getGradNode().native_grad().isPresent()), p), other);
                return adapted;
            };
        }

        final BiFunction<V, Pair<V, V>, V> backThisAdaptedFinal = backThisAdapted;

        final BiFunction<V, Pair<V, V>, V> backOtherAdaptedFinal = backOtherAdapted;


        UnaryOperator<V> backThisKeepGraph = g -> backThisAdaptedFinal.apply(g, new ImmutablePair<>(self(), other));
        UnaryOperator<V> backThisNonKeepGraph = (g) -> backThisAdaptedFinal.apply(g, new ImmutablePair<>(createAutogradValue(this.data(), inputContext, new ArrayList<>(), false, false).self(), createAutogradValue(other.data(), otherContext, new ArrayList<>(), false, false).self()));
        UnaryOperator<V> backOtherKeepGraph = (g) -> backOtherAdaptedFinal.apply(g, new ImmutablePair<>(self(), other));
        UnaryOperator<V> backOtherNonKeepGraph = g -> backOtherAdaptedFinal.apply(g, new ImmutablePair<>(createAutogradValue(this.data(), inputContext, new ArrayList<>(), false, false).self(), createAutogradValue(other.data(), otherContext, new ArrayList<>(), false, false).self()));

        Consumer<GradNode<V>> outBackwardKeepGraph = outGrad -> {
            addToGrad(backThisKeepGraph.apply(outGrad.getValue().get()));
            if (other.requires_grad()) {
                other.getGradNode().add_(backOtherKeepGraph.apply(outGrad.getValue().get()), (f, s) -> f.add(s));
            }
        };

        Consumer<V> outBackward = outGrad -> {
            addToGrad(backThisNonKeepGraph.apply(outGrad));
            if (other.requires_grad()) {
                other.getGradNode().add_(backOtherNonKeepGraph.apply(outGrad), (f, s) -> f.add(s)); // here1
            }
        };

        final BiConsumer<GradNode<V>, Boolean> backwardFunction = (out1, keep_graph) -> {
            if (keep_graph) {
                outBackwardKeepGraph.accept(out1);
            } else {
                if (out1 != null && out1.getValue() != null && out1.getValue().get() != null) {
                    // HERE
                    outBackward.accept(this.createAutogradValue(out1.getValue().get().data(), outputContext, new ArrayList<>(), false, false).self()); //here2
                }
            }
        };

        return this.convertBackward(wrapBackward(backwardFunction));

    }

    protected abstract V createAutogradValue(Supplier<D> data, C context, List<Node<?>> children, boolean requires_grad, boolean create_graph);

    /**
     * Apply an inline binary operator to this AutogradValue.
     *
     * @param other   The other value participating in this binary operation.
     * @param forward The forward propagation operator to apply to the data wrapped by this value.
     * @param op      The name of the operation.
     * @return This AutogradValue.
     */
    public V applyInlineBinaryOperator(V other, BinaryOperator<D> forward, String op) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    /**
     * Apply a unary operator to this AutogradValue.
     *
     * @param forward       The forward propagation operator to apply to the data wrapped by this value.
     * @param backThis      The backward propagation function to apply to this AutogradValue, to be applied
     *                      to the inbound gradient, and the value to which operation applies,
     *                      and returning the gradient to be accumulated by this AutogradValue.
     * @param op            The name of the operation.
     * @param contextMapper A function that specifies how to map the context of the this AutogradValues into the
     *                      context for the resultant AutogradValue (eg. to specify a size transformation).
     * @return The resultant AutogradValue.
     */
    public V applyUnaryOperator(UnaryOperator<D> forward, BiFunction<V, V, V> backThis, String op, UnaryOperator<C> contextMapper) {
        V autogradValue = createAutogradValue(() -> forward.apply(data().get()), contextMapper.apply(context()), Arrays.asList(getValueNode()), requires_grad, create_graph);
        BiConsumer<V, BackwardConfig> backwardFunction = createUnaryBackwardFunction(backThis, context);
        autogradValue.getValueNode().setBackwardFunction(backwardFunction);
        return autogradValue;
    }


    /**
     * Apply an inline unary operator to this AutogradValue.
     *
     * @param forward The forward propagation operator to apply to the data wrapped by this value.
     * @param op      The name of the operation.
     * @return This AutogradValue.
     */
    protected V applyInlineUnaryOperator(UnaryOperator<D> forward, String op) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public Supplier<D> data() {
        return data;
    }

    @Override
    public V requires_grad_(boolean requires_grad) {
        this.requires_grad = requires_grad;
        return self();
    }

    @Override
    public V grad() {
        V grad = getGradNode().getValue().get();
        if (cachedGrad != null && grad != cachedGrad) {
            cachedGrad.swapWith(grad);
        } else {
            this.cachedGrad = grad;
        }

        return cachedGrad;

    }

    @Override
    public void backward() {
        backward(new BackwardConfig());
    }

    @Override
    public void backward(BackwardConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("Config must not be null");
        }
        backward(createAutogradValue(() -> multiplicativeIdentity().get(), context, new ArrayList<>(), false, config.keep_graph()).self(), config);
    }

    @Override
    public void backward(V g, BackwardConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("Config must not be null");
        }
        if (!requires_grad()) {
            throw new IllegalStateException("Cannot backprogate through node without requires_grad=true");
        }
        this.create_graph = config.keep_graph();
        // topological order all of the children in the graph
        List<Node<?>> topo = new ArrayList<>();
        Set<Node<?>> visited = new HashSet<>();
        build_topo(topo, visited, getValueNode(), config);

        // go one variable at a time and apply the chain rule to get its gradient
        getGradNode().setValue(() -> g);

        //grad = keep_graph ? create(identity.get(), c, "one", false).self() : create(identity.get(), c, "one", false).self();
        List<Node<?>> reversed = new ArrayList<>();
        reversed.addAll(topo);
        Collections.reverse(reversed);
        for (Node<?> value : reversed) {
            value.backward(config);
        }
    }

    @Override
    public void backward(V g) {
        backward(g, new BackwardConfig());
    }

    protected abstract Supplier<D> multiplicativeIdentity();

    private void build_topo(List<Node<?>> topo, Set<Node<?>> visited, Node<?> v, BackwardConfig config) {
        if (!visited.contains(v)) {
            visited.add(v);

            // Set to grad to zero to prevent previous values affecting the result.
            if (config.zero_grad()) {
                //v.grad_(create(zero.get(), prev(), "zero", requires_grad).self());
            }
            for (Node<?> child : v.prev()) {
                build_topo(topo, visited, child, config);
            }
            topo.add(v);
        }
    }

    public void addToGrad(V other) {
        if (this.requires_grad() || this.create_graph) {
            if (getGradNode().getValue() == null) {
                getGradNode().setValue(() -> createAutogradValue(() -> additiveIdentity().get(), context(), new ArrayList<>(), false, false));
            }
            getGradNode().add_(other, (f, s) -> f.add(s).self());
        }
    }


    protected abstract Supplier<D> additiveIdentity();

    private BiConsumer<V, BackwardConfig> createUnaryBackwardFunction(BiFunction<V, V, V> backThis, C inputContext) {

        BiFunction<V, V, V> backThisAdapted = (g, p) -> {
            D dat = g.data().get();
            V adapted = adapt(dat, backThis.apply(dummy(g, !this.getGradNode().isDisableNativeGradient() && this.self().getGradNode().native_grad().isPresent()), p), this.self());
            return adapted;
        };

        UnaryOperator<V> backThisKeepGraph = (g) -> backThisAdapted.apply(g, self());
        UnaryOperator<V> backThisNonKeepGraph = (g) -> backThisAdapted.apply(g, createAutogradValue(this.data(), inputContext, new ArrayList<>(), false, false).self());

        Consumer<GradNode<V>> outBackwardKeepGraph = outGrad -> {
            addToGrad(backThisKeepGraph.apply(outGrad.getValue().get()));
        };

        Consumer<V> outBackward = outGrad -> {
            addToGrad(backThisNonKeepGraph.apply(outGrad));
        };

        final BiConsumer<GradNode<V>, Boolean> backwardFunction = (out, keep_graph) -> {
            if (keep_graph) {
                outBackwardKeepGraph.accept(out);
            } else {
                outBackward.accept(out.getValue().get());
            }
        };

        return convertBackward(wrapBackward(backwardFunction));
    }

    protected BiConsumer<V, BackwardConfig> convertBackward(BiConsumer<GradNode<V>, Boolean> backwardFunction) {
        return (v, b) -> backwardFunction.accept(v.getGradNode(), b.keep_graph());
    }

    protected V adapt(D dat, V apply, V self) {
        return apply;
    }

    private V dummy(V g, boolean b) {
        return g;
    }

    protected BiConsumer<GradNode<V>, Boolean> wrapBackward(BiConsumer<GradNode<V>, Boolean> backwardFunction) {
        return backwardFunction;
    }

    @Override
    public C context() {
        return context;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public V name_(String name) {
        this.name = name;
        return self();
    }

    @Override
    public boolean requires_grad() {
        return requires_grad;
    }

    @Override
    public V data_(Supplier<D> data) {
        this.data = new CachingDataSupplierImpl<>(data);
        return self();
    }

    @Override
    public V apply(DifferentiableUnaryOperator<V, D, C> op) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public V apply(DifferentiableBinaryOperator<V, D, C> op, V other) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public ValueNode<V> getValueNode() {
        return valueNode;
    }

    @Override
    public GradNode<V> getGradNode() {
        return gradNode;
    }

    @Override
    public void swapWith(V other) {
        V otherCurrentInstance = other.self();
        GradNode<V> otherCurrentGradNode = other.getGradNode() == null ? null : other.getGradNode();
        ValueNode<V> otherCurrentValueNode = other.getValueNode();
        D otherData = other.data().get();

        this.replaceInstance(otherCurrentInstance);
        this.replaceGradNode(otherCurrentGradNode);
        this.replaceValueNode(otherCurrentValueNode);
        this.data_(() -> otherData);

        GradNode<V> thisCurrentGradNode = getGradNode() == null ? null : getGradNode();
        ValueNode<V> thisCurrentValueNode = getValueNode();
        V thisCurrentInstance = this.self();
        if (other instanceof AutogradValueImpl) {
            @SuppressWarnings("unchecked")
            AutogradValueImpl<V, D, C> otherImpl = (AutogradValueImpl<V, D, C>) other;
            otherImpl.replaceInstance(thisCurrentInstance);
            otherImpl.replaceGradNode(thisCurrentGradNode);
            otherImpl.replaceValueNode(thisCurrentValueNode);
        } else {
            throw new UnsupportedOperationException("Swap not supported for this instance");
        }
        other.data_(this.data());
    }


    protected void replaceValueNode(ValueNode<V> thisCurrentValueNode) {
        this.valueNode = thisCurrentValueNode;
    }

    protected void replaceGradNode(GradNode<V> otherCurrentGradNode) {
        this.gradNode = otherCurrentGradNode;
    }

    protected void replaceInstance(V otherCurrentInstance) {
        this.currentInstance = otherCurrentInstance;
    }
}
