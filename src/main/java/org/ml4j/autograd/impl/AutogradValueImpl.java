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

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.BackwardConfig;
import org.ml4j.autograd.CachingDataSupplier;
import org.ml4j.autograd.CachingDataSupplierImpl;
import org.ml4j.autograd.node.GradNode;
import org.ml4j.autograd.node.Node;
import org.ml4j.autograd.node.ValueNode;
import org.ml4j.autograd.operators.DifferentiableBinaryOperator;
import org.ml4j.autograd.operators.DifferentiableUnaryOperator;

import java.util.*;
import java.util.function.*;

/**
 * Base class for AutogradValues.
 *
 * @param <V> The concrete type of this AutogradValue.
 * @param <D> The type of data wrapped by this AutogradValue, eg. Float, Matrix, Tensor
 * @param <C> The type of context required for this AutogradValue, eg. Size,
 * @author Michael Lavelle
 */
public abstract class AutogradValueImpl<V extends AutogradValue<V, D, C>, D, C> implements AutogradValue<V, D, C> {

    private AutogradValueProperties<C> properties;
    private V currentInstance;
    private GradNode<V> gradNode;
    private ValueNode<V> valueNode;
    private CachingDataSupplier<D> data;
    private V cachedGrad;

    public <X extends AutogradValue<X, Y, Z>, Y, Z> AutogradValueImpl(AutogradValue<X, Y, Z> other, Function<Y, D> dataMapper, Function<Z, C> contextMapper, Function<X, V> valueMapper, Function<V, X> valueReverseMapper, Supplier<Optional<V>> nativeGradientSupplier) {
        D otherDat = dataMapper.apply(other.data().get());
        this.properties = new AutogradValueProperties<>();
        if (other.isClosed()) {
            throw new IllegalStateException("Other is closed");
        }
        if (other.isClosing()) {
            throw new IllegalStateException("Other is closing");
        }
        this.data = new CachingDataSupplierImpl<>(() -> otherDat);
        this.properties.setContext(contextMapper.apply(other.context()));
        this.valueNode = new NodeImpl<>(() -> self(), other.getValueNode().prev(), other.getValueNode().next());
        if (((NodeImpl<X>)other.getValueNode()).getBackwardFunction() != null){
            valueNode.setBackwardFunction((v, c) -> {
                ((NodeImpl<X>) other.getValueNode()).getBackwardFunction().accept(valueReverseMapper.apply(v), c);
            });
        }
        GradNodeImpl<V> g = new GradNodeImpl<V>(() -> null, nativeGradientSupplier);
        g.prev = other.getGradNode().prev();
        g.next = other.getGradNode().next();
        if (((GradNodeImpl<X>)other.getGradNode()).getBackwardFunction() != null) {
            g.setBackwardFunction((v, c) -> ((GradNodeImpl<X>)other.getGradNode()).getBackwardFunction().accept(valueReverseMapper.apply(v), c));
        }
        this.gradNode = g;
        this.gradNode.setNativeGradientSupplier(nativeGradientSupplier);
        if (data == null) {
            throw new IllegalArgumentException("Data supplier can not be null");
        }
        this.currentInstance = getInitialInstance();
        this.properties.setRequires_grad(other.requires_grad());
        this.properties.setCreate_graph(other.create_graph());
        this.properties.setName(other.name());
        this.properties.addLink(other.getValueNode());
        other.properties().addLink(getValueNode());
        properties().setRegistry(other.properties().getRegistry());
        properties().register(this);
    }

    public void close() {
        setClosed(true);
    }

    @Override
    public void setClosed(boolean closed) {
        this.valueNode.setClosed(closed);
    }

    @Override
    public boolean isClosed() {
        return valueNode.isClosed();
    }

    @Override
    public boolean isClosing() {
        return valueNode.isClosing();
    }

    protected abstract void close(D data);

    public boolean create_graph() {
        return properties.isCreate_graph();
    }

    protected AutogradValueImpl(V other) {
        if (other.isClosed()) {
            throw new IllegalStateException("Other is closed");
        }
        this.data = other.data();
        this.properties = other.properties();
        this.valueNode = other.getValueNode();
        this.gradNode = other.getGradNode();;
        this.currentInstance = getInitialInstance();
        properties().register(this);
    }

    protected AutogradValueImpl(AutogradValueProperties<C> properties, Supplier<D> data) {
        this.data = new CachingDataSupplierImpl<>(data);
        this.properties = properties;
        this.valueNode = new NodeImpl<>(() -> self(), properties.getChildren(), properties.getNext());
        this.gradNode = new GradNodeImpl<V>(() -> null, () -> Optional.empty());
        if (data == null) {
            throw new IllegalArgumentException("Data supplier can not be null");
        }
        this.currentInstance = getInitialInstance();
            properties().register(this);
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
        V gradValue = createAutogradValue(() -> forward.apply(data().get(), other.data().get()),
                new AutogradValueProperties<C>().setContext(contextMapper.apply(context(), other.context()))
                        .setChildren(Arrays.asList(getValueNode(), other.getValueNode()))
                        .setRequires_grad(requires_grad() || other.requires_grad())
                        .setRegistry(properties.getRegistry())
                        .setName("resultOf:" + name() + op + other.name()));


        for (Node<?> n :gradValue.getValueNode().prev()) {
            n.next().add(gradValue.getValueNode());
        }

        gradValue.getValueNode().setBackwardFunction(createBinaryBackwardFunction(other, backThis, backOther, properties.getContext(), other.context(), contextMapper.apply(context(), other.context()), op));

        return gradValue;
    }


    private BiConsumer<V, BackwardConfig> createBinaryBackwardFunction(V other, BiFunction<V, Pair<V, V>, V> backThis,
                                                                       BiFunction<V, Pair<V, V>, V> backOther, C inputContext, C otherContext, C outputContext, String op) {

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
        UnaryOperator<V> backThisNonKeepGraph = (g) -> backThisAdaptedFinal.apply(g, new ImmutablePair<>(addLink(createAutogradValue(this.data(), new AutogradValueProperties<C>().setRegistry(this.properties.getRegistry()).setContext(inputContext).setName("binaryBack3")).self()), addLink(createAutogradValue(other.data(), new AutogradValueProperties<C>().setContext(otherContext).setRegistry(this.properties.getRegistry()).setName("binaryBack1")).self())));
        UnaryOperator<V> backOtherKeepGraph = (g) -> backOtherAdaptedFinal.apply(g, new ImmutablePair<>(self(), other));
        UnaryOperator<V> backOtherNonKeepGraph = g -> backOtherAdaptedFinal.apply(g, new ImmutablePair<>(addLink(createAutogradValue(this.data(), new AutogradValueProperties<C>().setRegistry(this.properties.getRegistry()).setContext(inputContext).setName("binaryBack4")).self()), addLink(createAutogradValue(other.data(), new AutogradValueProperties<C>().setContext(inputContext).setRegistry(this.properties.getRegistry()).setName("binaryBack2")).self())));

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
                    outBackward.accept(addLink(this.createAutogradValue(out1.getValue().get().data(), new AutogradValueProperties<C>().setContext(outputContext).setRegistry(properties().getRegistry()).setName("binaryBack5")).self())); //here2
                }
            }
        };

        return convertBackward(wrapBackward(backwardFunction));

    }

    /**
     * Creates an AutogradValue of type V from a supplier of data of type D, and an instance of
     * AutogradValueProperties wrapping a context of type C
     *
     * @param data The supplier of data.
     * @param properties The properties.
     * @return an AutogradValue of type V for these data and properties.
     */
    protected abstract V createAutogradValue(Supplier<D> data, AutogradValueProperties<C> properties);

    /**
     * Apply an inline binary operator to this AutogradValue.
     *
     * @param other   The other value participating in this binary operation.
     * @param forward The forward propagation operator to apply to the data wrapped by this value.
     * @param op      The name of the operation.
     * @return This AutogradValue.
     */
    public V applyInlineBinaryOperator(V other, BinaryOperator<D> forward, String op) {
        D result = forward.apply(data().get(), other.data().get());
        data_(() -> result);
        return self();
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
        V autogradValue = createAutogradValue(() -> forward.apply(data().get()), new AutogradValueProperties<C>().setContext(contextMapper.apply(context())).setName("resultOf:" + name() + ":" + op).setChildren(Arrays.asList(getValueNode())).setRegistry(this.properties.getRegistry()).setRequires_grad(properties.isRequires_grad()).setCreate_graph(properties.isCreate_graph()));

        for (Node<?> n :autogradValue.getValueNode().prev()) {
            n.next().add(autogradValue.getValueNode());
        }

        BiConsumer<V, BackwardConfig> backwardFunction = createUnaryBackwardFunction(backThis, properties.getContext());
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
        D result = forward.apply(data().get());
        data_(() -> result);
        return self();
    }

    @Override
    public CachingDataSupplier<D> data() {
        return data;
    }

    @Override
    public V requires_grad_(boolean requires_grad) {
        this.properties.setRequires_grad(requires_grad);
        return self();
    }

    @Override
    public V grad() {
        return grad(false);
    }

    @Override
    public V grad(boolean close) {
        Optional<V> nativeGradient = (!requires_grad() || getGradNode().isDisableNativeGradient()) ? Optional.empty() : getGradNode().native_grad();
        V grad = nativeGradient.isPresent() && nativeGradient.get().data().get() != null ? nativeGradient.get() : getGradNode().getValue().get();
        if (cachedGrad != null && grad != cachedGrad) {
            cachedGrad.swapWith(grad);
        } else {
            this.cachedGrad = grad;
        }
        if (cachedGrad != null) {
            this.cachedGrad.getValueNode().getValue().get().data().get();
        }

        if (close && !properties().isUncloseable()) {
            close();
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
        backward(createAutogradValue(() -> multiplicativeIdentity().get(), new AutogradValueProperties<C>().setContext(properties.getContext()).setRegistry(this.properties.getRegistry()).setName("backwardStart")).self(), config);
    }

    @Override
    public void backward(V g, BackwardConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("Config must not be null");
        }
        if (!requires_grad()) {
            throw new IllegalStateException("Cannot backprogate through node without requires_grad=true");
        }
        this.properties.setCreate_graph(config.keep_graph());
        // topological order all of the children in the graph
        List<Node<?>> topo = new ArrayList<>();
        Set<Node<?>> visited = new HashSet<>();
        build_topo(topo, visited, getValueNode(), config);

        // go one variable at a time and apply the chain rule to get its gradient
        getGradNode().setValue(() -> g);

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
        if (this.requires_grad() || this.properties.isCreate_graph()) {
            if (getGradNode().getValue() == null) {
                getGradNode().setValue(() -> createAutogradValue(() -> additiveIdentity().get(), new AutogradValueProperties<C>().setContext(context()).setRegistry(this.properties.getRegistry()).setName("addToGrad")));
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
        UnaryOperator<V> backThisNonKeepGraph = (g) -> backThisAdapted.apply(g, addLink(createAutogradValue(this.data(), new AutogradValueProperties<C>().setContext(inputContext).setRegistry(this.properties.getRegistry()).addLink(this.getValueNode()).setName("unaryBack1")).self()));

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

    protected V addLink(V v) {
        this.properties.addLink(v.getValueNode());
        return v;
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
        return properties.getContext();
    }

    @Override
    public String name() {
        return properties.getName();
    }

    @Override
    public V name_(String name) {
        this.properties.setName(name);
        return self();
    }

    @Override
    public boolean requires_grad() {
        return properties.isRequires_grad();
    }

    @Override
    public V data_(Supplier<D> data) {
        this.data = new CachingDataSupplierImpl<>(data);
        return self();
    }

    @Override
    public V apply(DifferentiableUnaryOperator<V, D, C> op) {
        return applyUnaryOperator(f -> op.getForward().apply(f), (g, v) -> op.getBackwardThis().apply(g, v), "n/a", i -> op.getContextMapper().apply(i));
    }

    @Override
    public V apply(DifferentiableBinaryOperator<V, D, C> op, V other) {
        return applyBinaryOperator(other, (f, s) -> op.getForward().apply(f, s), (g, v) -> op.getBackwardThis().apply(g, v), (g, v) -> op.getBackwardOther().apply(g, v), "n/a", (i, j) -> op.getContextMapper().apply(i, j));
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

    public AutogradValueProperties<C> properties() {
        return properties;
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
