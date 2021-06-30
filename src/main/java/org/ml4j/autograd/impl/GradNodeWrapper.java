package org.ml4j.autograd.impl;

import org.ml4j.autograd.BackwardConfig;
import org.ml4j.autograd.node.GradNode;
import org.ml4j.autograd.node.Node;
import org.ml4j.autograd.node.ValueNode;

import java.util.List;
import java.util.Optional;
import java.util.function.BiConsumer;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.Supplier;

public class GradNodeWrapper<S, T> implements GradNode<T> {

    private GradNode<S> gradNode;
    private Function<S, T> mapper;
    private Function<T, S> reverseMapper;

    public GradNodeWrapper(GradNode<S> gradNode, Function<S, T> mapper, Function<T, S> reverseMapper) {
        this.gradNode = gradNode;
        this.mapper = mapper;
        this.reverseMapper = reverseMapper;
    }

    @Override
    public Supplier<T> getValue() {
        return () -> mapper.apply(gradNode.getValue().get());
    }

    @Override
    public void backward(BackwardConfig config) {
        gradNode.backward(config);
    }

    @Override
    public List<Node<?>> prev() {
        return gradNode.prev();
    }

    @Override
    public List<Node<?>> next() {
        return gradNode.next();
    }

    @Override
    public void close() {
        gradNode.close();
    }

    @Override
    public boolean isClosed() {
        return gradNode.isClosed();
    }

    @Override
    public boolean isClosing() {
        return gradNode.isClosing();
    }

    @Override
    public void setClosing(boolean closing) {
        gradNode.setClosing(closing);
    }

    @Override
    public void setClosed(boolean closed) {
        gradNode.setClosed(closed);
    }

    @Override
    public Optional<T> native_grad() {
        Optional<S> grad = gradNode.native_grad();
        if (grad.isPresent()) {
            return Optional.of(mapper.apply(grad.get()));
        } else {
            return Optional.empty();
        }
    }

    @Override
    public boolean isDisableNativeGradient() {
        return gradNode.isDisableNativeGradient();
    }

    @Override
    public void setDisableNativeGradient(boolean disableNativeGradient) {
        gradNode.setDisableNativeGradient(disableNativeGradient);
    }

    @Override
    public void setNativeGradientSupplier(Supplier<Optional<T>> nativeGradientSupplier) {

        gradNode.setNativeGradientSupplier(() -> { Optional<T> g = nativeGradientSupplier.get(); return g.isPresent() ? Optional.of(reverseMapper.apply(g.get())) : Optional.empty();});
    }

    @Override
    public GradNode<T> setValue(Supplier<T> value) {
        gradNode.setValue(() -> reverseMapper.apply(value.get()));
        return this;
    }

    @Override
    public GradNode<T> add_(T delta, BinaryOperator<T> addFunction) {
        gradNode.add_(reverseMapper.apply(delta), (f, s) -> reverseMapper.apply(addFunction.apply(mapper.apply(f), mapper.apply(s))));
        return this;
    }
}
