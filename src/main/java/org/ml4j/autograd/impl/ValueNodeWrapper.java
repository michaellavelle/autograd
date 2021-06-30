package org.ml4j.autograd.impl;

import org.ml4j.autograd.BackwardConfig;
import org.ml4j.autograd.node.Node;
import org.ml4j.autograd.node.ValueNode;

import java.util.List;
import java.util.function.BiConsumer;
import java.util.function.Function;
import java.util.function.Supplier;

public class ValueNodeWrapper<S, T> implements ValueNode<T> {

    private ValueNode<S> valueNode;
    private Function<S, T> mapper;
    private Function<T, S> reverseMapper;

    public ValueNodeWrapper(ValueNode<S> valueNode, Function<S, T> mapper, Function<T, S> reverseMapper) {
        this.valueNode = valueNode;
        this.mapper = mapper;
        this.reverseMapper = reverseMapper;
    }


    @Override
    public void setBackwardFunction(BiConsumer<T, BackwardConfig> backwardFunction) {
        valueNode.setBackwardFunction((v, c) -> backwardFunction.accept(mapper.apply(v),c));
    }

    @Override
    public BiConsumer<T, BackwardConfig> getBackwardFunction() {
        return (v, c) -> valueNode.getBackwardFunction().accept(reverseMapper.apply(v), c);
    }

    @Override
    public Supplier<T> getValue() {
        return () -> mapper.apply(valueNode.getValue().get());
    }

    @Override
    public void backward(BackwardConfig config) {
        valueNode.backward(config);
    }

    @Override
    public List<Node<?>> prev() {
        return valueNode.prev();
    }

    @Override
    public List<Node<?>> next() {
        return valueNode.next();
    }

    @Override
    public void close() {
        valueNode.close();
    }

    @Override
    public boolean isClosed() {
        return valueNode.isClosed();
    }

    @Override
    public boolean isClosing() {
        return valueNode.isClosing();
    }

    @Override
    public void setClosing(boolean closing) {
        valueNode.setClosing(closing);
    }

    @Override
    public void setClosed(boolean closed) {
        valueNode.setClosed(closed);
    }
}
