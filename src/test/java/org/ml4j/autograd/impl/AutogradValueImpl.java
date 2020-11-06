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

import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;

import org.apache.commons.lang3.tuple.Pair;
import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.BackwardConfig;
import org.ml4j.autograd.node.GradNode;
import org.ml4j.autograd.node.Node;
import org.ml4j.autograd.node.ValueNode;
import org.ml4j.autograd.operators.DifferentiableBinaryOperator;
import org.ml4j.autograd.operators.DifferentiableUnaryOperator;

/**
 * Not yet implemented base class for AuotgradValues.
 * 
 * @author Michael Lavelle
 *
 * @param <V> The concrete type of this AutogradValue.
 * @param <D> The type of data wrapped by this AutogradValue, eg. Float, Matrix, Tensor
 * @param <C> The type of context required for this AutogradValue, eg. Size,
 * 
 */
public abstract class AutogradValueImpl<V extends AutogradValue<V, D, C>, D, C> implements AutogradValue<V, D, C> {

	private GradNode<V> gradNode;
	private ValueNode<V> valueNode;
	private C context;
	private Supplier<D> data;
	private boolean requires_grad;
	private String name;
	
	protected AutogradValueImpl(Supplier<D> data, C context, List<Node<?>> children) {
		this.data = data; 
		this.context = context;
		this.valueNode = new NodeImpl<>(() -> self(), children);
		this.gradNode = new GradNodeImpl<V>(() -> null, () -> Optional.empty());
		if (data == null) {
			throw new IllegalArgumentException("Data supplier can not be null");
		}
	}

	/**
     * Apply a binary operator to this AutogradValue.
     *
     * @param other The other value participating in this binary operation.
     * @param forward The forward propagation operator to apply to the data wrapped by this value.
     * @param backThis The backward propagation function to apply to this AutogradValue, to be applied
     *                 to the inbound gradient, and the pair of AutogradValues to which this operation applies,
     *                 and returning the gradient to be accumulated by this AutogradValue.
     * @param backOther The backward propagation function to apply to the other AutogradValue, to be applied
     *                 to the inbound gradient, and the pair of AutogradValues to which this operation applies,
     *                 and returning the gradient to be accumulated by the other AutogradValue.
     * @param op The name of the operation.
     * @param contextMapper A function that specifies how to map the context of the two AutogradValues into the
     *                      context for the resultant AutogradValue (eg. to specify a size transformation).
     * @return The resultant AutogradValue.
     */
	public V applyBinaryOperator(V other, BinaryOperator<D> forward, BiFunction<V, Pair<V, V>, V> backThis,
                                    BiFunction<V, Pair<V, V>, V> backOther, String op, BinaryOperator<C> contextMapper) {
        return createAutogradValue(() -> forward.apply(data().get(), other.data().get()), contextMapper.apply(context(), other.context()),
        		Arrays.asList(getValueNode(), other.getValueNode()));
    }
	
	protected abstract V createAutogradValue(Supplier<D> data, C context, List<Node<?>> children);
    
    /**
     * Apply an inline binary operator to this AutogradValue.
     *
     * @param other The other value participating in this binary operation.
     * @param forward The forward propagation operator to apply to the data wrapped by this value.
     * @param op The name of the operation.
     *
     * @return This AutogradValue.
     */
    public V applyInlineBinaryOperator(V other, BinaryOperator<D> forward, String op) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    /**
     * Apply a unary operator to this AutogradValue.
     *
     * @param forward The forward propagation operator to apply to the data wrapped by this value.
     * @param backThis The backward propagation function to apply to this AutogradValue, to be applied
     *                 to the inbound gradient, and the value to which operation applies,
     *                 and returning the gradient to be accumulated by this AutogradValue.
     * @param op The name of the operation.
     * @param contextMapper A function that specifies how to map the context of the this AutogradValues into the
     *                      context for the resultant AutogradValue (eg. to specify a size transformation).
     * @return The resultant AutogradValue.
     */
    public V applyUnaryOperator(UnaryOperator<D> forward, BiFunction<V, V, V> backThis, String op, UnaryOperator<C> contextMapper) {
        V autogradValue = createAutogradValue(() -> forward.apply(data().get()), contextMapper.apply(context()), Arrays.asList(getValueNode()));
    	BiConsumer<V, BackwardConfig> backwardFunction = null;
		//throw new UnsupportedOperationException("Not yet implemented");
        autogradValue.getValueNode().setBackwardFunction(backwardFunction);
        return autogradValue;
    }
    
    /**
     * Apply an inline unary operator to this AutogradValue.
     *
     * @param forward The forward propagation operator to apply to the data wrapped by this value.
     * @param op The name of the operation.
     *
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
    	return getGradNode().getValue().get();
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
        // TODO
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
	public V add(V other) {
        throw new UnsupportedOperationException("Not yet implemented");
	}

	@Override
	public V data_(Supplier<D> data) {
        this.data = data;
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
        throw new UnsupportedOperationException("Not yet implemented");
	}
}
