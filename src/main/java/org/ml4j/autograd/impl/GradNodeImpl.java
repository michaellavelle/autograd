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
import java.util.List;
import java.util.Optional;
import java.util.function.BiConsumer;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.Supplier;
import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.BackwardConfig;
import org.ml4j.autograd.node.GradNode;
import org.ml4j.autograd.node.Node;

/**
 * Default implementation of GradNode.
 * 
 * @author Michael Lavelle
 *
 * @param <V> The type of AutogradValue associated with this Node
 */
public class GradNodeImpl<V extends AutogradValue<V, ?, ?>> extends NodeImpl<V> implements GradNode<V> {

    private Supplier<Optional<V>> nativeGradientSupplier;
    private boolean disableNativeGradient;

    public GradNodeImpl(Supplier<V> value, Supplier<Optional<V>> nativeGradientSupplier) {
        super(value);
        this.nativeGradientSupplier = nativeGradientSupplier;
    }
    
    protected <W> GradNodeImpl(Supplier<W> value, List<Node<?>> children, 
    		BiConsumer<W, BackwardConfig> wrapBackward,
    		Function<W, V> valueMapper, Function<V, W> valueMapper2, Supplier<Optional<W>> nativeGradientSupplier) {
    	super(value, children, wrapBackward, valueMapper, valueMapper2);
       
    }
    
    @Override
    public <W extends AutogradValue<W, ?, ?>> GradNode<W> convert(Function<V, W> valueMapper, Function<W, V> valueMapper2) {
    	return new GradNodeImpl<W>(value, prev(), wrapBackward, valueMapper, valueMapper2, () -> Optional.empty());
    }

    @Override
    public GradNode<V> setValue(Supplier<V> value) {
        if (this.value != null && this.value.get() != null) {
            throw new IllegalStateException();
        }
        if (this.prev == null) {
            this.prev = new ArrayList<>();
        }
        V val = value.get();
        this.value = () -> val;
        return this;
    }

    @Override
    public GradNode<V> add_(V value, BinaryOperator<V> addFunction) {
        if (this.prev == null) {
            this.prev = new ArrayList<>();
        }
        if (this.value == null) {
            this.value = () -> value;
        } else {
            V v = this.value.get();
            if (v == null) {
                this.value = () -> value;
            } else {
                // New value
                V res = addFunction.apply(v, value);
                this.value = () -> res;
                if (!this.prev.isEmpty() || wrapBackward != null) {
                    throw new IllegalStateException();
                }
            }
        }
        return this;
    }


    @Override
    public Optional<V> native_grad() {
        return nativeGradientSupplier != null ? nativeGradientSupplier.get() : Optional.empty();
    }

    public Supplier<Optional<V>> getNativeGradientSupplier() {
        return nativeGradientSupplier;
    }

    public void setNativeGradientSupplier(Supplier<Optional<V>> nativeGradientSupplier) {
        this.nativeGradientSupplier = nativeGradientSupplier;
    }

    public boolean isDisableNativeGradient() {
        return disableNativeGradient;
    }

    public void setDisableNativeGradient(boolean disableNativeGradient) {
        this.disableNativeGradient = disableNativeGradient;
    }
}
