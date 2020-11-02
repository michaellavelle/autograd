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

import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;

import org.apache.commons.lang3.tuple.Pair;
import org.ml4j.autograd.BackwardConfig;
import org.ml4j.autograd.AutogradValue;

/**
 * Not yet implemented base class for AuotgradValues.
 *
 * @param <V> The concrete type of this AutogradValue.
 * @param <D> The type of data wrapped by this AutogradValue, eg. Float, Matrix, Tensor
 * @param <C> The type of context required for this AutogradValue, eg. Size,
 */
public abstract class AutogradValueImpl<V, D, C> implements AutogradValue<V, D, C> {

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
    protected V applyBinaryOperator(V other, BinaryOperator<D> forward, BiFunction<V, Pair<V, V>, V> backThis,
                                    BiFunction<V, Pair<V, V>, V> backOther, String op, BinaryOperator<C> contextMapper) {
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
    protected V applyUnaryOperator(UnaryOperator<D> forward, BiFunction<V, V, V> backThis, String op, UnaryOperator<C> contextMapper) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public Supplier<D> data() {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public V requires_grad_(boolean requires_grad) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public V grad() {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public void backward() {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public void backward(BackwardConfig config) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public C context() {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public String name() {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public V name_(String name) {
        throw new UnsupportedOperationException("Not yet implemented");
    }
}
