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

package org.ml4j.autograd.operators;

import java.util.function.BiFunction;
import java.util.function.UnaryOperator;

/**
 * Represents a differentiable unary operator for an AutogradValue,
 * and the logic required for both the forward and backward passes.
 *
 * @author Michael Lavelle
 *
 * @param <V> The concrete type of AutogradValue to which this operator applies.
 * @param <D> The type of data wrapped by the AutogradValue, and to which we apply the forward propagation.
 * @param <C> The type of context for the AutogradValue (eg. a size attribute).
 */
public interface DifferentiableUnaryOperator<V, D, C> {

    /**
     * Obtain the unary operator to be applied to the data within an AutogradValue on forward pass.
     *
     * @return The unary operator to be applied to the data within an AutogradValue on forward pass.
     */
    UnaryOperator<D> getForward();

    /**
     * Obtain the function to be applied to an inbound gradient and to the value of an AutogradValue on
     * backward pass, resulting in the back propagated gradient.
     *
     * @return The back propagation function for this operation.
     */
    BiFunction<V, V, V> getBackwardThis();

    /**
     * Obtain the unary operator to be applied to an AutogradValue context for this operation - eg.
     * for size mapping.
     *
     * @return The context mapper for this operation.
     */
    UnaryOperator<C> getContextMapper();
}
