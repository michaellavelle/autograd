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

package org.ml4j.autograd.op;

import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import org.apache.commons.lang3.tuple.Pair;

/**
 * Represents a differentiable binary operator for AutogradValues
 * and the logic required for both the forward and backward passes.
 *
 * @param <V> The concrete type of AutogradValue to which this operator applies.
 * @param <D> The type of data wrapped by the AutogradValue, and to which we apply the forward propagation.
 * @param <C> The type of context for the AutogradValue (eg. a size attribute).
 */
public interface DifferentiableBinaryOperator<V, D, C> {

    /**
     * Obtain the binary operator to be applied to the data within a pair of AutogradValues on forward pass.
     *
     * @return The binary operator to be applied to the data within a pair of AutogradValues on forward pass.
     */
    BinaryOperator<D> getForward();

    /**
     * Obtain the function to be applied to an inbound gradient and to the values of the AutogradValues participating
     * in the operation on backward pass, resulting in the back propagated gradient for the first AutogradValue.
     *
     * @return The back propagation function for this operation, for the first AutogradValue.
     */
    BiFunction<V, Pair<V, V>, V> getBackwardThis();

    /**
     * Obtain the function to be applied to an inbound gradient and to the values of the AutogradValues participating
     * in the operation on backward pass, resulting in the back propagated gradient for the second AutogradValue.
     *
     * @return The back propagation function for this operation, for the second AutogradValue.
     */
    BiFunction<V, Pair<V, V>, V> getBackwardOther();

    /**
     * Obtain the binary operator to be applied to the pair of AutogradValue contexts for this operation - eg.
     * for size mapping.
     *
     * @return The context mapper for this operation.
     */
    BinaryOperator<C> getContextMapper();
}
