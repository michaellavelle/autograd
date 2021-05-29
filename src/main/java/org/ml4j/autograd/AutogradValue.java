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

package org.ml4j.autograd;

import org.ml4j.autograd.node.GradNode;
import org.ml4j.autograd.node.ValueNode;
import org.ml4j.autograd.operators.DifferentiableBinaryOperator;
import org.ml4j.autograd.operators.DifferentiableUnaryOperator;

/**
 * Represents an AutogradValue of type V, with a Pytorch-like API.
 *
 * @param <V> The concrete type of this AutogradValue.
 * @param <D> The type of data wrapped by this AutogradValue, eg. Float, Matrix, Tensor
 * @param <C> The type of context required for this AutogradValue, eg. Size,
 *
 * @author Michael Lavelle
 */
public interface AutogradValue<V, D, C> extends Value<V, D, C>, Accumulatable<V> {


    /**
     * Obtain the ValueNode in the computation graph that references this AutogradValue.
     *
     * @return The ValueNode in the computation graph that references this AutogradValue.
     */
    ValueNode<V> getValueNode();

    /**
     * Obtain the GradNode in the computation graph that references the gradient AutogradValue of this AutogradValue.
     *
     * @return The GradNode in the computation graph that references the gradient AutogradValue of this AutogradValue.
     */
    GradNode<V> getGradNode();

    /**
     * Sets whether this AutogradValue requires a gradient to be calculated.
     *
     * @param requires_grad Whether or not to require a gradient for this AutogradValue.
     * @return This AutogradValue.
     */
    V requires_grad_(boolean requires_grad);

    /**
     * Determines whether this AutogradValue requires a gradient to be calculated.
     *
     * @return Whether this AutogradValue requires a gradient to be calculated.
     */
    boolean requires_grad();

    /**
     * Obtains the gradient of this AutogradValue (after backpropagation has been performed), or null otherwise.
     *
     * @return The gradient, or null if no gradient.
     */
    V grad();

    /**
     * Backpropagate the gradient of this AutogradValue back to previous nodes in the back propagation chain,
     * with the default BackwardConfig ( false, false).
     */
    void backward();

    /**
     * Backpropagate the gradient of this AutogradValue back to previous nodes in the back propagation chain,
     * using the specified BackwardConfig.
     */
    void backward(BackwardConfig config);

    /**
     * Backpropagate the gradient of this AutogradValue back to previous nodes in the back propagation chain,
     * with the default BackwardConfig ( false, false).
     */
    void backward(V gradient);

    /**
     * Backpropagate the gradient of this AutogradValue back to previous nodes in the back propagation chain,
     * using the specified BackwardConfig.
     */
    void backward(V gradient, BackwardConfig config);

    /**
     * Applies the operator to this AutogradValue.
     *
     * @param op The operator to apply.
     * @return A new AutogradValue containing the result of the operation.
     */
    V apply(DifferentiableUnaryOperator<V, D, C> op);

    /**
     * Applies the operator to this AutogradValue, and the specified other AutogradValue.
     *
     * @param op The operator to apply.
     * @return A new AutogradValue containing the result of the operation.
     */
    V apply(DifferentiableBinaryOperator<V, D, C> op, V other);

    /**
     * Swaps this AutogradValue with the other AutogradValue in such a way that
     * all the attributes of the values are swapped.
     *
     * @param other The other AutogradValue.
     */
    void swapWith(V other);

}
