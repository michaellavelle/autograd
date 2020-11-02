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

package org.ml4j.autograd.node;

import java.util.function.BinaryOperator;
import java.util.function.Supplier;

/**
 * Represents a Node that is used for accumulating deltas added to an AutogradValue.
 *
 * @param <V> The concrete type of the AutogradValue being accumulated.
 * @param <A> The concrete type of this AccumulatorNode.
 */
public interface AccumulatorNode<V, A extends AccumulatorNode<V, A>> extends Node<V> {

    /**
     * Initialise the value within this AccumulatorNode.
     *
     * @param value The value with which to initialise this Node.
     * @return This Node.
     */
    A setValue(Supplier<V> value);

    /**
     * Add a delta value to the value within this Node.
     *
     * @param delta The delta to add.
     * @param addFunction The add function that performs the addition.
     * @return This node.
     */
    A add_(V delta, BinaryOperator<V> addFunction);

}
