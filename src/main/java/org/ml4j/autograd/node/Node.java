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

import java.util.List;
import java.util.function.Supplier;
import org.ml4j.autograd.BackwardConfig;


/**
 * Represents a node in a compulation graph for an AutogradValue of type V.
 *
 * @author Michael Lavelle
 *
 * @param <V> The concrete type of the AutogradValue that is referenced by this Node.
 */
public interface Node<V> {

    /**
     * Allows access to the AutogradValue referenced by this Node.
     *
     * @return A supplier of the AutogradValue referenced by this Node.
     */
    Supplier<V> getValue();

    /**
     * Performs backpropagation on this Node.
     *
     * @param config The BackwardConfig that controls the behaviour of back propagation.
     */
    void backward(BackwardConfig config);

    /**
     * The previous nodes in the computation graph.
     *
     * @return A list of all of the immediate Nodes that caused this Node to be created - used
     *     by the backward method to chain backpropagation to previous nodes in the computation graph.
     */
    List<Node<?>> prev();


}
