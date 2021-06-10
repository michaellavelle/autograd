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

import java.util.function.BiConsumer;
import org.ml4j.autograd.BackwardConfig;

/**
 * Represents a Node wrapping a (non-accumulating) AutogradValue.
 *
 * @author Michael Lavelle
 *
 * @param <V> The concret type of the AutogradValue referenced by this Node.
 */
public interface ValueNode<V> extends Node<V> {

    /**
     * Set the backward function on this ValueNode, to be executed on backpropagation.
     *
     *
     * @param backwardFunction A backward function which consumes the AutogradValue referenced by this node
     *                         and the BackwardConfig specified during back propagation.  An example backward function
     *                          may pass the gradient of the value back to previous nodes in the compulation graph.s
     */
    void setBackwardFunction(BiConsumer<V, BackwardConfig> backwardFunction);

}
