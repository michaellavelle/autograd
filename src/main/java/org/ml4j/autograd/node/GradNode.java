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

import java.util.Optional;

/**
 * Represents the Node in the computation graph for the gradient of an AutogradValue.
 * Allows gradients to be accumulated.
 *
 * @param <V> The concrete of AutogradValue referenced by this Node.
 */
public interface GradNode<V> extends AccumulatorNode<V, GradNode<V>> {

    /**
     * A hook to obtain an optional natively generated gradient from the AutogradValue referenced by this Node.
     *
     * @return
     */
    Optional<V> native_grad();

    /**
     * Whether notively generated gradients are disabled for this Node.
     * @return Whether notively generated gradients are disabled for this Node.
     */
    boolean isDisableNativeGradient();

    /**
     * Allow native gradient calculation to be disabled for this Node.
     *
     * @param disableNativeGradient Whether to disable native gradient calculation.
     */
    void setDisableNativeGradient(boolean disableNativeGradient);

}
