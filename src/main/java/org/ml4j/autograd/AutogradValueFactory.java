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

import java.util.function.Supplier;

/**
 * Represents a factory for AutogradValues.
 *
 * @param <V> The concrete type of the Values created by this factory..
 * @param <D> The type of data wrapped by the Values, eg. Float, Matrix, Tensor
 * @param <C> The type of context required for these Value, eg. Size,
 * 
 * @author Michael Lavelle
 */
public interface AutogradValueFactory<V, D, C> {

    /**
     * Create an AutogradValue.
     *
     * @param data    A supplier of data for the value to be created.
     * @param context The context for the value to be created.
     * @return A new AutogradValue
     */
    V create(Supplier<D> data, C context);

}
