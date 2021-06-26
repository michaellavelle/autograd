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
 * Represents a Value of type V that wraps some data of type D, with a Pytorch-like API.
 *
 * @param <V> The concrete type of this Value.
 * @param <D> The type of data wrapped by this Value, eg. Float, Matrix, Tensor
 * @param <C> The type of context required for this Value, eg. Size,
 *
 * @author Michael Lavelle
 */
public interface Value<V, D, C> extends DataSupplier<D> {

    /**
     * A supplier of data.
     */
    Supplier<D> data();

    /**
     * Sets the supplier of data on this Value.
     *
     * @param data The supplier of data.
     * @return This Value.
     */
    V data_(Supplier<D> data);

    /**
     * Returns the name of this Value, or null if no name defined.
     *
     * @return The name of this Value, or null if no name defined.
     */
    String name();

    /**
     * Set the name of this Value.
     *
     * @param name The name of this Value.
     * @return This Value.
     */
    V name_(String name);

    /**
     * Returns the value itself.
     *
     * @return This Value.
     */
    V self();

    /**
     * Return the context of the data wrapped by this value, eg. a size attribute.
     *
     * @return The context.
     */
    C context();

    boolean create_graph();

}
