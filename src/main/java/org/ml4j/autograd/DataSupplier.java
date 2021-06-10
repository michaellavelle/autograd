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
 * Specifies the signature of the data-accessor method for a Value.
 * 
 * @author Michael Lavelle
 *
 * @param <D> The type of data being wrapped by this data supplier.
 */
public interface DataSupplier<D> {

    /**
     * Returns a supplier of data, allowing for lazy, or cached data retrieval.
     *
     * @return A supplier of data.
     */
    Supplier<D> data();
}
