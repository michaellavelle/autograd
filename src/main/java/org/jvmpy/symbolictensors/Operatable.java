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
package org.jvmpy.symbolictensors;

public interface Operatable<T, S, O> {

    /**
     * Perform an inline operation on the underlying tensor,
     * potentially lazily.
     *
     * @param operation The operation to perform.
     */
    void performInlineOperation(Operation<T, S> operation);

    /**
     * Perform an operation
     *
     * @param newTensorName
     * @param operation
     * @return
     */
    O performUnaryMappingOperation(Operation<T, S> operation);
}
