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

import java.util.function.UnaryOperator;

public class OperationImpl<T, S> implements Operation<T, S> {

    private UnaryOperator<T> operation;
    private UnaryOperator<S> dimensionsMapping;
    private String name;

    public OperationImpl(String name, UnaryOperator<T> operation, UnaryOperator<S> dimensionsMapping) {
        this.operation = operation;
        this.name = name;
        this.dimensionsMapping = dimensionsMapping;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public T apply(T t) {
        return operation.apply(t);
    }



	@Override
	public UnaryOperator<S> dimensionsMapping() {
		return dimensionsMapping;
	}

}
