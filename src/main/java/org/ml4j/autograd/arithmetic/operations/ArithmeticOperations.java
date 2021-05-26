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

package org.ml4j.autograd.arithmetic.operations;

import org.ml4j.autograd.Accumulatable;

/**
 * The API for classes implementing ArithmeticOperations.
 *
 * @param <T> The concrete type of the class implementing these
 *            ArithmeticOperations.
 *
 * @author Michael Lavelle
 */
public interface ArithmeticOperations<V> extends Accumulatable<V> {

    float[] getDataAsFloatArray();

    //V add(V other);
    
    V add(float other);

    V sub(V other);
    
    V sub(float other);

    V mul(V other);
    
    V mul(float other);

    V div(V other);

    V div(float other);

    V add_(V other);

    V sub_(V other);

    V neg();

    V gt(float value);

    V gte(float value);

}
