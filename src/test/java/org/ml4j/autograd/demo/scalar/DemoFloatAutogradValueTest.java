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

package org.ml4j.autograd.demo.scalar;

import org.junit.jupiter.api.Assertions;
import org.ml4j.autograd.demo.DemoAutogradValue;
import org.ml4j.autograd.demo.DemoAutogradValueTestBase;

public class DemoFloatAutogradValueTest extends DemoAutogradValueTestBase<Float> {

	@Override
	protected DemoAutogradValue<Float> createGradValue(float value, boolean requires_grad) {
        return gradValueFactory.create(() -> value, size).requires_grad_(requires_grad);
	}

	@Override
	protected DemoAutogradValue<Float> createGradValue(Float value, boolean requires_grad) {
        return gradValueFactory.create(() -> value, size).requires_grad_(requires_grad);
	}

	@Override
	protected Float createData(float value) {
		return value;
	}

	@Override
	protected void assertEquals(Float value1, Float value2) {
		Assertions.assertEquals(value1, value2, 0.01f);
	}

	@Override
	protected Float add(Float value1, Float value2) {
		return value1.floatValue() + value2.floatValue();
	}

	@Override
	protected Float mul(Float value1, float value2) {
		return value1.floatValue() * value2;
	}

}
