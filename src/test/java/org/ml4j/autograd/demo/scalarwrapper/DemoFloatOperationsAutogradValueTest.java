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

package org.ml4j.autograd.demo.scalarwrapper;

import org.junit.jupiter.api.Assertions;
import org.ml4j.autograd.demo.DemoAutogradValue;
import org.ml4j.autograd.demo.DemoAutogradValueTestBase;

public class DemoFloatOperationsAutogradValueTest extends DemoAutogradValueTestBase<DemoFloatOperations> {

	@Override
	protected DemoAutogradValue<DemoFloatOperations> createGradValue(float value, boolean requires_grad) {
        return new DemoFloatOperationsAutogradValueImpl(() -> createData(value), size, requires_grad, false);
	}

	@Override
	protected DemoAutogradValue<DemoFloatOperations> createGradValue(DemoFloatOperations value, boolean requires_grad) {
        return new DemoFloatOperationsAutogradValueImpl(() -> value, size, requires_grad, false);
	}

	@Override
	protected DemoFloatOperations createData(float value) {
		return new DemoFloatOperations(value, size);
	}

	@Override
	protected void assertEquals(DemoFloatOperations value1, DemoFloatOperations value2) {
		Assertions.assertEquals(value1.getValue(), value2.getValue(), 0.01f);
	}

	@Override
	protected DemoFloatOperations add(DemoFloatOperations value1, DemoFloatOperations value2) {
		return value1.add(value2);
	}

	@Override
	protected DemoFloatOperations mul(DemoFloatOperations value1, float value2) {
		return value1.mul(value2);
	}

}
