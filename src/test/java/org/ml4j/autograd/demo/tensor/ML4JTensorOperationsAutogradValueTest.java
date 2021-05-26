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

package org.ml4j.autograd.demo.tensor;

import org.junit.Assert;
import org.junit.jupiter.api.Assertions;
import org.jvmtorch.torch.Tensor;
import org.jvmtorch.torch.Torch;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.autograd.demo.TensorAutogradValueTestBase;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.DirectedComponentsContextImpl;

public class ML4JTensorOperationsAutogradValueTest extends TensorAutogradValueTestBase<ML4JTensorOperations> {

	private static MatrixFactory matrixFactory = new JBlasRowMajorMatrixFactory();

	private static DirectedComponentsContext context =  new DirectedComponentsContextImpl(matrixFactory, true);

	@Override
	protected Tensor<ML4JTensorOperations> createGradValue(float value, boolean requires_grad) {
        return new ML4JTensor(context, () -> createData(value), size).requires_grad_(requires_grad);
	}

	@Override
	protected Tensor<ML4JTensorOperations> createGradValue(ML4JTensorOperations value, boolean requires_grad) {
        return new ML4JTensor(context, () -> value, size).requires_grad_(requires_grad);
	}

	@Override
	protected ML4JTensorOperations createData(float value) {
		return new ML4JTensorOperationsImpl(context, value, size);
	}

	@Override
	protected void assertEquals(ML4JTensorOperations value1, ML4JTensorOperations value2) {
		Matrix m1 = value1.getMatrix();
		Matrix m2 = value2.getMatrix();
		Assert.assertEquals(m1.getLength(), m2.getLength());
		for (int i = 0; i < m1.getLength(); i++) {
			Assert.assertEquals(m1.get(i), m2.get(i), 0.01f);
		}
	}

	@Override
	protected ML4JTensorOperations add(ML4JTensorOperations value1, ML4JTensorOperations value2) {
		return value1.add(value2);
	}

	@Override
	protected ML4JTensorOperations mul(ML4JTensorOperations value1, float value2) {
		return value1.mul(value2);
	}

}
