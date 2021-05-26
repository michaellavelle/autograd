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
package org.jvmtorch.torch;

public class ScalarOperations implements TensorOperations<ScalarOperations> {

	private float value;
	private Torch torch;
	
	public ScalarOperations(Torch torch, float value) {
		this.value = value;
		this.torch = torch;
	}
	
	@Override
	public ScalarOperations get() {
		return this;
	}

	@Override
	public float[] getDataAsFloatArray() {
		return new float[] {value};
	}

	@Override
	public ScalarOperations mul(float otherValue) {
		return new ScalarOperations(torch, value * otherValue);
	}
	
	public ScalarOperations div(float otherValue) {
		return new ScalarOperations(torch, value / otherValue);
	}

	@Override
	public ScalarOperations add(float otherValue) {
		return new ScalarOperations(torch, value + otherValue);
	}

	@Override
	public ScalarOperations mul(ScalarOperations other) {
		return mul(other.value);
	}
	
	@Override
	public ScalarOperations div(ScalarOperations other) {
		return div(other.value);
	}

	@Override
	public int numel() {
		return 1;
	}

	@Override
	public ScalarOperations add(ScalarOperations other) {
		return add(other.value);
	}

	@Override
	public ScalarOperations sub(float otherValue) {
		return new ScalarOperations(torch, value - otherValue);
	}

	@Override
	public ScalarOperations sub(ScalarOperations other) {
		return sub(other.value);
	}

	@Override
	public ScalarOperations mean() {
		return new ScalarOperations(torch, value);
	}

	@Override
	public ScalarOperations mul_(ScalarOperations other) {
		value = value * other.value;
		return this;
	}

	@Override
	public ScalarOperations fill_(float value) {
		value = value;
		return this;
	}

	@Override
	public ScalarOperations zero_() {
		value = 0;
		return this;
	}

	@Override
	public ScalarOperations normal_(float v1, float v2) {
		// TODO
		return this;
	}


	@Override
	public ScalarOperations sub_(ScalarOperations other) {
		value = value - other.value;
		return this;
	}

	@Override
	public ScalarOperations neg() {
		return null;
	}

	@Override
	public ScalarOperations gt(float value) {
		return null;
	}

	@Override
	public ScalarOperations gte(float value) {
		return null;
	}

	@Override
	public ScalarOperations add_(ScalarOperations other) {
		value = value + other.value;
		return this;
	}

	@Override
	public ScalarOperations matmul(ScalarOperations other) {
		throw new UnsupportedOperationException("Not yet implemented");
	}

	@Override
	public ScalarOperations t() {
		return new ScalarOperations(torch, value);
	}

	@Override
	public Size size() {
		return torch.Size();
	}

	@Override
	public int size(int dim) {
		return 0;
	}

	@Override
	public ScalarOperations size_(Size size) {
		throw new UnsupportedOperationException("Not yet implemented");
	}

	@Override
	public ScalarOperations view(Size size) {
		if (size.dimensions().length == 0) {
			return new ScalarOperations(torch, value);
		} else {
			throw new IllegalArgumentException("Size is not compatible");
		}
	}

	@Override
	public ScalarOperations view(int... dims) {
		// TODO
		return this;
	}

	@Override
	public String toString() {
		return Float.valueOf(value).toString();
	}

	@Override
	public ScalarOperations sum() {
		return new ScalarOperations(torch, value);
	}

	@Override
	public ScalarOperations columnSums() {
		throw new UnsupportedOperationException("Not yet implemented");
	}
	
	@Override
	public ScalarOperations rowSums() {
		throw new UnsupportedOperationException("Not yet implemented");
	}

	@Override
	public void close() {
		//this.torch = null;
	}

	@Override
	public ScalarOperations relu() {
		return null;
	}

	@Override
	public ScalarOperations bernoulli() {
		return null;
	}

	@Override
	public ScalarOperations sigmoid() {
		return null;
	}

	@Override
	public ScalarOperations norm() {
		throw new UnsupportedOperationException("Not yet implemented");
	}

	@Override
	public ScalarOperations cloneTensor() {
		throw new UnsupportedOperationException("Not yet implemented");
	}

}
