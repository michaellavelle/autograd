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

package org.ml4j.autograd.demo.providertensor;

public class DJLTensorOperations implements TensorOperations<DJLTensorOperations> {
	
	private float value;
	private Size size;
	
	public DJLTensorOperations(float value, Size size) {
		this.value = value;
		this.size = size;
	}
	
	public float getValue() {
		return value;
	}
	
	protected DJLTensorOperations create(float value) {
		return new DJLTensorOperations(value, size);
	}
	
	@Override
	public float[] getDataAsFloatArray() {
		return new float[] {value};
	}

	@Override
	public DJLTensorOperations add(DJLTensorOperations other) {
		return create(value + other.value);
	}

	@Override
	public DJLTensorOperations add(float other) {
		return create(value + other);
	}

	@Override
	public DJLTensorOperations sub(DJLTensorOperations other) {
		return create(value - other.value);
	}

	@Override
	public DJLTensorOperations sub(float other) {
		return create(value - other);
	}

	@Override
	public DJLTensorOperations mul(DJLTensorOperations other) {
		return create(value * other.value);
	}

	@Override
	public DJLTensorOperations mul(float other) {
		return create(value * other);
	}

	@Override
	public DJLTensorOperations div(DJLTensorOperations other) {
		return create(value / other.value);
	}

	@Override
	public DJLTensorOperations div(float other) {
		return create(value / other);
	}

	@Override
	public DJLTensorOperations add_(DJLTensorOperations other) {
		value = value + other.value;
		return this;
	}

	@Override
	public DJLTensorOperations sub_(DJLTensorOperations other) {
		value = value - other.value;
		return this;
	}

	@Override
	public DJLTensorOperations neg() {
		return create(-value);
	}

	@Override
	public DJLTensorOperations gt(float value) {
		return create(this.value > value ? 1f : 0f);
	}

	@Override
	public DJLTensorOperations gte(float value) {
		return create(this.value >= value ? 1f : 0f);
	}

	@Override
	public Size size() {
		return size;
	}

	@Override
	public DJLTensorOperations relu() {
		return create(value < 0 ? 0f : 1f);
	}

}
