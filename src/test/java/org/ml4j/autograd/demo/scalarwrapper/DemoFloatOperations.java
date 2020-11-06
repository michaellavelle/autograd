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

import org.ml4j.autograd.demo.DemoOperations;
import org.ml4j.autograd.demo.DemoSize;

public class DemoFloatOperations implements DemoOperations<DemoFloatOperations> {
	
	private float value;
	private DemoSize size;
	
	public DemoFloatOperations(float value, DemoSize size) {
		this.value = value;
		this.size = size;
	}
	
	public float getValue() {
		return value;
	}
	
	protected DemoFloatOperations create(float value) {
		return new DemoFloatOperations(value, size);
	}
	
	@Override
	public float[] getDataAsFloatArray() {
		return new float[] {value};
	}

	@Override
	public DemoFloatOperations add(DemoFloatOperations other) {
		return create(value + other.value);
	}

	@Override
	public DemoFloatOperations add(float other) {
		return create(value + other);
	}

	@Override
	public DemoFloatOperations sub(DemoFloatOperations other) {
		return create(value - other.value);
	}

	@Override
	public DemoFloatOperations sub(float other) {
		return create(value - other);
	}

	@Override
	public DemoFloatOperations mul(DemoFloatOperations other) {
		return create(value * other.value);
	}

	@Override
	public DemoFloatOperations mul(float other) {
		return create(value * other);
	}

	@Override
	public DemoFloatOperations div(DemoFloatOperations other) {
		return create(value / other.value);
	}

	@Override
	public DemoFloatOperations div(float other) {
		return create(value / other);
	}

	@Override
	public DemoFloatOperations add_(DemoFloatOperations other) {
		value = value + other.value;
		return this;
	}

	@Override
	public DemoFloatOperations sub_(DemoFloatOperations other) {
		value = value - other.value;
		return this;
	}

	@Override
	public DemoFloatOperations neg() {
		return create(-value);
	}

	@Override
	public DemoFloatOperations gt(float value) {
		return create(this.value > value ? 1f : 0f);
	}

	@Override
	public DemoFloatOperations gte(float value) {
		return create(this.value >= value ? 1f : 0f);
	}

	@Override
	public DemoSize size() {
		return size;
	}

	@Override
	public DemoFloatOperations relu() {
		return create(value < 0 ? 0f : 1f);
	}

}
