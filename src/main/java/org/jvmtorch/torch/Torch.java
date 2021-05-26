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


import java.util.List;

public interface Torch {

	public static int LONG = 1;
	
	/**
	 * Create a new normally distributed random Tensor.
	 * 
	 * @param size The size of the tensor.
	 * @return a new normally distributed random Tensor.
	 */
	Tensor randn(Size size);
	

	/**
	 * Create a new normally distributed random Tensor.
	 * 
	 * @param dimensions The dimensions of the tensor.
	 * @return a new normally distributed random Tensor.
	 */
	Tensor randn(int... dimensions);

	/**
	 * Create a new normally distributed random Tensor.
	 *
	 * @param values The values of the tensor.
	 * @return a new normally distributed random Tensor.
	 */
	Tensor tensor(List<Long> values);
	
	/**
	 * Create a new uniformly distributed random Tensor.
	 * 
	 * @param size The size of the tensor.
	 * @return a new uniformly distributed random Tensor.
	 */
	Tensor rand(Size size);
	
	/**
	 * Create a new uniformly distributed random Tensor.
	 * 
	 * @param dimensions The dimensions of the tensor.
	 * @return a new uniformly distributed random Tensor.
	 */
	Tensor rand(int... dimensions);
	
	/**
	 * Create a new uninitialised Tensor.
	 * @param size The size of the tensor.
	 * @return a new uninitialised Tensor.
	 */
	Tensor empty(Size size);
	
	/**
	 * Create a new uninitialised Tensor.
	 * @param dimensions The dimensions of the tensor.
	 * @return a new uninitialised Tensor.
	 */
	Tensor empty(int... dimensions);
	
	/**
	 * Create a new Tensor of zeros.
	 * @param size The size of the tensor.
	 * @return a new Tensor of zeros.
	 */
	Tensor zeros(Size size);
	
	/**
	 * Create a new Tensor of zeros.
	 * @param dimensions The size of the tensor.
	 * @return a new Tensor of zeros.
	 */
	Tensor zeros(int... dimensions);
	
	/**
	 * Create a new Tensor of ones.
     *
	 * @param size The size of the tensor.
	 * @return a new Tensor of ones.
	 */
	Tensor ones(Size size);
	
	/**
	 * Create a new Tensor of ones.
     *
	 * @param dimensions The size of the tensor.
	 * @return a new Tensor of ones.
	 */
	Tensor ones(int... dimensions);
	
	/**
	 * Create a new Tensor from the provided size
	 * and data.
	 * 
	 * @param data The data within the tensor.
	 * @param size The size of the tensor.
	 * @return a new Tensor from the provided size
	 * and data.
	 */
	Tensor tensor(float[] data, Size size);
	
	/**
	 * Create a new Tensor from the provided size
	 * and data.
	 * 
	 * @param data The data within the tensor.
	 * @param dimensions The dimensions of the tensor.
	 * @return a new Tensor from the provided size
	 * and data.
	 */
	Tensor tensor(float[] data, int... dimensions);
	
	/**
	 * Create a new scalar Tensor from the 
	 * provided value.
	 * 
	 * @param value The scalar value.
	 * @return  a new scalar Tensor from the 
	 * provided value.
	 */
	Tensor tensor(float value);
	
	/**
	 * Create a new Tensor from the provided
	 * TensorData instance.
	 * 
	 * @param tensorData The tensor data.
	 * @return a new Tensor from the provided
	 * TensorData instance.
	 */
	Tensor tensor(TensorData tensorData);
	
	/**
	 * Create a new Size instance from the
	 * provided dimension sizes.
	 * 
	 * @param dimensionSizes The dimension sizes.
	 * @return a new Size instance from the
	 * provided dimension sizes.
	 */
	default Size Size(int... dimensionSizes) {
		return new Size(dimensionSizes);
	}
	
	
	default Size Size() {
		return new Size();
	}
	
	/**
	 * Create a new Size instance from the
	 * provided Sizes containing dimension
	 * sizes.  Allows dimensions to be grouped
	 * to allow for matrix representation.
	 * 
	 * @param sizes The sizes containing
	 * the dimension sizes.
	 * @return new Size instance from the
	 * provided Sizes
	 */
	default Size Size(Size... sizes) {
		return new Size(sizes);
	}
	
	Tensor add(Tensor first, Tensor second);
	
	Tensor add(Tensor first, float value);

	Tensor mul(Tensor first, Tensor second);
	
	Tensor mul(Tensor first, float value);


}
