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

import org.jvmtorch.torch.Size;

import java.util.List;
import java.util.function.Supplier;

/**
 * A SymbolicTensor allows an underlying TensorDataContainer of type T and size of type S to be
 * manipulated in mathematical expressions, and for gradients
 * to be calculated.
 * 
 * SymbolicTensor instances can be lazy-evaulated - for this use
 * case the evaluate()/get() methods evaluate the value of the underlying
 * tensor data.
 * 
 * @author Michael Lavelle
 * 
 * @param <T> The type of tensor to wrap inside the SymbolicTensor
 */
public interface SymbolicTensor<T extends TensorDataContainer, S> extends Supplier<T>, Operatable<T, S, SymbolicTensor<T, S>> {

	/**
	 * @return The current evaluation of this tensor.
	 */
	T evaluate();

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
	SymbolicTensor<T, S> performUnaryMappingOperation(Operation<T, S> operation);
			
	List<String> getOperationNames();

	List<String> getAllOperationNames();
		
	List<Operation<T, S>> getOperations();

	SymbolicTensor<T, S> detach(String tensorName);
	
	Supplier<T> getInput();
	
	SymbolicTensor<T, S> size_(Size size);
	
	S size();
	
	@Override
	default T get() {
		return evaluate();
	}

}
