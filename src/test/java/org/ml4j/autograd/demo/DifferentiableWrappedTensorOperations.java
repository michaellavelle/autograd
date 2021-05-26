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

package org.ml4j.autograd.demo;

import org.jvmtorch.torch.TensorOperations;
import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.arithmetic.operations.DifferentiableWrappedArithmeticOperations;

public interface DifferentiableWrappedTensorOperations<V extends TensorOperations<V>, D extends TensorOperations<D>, C> extends DifferentiableWrappedArithmeticOperations<V, D, C>, AutogradValue<V, D, C>, TensorOperations<V> {

	@Override
	default V relu() {
        return applyUnaryOperator(D::relu, (g, v) -> g.mul(v.gt(0)), "gt", s -> s);
	}
	
	@Override
	default V sigmoid() {
        return applyUnaryOperator(D::sigmoid, (g, v) -> g.mul(sigGrad(v.getDataAsFloatArray()[0])), "gt", s -> s);
	}
	
	@Override
	default V bernoulli() {
        return applyUnaryOperator(D::bernoulli, (g, v) -> g, "gt", s -> s);
	}
	
	 private float sig(float x) {
	    	return 1f / (1f + (float)Math.exp(-x));
	    }
	    
	    private float sigGrad(float x) {
	    	float s = sig(x);
	    	return s * ( 1 - s);
	    }

}
