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

import java.util.Random;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.demo.DemoAutogradValue;
import org.ml4j.autograd.demo.DemoAutogradValueTestBase;

public class DemoFloatAutogradValueTest extends DemoAutogradValueTestBase<Float> {

	@Override
	protected DemoAutogradValue<Float> createGradValue(float value, boolean requires_grad) {
        return new DemoFloatAutogradValueImpl(() -> value, size).requires_grad_(requires_grad);
	}

	@Override
	protected DemoAutogradValue<Float> createGradValue(Float value, boolean requires_grad) {
        return new DemoFloatAutogradValueImpl(() -> value, size).requires_grad_(requires_grad);
	}
	
	@Test
	public void blah() {
		
		var weight = createGradValue(0, true);
		var bias = createGradValue(-0.03f, true);
		
		var x = createGradValue(4f, false);
		var x2 = createGradValue(3f, false);

		var target = createGradValue(11f, false);
		var target2 = createGradValue(16f, false);
		
		
		//var y = x.apply(DemoAutogradValue::neg);

		for (int i = 0; i < 900; i++) {
						
			var sig = weight.sigmoid().bernoulli().sub(0.5f).mul(2);

			//System.out.println("Weight:" + weight.getDataAsFloatArray()[0]);
			DemoAutogradValue<Float> y = null;
			DemoAutogradValue<Float> y2 = null;

			if (sig.data().get() == 0) {
				y = bias.relu();
				y2 = bias.relu();

			} else {
				y = x.add(bias).relu();
				y2 = x2.add(bias).relu();

			}
			
			
			var diff = target.sub(y).add(target2.sub(y2)).div(2);
			//System.out.println("Target:" + y.getDataAsFloatArray()[0] + " Loss:" + diff.getDataAsFloatArray()[0]);
			System.out.println(diff.getDataAsFloatArray()[0]);

			diff.backward();
			var weightGrad = weight.grad();
			var biasGrad = bias.grad();
			if (weightGrad != null) {
				weight = weight.sub(weightGrad.mul(0.005f));
			}
			if (biasGrad != null) {
				bias = bias.sub(biasGrad.mul(0.005f));
			}
		}
		
		
		
		
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
