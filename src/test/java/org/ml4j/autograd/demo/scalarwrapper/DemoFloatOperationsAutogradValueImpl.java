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

import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.demo.DemoAutogradValue;
import org.ml4j.autograd.demo.DemoSize;
import org.ml4j.autograd.demo.DifferentiableWrappedDemoOperations;
import org.ml4j.autograd.demo.scalar.DemoFloatOperations;
import org.ml4j.autograd.impl.AutogradValueImpl;

/**
 * An AutogradValue implementation that supports the operations defined bt DemoOperations.
 * 
 * @author Michael Lavelle
 */
public class DemoFloatOperationsAutogradValueImpl extends AutogradValueImpl<DemoAutogradValue<DemoFloatOperations>, DemoFloatOperations, DemoSize> implements AutogradValue<DemoAutogradValue<DemoFloatOperations>, DemoFloatOperations, DemoSize>, DifferentiableWrappedDemoOperations<DemoAutogradValue<DemoFloatOperations>, DemoFloatOperations, DemoSize>, DemoAutogradValue<DemoFloatOperations> {

	private DemoSize size;
	
	public DemoFloatOperationsAutogradValueImpl(DemoSize size) {
		this.size = size;
	}
	
	@Override
	public DemoAutogradValue<DemoFloatOperations> self() {
		return this;
	}

	@Override
	public DemoSize size() {
		return size;
	}	
}
