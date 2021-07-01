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
import org.ml4j.autograd.impl.AutogradValueImpl;
import org.ml4j.autograd.impl.AutogradValueProperties;

import java.util.function.Supplier;

/**
 * An AutogradValue implementation that supports the operations defined by DemoOperations, 
 * and that takes advantage of the fact that the wrapped data also implements DemoOperations
 * by implementing default DifferentiableWrappedDemoOperations methods. 
 * 
 * @author Michael Lavelle
 */
public class DemoFloatOperationsAutogradValueImpl extends AutogradValueImpl<DemoAutogradValue<DemoFloatOperations>, DemoFloatOperations, DemoSize> implements AutogradValue<DemoAutogradValue<DemoFloatOperations>, DemoFloatOperations, DemoSize>, DifferentiableWrappedDemoOperations<DemoAutogradValue<DemoFloatOperations>, DemoFloatOperations, DemoSize>, DemoAutogradValue<DemoFloatOperations> {
	
	public DemoFloatOperationsAutogradValueImpl(AutogradValueProperties<DemoSize> properties, Supplier<DemoFloatOperations> data) {
		super(properties, data);
	}
	
	public DemoFloatOperationsAutogradValueImpl(AutogradValueProperties<DemoSize> properties, float data) {
		this(properties, () -> new DemoFloatOperations(data, properties.getContext()));
	}

	@Override
	protected void close(DemoFloatOperations data) {
		data.setValue(-1);
	}

	@Override
	public DemoSize size() {
		return context();
	}

	@Override
	protected DemoAutogradValue<DemoFloatOperations> createAutogradValue(Supplier<DemoFloatOperations> data, AutogradValueProperties<DemoSize> properties) {
		return new DemoFloatOperationsAutogradValueImpl(properties, data);
	}

	@Override
	protected DemoAutogradValue<DemoFloatOperations> getInitialInstance() {
		return this;
	}

	@Override
	protected Supplier<DemoFloatOperations> multiplicativeIdentity() {
		return () -> new DemoFloatOperations(1, size());
	}

	@Override
	protected Supplier<DemoFloatOperations> additiveIdentity() {
		return () -> new DemoFloatOperations(0, size());
	}	
}
