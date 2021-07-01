package org.ml4j.autograd.demo.scalar;

import org.ml4j.autograd.demo.DemoAutogradValue;
import org.ml4j.autograd.demo.DemoAutogradValueFactory;
import org.ml4j.autograd.demo.DemoSize;
import org.ml4j.autograd.impl.AutogradValueProperties;

import java.util.function.Supplier;

public class DemoFloatAutogradValueFactoryImpl implements DemoAutogradValueFactory<Float> {

	@Override
	public DemoAutogradValue<Float> create(Supplier<Float> data,
			DemoSize size) {
		DemoFloatAutogradValueImpl value = new DemoFloatAutogradValueImpl(new AutogradValueProperties<DemoSize>().setContext(size),data);
		value.data_(data);
		return value;
	}

}
