package org.ml4j.autograd.demo.providertensor;

import java.util.ArrayList;
import java.util.function.Supplier;

public class DJLTensorFactoryImpl implements DJLTensorFactory {

	@Override
	public DJLTensor create(Supplier<DJLTensorOperations> data,
			Size size) {
		DJLTensorImpl value = new DJLTensorImpl(data, size, new ArrayList<>());
		value.data_(data);
		return value;
	}

}
