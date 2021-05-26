package org.ml4j.autograd.demo.providertensor;

import java.util.List;
import java.util.function.Supplier;

import org.ml4j.autograd.node.Node;

public class DJLTensorImpl extends ProviderTensorImpl<DJLTensor, DJLTensorOperations> implements DJLTensor {

	public DJLTensorImpl(Supplier<DJLTensorOperations> data, Size size, List<Node<?>> children) {
		super(data, size, children);
	}

	@Override
	public DJLTensor getInitialInstance() {
		return this;
	}

	@Override
	protected DJLTensor createAutogradValue(Supplier<DJLTensorOperations> data, Size size, List<Node<?>> children) {
		return new DJLTensorImpl(data, size, children);
	}

	@Override
	protected Supplier<DJLTensorOperations> multiplicativeIdentity() {
		return () -> new DJLTensorOperations(1f, size());
	}

	@Override
	protected Supplier<DJLTensorOperations> additiveIdentity() {
		return () -> new DJLTensorOperations(0f, size());
	}
}
