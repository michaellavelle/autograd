package org.jvmpy.symbolictensors;

import org.jvmtorch.torch.Size;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

public class SymbolicTensorAdapter<T extends TensorDataContainer, S> implements SymbolicTensor<T, S>, TensorDimensionsContainer<S> {

	private T tensorData;
	private S size;
	private List<Operation<T, S>> outstandingOperations;
	
	private SymbolicTensorAdapter(T tensorData, List<Operation<T, S>> outstandingOperations) {
		this.tensorData = tensorData;
		this.outstandingOperations = outstandingOperations;
	}
	
	public SymbolicTensorAdapter(T tensorData, S size) {
		this.size = size;
		this.tensorData = tensorData;
		this.outstandingOperations = new ArrayList<>();
	}
	
	@Override
	public T evaluate() {
		T evaluation = tensorData;
		if (outstandingOperations.size() == 2 && outstandingOperations.get(0).name().equals("T") && outstandingOperations.get(0).name().equals("T")) {
			outstandingOperations.clear();
		}
		for (Operation<T, S> outstandingOperation : outstandingOperations) {
			evaluation = outstandingOperation.apply(evaluation);
		}
		outstandingOperations.clear();
		this.tensorData  = evaluation;
		return evaluation;
	}

	@Override
	public void performInlineOperation(Operation<T, S> operation) {
		operation.apply(evaluate());
		this.size = operation.dimensionsMapping().apply(this.size);
	}

	@Override
	public SymbolicTensor<T, S> performUnaryMappingOperation(Operation<T, S> operation) {
		if (operation.name().equals("T")) {
			T result = tensorData;
			List<Operation<T, S>> newOutstanding = new ArrayList<>();
			newOutstanding.addAll(this.outstandingOperations);
			newOutstanding.add(operation);
			SymbolicTensorAdapter<T, S> adapter = new SymbolicTensorAdapter<>(result, newOutstanding);
			adapter.size = operation.dimensionsMapping().apply(this.size);
			return adapter;
		
		} else {
			T result = operation.apply(evaluate());
			SymbolicTensorAdapter<T, S> adapter = new SymbolicTensorAdapter<>(result, outstandingOperations);
			adapter.size = operation.dimensionsMapping().apply(this.size);
			return adapter;
		}
	}
	
	@Override
	public S size() {
		return size;
	}

	@Override
	public List<String> getOperationNames() {
		return new ArrayList<>();
	}

	@Override
	public List<String> getAllOperationNames() {
		return new ArrayList<>();	}

	@Override
	public List<Operation<T, S>> getOperations() {
		return new ArrayList<>();
	}

	@Override
	public SymbolicTensor<T, S> detach(String tensorName) {
		return this;
	}

	@Override
	public Supplier<T> getInput() {
		return this;
	}

	@Override
	public SymbolicTensor<T, S> size_(Size size) {
		return this;
	}


}
