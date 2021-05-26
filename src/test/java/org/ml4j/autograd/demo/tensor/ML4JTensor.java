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

package org.ml4j.autograd.demo.tensor;

import org.jvmtorch.torch.Size;
import org.jvmtorch.torch.Tensor;
import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.demo.*;
import org.ml4j.autograd.impl.AutogradValueImpl;
import org.ml4j.autograd.node.Node;
import org.ml4j.nn.components.DirectedComponentsContext;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

/**
 * An AutogradValue implementation that supports the operations defined by DemoOperations, 
 * and that takes advantage of the fact that the wrapped data also implements DemoOperations
 * by implementing default DifferentiableWrappedDemoOperations methods. 
 * 
 * @author Michael Lavelle
 */
public class ML4JTensor extends AutogradValueImpl<Tensor<ML4JTensorOperations>, ML4JTensorOperations, Size> implements AutogradValue<Tensor<ML4JTensorOperations>, ML4JTensorOperations, Size>, DifferentiableWrappedTensorOperations<Tensor<ML4JTensorOperations>, ML4JTensorOperations, Size>, org.jvmtorch.torch.TensorOperations<Tensor<ML4JTensorOperations>>, org.ml4j.autograd.DataSupplier<ML4JTensorOperations>, Tensor<ML4JTensorOperations> {

	private DirectedComponentsContext context;

	public ML4JTensor(DirectedComponentsContext context, Supplier<ML4JTensorOperations> data, Size size) {
		this(context, data, size, new ArrayList<>());
	}

	public ML4JTensor(DirectedComponentsContext context, float data, Size size) {
		this(context, () -> new ML4JTensorOperationsImpl(context, data, size), size, new ArrayList<>());
	}

	protected ML4JTensor(DirectedComponentsContext context, Supplier<ML4JTensorOperations> data, Size size, List<Node<?>> children) {
		super(data, size, children);
		this.context = context;
	}

	@Override
	public int numel() {
		return size().numel();
	}

	@Override
	public Tensor<ML4JTensorOperations> sum() {
		return null;
	}

	@Override
	public Tensor<ML4JTensorOperations> mean() {
		return null;
	}

	@Override
	public Tensor<ML4JTensorOperations> norm() {
		return null;
	}

	@Override
	public Tensor<ML4JTensorOperations> mul_(Tensor<ML4JTensorOperations> other) {
		return null;
	}

	@Override
	public Tensor<ML4JTensorOperations> columnSums() {
		return null;
	}

	@Override
	public Tensor<ML4JTensorOperations> rowSums() {
		return null;
	}

	@Override
	public Tensor<ML4JTensorOperations> cloneTensor() {
		return null;
	}

	@Override
	public Tensor<ML4JTensorOperations> matmul(Tensor<ML4JTensorOperations> other) {
		return null;
	}

	@Override
	public Tensor<ML4JTensorOperations> t() {
		return applyUnaryOperator((f, s) -> f.t(), 0, (g, p) -> g, "t", f -> f.t());
	}

	@Override
	public Size size() {
		return context();
	}

	@Override
	public int size(int dim) {
		return size().get(dim);
	}
	@Override
	public Tensor<ML4JTensorOperations> size_(Size size) {
		return null;
	}

	@Override
	public Tensor<ML4JTensorOperations> zero_() {
		return null;
	}

	@Override
	public Tensor<ML4JTensorOperations> normal_(float v1, float v2) {
		return null;
	}

	@Override
	public Tensor<ML4JTensorOperations> fill_(float value) {
		return null;
	}

	@Override
	public Tensor<ML4JTensorOperations> view(Size size) {
		return null;
	}

	@Override
	public Tensor<ML4JTensorOperations> view(int... dims) {
		return null;
	}

	@Override
	public void close() {

	}

	@Override
	protected Tensor<ML4JTensorOperations> createAutogradValue(Supplier<ML4JTensorOperations> data, Size size, List<Node<?>> children) {
		return new ML4JTensor(context, data, size, children);
	}

	@Override
	protected Tensor<ML4JTensorOperations> getInitialInstance() {
		return this;
	}

	@Override
	protected Supplier<ML4JTensorOperations> multiplicativeIdentity() {
		return () -> new ML4JTensorOperationsImpl(context, 1, size());
	}

	@Override
	protected Supplier<ML4JTensorOperations> additiveIdentity() {
		return () -> new ML4JTensorOperationsImpl(context, 0, size());
	}

	@Override
	public Tensor<ML4JTensorOperations> add(Tensor<ML4JTensorOperations> other) {
		return applyBinaryOperator(other, (f, s) -> f.add(s), (g, p) -> g, (g, p) -> g, "add", (f, s) -> f);
	}

	@Override
	public Tensor<ML4JTensorOperations> get() {
		return this;
	}
}
