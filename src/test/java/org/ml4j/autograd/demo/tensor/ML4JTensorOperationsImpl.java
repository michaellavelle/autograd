package org.ml4j.autograd.demo.tensor;

import org.jvmpy.symbolictensors.Operatable;
import org.jvmpy.symbolictensors.Operation;
import org.jvmtorch.torch.Size;
import org.jvmtorch.torch.Torch;
import org.ml4j.EditableMatrix;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.neurons.*;
import org.ml4j.nn.neurons.format.ImageNeuronsActivationFormat;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.UnaryOperator;


public class ML4JTensorOperationsImpl implements ML4JTensorOperations, Operatable<ML4JTensorOperations, Size, ML4JTensorOperations> {

	private MatrixFactory matrixFactory;
	private DirectedComponentsContext directedComponentsContext;
	private Matrix matrix;
	private Size size;

	public ML4JTensorOperationsImpl(DirectedComponentsContext directedComponentsContext, Matrix matrix, Size size) {
		this.matrixFactory = directedComponentsContext.getMatrixFactory();
		this.directedComponentsContext = directedComponentsContext;
		this.matrix = matrix;
		this.size = size;
		if (matrix.getRows() == 0 || matrix.getColumns() ==0) {
			throw new IllegalArgumentException(matrix.getRows() + ":" + matrix.getColumns());
		}
	}

	public ML4JTensorOperationsImpl(DirectedComponentsContext directedComponentsContext, float value, Size size) {
		this.matrixFactory = directedComponentsContext.getMatrixFactory();
		this.directedComponentsContext = directedComponentsContext;
		this.matrix = directedComponentsContext.getMatrixFactory().createOnes(size.asMatrixSize().get(0), size.asMatrixSize().get(1)).mul(value);
		this.size = size;
		if (matrix.getRows() == 0 || matrix.getColumns() ==0) {
			throw new IllegalArgumentException(matrix.getRows() + ":" + matrix.getColumns());
		}
	}
	
	public ML4JTensorOperationsImpl(DirectedComponentsContext directedComponentsContext, NeuronsActivation neuronsActivation) {
		this.matrixFactory = directedComponentsContext.getMatrixFactory();
		this.directedComponentsContext = directedComponentsContext;

		this.matrix = neuronsActivation.getActivations(directedComponentsContext.getMatrixFactory());
		if (matrix.getRows() == 0 || matrix.getColumns() ==0) {
			throw new IllegalArgumentException();
		}
		this.size = NeuronsActivationSize.getSize(neuronsActivation);

	}
	
	public Matrix getMatrix() {
		return matrix;
	}
	
	public static ML4JTensorOperations fromNeuronsActivation(DirectedComponentsContext directedComponentsContext, NeuronsActivation neuronsActivation) {
		return new org.ml4j.autograd.demo.tensor.ML4JTensorOperationsImpl(directedComponentsContext, neuronsActivation);
	}
	
	public NeuronsActivation toNeuronsActivation(Neurons neurons) {
		return new NeuronsActivationImpl(neurons, matrix, NeuronsActivationFormat.ROWS_SPAN_FEATURE_SET);
	}
	
	public NeuronsActivation toImageNeuronsActivation(Neurons3D neurons) {
		return new ImageNeuronsActivationImpl(matrix, neurons, ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT, false);
	}
	
	public DirectedComponentsContext getDirectedComponentsContext() {
		return directedComponentsContext;
	}

	private ML4JTensorOperations toML4JTensorOperations(Matrix matrix, Size size) {
		return new ML4JTensorOperationsImpl(directedComponentsContext, matrix, size);
	}
	
	@Override
	public ML4JTensorOperations mul(float value) {
		return toML4JTensorOperations(matrix.mul(value), size);
	}
	
	public ML4JTensorOperations norm() {
		
		EditableMatrix norm = matrix.dup().asEditableMatrix();
		for (int i = 0; i < norm.getLength(); i++) {
			norm.put(i, (float) Math.sqrt(norm.get(i) * norm.get(i)));
		}
		return toML4JTensorOperations(norm, size);
	}
	@Override
	public ML4JTensorOperations add(float value) {
		return toML4JTensorOperations(matrix.add(value), size);
	}
	
	@Override
	public ML4JTensorOperations sub(float value) {
		return toML4JTensorOperations(matrix.sub(value), size);
	}

	@Override
	public ML4JTensorOperations sub_(ML4JTensorOperations other) {
		return apply_(other, m -> m.subiColumnVector(other.getMatrix()),
				m -> m.subiRowVector(other.getMatrix()), m -> m.subi(other.getMatrix()));
	}

	@Override
	public ML4JTensorOperations neg() {
		EditableMatrix r = matrix.dup().asEditableMatrix();
		for (int i = 0; i < r.getLength(); i++) {
			r.put(i, -matrix.get(i));
		}
		return toML4JTensorOperations(r, size());
	}

	@Override
	public ML4JTensorOperations gt(float value) {
		EditableMatrix r = matrix.dup().asEditableMatrix();
		for (int i = 0; i < r.getLength(); i++) {
			if (r.get(i) <= 0) {
				r.put(i, 0);
			} else {
				r.put(i, 1);
			}
		}
		return toML4JTensorOperations(r, size());
	}

	@Override
	public ML4JTensorOperations gte(float value) {
		return null;
	}

	@Override
	public ML4JTensorOperations mul_(ML4JTensorOperations other) {
		return apply_(other, m -> m.muliColumnVector(other.getMatrix()),
				m -> m.muliRowVector(other.getMatrix()), m -> m.muli(other.getMatrix()));
	}

	@Override
	public ML4JTensorOperations fill_(float value) {
		// TODO
		return this;
	}

	@Override
	public ML4JTensorOperations zero_() {
		// TODO
		return this;
	}

	@Override
	public ML4JTensorOperations normal_(float v1, float v2) {
		// TODO
		return this;
	}

	@Override
	public ML4JTensorOperations add_(ML4JTensorOperations other) {
		return apply_(other, m -> m.addiColumnVector(other.getMatrix()),
				m -> m.addiRowVector(other.getMatrix()), m -> m.addi(other.getMatrix()));
	}
	
	private ML4JTensorOperations apply_(ML4JTensorOperations other, Consumer<EditableMatrix> columnVectorOp,
                                        Consumer<EditableMatrix> rowVectorOp, Consumer<EditableMatrix> matrixOp) {
		if (requiresSecondMatrixColumnBroadcast(matrix, other.getMatrix())) {
			columnVectorOp.accept(matrix.asEditableMatrix());
							
		} else if ( requiresSecondMatrixRowsBroadcast(matrix, other.getMatrix())) {
			rowVectorOp.accept(matrix.asEditableMatrix());

		} else {
			matrixOp.accept(matrix.asEditableMatrix());
		}
		return this;
	}

	@Override
	public ML4JTensorOperations matmul(ML4JTensorOperations other) {
	
		return toML4JTensorOperations(matrix.mmul(other.getMatrix()), size().matmul(other.size()));
	}

	@Override
	public ML4JTensorOperations mul(ML4JTensorOperations other) {
		return apply(other, m -> m.mulColumnVector(other.getMatrix()),
				m -> m.mulRowVector(other.getMatrix()), m -> m.mul(other.getMatrix()));
	}

	@Override
	public int numel() {
		return matrix.getLength();
	}

	@Override
	public ML4JTensorOperations add(ML4JTensorOperations other) {
		return apply(other, m -> m.addColumnVector(other.getMatrix()),
				m -> m.addRowVector(other.getMatrix()), m -> m.add(other.getMatrix()));
	}
	
	private ML4JTensorOperations apply(ML4JTensorOperations other, UnaryOperator<Matrix> columnVectorOp,
                                       UnaryOperator<Matrix> rowVectorOp, UnaryOperator<Matrix> matrixOp) {
		
		if (requiresSecondMatrixColumnBroadcast(matrix, other.getMatrix())) {
			return toML4JTensorOperations(columnVectorOp.apply(matrix), size);
					
		} else if ( requiresSecondMatrixRowsBroadcast(matrix, other.getMatrix())) {
			return toML4JTensorOperations(rowVectorOp.apply(matrix), size);

		} else {
			return toML4JTensorOperations(matrixOp.apply(matrix), size);
		}
	}
	
	@Override
	public ML4JTensorOperations div(ML4JTensorOperations other) {
		return apply(other, m -> m.divColumnVector(other.getMatrix()),
				m -> m.divRowVector(other.getMatrix()), m -> m.div(other.getMatrix()));
	}

	@Override
	public ML4JTensorOperations div(float value) {
		return toML4JTensorOperations(matrix.div(value), size);
	}


	@Override
	public ML4JTensorOperations sub(ML4JTensorOperations other) {
		return apply(other, m -> m.subColumnVector(other.getMatrix()),
				m -> m.subRowVector(other.getMatrix()), m -> m.sub(other.getMatrix()));
	}
	
	@Override
	public String toString() {
		Object s = null;
		List<String> lists = new ArrayList<>();

		
		for (int r = 0; r < Math.min(2, matrix.getRows()); r++) {
			List<String> vals = new ArrayList<>();
			for (int c = 0; c < Math.min(matrix.getColumns(), 2); c++) {
				vals.add(Float.valueOf(matrix.get(r, c)).toString());
			}
			if (matrix.getColumns() > 2) {
				vals.add("...");
			}
			lists.add(vals.toString());
		}
		if (matrix.getRows() > 2) {
			lists.add("...");
		}
	
		s = this.size.asList().size() == 0 ? matrix.get(0, 0) : lists;
		
		return s.toString();
	}
	
	
	//@Override
	public ML4JTensorOperations view(int i, int j) {

		if (i == -1 && j == -1) {
			throw new RuntimeException("only one dimension can be inferred");
		} else {
			if (i == -1) {
				i = this.numel() / j;
			}
			if (j == -1) {
				j = this.numel() / i;
			}
		}
	
		
		final int finalI = i;
		final int finalJ = j;


		Matrix view = matrix.softDup();
		view.asEditableMatrix().reshape(i, j);
		
	    return toML4JTensorOperations(view, new Size(finalI, finalJ));
		
	}
	
	public ML4JTensorOperations view(Size size) {
	    return toML4JTensorOperations(matrix, size);		
	}

	@Override
	public ML4JTensorOperations view(int... dims) {
		if (dims.length == 1 && dims[0] == -1) {
			return toML4JTensorOperations(matrix, new Size(size().numel()));
		} else {
			return toML4JTensorOperations(matrix, new Size(dims));
		}
	}

	private boolean requiresSecondMatrixColumnBroadcast(Matrix first, Matrix second) {
		if (first.getRows() == second.getRows() 
				&& first.getColumns() == second.getColumns()) {
			return false;
		} else {
			if (first.getRows() == second.getRows()) {
				if (second.getColumns() == 1) {
					// Need to broadcast second columns
					return true;
				} else if (first.getColumns() == 1) {
					// Need to broadcast first columns
					return false;
				}
			}
			if (first.getColumns() == second.getColumns()) {
				if (second.getRows() == 1) {
					// Need to broadcast second rows
					return false;
				} else if (first.getRows() == 1) {
					// Need to broadcast first rows
					return false;
				}
			}
		}
		return false;
	}
	
	private boolean requiresSecondMatrixRowsBroadcast(Matrix first, Matrix second) {
		if (first.getRows() == second.getRows() 
				&& first.getColumns() == second.getColumns()) {
			return false;
		} else {
			if (first.getRows() == second.getRows()) {
				if (second.getColumns() == 1) {
					// Need to broadcast second columns
					return false;
				} else if (first.getColumns() == 1) {
					// Need to broadcast first columns
					return false;
				}
			}
			if (first.getColumns() == second.getColumns()) {
				if (second.getRows() == 1) {
					// Need to broadcast second rows
					return true;
				} else if (first.getRows() == 1) {
					// Need to broadcast first rows
					return false;
				}
			}
		}
		return false;
	}
	

	@Override
	public ML4JTensorOperations mean() {
		return toML4JTensorOperations(matrixFactory.createOnes(1, 1).mul(matrix.sum() / matrix.getLength()), new Size(1, 1));
	}
	
	@Override
	public ML4JTensorOperations sum() {
		return toML4JTensorOperations(matrixFactory.createOnes(1, 1).mul(matrix.sum()), new Size(1, 1));
	}

	@Override
	public ML4JTensorOperations t() {
		return toML4JTensorOperations(matrix.transpose(), size.t());
	}

	@Override
	public Size size() {
		return size;
	}

	@Override
	public int size(int dim) {
		if (dim < 0) {
			return size().get(size().length() + dim);
		} else {
			return size().get(dim);
		}
	}

	@Override
	public ML4JTensorOperations get() {
		return this;
	}

	@Override
	public void performInlineOperation(Operation<ML4JTensorOperations, Size> operation) {
		operation.apply(this);
	}

	@Override
	public ML4JTensorOperations performUnaryMappingOperation(Operation<ML4JTensorOperations, Size> operation) {
		return operation.apply(this);
	}

	@Override
	public float[] getDataAsFloatArray() {
		float[] data = matrix.getRowByRowArray();
		return data;
	}

	@Override
	public ML4JTensorOperations size_(Size size) {
		if (size.numel() != this.numel()) {
			throw new IllegalArgumentException();
		} else {
			this.size = size;
		}
		return this;
	}

	@Override
	public ML4JTensorOperations columnSums() {
		if (size.dimensions().length != 2) {
			throw new IllegalStateException("Tensor must be 2 dimensional");
		}
		return toML4JTensorOperations(matrix.columnSums(), new Size(1, size().get(1)));
	}
	
	@Override
	public ML4JTensorOperations rowSums() {
		if (size.dimensions().length != 2) {
			throw new IllegalStateException("Tensor must be 2 dimensional");
		}
		return toML4JTensorOperations(matrix.rowSums(), new Size(size().get(0), 1));
	}

	@Override
	public void close() {
		//matrix.close();
	}

	@Override
	public ML4JTensorOperations relu() {
		EditableMatrix r = matrix.dup().asEditableMatrix();
		for (int i = 0; i < r.getLength(); i++) {
			if (r.get(i) < 0) {
				r.put(i, 0);
			}
		}
		return toML4JTensorOperations(r, size());
	}

	@Override
	public ML4JTensorOperations bernoulli() {
		throw new UnsupportedOperationException();
	}

	@Override
	public ML4JTensorOperations sigmoid() {
		return toML4JTensorOperations(matrix.sigmoid(), size());
	}

	@Override
	public ML4JTensorOperations cloneTensor() {
		return toML4JTensorOperations(matrix.dup(), size());
	}

}
