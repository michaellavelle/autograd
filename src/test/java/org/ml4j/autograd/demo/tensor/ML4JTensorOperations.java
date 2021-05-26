package org.ml4j.autograd.demo.tensor;

import org.jvmpy.symbolictensors.Operatable;
import org.jvmtorch.torch.Size;
import org.jvmtorch.torch.TensorOperations;
import org.ml4j.Matrix;
import org.ml4j.nn.components.DirectedComponentsContext;


public interface ML4JTensorOperations extends TensorOperations<ML4JTensorOperations>, Operatable<ML4JTensorOperations, Size, ML4JTensorOperations> {

	Matrix getMatrix();

	DirectedComponentsContext getDirectedComponentsContext();
		

}
