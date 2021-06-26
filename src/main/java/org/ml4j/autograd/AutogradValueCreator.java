package org.ml4j.autograd;

import org.ml4j.autograd.node.Node;

import java.util.List;
import java.util.function.Supplier;

public interface AutogradValueCreator<V, D, C> {
    V createAutogradValue(Supplier<D> data, C context, List<Node<?>> children, boolean requires_grad, boolean create_graph);
}
