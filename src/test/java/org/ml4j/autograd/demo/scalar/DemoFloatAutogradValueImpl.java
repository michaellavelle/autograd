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

package org.ml4j.autograd.demo.scalar;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.demo.DemoAutogradValue;
import org.ml4j.autograd.demo.DemoOperations;
import org.ml4j.autograd.demo.DemoSize;
import org.ml4j.autograd.impl.AutogradValueImpl;
import org.ml4j.autograd.node.Node;

/**
 * An AutogradValue implementation that supports the operations defined bt DemoOperations.
 * 
 * @author Michael Lavelle
 */
public class DemoFloatAutogradValueImpl extends AutogradValueImpl<DemoAutogradValue<Float>, Float, DemoSize> implements AutogradValue<DemoAutogradValue<Float>, Float, DemoSize>, DemoOperations<DemoAutogradValue<Float>>, DemoAutogradValue<Float> {
	
	public DemoFloatAutogradValueImpl(Supplier<Float> data, DemoSize size) {
		this(data, size, new ArrayList<>());
	}
	
    protected DemoFloatAutogradValueImpl(Supplier<Float> data, DemoSize size, List<Node<?>> childen) {
		super(data, size, childen);
	}

	@Override
    public float[] getDataAsFloatArray() {
        return new float[]{data().get()};
    }

    @Override
    public DemoAutogradValue<Float> add(DemoAutogradValue<Float> other) {
        return applyBinaryOperator(other, (f, s) -> f + s, (g, p) -> g, (g, p) -> g, "add", (f, s) -> f);
    }

    @Override
    public DemoAutogradValue<Float> div(DemoAutogradValue<Float> other) {
        return applyBinaryOperator(other, (f, s) -> f / s, (g, p) -> g.div(p.getRight()), (g, p) -> g.neg().mul(p.getLeft()).div(p.getRight().mul(p.getRight())), "div", (f, s) -> f);
    }

    @Override
    public DemoAutogradValue<Float> mul(DemoAutogradValue<Float> other) {
        return applyBinaryOperator(other, (f, s) -> f * s, (g, p) -> g.mul(p.getRight()), (g, p) -> g.mul(p.getLeft()), "mul", (f, s) -> f);
    }

    @Override
    public DemoAutogradValue<Float> add(float other) {
        return applyUnaryOperator(f -> f + other, (g, v) -> g, "add", s -> s);
    }

    @Override
    public DemoAutogradValue<Float> div(float other) {
        return applyUnaryOperator(f -> f / other, (g, v) -> g.div(other), "div", s -> s);
    }

    @Override
    public DemoAutogradValue<Float> mul(float other) {
        return applyUnaryOperator(f -> f * other, (g, v) -> g.mul(other), "mul", s -> s);
    }

    @Override
    public DemoAutogradValue<Float> neg() {
        return applyUnaryOperator(f -> -f, (g, v) -> g.neg(), "neg", s -> s);
    }

    @Override
    public DemoAutogradValue<Float> sub(DemoAutogradValue<Float> other) {
        return applyBinaryOperator(other, (f, s) -> f - s, (g, p) -> g, (g, p) -> g.neg(), "sub", (f, s) -> f);
    }

    @Override
    public DemoAutogradValue<Float> relu() {
        return applyUnaryOperator(f -> f < 0 ? 0 : f, (g, v) -> g.mul(v.gt(0)), "relu", s -> s);
    }

    @Override
    public DemoAutogradValue<Float> sub(float other) {
        return applyUnaryOperator(f -> f - other, (g, v) -> g, "sub", s -> s);
    }

    @Override
    public DemoSize size() {
        return context();
    }

    @Override
    public DemoAutogradValue<Float> self() {
        return this;
    }

	@Override
	public DemoAutogradValue<Float> add_(DemoAutogradValue<Float> other) {
        return applyInlineBinaryOperator(other, (f, s) -> f + s, "add");
	}

	@Override
	public DemoAutogradValue<Float> sub_(DemoAutogradValue<Float> other) {
        return applyInlineBinaryOperator(other, (f, s) -> f + s, "sub");
	}

	@Override
	public DemoAutogradValue<Float> gt(float value) {
        return applyUnaryOperator(f -> f > value ? 1f : 0f, (g, v) -> g.mul(v.gt(value)), "gt", s -> s);
	}

	@Override
	public DemoAutogradValue<Float> gte(float value) {
        return applyUnaryOperator(f -> f >= value ? 1f : 0f, (g, v) -> g.mul(v.gte(value)), "gt", s -> s);
	}

	@Override
	protected DemoAutogradValue<Float> createAutogradValue(Supplier<Float> data, DemoSize size, List<Node<?>> childen) {
		return new DemoFloatAutogradValueImpl(data, size, childen);
	}
}
