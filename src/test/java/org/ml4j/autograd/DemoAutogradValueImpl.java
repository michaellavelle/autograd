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

package org.ml4j.autograd;

import org.ml4j.autograd.impl.AutogradValueImpl;

/**
 * An AutogradValue implementation that supports the operations defined bt DemoOperations.
 */
public class DemoAutogradValueImpl extends AutogradValueImpl<DemoAutogradValue, Float, DemoSize> implements AutogradValue<DemoAutogradValue, Float, DemoSize>, DemoOperations<DemoAutogradValue>, DemoAutogradValue {

    @Override
    public float[] getDataAsFloatArray() {
        return new float[]{data().get()};
    }

    @Override
    public DemoAutogradValue add(DemoAutogradValue other) {
        return applyBinaryOperator(other, (f, s) -> f + s, (g, p) -> g, (g, p) -> g, "add", (f, s) -> f);
    }

    @Override
    public DemoAutogradValue div(DemoAutogradValue other) {
        return applyBinaryOperator(other, (f, s) -> f / s, (g, p) -> g.div(p.getRight()), (g, p) -> g.neg().mul(p.getLeft()).div(p.getRight().mul(p.getRight())), "div", (f, s) -> f);
    }

    @Override
    public DemoAutogradValue mul(DemoAutogradValue other) {
        return applyBinaryOperator(other, (f, s) -> f * s, (g, p) -> g.mul(p.getRight()), (g, p) -> g.mul(p.getLeft()), "mul", (f, s) -> f);
    }

    @Override
    public DemoAutogradValue add(float other) {
        return applyUnaryOperator(f -> f + other, (g, v) -> g, "add", s -> s);
    }

    @Override
    public DemoAutogradValue div(float other) {
        return applyUnaryOperator(f -> f / other, (g, v) -> g.div(other), "div", s -> s);
    }

    @Override
    public DemoAutogradValue mul(float other) {
        return applyUnaryOperator(f -> f * other, (g, v) -> g.mul(other), "mul", s -> s);
    }

    @Override
    public DemoAutogradValue neg() {
        return applyUnaryOperator(f -> -f, (g, v) -> g.neg(), "neg", s -> s);
    }

    @Override
    public DemoAutogradValue sub(DemoAutogradValue other) {
        return applyBinaryOperator(other, (f, s) -> f - s, (g, p) -> g, (g, p) -> g.neg(), "sub", (f, s) -> f);
    }

    @Override
    public DemoAutogradValue relu() {
        return applyUnaryOperator(f -> f < 0 ? 0 : f, (g, v) -> v.data().get() < 0 ? g.mul(0) : g, "relu", s -> s);
    }

    @Override
    public DemoAutogradValue sub(float other) {
        return applyUnaryOperator(f -> f - other, (g, v) -> g, "sub", s -> s);
    }

    @Override
    public DemoSize size() {
        return context();
    }

    @Override
    public DemoAutogradValue self() {
        return this;
    }
}
