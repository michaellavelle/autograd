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

package org.ml4j.autograd.arithmetic.operations;

import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.UnaryOperator;
import org.apache.commons.lang3.tuple.Pair;
import org.ml4j.autograd.AutogradValue;

/**
 * Wraps an ArithmeticOperations instance with differentiation.
 * 
 * @author Michael Lavelle
 */
public interface DifferentiableWrappedArithmeticOperations<V extends ArithmeticOperations<V>, D extends ArithmeticOperations<D>, C> extends AutogradValue<V, D, C>, ArithmeticOperations<V> {

    @Override
    default float[] getDataAsFloatArray() {
        return data().get().getDataAsFloatArray();
    }
 
    @Override
    default V add(V other) {
        return applyBinaryOperator(other, D::add, (g, p) -> g, (g, p) -> g, "add", (f, s) -> f);
    }
   
    @Override
    default V add(float other) {
        return applyUnaryOperator(D::add, other, (g, v) -> g, "add", s -> s);
    }

    @Override
    default V div(V other) {
        return applyBinaryOperator(other, D::div, (g, p) -> g.div(p.getRight()), (g, p) -> g.neg().mul(p.getLeft()).div(p.getRight().mul(p.getRight())), "div", (f, s) -> f);
    }
    
    @Override
    default V div(float other) {
        return applyUnaryOperator(D::div, other, (g, v) -> g.div(other), "div", s -> s);
    }

    @Override
    default V mul(V other) {
        return applyBinaryOperator(other, D::mul, (g, p) -> g.mul(p.getRight()), (g, p) -> g.mul(p.getLeft()), "mul", (f, s) -> f);
    }
    
    @Override
    default V mul(float other) {
        return applyUnaryOperator(D::mul, other, (g, v) -> g.mul(other), "mul", s -> s);
    }
    
    private UnaryOperator<D> unary(BiFunction<D, Float, D> op, float other) {
        return v -> op.apply(v, other);
    }
    
    V applyBinaryOperator(V other, BinaryOperator<D> forward, BiFunction<V, Pair<V, V>, V> backThis,
            BiFunction<V, Pair<V, V>, V> backOther, String op, BinaryOperator<C> contextMapper);

    V applyInlineBinaryOperator(V other, BinaryOperator<D> forward, String op);

    
    default V applyUnaryOperator(BiFunction<D, Float, D> forward, float other, BiFunction<V, V, V> backThis, String op, UnaryOperator<C> contextMapper) {
        return applyUnaryOperator(unary(forward, other), backThis, op, contextMapper);
    }
    
    V applyUnaryOperator(UnaryOperator<D> forward, BiFunction<V, V, V> backThis, String op, UnaryOperator<C> contextMapper);

    @Override
    default V neg() {
        return applyUnaryOperator(D::neg, (g, v) -> g.neg(), "neg", s -> s);
    }

    @Override
    default V sub(V other) {
        return applyBinaryOperator(other,  D::sub, (g, p) -> g, (g, p) -> g.neg(), "sub", (f, s) -> f);
    }

    @Override
    default V sub(float other) {
        return applyUnaryOperator(D::sub, other, (g, v) -> g, "sub", s -> s);
    }

    @Override
    default V add_(V other) {
        return applyInlineBinaryOperator(other, D::add_, "add");
    }

    @Override
    default V sub_(V other) {
        return applyInlineBinaryOperator(other, D::sub, "sub");
    }

    @Override
    default V gt(float value) {
        return applyUnaryOperator(D::gt, value, (g, v) -> g.mul(v.gt(value)), "gt", s -> s);
    }

    @Override
    default V gte(float value) {
        return applyUnaryOperator(D::gte, value, (g, v) -> g.mul(v.gte(value)), "gt", s -> s);
    }
}
