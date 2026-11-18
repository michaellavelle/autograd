package org.ml4j.autograd.impl;

import org.junit.jupiter.api.Test;
import org.ml4j.autograd.node.ValueNode;
import org.ml4j.autograd.operators.BinaryGradient;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.*;

class OperationOriginTest {
    private static final class TestValue extends AutogradValueImpl<TestValue, Float, Void> {
        TestValue(float data, boolean requiresGrad) { this(() -> data, requiresGrad, List.of()); }
        TestValue(Supplier<Float> data, boolean requiresGrad, List<? extends ValueNode<?>> inputs) {
            super(data, null, requiresGrad, inputs);
        }
        TestValue operation(TestValue other, BinaryGradient<TestValue> left, BinaryGradient<TestValue> right) {
            return applyBinaryOperator(other, (leftData, rightData) -> leftData + rightData, left, right, this::mapContext);
        }
        @Override public TestValue add(TestValue other) {
            return applyBinaryOperator(other, (leftData, rightData) -> leftData + rightData,
                    (incoming, left, right) -> incoming, (incoming, left, right) -> incoming, this::mapContext);
        }
        @Override protected TestValue self() { return this; }
        @Override public Void mapContext(Void left, Void right) { return null; }
        @Override protected TestValue createAutogradValue(Supplier<Float> data, Void context,
                boolean requiresGrad, List<? extends ValueNode<?>> inputs) {
            return new TestValue(data, requiresGrad, inputs);
        }
    }

    @Test void lazyNestedFailureKeepsOnlyFirstOriginAndExceptionIdentity() {
        var failure = new IllegalStateException("backend failed");
        var calls = new AtomicInteger();
        var leaf = new TestValue(() -> { calls.incrementAndGet(); throw failure; }, false, List.of());
        var result = leaf.add(new TestValue(2f, false)).add(new TestValue(3f, false));
        assertEquals(0, calls.get());
        assertSame(failure, assertThrows(IllegalStateException.class, () -> result.data().get()));
        assertEquals(1, failure.getSuppressed().length);
        assertEquals("lazyNestedFailureKeepsOnlyFirstOriginAndExceptionIdentity", originTestFrame(failure).getMethodName());
        assertSame(failure, assertThrows(IllegalStateException.class, () -> result.data().get()));
        assertEquals(1, failure.getSuppressed().length);
        assertEquals(2, calls.get());
    }

    @Test void binaryForwardNullResultHasConstructionOrigin() {
        var leaf = new TestValue(2f, true);
        var result = leaf.applyBinaryOperator(leaf, (a, b) -> null,
                (g, a, b) -> g, (g, a, b) -> g, leaf::mapContext);
        var failure = assertThrows(NullPointerException.class, () -> result.data().get());
        assertEquals(1, failure.getSuppressed().length);
        assertTrue(failure.getSuppressed()[0].getMessage().contains("forward evaluation"));
        assertEquals("binaryForwardNullResultHasConstructionOrigin", originTestFrame(failure).getMethodName());
    }

    @Test void unaryAndBinaryBackwardFailuresIncludeAccumulation() {
        var leaf = new TestValue(2f, true);
        var failure = new IllegalArgumentException("derivative failed");
        var unary = leaf.applyUnaryOperator(x -> -x, (g, x) -> { throw failure; }, c -> c);
        assertSame(failure, assertThrows(IllegalArgumentException.class,
                () -> unary.backward(new TestValue(1f, false))));
        assertEquals(1, failure.getSuppressed().length);

        assertTrue(failure.getSuppressed()[0].getMessage().contains("unary gradient"));

        // A null contribution fails inside accumulate(), outside the derivative rule.
        var binary = leaf.operation(leaf, (g, a, b) -> g, (g, a, b) -> null);
        var accumulationFailure = assertThrows(NullPointerException.class,
                () -> binary.backward(new TestValue(1f, false)));
        assertEquals(1, accumulationFailure.getSuppressed().length);
        assertTrue(accumulationFailure.getSuppressed()[0].getMessage().contains("right gradient"));
        assertEquals("unaryAndBinaryBackwardFailuresIncludeAccumulation",
                originTestFrame(accumulationFailure).getMethodName());
    }

    @Test void lazyGradientDataAndNonDifferentiableOperationsReceiveOrigins() {
        var leaf = new TestValue(2f, true);
        var failure = new IllegalStateException("lazy derivative failed");
        var result = leaf.applyUnaryOperator(x -> x, (g, x) ->
                x.applyNonDifferentiableOperator(value -> { throw failure; }, c -> c), c -> c);
        result.backward(new TestValue(1f, false));
        assertEquals(0, failure.getSuppressed().length);
        assertSame(failure, assertThrows(IllegalStateException.class,
                () -> leaf.gradient().orElseThrow().data().get()));
        assertEquals(1, failure.getSuppressed().length);
    }

    @Test void leftGradientFailureHasDistinctPhase() {
        var leaf = new TestValue(2f, true);
        var failure = new IllegalStateException("left failed");
        var result = leaf.operation(leaf, (g, a, b) -> { throw failure; }, (g, a, b) -> g);
        assertSame(failure, assertThrows(IllegalStateException.class,
                () -> result.backward(new TestValue(1f, false))));
        assertEquals(1, failure.getSuppressed().length);
        assertTrue(failure.getSuppressed()[0].getMessage().contains("left gradient"));
    }

    private static StackTraceElement originTestFrame(Throwable failure) {
        return java.util.Arrays.stream(failure.getSuppressed()[0].getStackTrace())
                .filter(frame -> frame.getClassName().equals(OperationOriginTest.class.getName()))
                .findFirst().orElseThrow();
    }
}
