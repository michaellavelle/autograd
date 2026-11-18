package org.ml4j.autograd.impl;

import org.junit.jupiter.api.Test;
import org.ml4j.autograd.cache.CachingDataSupplier;
import org.ml4j.autograd.node.ValueNode;
import org.ml4j.autograd.scalar.FloatAutogradValue;
import org.ml4j.autograd.scalar.impl.FloatAutogradValueImpl;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.*;

class UntrackedOperationTest {
    @Test
    void unaryBinaryAndChainsHaveNoBackwardStateAndRemainLazy() {
        AtomicInteger reads = new AtomicInteger();
        FloatAutogradValue left = new FloatAutogradValueImpl(() -> {
            reads.incrementAndGet();
            return 3f;
        }, false);
        FloatAutogradValue right = new FloatAutogradValueImpl(2f, false);
        FloatAutogradValue unary = left.neg();
        FloatAutogradValue binary = left.mul(right);
        FloatAutogradValue chain = binary.add(unary).mul(right).neg();

        assertEquals(0, reads.get());
        for (FloatAutogradValue result : List.of(unary, binary, chain)) {
            assertUntracked(result);
        }
        assertEquals(-3f, unary.data().get());
        assertEquals(6f, binary.data().get());
        assertEquals(-6f, chain.data().get());
        assertEquals(-6f, chain.data().get());
        assertUntracked(chain);
    }

    @Test
    void cachedResultReleasesForwardClosureAfterSuccess() throws ReflectiveOperationException {
        AtomicInteger reads = new AtomicInteger();
        CachedValue leaf = new CachedValue(() -> {
            reads.incrementAndGet();
            return 4f;
        }, "shape", false, List.of());
        CachedValue result = leaf.applyUnaryOperator(value -> value * 2,
                (incoming, operand) -> incoming, context -> context + " mapped");
        assertEquals(0, reads.get());
        assertEquals("shape mapped", result.context());
        assertTrue(result.valueNode().inputs().isEmpty());
        assertNull(result.valueNode().backwardFunction());
        java.lang.reflect.Field source = CachingDataSupplier.class.getDeclaredField("source");
        source.setAccessible(true);
        assertNotNull(source.get(result.cache));
        assertEquals(8f, result.data().get());
        assertNull(source.get(result.cache));
        assertEquals(8f, result.data().get());
        assertEquals(1, reads.get());
        assertThrows(IllegalStateException.class, () -> result.setRequiresGrad(true));
    }

    @Test
    void laterLeafEnablingOnlyAffectsNewOperations() {
        FloatAutogradValue leaf = new FloatAutogradValueImpl(3f, false);
        FloatAutogradValue old = leaf.mul(leaf);
        leaf.setRequiresGrad(true);
        assertUntracked(old);
        assertEquals(9f, old.data().get());
        assertThrows(IllegalStateException.class, () -> old.setRequiresGrad(true));
        leaf.mul(leaf).backward();
        assertEquals(6f, leaf.gradient().orElseThrow().data().get());
    }

    @Test
    void mixedOperandsKeepBothInputsAndAccumulateHigherDifferentiableGradients() {
        for (boolean trackedLeft : List.of(true, false)) {
            FloatAutogradValue variable = new FloatAutogradValueImpl(3f, true);
            FloatAutogradValue constant = new FloatAutogradValueImpl(2f, false);
            FloatAutogradValue product = trackedLeft ? variable.mul(constant) : constant.mul(variable);
            assertEquals(2, product.valueNode().inputs().size());
            assertNotNull(product.valueNode().backwardFunction());
            FloatAutogradValue result = product.mul(variable).neg();
            result.backward();
            FloatAutogradValue derivative = variable.gradient().orElseThrow();
            assertEquals(-12f, derivative.data().get());
            assertTrue(constant.gradient().isEmpty());
            result.backward();
            assertEquals(-24f, variable.gradient().orElseThrow().data().get());
            variable.resetGradient();
            derivative.backward();
            assertEquals(-4f, variable.gradient().orElseThrow().data().get());
        }
    }

    @Test
    void nonDifferentiableUnaryRetainsLazyNumericalInputWithoutGradientPropagation() {
        for (boolean tracked : List.of(false, true)) {
            AtomicInteger operandReads = new AtomicInteger();
            AtomicInteger contextMappings = new AtomicInteger();
            CachedValue operand = new CachedValue(() -> {
                operandReads.incrementAndGet();
                return 4f;
            }, "input", tracked, List.of());

            CachedValue result = operand.applyNonDifferentiableOperator(
                    numericalData -> numericalData * 3f,
                    inputContext -> {
                        contextMappings.incrementAndGet();
                        assertEquals("input", inputContext);
                        return inputContext + " mapped";
                    });

            assertEquals("input mapped", result.context());
            assertEquals(1, contextMappings.get());
            assertEquals(0, operandReads.get());
            assertFalse(result.requiresGrad());
            assertTrue(result.valueNode().inputs().isEmpty());
            assertNull(result.valueNode().backwardFunction());
            assertTrue(result.gradient().isEmpty());
            assertThrows(IllegalStateException.class, () -> result.backward(operand));

            assertEquals(12f, result.data().get());
            assertEquals(12f, result.data().get());
            assertEquals(1, operandReads.get());
            assertEquals(1, contextMappings.get());

            CachedValue independent = new CachedValue(() -> 2f, "independent", true, List.of());
            result.add(independent).backward(new CachedValue(() -> 1f, "seed", false, List.of()));
            assertEquals(1f, independent.gradient().orElseThrow().data().get());
            assertTrue(result.gradient().isEmpty());
            assertTrue(operand.gradient().isEmpty());
        }
    }

    @Test
    void nonDifferentiableBinaryRetainsBothNumericalOperandsForEveryTrackingCombination() {
        for (boolean leftTracked : List.of(false, true)) {
            for (boolean rightTracked : List.of(false, true)) {
                AtomicInteger leftReads = new AtomicInteger();
                AtomicInteger rightReads = new AtomicInteger();
                AtomicInteger operatorCalls = new AtomicInteger();
                AtomicInteger contextMappings = new AtomicInteger();
                CachedValue left = new CachedValue(() -> {
                    leftReads.incrementAndGet();
                    return 9f;
                }, "left", leftTracked, List.of());
                CachedValue right = new CachedValue(() -> {
                    rightReads.incrementAndGet();
                    return 4f;
                }, "right", rightTracked, List.of());

                CachedValue result = left.applyNonDifferentiableOperator(right,
                        (leftData, rightData) -> {
                            operatorCalls.incrementAndGet();
                            return leftData - rightData;
                        },
                        (leftContext, rightContext) -> {
                            contextMappings.incrementAndGet();
                            return leftContext + "/" + rightContext;
                        });

                assertEquals("left/right", result.context());
                assertEquals(1, contextMappings.get());
                assertEquals(0, leftReads.get());
                assertEquals(0, rightReads.get());
                assertEquals(0, operatorCalls.get());
                assertFalse(result.requiresGrad());
                assertTrue(result.valueNode().inputs().isEmpty());
                assertNull(result.valueNode().backwardFunction());
                assertTrue(result.gradient().isEmpty());
                assertThrows(IllegalStateException.class, () -> result.backward(left));

                CachedValue independent = new CachedValue(() -> 2f, "independent", true, List.of());
                CachedValue combined = result.add(independent);
                combined.backward(new CachedValue(() -> 1f, "seed", false, List.of()));
                assertTrue(left.gradient().isEmpty());
                assertTrue(right.gradient().isEmpty());
                assertTrue(result.gradient().isEmpty());
                assertEquals(1f, independent.gradient().orElseThrow().data().get());
                assertEquals(0, operatorCalls.get());

                assertEquals(7f, combined.data().get());
                assertEquals(5f, result.data().get());
                assertEquals(5f, result.data().get());
                assertEquals(1, leftReads.get());
                assertEquals(1, rightReads.get());
                assertEquals(1, operatorCalls.get());
                assertEquals(1, contextMappings.get());
            }
        }
    }

    @Test
    void nonDifferentiableBinaryRejectsNullArgumentsWithoutNumericalEvaluation() {
        CachedValue operand = new CachedValue(() -> {
            throw new AssertionError("Unexpected numerical evaluation");
        }, "input", true, List.of());

        assertThrows(NullPointerException.class, () -> operand.applyNonDifferentiableOperator(
                null, Float::sum, (leftContext, rightContext) -> leftContext));
        assertThrows(NullPointerException.class, () -> operand.applyNonDifferentiableOperator(
                operand, null, (leftContext, rightContext) -> leftContext));
        assertThrows(NullPointerException.class, () -> operand.applyNonDifferentiableOperator(
                operand, Float::sum, null));
    }

    private static void assertUntracked(FloatAutogradValue value) {
        assertFalse(value.requiresGrad());
        assertTrue(value.valueNode().inputs().isEmpty());
        assertNull(value.valueNode().backwardFunction());
        assertTrue(value.gradient().isEmpty());
    }

    // Demonstrates a downstream result factory wrapping each forward supplier in a cache.
    private static final class CachedValue extends AutogradValueImpl<CachedValue, Float, String> {
        private final CachingDataSupplier<Float> cache;

        private CachedValue(Supplier<Float> source, String context, boolean tracked,
                            List<? extends ValueNode<?>> inputs) {
            this(new CachingDataSupplier<>(source), context, tracked, inputs);
        }

        private CachedValue(CachingDataSupplier<Float> cache, String context, boolean tracked,
                            List<? extends ValueNode<?>> inputs) {
            super(cache, context, tracked, inputs);
            this.cache = cache;
        }

        @Override protected CachedValue self() { return this; }
        @Override public String mapContext(String left, String right) { return left; }
        @Override public CachedValue add(CachedValue other) {
            return applyBinaryOperator(other, Float::sum, (incoming, left, right) -> incoming,
                    (incoming, left, right) -> incoming, this::mapContext);
        }
        @Override protected CachedValue createAutogradValue(Supplier<Float> source, String context,
                boolean tracked, List<? extends ValueNode<?>> inputs) {
            return new CachedValue(source, context, tracked, inputs);
        }
    }
}
