package org.ml4j.autograd.scalar;

import org.ml4j.autograd.scalar.impl.FloatWrapperAutogradValueFactoryImpl;
import org.ml4j.autograd.scalar.impl.FloatWrapperImpl;

import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

class SharedScalarDefaultsTest {
    private final FloatWrapperAutogradValueFactory factory =
            new FloatWrapperAutogradValueFactoryImpl();

    private FloatWrapperAutogradValue value(float number, boolean tracked) {
        return factory.create(() -> new FloatWrapperImpl(number), null, tracked);
    }

    @Test
    void scalarOperationsAreLazyAndHaveOneInputWithCorrectSeededContributions() {
        for (int operation = 0; operation < 4; operation++) {
            var reads = new AtomicInteger();
            var input = factory.create(() -> {
                reads.incrementAndGet(); return new FloatWrapperImpl(6f);
            }, null, true);
            var output = switch (operation) {
                case 0 -> input.add(2f);
                case 1 -> input.sub(2f);
                case 2 -> input.mul(2f);
                default -> input.div(2f);
            };
            assertEquals(0, reads.get());
            assertEquals(1, output.valueNode().inputs().size());
            assertSame(input.valueNode(), output.valueNode().inputs().getFirst());
            assertTrue(output.requiresGrad());
            assertNull(output.context());
            assertEquals(new float[]{8, 4, 12, 3}[operation], output.data().get().floatValue());
            output.backward(value(3f, false));
            assertEquals(new float[]{3, 3, 6, 1.5f}[operation],
                    input.gradNode().gradient().orElseThrow().data().get().floatValue());
            assertFalse(value(6f, false).mul(2f).requiresGrad());
        }
    }

    @Test
    void composedScalarRulesPreserveSecondDerivativesAndRepeatedPasses() {
        var input = value(3f, true);
        var result = input.mul(input).mul(2f).div(4f).add(7f).sub(1f);
        result.backward(value(1f, false));
        var first = input.gradNode().gradient().orElseThrow();
        assertEquals(3f, first.data().get().floatValue());
        input.gradNode().reset();
        first.backward(value(1f, false));
        assertEquals(1f, input.gradNode().gradient().orElseThrow().data().get().floatValue());
        input.gradNode().reset();
        result.backward(value(1f, false));
        result.backward(value(1f, false));
        assertEquals(6f, input.gradNode().gradient().orElseThrow().data().get().floatValue());
    }

    @Test
    void comparisonsAreLazyConstantLeavesWithNoOperandMutation() {
        var reads = new AtomicInteger();
        var input = factory.create(() -> {
            reads.incrementAndGet(); return new FloatWrapperImpl(2f);
        }, null, true);
        var stored = value(4f, false);
        input.gradNode().accumulate(stored);
        var strict = input.gt(2f);
        var inclusive = input.gte(2f);
        assertEquals(0, reads.get());
        for (var mask : new FloatWrapperAutogradValue[]{strict, inclusive}) {
            assertFalse(mask.requiresGrad());
            assertTrue(mask.valueNode().inputs().isEmpty());
            assertNull(mask.valueNode().backwardFunction());
            assertTrue(mask.gradNode().gradient().isEmpty());
            assertThrows(IllegalStateException.class, mask::backward);
        }
        assertEquals(0f, strict.data().get().floatValue());
        assertEquals(1f, inclusive.data().get().floatValue());
        assertEquals(0f, value(Float.NaN, true).gt(0f).data().get().floatValue());
        assertSame(stored, input.gradNode().gradient().orElseThrow());
        assertTrue(input.requiresGrad());
    }

    @Test
    void nonDifferentiableUnaryMapsContextOnceAndRejectsNullFunctionsWithoutEvaluation() {
        var reads = new AtomicInteger();
        var maps = new AtomicInteger();
        var input = factory.create(() -> {
            reads.incrementAndGet(); return new FloatWrapperImpl(2f);
        }, null, true);
        var result = input.applyNonDifferentiableOperator(data -> data.gte(0f), context -> {
            maps.incrementAndGet(); return context;
        });
        assertEquals(1, maps.get());
        assertEquals(0, reads.get());
        assertThrows(NullPointerException.class, () -> input.applyNonDifferentiableOperator(null, context -> context));
        assertThrows(NullPointerException.class, () -> input.applyNonDifferentiableOperator(data -> data, null));
        assertEquals(0, reads.get());
        assertEquals(1f, result.data().get().floatValue());
        assertEquals(1, maps.get());
    }

    @Test
    void nonDifferentiableBinaryDispatchesThroughTheSharedCapability() {
        FloatWrapperAutogradValue left = value(9f, true);
        FloatWrapperAutogradValue right = value(4f, true);
        FloatWrapperAutogradValue result = left.applyNonDifferentiableOperator(
                right, FloatWrapper::sub, (leftContext, rightContext) -> null);

        assertFalse(result.requiresGrad());
        assertTrue(result.valueNode().inputs().isEmpty());
        assertNull(result.valueNode().backwardFunction());
        assertNull(result.context());
        assertEquals(5f, result.data().get().floatValue());
        assertTrue(left.gradient().isEmpty());
        assertTrue(right.gradient().isEmpty());
    }

}
