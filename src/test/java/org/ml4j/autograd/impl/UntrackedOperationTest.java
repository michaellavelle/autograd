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
