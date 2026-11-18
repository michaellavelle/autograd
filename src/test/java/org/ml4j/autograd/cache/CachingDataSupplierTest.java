package org.ml4j.autograd.cache;

import org.ml4j.autograd.scalar.FloatAutogradValueFactory;

import org.ml4j.autograd.scalar.impl.FloatAutogradValueFactoryImpl;


import org.junit.jupiter.api.Test;
import org.ml4j.autograd.scalar.FloatAutogradValue;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

class CachingDataSupplierTest {
    private final FloatAutogradValueFactory rawFactory =
            new FloatAutogradValueFactoryImpl();

    @Test
    void inspectingUnevaluatedCacheDoesNotInvokeSupplier() {
        AtomicInteger evaluations = new AtomicInteger();
        CachingDataSupplier<Object> supplier = new CachingDataSupplier<>(() -> {
            evaluations.incrementAndGet();
            return new Object();
        });

        assertTrue(supplier.cachedValue().isEmpty());
        assertTrue(supplier.cachedValue().isEmpty());
        assertEquals(0, evaluations.get());

        assertNotNull(supplier.get());
        assertEquals(1, evaluations.get());
    }

    @Test
    void inspectingPopulatedCacheReturnsExactObjectWithoutReevaluation() {
        AtomicInteger evaluations = new AtomicInteger();
        float[] numericalData = new float[]{2f, 3f};
        CachingDataSupplier<float[]> supplier = new CachingDataSupplier<>(() -> {
            evaluations.incrementAndGet();
            return numericalData;
        });

        assertSame(numericalData, supplier.get());
        assertSame(numericalData, supplier.cachedValue().orElseThrow());
        assertSame(numericalData, supplier.cachedValue().orElseThrow());
        assertSame(numericalData, supplier.get());
        assertEquals(1, evaluations.get());
    }

    @Test
    void inspectingCacheAfterExceptionOrNullResultPreservesRetry() {
        AtomicInteger evaluations = new AtomicInteger();
        IllegalStateException failure = new IllegalStateException("not ready");
        Object successfulResult = new Object();
        CachingDataSupplier<Object> supplier = new CachingDataSupplier<>(() -> {
            int attempt = evaluations.incrementAndGet();
            if (attempt == 1) {
                throw failure;
            }
            if (attempt == 2) {
                return null;
            }
            return successfulResult;
        });

        assertSame(failure, assertThrows(IllegalStateException.class, supplier::get));
        assertTrue(supplier.cachedValue().isEmpty());
        assertTrue(supplier.cachedValue().isEmpty());
        assertEquals(1, evaluations.get());

        assertThrows(NullPointerException.class, supplier::get);
        assertTrue(supplier.cachedValue().isEmpty());
        assertTrue(supplier.cachedValue().isEmpty());
        assertEquals(2, evaluations.get());

        assertSame(successfulResult, supplier.get());
        assertSame(successfulResult, supplier.cachedValue().orElseThrow());
        assertSame(successfulResult, supplier.get());
        assertEquals(3, evaluations.get());
    }

    @Test
    void defersEvaluationAndReusesTheFirstResult() {
        var reads = new AtomicInteger();
        var supplier = new CachingDataSupplier<>(() -> {
            reads.incrementAndGet();
            return new float[]{2f, 3f};
        });
        assertEquals(0, reads.get());
        var first = supplier.get();
        assertArrayEquals(new float[]{2f, 3f}, first);
        assertSame(first, supplier.get());
        assertSame(first, supplier.get());
        assertEquals(1, reads.get());
    }

    @Test
    void propagatesFailureAndRetriesUntilSuccessfulThenCaches() {
        var reads = new AtomicInteger();
        var failure = new IllegalStateException("not ready");
        var result = new Object();
        var supplier = new CachingDataSupplier<>(() -> {
            if (reads.incrementAndGet() < 3) throw failure;
            return result;
        });
        assertSame(failure, assertThrows(IllegalStateException.class, supplier::get));
        assertSame(failure, assertThrows(IllegalStateException.class, supplier::get));
        assertSame(result, supplier.get());
        assertSame(result, supplier.get());
        assertEquals(3, reads.get());
    }

    @Test
    void rejectsNullSourceAndRetriesNullResults() {
        assertThrows(NullPointerException.class, () -> new CachingDataSupplier<>(null));
        var reads = new AtomicInteger();
        var supplier = new CachingDataSupplier<>(() -> reads.incrementAndGet() == 1 ? null : 4f);
        assertThrows(NullPointerException.class, supplier::get);
        assertEquals(4f, supplier.get());
        assertEquals(4f, supplier.get());
        assertEquals(2, reads.get());
    }

    @Test
    void canSupplyLazyAutogradDataWithoutChangingGraphBehaviour() {
        var reads = new AtomicInteger();
        var cached = new CachingDataSupplier<>(() -> { reads.incrementAndGet(); return 3f; });
        var leaf = rawFactory.create(cached, null, true);
        var square = leaf.mul(leaf);
        assertEquals(0, reads.get());
        assertEquals(9f, square.data().get());
        square.backward();
        assertEquals(6f, leaf.gradNode().gradient().orElseThrow().data().get());
        assertEquals(9f, square.data().get());
        assertEquals(1, reads.get());
    }
}
