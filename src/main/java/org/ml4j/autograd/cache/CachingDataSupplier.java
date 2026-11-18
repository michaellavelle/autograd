package org.ml4j.autograd.cache;

import java.util.Objects;
import java.util.Optional;
import java.util.function.Supplier;

/**
 * Lazily caches the first successful non-null result of a data supplier.
 *
 * <p>A scalar supplier can cache a computed Float. A tensor adapter can wrap a
 * random-data or native-array supplier to allocate its numerical data once.
 * Repeated reads return the same cached object; callers must preserve its contents
 * when using it as stable Value data.
 *
 * <p>Failed or null evaluations are not cached and can be retried. There is no
 * reset or invalidation. This utility does not manage native resources and is
 * intended for sequential use; instances are not thread-safe.
 *
 * @param <D> the supplied data type
 */
public final class CachingDataSupplier<D> implements Supplier<D> {
    private Supplier<? extends D> source;
    private D cached;

    /**
     * Wraps a supplier without evaluating it.
     * @param source a non-null data supplier
     * @throws NullPointerException if source is null
     */
    public CachingDataSupplier(Supplier<? extends D> source) {
        this.source = Objects.requireNonNull(source, "source");
    }

    /**
     * Inspects the cache without triggering evaluation.
     *
     * <p>Returns empty until an evaluation succeeds, including after a supplier
     * exception or null result. After success, returns the exact cached object
     * without copying or transformation. Does not invoke the underlying supplier
     * or modify the cache or supplier. Adds no thread-safety guarantee.
     *
     * @return the cached value, or empty before successful evaluation
     */
    public Optional<D> cachedValue() {
        return Optional.ofNullable(cached);
    }

    /**
     * Returns the cached data, evaluating the source only until it succeeds.
     *
     * <p>On success, releases this wrapper's reference to the source closure.
     * Evaluation failures propagate without changing the cache; subsequent calls
     * retry. A null result also leaves the cache empty.
     *
     * @return the same non-null data object after the first successful evaluation
     * @throws NullPointerException if the source returns null
     */
    @Override
    public D get() {
        if (cached == null) {
            D result = Objects.requireNonNull(source.get(), "supplied data");
            cached = result;
            source = null;
        }
        return cached;
    }
}
