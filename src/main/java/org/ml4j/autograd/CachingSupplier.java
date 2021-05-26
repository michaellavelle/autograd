package org.ml4j.autograd;

import java.util.function.Supplier;

public interface CachingSupplier<T> extends Supplier<T> {

    void clearCache();

}
