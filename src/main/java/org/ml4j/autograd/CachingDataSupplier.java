package org.ml4j.autograd;

import java.util.function.Supplier;

public interface CachingDataSupplier<T> extends Supplier<T> {

    void clearCache();

}
