package org.ml4j.autograd;

import java.util.function.Supplier;

public class CachingDataSupplierImpl<T> implements CachingDataSupplier<T> {

    private Supplier<T> supplier;
    private T value;
    private boolean calc;

    public CachingDataSupplierImpl(Supplier<T> supplierT) {
        this.supplier = supplierT;
        //this.value = supplierT.get();
    }

    public void clearCache() {
        this.calc = false;
        this.value = null;
    }

    @Override
    public T get() {
        if (calc) {
            return value;
        } else {
            value = supplier.get();
            calc = true;
            return value;
        }
    }
}
