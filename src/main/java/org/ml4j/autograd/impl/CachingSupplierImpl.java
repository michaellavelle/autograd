package org.ml4j.autograd.impl;

import java.util.function.Supplier;

import org.ml4j.autograd.CachingSupplier;

public class CachingSupplierImpl<T> implements CachingSupplier<T> {

    private Supplier<T> supplier;
    private T value;
    private boolean calc;

    public CachingSupplierImpl(Supplier<T> supplierT) {
        this.supplier = supplierT;
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
