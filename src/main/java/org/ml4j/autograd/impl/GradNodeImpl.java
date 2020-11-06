package org.ml4j.autograd.impl;

import java.util.ArrayList;
import java.util.Optional;
import java.util.function.BinaryOperator;
import java.util.function.Supplier;
import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.node.GradNode;

public class GradNodeImpl<V extends AutogradValue<V, ?, ?>> extends NodeImpl<V> implements GradNode<V> {

    private Supplier<Optional<V>> nativeGradientSupplier;
    private boolean disableNativeGradient;

    public GradNodeImpl(Supplier<V> value, Supplier<Optional<V>> nativeGradientSupplier) {
        super(value);
        this.nativeGradientSupplier = nativeGradientSupplier;
    }

    @Override
    public GradNode<V> setValue(Supplier<V> value) {
        if (this.value != null) {
            throw new IllegalStateException();  // Put back
        }
        if (this.prev == null) {
            this.prev = new ArrayList<>();
        }
        V val = value.get();
        this.value = () -> val;
        return this;
    }

    @Override
    public GradNode<V> add_(V value, BinaryOperator<V> addFunction) {
        if (this.prev == null) {
            this.prev = new ArrayList<>();
        }
        if (this.value == null) {
            this.value = () -> value;
        } else {
            V v = this.value.get();
            if (v == null) {
                this.value = () -> value;
            } else {
                // New value
                V res = addFunction.apply(v, value);
                this.value = () -> res;
                if (!this.prev.isEmpty() || wrapBackward != null) {
                    throw new IllegalStateException();
                }
            }
        }
        return this;
    }


    @Override
    public Optional<V> native_grad() {
        return nativeGradientSupplier != null ? nativeGradientSupplier.get() : Optional.empty();
    }

    public Supplier<Optional<V>> getNativeGradientSupplier() {
        return nativeGradientSupplier;
    }

    public void setNativeGradientSupplier(Supplier<Optional<V>> nativeGradientSupplier) {
        this.nativeGradientSupplier = nativeGradientSupplier;
    }

    public boolean isDisableNativeGradient() {
        return disableNativeGradient;
    }

    public void setDisableNativeGradient(boolean disableNativeGradient) {
        this.disableNativeGradient = disableNativeGradient;
    }
}
