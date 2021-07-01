package org.ml4j.autograd;

import org.ml4j.autograd.impl.DefaultAutogradValueRegistry;

public interface AutogradValueRegistry extends Iterable<AutogradValue<?, ?, ?>> {

    void registerAutogradValue(AutogradValue<?, ?, ?> autogradValue);

    static AutogradValueRegistry create(String name) {
        return DefaultAutogradValueRegistry.create(name);
    }

    static void status(boolean print) {
        DefaultAutogradValueRegistry.status(print);
    }

    static void close() {
        DefaultAutogradValueRegistry.close();
    }

    static void clear() {
        DefaultAutogradValueRegistry.clear();
    }

    static boolean allClosed() {
        return DefaultAutogradValueRegistry.allClosed();
    }
}
