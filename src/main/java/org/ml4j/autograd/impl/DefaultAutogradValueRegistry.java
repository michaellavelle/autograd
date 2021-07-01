package org.ml4j.autograd.impl;

import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.AutogradValueRegistry;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class DefaultAutogradValueRegistry implements AutogradValueRegistry, Iterable<AutogradValue<?, ?, ?>> {

    private List<AutogradValue<?, ?, ?>> registry;
    private static List<DefaultAutogradValueRegistry> registries = new ArrayList<>();
    private String name;

    public static AutogradValueRegistry create(String name) {
        DefaultAutogradValueRegistry registry = new DefaultAutogradValueRegistry(name);
        registries.add(registry);
        return registry;
    }

    public DefaultAutogradValueRegistry(String name) {
        this.registry = new ArrayList<>();
        this.name = name;
    }

    public static boolean allClosed() {
        for (DefaultAutogradValueRegistry registry : registries) {
            if (!registry.allClosedLocal()) {
                return false;
            }
        }
        return true;
    }

    public boolean allClosedLocal() {
        for (AutogradValue<?, ?, ?> autogradValue : registry) {
            if (!autogradValue.isClosed()) {
                return false;
            }
        }
        return true;
    }

    public static void status(boolean print) {
        for (DefaultAutogradValueRegistry registry : registries) {
            if (print) {
                System.out.println("-----");
            }
            registry.statusLocal(print);
            if (print) {
                System.out.println("-----");
            }
        }
    }

    public static void close() {
        for (DefaultAutogradValueRegistry registry : registries) {
            registry.closeLocal();
        }
    }

    public static void clear() {
        for (DefaultAutogradValueRegistry registry : registries) {
            registry.clearLocal();
        }
    }

    public void statusLocal(boolean print) {
        int notYetClosed = 0;
        int uncloseable = 0;

        for (AutogradValue<?, ?, ?> b : registry) {
            if (!b.isClosed() && !b.properties().isUncloseable()) {
                if (print) {
                    System.out.println(name + ":" + "Not yet closed:" + b.name() + ":" + b.requires_grad());
                }
                notYetClosed++;
            } else if (!b.isClosed() && b.properties().isUncloseable()) {
                System.out.println(name + ":" + "Uncloseable:" + b.name() + ":" + b.requires_grad());
                uncloseable++;
            }
        }
        if (print) {
            System.out.println(name + ":" + "Not yet closed total:" + notYetClosed);
        }
        if (print) {
            System.out.println(name + ":" + "Unclosable total:" + uncloseable);
        }
    }

    public void clearLocal() {
        List<AutogradValue<?, ?, ?>> toClose = new ArrayList<>();
        for (AutogradValue<?, ?, ?> value : registry) {
            if (value.isClosed() && !value.properties().isUncloseable()) {
                toClose.add(value);
            }
        }
        registry.removeAll(toClose);
    }

    public void closeLocal() {
        for (AutogradValue<?, ?, ?> b : registry) {
            if (!b.isClosed() && !b.properties().isUncloseable()) {
                b.close();
            }
        }
    }

    @Override
    public void registerAutogradValue(AutogradValue<?, ?, ?> autogradValue) {
        registry.add(autogradValue);
    }

    @Override
    public Iterator<AutogradValue<?, ?, ?>> iterator() {
        return registry.iterator();
    }
}
