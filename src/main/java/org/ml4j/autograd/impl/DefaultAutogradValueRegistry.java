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
        int not = 0;
        for (AutogradValue<?, ?, ?> b : registry) {
            if (!b.isClosed()) {
                if (print) {
                    System.out.println(name + ":" + "Not closed:" + b.name() + ":" + b.requires_grad());
                }
                not++;
            }
        }
        if (print) {
            System.out.println(name + ":" + "Not closed total:" + not);
        }
    }

    public void clearLocal() {
        for (AutogradValue<?, ?, ?> b : registry) {
            if (!b.isClosed()) {
                throw new IllegalStateException("Autograd value not closed");
            }
        }
        registry = new ArrayList<>();
    }

    public void closeLocal() {
        for (AutogradValue<?, ?, ?> b : registry) {
            if (!b.isClosed()) {
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
