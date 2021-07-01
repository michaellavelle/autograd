package org.ml4j.autograd.impl;

import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.AutogradValueRegistry;
import org.ml4j.autograd.node.Node;

import java.util.ArrayList;
import java.util.List;

public class AutogradValueProperties<C> {

    private String name;
    protected boolean create_graph;
    private boolean requires_grad;
    private C context;
    private List<Node<?>> children;
    private List<Node<?>> next;
    private List<Node<?>> links;
    private AutogradValueRegistry registry;
    private boolean uncloseable;

    public AutogradValueProperties() {
        this.children = new ArrayList<>();
        this.next = new ArrayList<>();
        this.links = new ArrayList<>();
    }

    public boolean isUncloseable() {
        return uncloseable;
    }

    public AutogradValueProperties<C> setUncloseable(boolean uncloseable) {
        this.uncloseable = uncloseable;
        return this;
    }

    public AutogradValueProperties<C> setRegistry(AutogradValueRegistry registry) {
        this.registry = registry;
        return this;
    }

    public AutogradValueRegistry getRegistry() {
        return registry;
    }

    public void register(AutogradValue<?, ?, ?> value) {
        if (registry != null) {
            this.registry.registerAutogradValue(value);
        } else {
            throw new IllegalStateException();
        }
    }

    public AutogradValueProperties<C> setName(String name) {
        this.name = name;
        return this;
    }

    public AutogradValueProperties<C> setCreate_graph(boolean create_graph) {
        this.create_graph = create_graph;
        return this;
    }

    public AutogradValueProperties<C> setRequires_grad(boolean requires_grad) {
        this.requires_grad = requires_grad;
        return this;
    }

    public AutogradValueProperties<C> setContext(C context) {
        this.context = context;
        return this;
    }

    public AutogradValueProperties<C> setChildren(List<Node<?>> children) {
        this.children = children;
        return this;
    }

    public AutogradValueProperties<C> addLink(Node<?> link) {
        this.links.add(link);
        return this;
    }

    public List<Node<?>> getLinks() {
        return links;
    }

    public AutogradValueProperties<C> setNext(List<Node<?>> next) {
        this.next = next;
        return this;
    }

    public boolean isCreate_graph() {
        return create_graph;
    }

    public boolean isRequires_grad() {
        return requires_grad;
    }

    public C getContext() {
        return context;
    }

    public List<Node<?>> getChildren() {
        return children;
    }

    public List<Node<?>> getNext() {
        return next;
    }

    public String getName() {
        return name;
    }
}
