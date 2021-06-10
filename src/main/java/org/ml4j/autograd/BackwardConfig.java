/*
 * Copyright 2020 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

package org.ml4j.autograd;

/**
 * Encapsulates the configuration parameters we use to control the behaviour of a backward pass.
 *
 * @author Michael Lavelle
 */
public class BackwardConfig {

    private boolean keep_graph;
    private boolean zero_grad;

    public BackwardConfig with_keep_graph(boolean keep_graph) {
        this.keep_graph = keep_graph;
        return this;
    }

    public BackwardConfig with_zero_grad(boolean zero_grad) {
        this.zero_grad = zero_grad;
        return this;
    }

    public boolean zero_grad() {
        return zero_grad;
    }

    public boolean keep_graph() {
        return keep_graph;
    }
}
