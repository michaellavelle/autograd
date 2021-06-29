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

import org.junit.Test;
import org.junit.jupiter.api.Assertions;

/**
 * @author Michael Lavelle
 */
public class BackwardConfigTest {

    @Test
    public void testDefaultConstructor() {
        BackwardConfig config = new BackwardConfig();
        Assertions.assertFalse(config.keep_graph());
        Assertions.assertFalse(config.zero_grad());
    }

    @Test
    public void testOnlyKeepGraphSetAsTrue() {
        BackwardConfig config = new BackwardConfig().with_keep_graph(true);
        Assertions.assertTrue(config.keep_graph());
        Assertions.assertFalse(config.zero_grad());
    }

    @Test
    public void testOnlyKeepGraphSetAsFalse() {
        BackwardConfig config = new BackwardConfig().with_keep_graph(false);
        Assertions.assertFalse(config.keep_graph());
        Assertions.assertFalse(config.zero_grad());
    }

    @Test
    public void testOnlyZeroGradSetAsTrue() {
        BackwardConfig config = new BackwardConfig().with_zero_grad(true);
        Assertions.assertFalse(config.keep_graph());
        Assertions.assertTrue(config.zero_grad());
    }

    @Test
    public void testOnlyZeroGradSetAsFalse() {
        BackwardConfig config = new BackwardConfig().with_zero_grad(false);
        Assertions.assertFalse(config.keep_graph());
        Assertions.assertFalse(config.zero_grad());
    }

    @Test
    public void testBothSetAsTrue() {
        BackwardConfig config = new BackwardConfig().with_zero_grad(true).with_keep_graph(true);
        Assertions.assertTrue(config.keep_graph());
        Assertions.assertTrue(config.zero_grad());
    }

    @Test
    public void testBothSetAsFalse() {
        BackwardConfig config = new BackwardConfig().with_zero_grad(false).with_keep_graph(false);
        Assertions.assertFalse(config.keep_graph());
        Assertions.assertFalse(config.zero_grad());
    }
}
