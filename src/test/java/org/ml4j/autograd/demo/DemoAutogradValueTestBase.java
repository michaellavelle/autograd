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

package org.ml4j.autograd.demo;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.ml4j.autograd.BackwardConfig;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

/**
 * A test for our DemoAutogradValue.
 * 
 * @author Michael Lavelle
 *
 */
public abstract class DemoAutogradValueTestBase<D> {

    @Mock
	protected DemoAutogradValueFactory<D> gradValueFactory;

    @Mock
    protected DemoAutogradValue<D> mockGradValue;

    @Mock
    protected DemoSize size;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
        Mockito.when(gradValueFactory.create(Mockito.any(), Mockito.any())).thenReturn(mockGradValue);
        Mockito.when(mockGradValue.requires_grad_(Mockito.anyBoolean())).thenReturn(mockGradValue);
        Mockito.when(mockGradValue.mul(Mockito.any())).thenReturn(mockGradValue);
        Mockito.when(mockGradValue.add(Mockito.any())).thenReturn(mockGradValue);
        Mockito.when(mockGradValue.sub(Mockito.any())).thenReturn(mockGradValue);
        Mockito.when(mockGradValue.div(Mockito.any())).thenReturn(mockGradValue);
        Mockito.when(mockGradValue.add(Mockito.anyFloat())).thenReturn(mockGradValue);
        Mockito.when(mockGradValue.mul(Mockito.anyFloat())).thenReturn(mockGradValue);
        Mockito.when(mockGradValue.div(Mockito.anyFloat())).thenReturn(mockGradValue);
        Mockito.when(mockGradValue.sub(Mockito.anyFloat())).thenReturn(mockGradValue);
        Mockito.when(mockGradValue.add(Mockito.anyInt())).thenReturn(mockGradValue);
        Mockito.when(mockGradValue.mul(Mockito.anyInt())).thenReturn(mockGradValue);
        Mockito.when(mockGradValue.div(Mockito.anyInt())).thenReturn(mockGradValue);
        Mockito.when(mockGradValue.sub(Mockito.anyInt())).thenReturn(mockGradValue);
        Mockito.when(mockGradValue.relu()).thenReturn(mockGradValue);
        Mockito.when(mockGradValue.grad()).thenReturn(mockGradValue);
        Mockito.when(mockGradValue.name_(Mockito.anyString())).thenReturn(mockGradValue);
        Mockito.when(mockGradValue.getDataAsFloatArray()).thenReturn(new float[]{1f});
    }

    protected abstract DemoAutogradValue<D> createGradValue(float value, boolean requires_grad);
   
    protected abstract DemoAutogradValue<D> createGradValue(D value, boolean requires_grad);

    
    protected abstract D createData(float value);

    
    @Test
    public void test_example() {

        var a = createGradValue(-4f, true).name_("a");

        var b = createGradValue(2.0f, true).name_("b");

        var c = a.add(b);

        var d = a.mul(b).add(b.mul(b).mul(b));

        c = c.add(c.add(1));

        c = c.add(one().add(c).sub(a));

        d = d.add(d.mul(2).add(b.add(a).relu()));

        d = d.add(d.mul(3).add(b.sub(a).relu()));

        var e = c.sub(d);

        var f = e.mul(e);

        var g = f.div(2f);

        g = g.add(ten().div(f));

        Mockito.when(mockGradValue.data()).thenReturn(() -> createData(24.70f));

        assertEquals(createData(24.70f), g.data().get());

        g.backward();

        Mockito.when(mockGradValue.data()).thenReturn(() ->createData(138.83f));

        assertEquals(createData(138.83f), a.grad().data().get());

        Mockito.when(mockGradValue.data()).thenReturn(() -> createData(645.58f));

        assertEquals(createData(645.58f), b.grad().data().get());
    }
    
    protected abstract void assertEquals(D value1, D value2);
    
    protected abstract D add(D value1, D value2);
    protected abstract D mul(D value1, float value2);


    @Test
    public void test_hessian_vector() {

        var x = createGradValue(0.5f, true).name_("x");

        var y = createGradValue(0.6f, true).name_("y");

        var z = x.mul(x).add(y.mul(x).add(y.mul(y))).name_("z");

        var two = createGradValue(2, true).name_("two");

        z.backward(new BackwardConfig().with_keep_graph(true));

        var xGradAfterFirstBackward = x.grad();
        var yGradAfterFirstBackward = y.grad();

        Mockito.when(mockGradValue.data()).thenReturn(() -> createData(1.6f));

        assertEquals(createData(1.6f), xGradAfterFirstBackward.data().get());

        Mockito.when(mockGradValue.data()).thenReturn(() -> createData(1.7f));

        assertEquals(createData(1.7f), yGradAfterFirstBackward.data().get());

        var x_grad = createGradValue(add(mul(x.data().get(),2f),y.data().get()), false);
        var y_grad = createGradValue(add(x.data().get(), mul(y.data().get(),2f)), false);

        var grad_sum = x.grad().mul(two).add(y.grad());

        grad_sum.backward(new BackwardConfig());

        var xGradAfterSecondBackward = x.grad();
        var yGradAfterSecondBackward = y.grad();

        Mockito.when(mockGradValue.data()).thenReturn(() -> createData(6.6f));

        Assertions.assertSame(xGradAfterFirstBackward, xGradAfterSecondBackward);
        assertEquals(createData(6.6f), xGradAfterSecondBackward.data().get());

        Assertions.assertSame(yGradAfterFirstBackward, yGradAfterSecondBackward);

        Mockito.when(mockGradValue.data()).thenReturn(() -> createData(5.7f));

        assertEquals(createData(5.7f), yGradAfterSecondBackward.data().get());

        var x_hv = 5;
        var y_hv = 4;

        Assertions.assertArrayEquals(x.grad().getDataAsFloatArray(), x_grad.add(createGradValue(x_hv, false)).getDataAsFloatArray(), 0.001f);
        Assertions.assertArrayEquals(y.grad().getDataAsFloatArray(), y_grad.add(createGradValue(y_hv, false)).getDataAsFloatArray(), 0.001f);
    }

    private DemoAutogradValue<D> one() {
        return createGradValue(1, false);
    }

    private DemoAutogradValue<D> ten() {
        return createGradValue(10, false);
    }
}
