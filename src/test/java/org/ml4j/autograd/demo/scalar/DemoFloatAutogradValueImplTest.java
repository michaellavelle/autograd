package org.ml4j.autograd.demo.scalar;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.BackwardConfig;
import org.ml4j.autograd.demo.DemoAutogradValue;
import org.ml4j.autograd.demo.DemoSize;
import org.ml4j.autograd.impl.AutogradValueProperties;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import java.util.ArrayList;
import java.util.List;

public class DemoFloatAutogradValueImplTest {
	
	@Mock
	private DemoSize size;

	protected List<AutogradValue<?, ?, ?>> registry;

	@BeforeEach
	public void setUp() {
		MockitoAnnotations.initMocks(this);
		this.registry = new ArrayList<>();
	}

	@Test
	public void testValidConstruction() {
		DemoAutogradValue<Float> autogradValue = new DemoFloatAutogradValueImpl(new AutogradValueProperties<DemoSize>().setRegistry(registry).setContext(size), () -> 2.6f);
		Assertions.assertEquals(2.6f, autogradValue.data().get());
		Assertions.assertArrayEquals(new float[] {2.6f } , autogradValue.getDataAsFloatArray(), 0.01f);
		Assertions.assertNotNull(autogradValue.getGradNode());
		Assertions.assertNotNull(autogradValue.getValueNode());
		Assertions.assertNotNull(autogradValue.getValueNode().prev());
		Assertions.assertTrue(autogradValue.getValueNode().prev().isEmpty());
		Assertions.assertSame(autogradValue, autogradValue.getValueNode().getValue().get());
		Assertions.assertNotNull(autogradValue.getGradNode().getValue());
		Assertions.assertNull(autogradValue.getGradNode().getValue().get());
		Assertions.assertFalse(autogradValue.getGradNode().isDisableNativeGradient());
		Assertions.assertNotNull(autogradValue.getGradNode().native_grad());
		Assertions.assertFalse(autogradValue.getGradNode().native_grad().isPresent());
		Assertions.assertNotNull(autogradValue.getGradNode().prev());
		Assertions.assertTrue(autogradValue.getGradNode().prev().isEmpty());
		Assertions.assertFalse(autogradValue.requires_grad());
		Assertions.assertNotNull(autogradValue.size());
		Assertions.assertSame(size, autogradValue.size());
		Assertions.assertSame(size, autogradValue.context());
		Assertions.assertNull(autogradValue.name());
		Assertions.assertNotNull(autogradValue.self());
		Assertions.assertEquals(autogradValue, autogradValue.self());
		Assertions.assertNotNull(autogradValue.getDataAsFloatArray());
		Assertions.assertTrue(autogradValue.getDataAsFloatArray().length == 1);
		Assertions.assertEquals(2.6f, autogradValue.getDataAsFloatArray()[0], 0.01f);
		Assertions.assertNull(autogradValue.grad());

	}
	
	@Test
	public void testConstructorWithNullSupplier_throwsIllegalArgumentException() {
		Assertions.assertThrows(IllegalArgumentException.class, () -> new DemoFloatAutogradValueImpl(new AutogradValueProperties<DemoSize>().setRegistry(registry).setContext(size), null));
	}
	
	@Test
	public void testRequiresGrad() {
		DemoAutogradValue<Float> autogradValue = new DemoFloatAutogradValueImpl(new AutogradValueProperties<DemoSize>().setRegistry(registry).setContext(size),() -> 2.6f);
		Assertions.assertFalse(autogradValue.requires_grad());
	
		autogradValue.requires_grad_(true);
		
		Assertions.assertTrue(autogradValue.requires_grad());
		
		autogradValue.requires_grad_(false);

		Assertions.assertFalse(autogradValue.requires_grad());

	}
	
	@Test
	public void testName() {
		DemoAutogradValue<Float> autogradValue = new DemoFloatAutogradValueImpl(new AutogradValueProperties<DemoSize>().setRegistry(registry).setContext(size), () -> 2.6f);
		Assertions.assertNull(autogradValue.name());
	
		autogradValue.name_("a");
		
		Assertions.assertEquals("a", autogradValue.name());
		
		autogradValue.name_("b");
		
		Assertions.assertEquals("b", autogradValue.name());

	}
	
	@Test
	public void testBackwardNoArgs() {
		DemoAutogradValue<Float> autogradValue = new DemoFloatAutogradValueImpl(new AutogradValueProperties<DemoSize>().setContext(size).setRegistry(registry).setRequires_grad(true), () -> 2.6f);
		autogradValue.backward();
	}
	
	@Test
	public void testBackwardDefaultBackwardConfig() {
		DemoAutogradValue<Float> autogradValue = new DemoFloatAutogradValueImpl(new AutogradValueProperties<DemoSize>().setContext(size).setRegistry(registry).setRequires_grad(true),() -> 2.6f);
		autogradValue.backward(new BackwardConfig());
	}
	
	@Test
	public void testBackwardNullBackwardConfig() {
		DemoAutogradValue<Float> autogradValue = new DemoFloatAutogradValueImpl(new AutogradValueProperties<DemoSize>().setContext(size).setRegistry(registry).setRequires_grad(true),() -> 2.6f);
		Assertions.assertThrows(IllegalArgumentException.class, () -> autogradValue.backward((BackwardConfig) null));
	}
	
	@Test
	public void testBackwardDefaultBackwardConfigWithKeepGraphTrue() {
		DemoAutogradValue<Float> autogradValue = new DemoFloatAutogradValueImpl(new AutogradValueProperties<DemoSize>().setContext(size).setRegistry(registry).setRequires_grad(true),() -> 2.6f);
		autogradValue.backward(new BackwardConfig().with_keep_graph(true));
	}
	
	@Test
	public void testBackwardDefaultBackwardConfigWithKeepGraphFalse() {
		DemoAutogradValue<Float> autogradValue = new DemoFloatAutogradValueImpl(new AutogradValueProperties<DemoSize>().setContext(size).setRegistry(registry).setRequires_grad(true),() -> 2.6f);
		autogradValue.backward(new BackwardConfig().with_keep_graph(false));
	}
	
	@Test
	public void testBackwardDefaultBackwardConfigWithZeroGradTrue() {
		DemoAutogradValue<Float> autogradValue = new DemoFloatAutogradValueImpl(new AutogradValueProperties<DemoSize>().setContext(size).setRegistry(registry).setRequires_grad(true),() -> 2.6f);
		autogradValue.backward(new BackwardConfig().with_zero_grad(true));
	}
	
	@Test
	public void testBackwardDefaultBackwardConfigWithZeroGradFalse() {
		DemoAutogradValue<Float> autogradValue = new DemoFloatAutogradValueImpl(new AutogradValueProperties<DemoSize>().setContext(size).setRegistry(registry).setRequires_grad(true),() -> 2.6f);
		autogradValue.backward(new BackwardConfig().with_zero_grad(false));
	}
	
	@Test
	public void testAdd() {
		DemoAutogradValue<Float> first = new DemoFloatAutogradValueImpl(new AutogradValueProperties<DemoSize>().setContext(size).setRegistry(registry).setRequires_grad(true),() -> 2.6f);
		DemoAutogradValue<Float> second = new DemoFloatAutogradValueImpl(new AutogradValueProperties<DemoSize>().setContext(size).setRegistry(registry).setRequires_grad(true),() -> 3.6f);

		DemoAutogradValue<Float> result = first.add(second);
		
		Assertions.assertNotNull(result);
		
		Assertions.assertNotNull(result.data().get());
		
		Assertions.assertEquals(6.2f, result.data().get().floatValue());
		
		Assertions.assertEquals(2, result.getValueNode().prev().size());
		
		Assertions.assertTrue(result.getValueNode().prev().contains(first.getValueNode()));
		Assertions.assertTrue(result.getValueNode().prev().contains(second.getValueNode()));
		Assertions.assertNotNull(result.size());
		Assertions.assertNotNull(result.context());

		// Assert that grads are null - no backward operation has yet been performed
		Assertions.assertNull(result.grad());
		Assertions.assertNull(first.grad());
		Assertions.assertNull(second.grad());
	}
	
	@Disabled
	@Test
	public void testAddWithBackward() {
		DemoAutogradValue<Float> first = new DemoFloatAutogradValueImpl(new AutogradValueProperties<DemoSize>().setContext(size).setRegistry(registry).setRequires_grad(true),() -> 2.6f);
		DemoAutogradValue<Float> second = new DemoFloatAutogradValueImpl(new AutogradValueProperties<DemoSize>().setContext(size).setRegistry(registry).setRequires_grad(true),() -> 3.6f);

		DemoAutogradValue<Float> result = first.add(second);
		
		Assertions.assertNotNull(result);
		
		Assertions.assertNotNull(result.data().get());
		
		Assertions.assertEquals(6.2f, result.data().get().floatValue());
		
		Assertions.assertEquals(2, result.getValueNode().prev().size());
		
		Assertions.assertTrue(result.getValueNode().prev().contains(first.getValueNode()));
		Assertions.assertTrue(result.getValueNode().prev().contains(second.getValueNode()));
		Assertions.assertNotNull(result.size());
		Assertions.assertNotNull(result.context());
		
		result.backward();

		// Assert that grads are null - no backward operation has yet been performed
		Assertions.assertNotNull(first.grad());
		Assertions.assertNotNull(second.grad());
		Assertions.assertNull(result.grad());
	}
	
	@Test
	public void testSetGradientDoesNotThrowIllegalStateException_WhenRequiresGradIsFalse_WhenGradNotAlreadySet() {
		DemoAutogradValue<Float> first = new DemoFloatAutogradValueImpl(new AutogradValueProperties<DemoSize>().setRegistry(registry).setContext(size),() -> 2.6f);
		DemoAutogradValue<Float> grad = new DemoFloatAutogradValueImpl(new AutogradValueProperties<DemoSize>().setRegistry(registry).setContext(size),() -> 3.1f);

		first.getGradNode().setValue(() -> grad);

		Assertions.assertSame(grad, first.getGradNode().getValue().get());
	}
	
	@Test
	public void testSetGradientDoesNotThrowIllegalStateException_WhenRequiresGradIsTrue_WhenGradNotAlreadySet() {
		DemoAutogradValue<Float> first = new DemoFloatAutogradValueImpl(new AutogradValueProperties<DemoSize>().setRegistry(registry).setContext(size).setRequires_grad(true),() -> 2.6f);
		DemoAutogradValue<Float> grad = new DemoFloatAutogradValueImpl(new AutogradValueProperties<DemoSize>().setRegistry(registry).setContext(size),() -> 3.1f);

		first.getGradNode().setValue(() -> grad);
	
		
		Assertions.assertSame(grad, first.getGradNode().getValue().get());
	}
	
	
	@Test
	public void testSetGradientThrowsIllegalStateException_WhenRequiresGradIsFalse_WhenGradAlreadySet() {
		DemoAutogradValue<Float> first = new DemoFloatAutogradValueImpl(new AutogradValueProperties<DemoSize>().setRegistry(registry).setContext(size),() -> 2.6f);
		DemoAutogradValue<Float> grad = new DemoFloatAutogradValueImpl(new AutogradValueProperties<DemoSize>().setRegistry(registry).setContext(size),() -> 3.1f);

		first.getGradNode().setValue(() -> grad);
		
		Assertions.assertThrows(IllegalStateException.class, () -> first.getGradNode().setValue(() -> grad));
	}
	
	@Test
	public void testSetGradientThrowsIllegalStateException_WhenRequiresGradIsTrue_WhenGradAlreadySet() {
		DemoAutogradValue<Float> first = new DemoFloatAutogradValueImpl(new AutogradValueProperties<DemoSize>().setContext(size).setRegistry(registry).setRequires_grad(true),() -> 2.6f);
		DemoAutogradValue<Float> grad = new DemoFloatAutogradValueImpl(new AutogradValueProperties<DemoSize>().setContext(size).setRegistry(registry),() -> 3.1f);

		first.getGradNode().setValue(() -> grad);
		
		Assertions.assertThrows(IllegalStateException.class, () -> first.getGradNode().setValue(() -> grad));
	}

}
