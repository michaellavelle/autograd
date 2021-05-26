package org.jvmpy.python;

import java.util.function.Supplier;

public interface Indexed<E> extends Supplier<E> {
	
	int getIndex();
}
