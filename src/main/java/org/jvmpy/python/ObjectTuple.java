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
package org.jvmpy.python;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.stream.IntStream;

public class ObjectTuple implements Tuple<Object>{

	private Object[] components;
	
	public ObjectTuple(Object first, Object...remaining) {
		this.components = new Object[1 + remaining.length];
		this.components[0] = first;
		IntStream.range(0, remaining.length).forEach(i -> this.components[i + 1] = remaining[i]);
	}

	@Override
	public Object[] getComponents() {
		return components;
	}
	
	public void put(int index, Object value) {
		this.components[index] = value;
	}

	@Override
	public Iterator<Object> iterator() {
		return Arrays.asList(components).iterator();
	}
	
	@SuppressWarnings("unchecked")
	@Override
	public <S> S[] getComponentsAsType(Class<S> type) {
		S[] ret = (S[]) Array.newInstance(type, components.length);
		for (int i = 0; i < components.length; i++) {
			ret[i] = (S)components[i];
		}
		return ret;
	}

	@Override
	public int length() {
		return components.length;
	}

	@Override
	public List<Object> asList() {
		return Arrays.asList(components);
	}

}
