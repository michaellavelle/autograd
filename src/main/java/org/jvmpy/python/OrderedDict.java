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

import org.apache.commons.lang3.tuple.Pair;

import java.util.ArrayList;

public class OrderedDict<T> extends ArrayList<Pair<String, T>> {

	/**
	 * Default serialization id
	 */
	private static final long serialVersionUID = 1L;
	
	private Class<?> elementClass;
	
	public <S extends T> OrderedDict(Class<?> elementClass) {
		this.elementClass = elementClass;
	}

	Class<?> getElementClass() {
		return elementClass;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = super.hashCode();
		result = prime * result + ((elementClass == null) ? 0 : elementClass.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (!super.equals(obj))
			return false;
		if (getClass() != obj.getClass())
			return false;
		OrderedDict<?> other = (OrderedDict<?>) obj;
		if (elementClass == null) {
			if (other.elementClass != null)
				return false;
		} else if (!elementClass.equals(other.elementClass)) {
			return false;
		}
		return true;
	}
}
