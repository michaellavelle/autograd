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

public interface Subscriptable<T> {

	T[] getComponents();
	
	<S extends T> S[] getComponentsAsType(Class<S> type);
	
	default T get(int first) {
		T firstOne = getComponents()[first];
		return firstOne;
	}
	
	int length();

	default Object get(int first, int... remaining) {
		T component = getComponents()[first];
		if (remaining.length == 0) {
			return component;
		} else {
			if (component instanceof Subscriptable) {

				Subscriptable<?> subscriptable = (Subscriptable<?>) component;
				if (remaining.length == 1) {
					return subscriptable.get(remaining[0]);
				} else {
					int[] rem = new int[remaining.length - 1];
					for (int r = 1; r < remaining.length; r++) {
						rem[r - 1] = remaining[r];
					}
					return subscriptable.get(remaining[0], rem);
				}
			} else {
				throw new IllegalArgumentException(
						"TypeError: '" + component.getClass().getName() + "' object is not subscriptable");
			}
		}
	}

}
