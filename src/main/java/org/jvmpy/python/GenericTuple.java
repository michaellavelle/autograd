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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

public class GenericTuple<T> implements Tuple<T>{

	protected T[] components;
	
	@SuppressWarnings("unchecked")
	public GenericTuple(T first, T...remaining) {
		if (first != null) {
			this.components = (T[]) Array.newInstance(first.getClass(), remaining.length + 1);
			this.components[0] = first;
			for (int i = 0; i < remaining.length; i++) {
				this.components[i + 1] = (T)remaining[i];
			}
		} else {
			throw new IllegalArgumentException("First component of generic tuple cannot be null");
		}
	}
	
	@SuppressWarnings("unchecked")
	public GenericTuple(List<T> components) {
			T first = components.isEmpty() ? null: components.get(0);
			if (first == null && !components.isEmpty()) {
				throw new IllegalArgumentException("First is null");
			} else {
				if (!components.isEmpty()) {
					this.components = (T[]) Array.newInstance(first.getClass(), components.size());
					for (int i = 0; i < components.size(); i++) {
						this.components[i] = (T)components.get(i);
					}
				}
			}
	}
	
	@Override
	public List<T> asList() {
		return components == null ? new ArrayList<>() : Arrays.asList(components);
	}
	
	@SuppressWarnings("unchecked")
	protected GenericTuple(T...all) {
		this.components = all;
	}
	

	public void put(int index, T value) {
		this.components[index] = value;
	}


	@Override
	public T[] getComponents() {
		return components;
	}

	@Override
	public Iterator<T> iterator() {
		return Arrays.asList(components).iterator();
	}
	
	@SuppressWarnings("unchecked")
	@Override
	public <S extends T> S[] getComponentsAsType(Class<S> type) {
		S[] ret = (S[]) Array.newInstance(type, components.length);
		for (int i = 0; i < components.length; i++) {
			ret[i] = (S)components[i];
		}
		return ret;
	}
	
	@Override
	public String toString() {
		return components == null ? "()" : Arrays.asList(components).toString().replace('[', '(').replace(']', ')');
	}

	@Override
	public int length() {
		return components == null ? 0 : components.length;
	}
	

}
