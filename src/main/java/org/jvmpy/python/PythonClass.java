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

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public abstract class PythonClass<M> {

	protected final M self;

	public PythonClass() {
		this.self = self();
	}

	protected abstract M self();
	
	@SuppressWarnings("unchecked")
	protected <S extends T, T> List<Pair<String, T>> getFields(Class<S> fieldClass) {
		try {

			List<Pair<String, T>> fields = new ArrayList<>();
			List<Field> fieldList = new ArrayList<>();
			populateAllFields(fieldList, self().getClass());

			for (Field field : fieldList) {
				field.setAccessible(true);
				if (fieldClass.isAssignableFrom(field.getType())) {
					T value = (T)field.get(self());
					fields.add(new ImmutablePair<>(field.getName(), (T) value));
				} else if (Attribute.class.isAssignableFrom(field.getType())) {
					Attribute<?> attribute = (Attribute<?>)field.get(self());
					if (attribute.value != null) {
						if (fieldClass.isAssignableFrom(attribute.value.getClass())) {
							T value = (T)attribute.value;
							fields.add(new ImmutablePair<>(field.getName(), (T) value));
						}
					}
				}
			}
			return fields;

		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public List<Field> populateAllFields(List<Field> fields, Class<?> type) {
		fields.addAll(Arrays.asList(type.getDeclaredFields()));

		if (type.getSuperclass() != null) {
			populateAllFields(fields, type.getSuperclass());
		}
		return fields;
	}
}
