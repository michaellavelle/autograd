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

import com.codepoetics.protonpack.StreamUtils;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Provides methods that can be statically imported
 * that emulate standard Python functions.
 *
 * @author Michael Lavelle
 */
public class Python {

	public static boolean True = true;
	public static boolean False = false;

	public static <T> void print(T o) {
		System.out.println(o == null ? "None" : o.toString());
	}

	public static <T> void print(Object...objects) {
		StringBuilder sb = new StringBuilder();
		if (objects == null) {
			print(objects, sb);
		} else {
			Arrays.stream(objects).forEach(o -> print(o, sb));
		}
		System.out.println(sb.toString().trim());
	}

	private static <T> void print(T o, StringBuilder sb) {
		sb.append(o == null ? " None" : " " + o.toString());
	}

   /*
   public static <T> void print(T[] o) {
      System.out.println(o == null ? "None" : Arrays.asList(o).toString());
   }
   */

	public static Tuple<Integer> inttuple(int first, int...remaining) {
		return new IntTuple(first, remaining);
	}

	public static <T> Iterable<T> enumerate(Iterable<T> iterable) {
		return iterable;
	}

	public static <T> Iterable<Tuple<?>> enumerate(List<T> components) {
		return StreamUtils.zipWithIndex(components.stream()).map((e) -> new ObjectTuple((int)e.getIndex(), e.getValue())).collect(Collectors.toList());
	}

	public static <T> Iterable<Tuple<?>> enumerate(List<T> components, int index) {
		return StreamUtils.zipWithIndex(components.subList(index, components.size()).stream()).map(e -> new ObjectTuple((int)e.getIndex(), e.getValue())).collect(Collectors.toList());
	}

	public static <T> Iterable<Indexed<T>> enumerate(Supplier<Stream<T>> componentStreamSupplier, int index) {
		return () -> StreamUtils.zipWithIndex(componentStreamSupplier.get()).filter(e -> e.getIndex() >= index).map(e -> IndexedImpl.create(e.getValue(), (int)e.getIndex())).iterator();
	}

	public static int[] range(int r) {
		return IntStream.range(0, r).toArray();
	}

	@SafeVarargs
	public static <T> Tuple<T> tuple(T first, T...components) {
		return new GenericTuple<T>(first, components);
	}


	public static <T> Tuple<T> tuple(List<T> components) {
		return new GenericTuple<T>(components);
	}

	@SuppressWarnings("unchecked")
	public static <T> T[] list(OrderedDict<T> orderedDict) {
		List<T> list = new ArrayList<>();
		orderedDict.forEach(i -> list.add(i.getRight()));
		return list.toArray((T[]) Array.newInstance(orderedDict.getElementClass(), list.size()));
	}

	public static <T> int len(T[] a) {
		return a.length;
	}

	public static <T> int len(Lengthable l) {
		return l.len();
	}

	public static <T> int len(Tuple<T> a) {
		return a.length();
	}

	public static <T> String type(T type) {
		return "<class '" + type.getClass().getSimpleName() + "'>";
	}

	public static String type(Tuple<?> type) {
		return "<class 'tuple'>";
	}
}