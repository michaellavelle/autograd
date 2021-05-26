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
package org.jvmtorch.torch;

import com.codepoetics.protonpack.StreamUtils;
import org.jvmpy.python.IntTuple;
import org.jvmpy.python.Tuple;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.jvmpy.python.Python.tuple;

public class Size extends IntTuple  {

	private int numel;
	public Size[] sizeComponents;
	private String name;
	private final List<Size> alternates;

	public Size() {
		super(new int[0]);
		alternates = new ArrayList<>();
		sizeComponents = new Size[0];
		this.numel = 1;
	}

	public Size(Size... components) {
		super(getDimensions(components));
		this.alternates = new ArrayList<>();
		this.sizeComponents = components;
		if (components.length == 1) {
			this.name = components[0].name;
			this.numel = components[0].numel;
		}
		this.numel = Arrays.stream(components).mapToInt(s -> s.numel).reduce(1, (l, r) -> l * r);
	}
	
	public Size(List<Size> components) {
		super(getDimensions(components));
		this.alternates = new ArrayList<>();
		if (components != null && components.size() == 1) {
			this.name = components.get(0).name;
			this.numel = components.get(0).numel;
		}
		this.sizeComponents = components.toArray(new Size[components.size()]);
		this.numel = Arrays.stream(this.sizeComponents).mapToInt(s -> s.numel).reduce(1, (l, r) -> l * r);
	}

	public Size(int... components) {
		this(IntStream.of(components).mapToObj(i -> new Size(i)).toArray(i -> new Size[i]));
	}

	public Size(int numel) {
		super(numel);
		this.alternates = new ArrayList<>();
		this.numel = numel;
		this.sizeComponents = new Size[] { this };
	}


	private Tuple<String> extractNames(String name) {
		if (name != null) {
			String[] parts = name.split(", ");
			List<String> names = new ArrayList<>();
			for (String part : parts) {
				names.add(part.replaceAll(",", "").trim());
			}
			return tuple(names);

		} else {
			return tuple(Arrays.asList("None"));
		}
	}

	public Tuple<String> dimensionNames() {
		if (sizeComponents.length == 1) {
			if (name != null) {
				return tuple(Arrays.asList(name));
			} else {
				return tuple(Arrays.asList("None"));
			}
		}
		if (this.components == null || this.components.length == 0) {
			return tuple(new ArrayList<>());
		}
		List<String> ns = new ArrayList<>();
		for (int i = 0; i < dimensions().length; i++) {
			ns.add("None");
		}
		// return tuple(ns);
		return tuple(
				decompose().stream().flatMap(s -> s.dimensionNames().asList().stream()).collect(Collectors.toList()));
	}

	public Size names_(Tuple<String> names) {
		if (names != null) {
			return names_(names, true);
		}

		return this;
	}

	private Size names_(Tuple<String> names, boolean extract) {
		List<Size> decomposed = decompose();
		if (names.length() != 0 && decomposed.size() != names.length()) {
			if (extract) {
				return names_(tuple(names.asList().stream().flatMap(n -> extractNames(n).asList().stream())
						.collect(Collectors.toList())), false);
			} else {
				throw new IllegalArgumentException(decomposed.size() + ":" + names.length());
			}
		} else {
			StreamUtils.zipWithIndex(decomposed.stream())
					.forEach(i -> i.getValue().name = names.get((int) i.getIndex()));
		}
		return this;
	}

	public Size t() {
		Size matrixSize = asMatrixSize();
		Size t = new Size(matrixSize.getSecondComponent(), matrixSize.getFirstComponent());
		return t;

	}

	public Size asMatrixSize() {
		return asMatrixSize(true);
	}

	public Size asMatrixSize(boolean throwException) {
		if (sizeComponents.length == 2) {
			return this;
		} else if (sizeComponents.length == 1 && sizeComponents[0].sizeComponents.length == 2) {
			return sizeComponents[0].asMatrixSize();
		}
		else if (components == null || components.length == 0) {
			return new Size(1, 1);
		} else if (components.length == 2) {
			return new Size(new Size(components[0]), new Size(components[1]));
		} else if (components.length == 1) {
			return new Size(new Size(1), new Size(components[0]));
		} else if (!this.alternates.isEmpty()) {
			List<Size> reversedAlternates = new ArrayList<>();
			reversedAlternates.addAll(this.alternates);
			Collections.reverse(reversedAlternates);
			for (Size alternateSize : reversedAlternates) {
				Size matrixSize = alternateSize.asMatrixSize(false);
				if (matrixSize != null) {
					return matrixSize;
				}
			}
		}
		if (throwException) {
			throw new IllegalStateException("Size cannot be converted into matrix size:" + this);
		} else {
			return null;
		}

	}

	public Size getFirstComponent() {
		if (sizeComponents.length < 1) {
			throw new IllegalStateException("Size does not have a first component");
		}
		return sizeComponents[0];
	}

	public Size getSecondComponent() {
		if (sizeComponents.length < 2) {
			throw new IllegalStateException("Size does not have a second component");
		}
		return sizeComponents[1];
	}

	
	public int numel() {
		return numel;
	}

	public int len() {
		return sizeComponents.length;
	}

	public List<Size> decompose() {
		if (sizeComponents.length == 1) {
			return Arrays.asList(this);
		} else {
			return Arrays.stream(sizeComponents).flatMap(c -> c.decompose().stream()).collect(Collectors.toList());
		}
	}

	private List<Integer> getDimensions() {
		if (this.components == null || this.components.length == 0) {
			return new ArrayList<>();
		}
		return decompose().stream().map(s -> s.numel()).collect(Collectors.toList());
	}

	private static int[] getDimensions(Size[] components) {
		List<Size> decomposed = Arrays.stream(components).flatMap(c -> c.decompose().stream())
				.collect(Collectors.toList());
		int[] dims = new int[decomposed.size()];
		for (int i = 0; i < decomposed.size(); i++) {
			dims[i] = decomposed.get(i).numel();
		}
		return dims;
	}

	private static int[] getDimensions(List<Size> components) {
		List<Size> decomposed = components.stream().flatMap(c -> c.decompose().stream()).collect(Collectors.toList());
		int[] dims = new int[decomposed.size()];
		for (int i = 0; i < decomposed.size(); i++) {
			dims[i] = decomposed.get(i).numel();
		}
		return dims;
	}

	public int[] dimensions() {
		List<Integer> dims = getDimensions();
		int[] ret = new int[dims.size()];
		for (int i = 0; i < ret.length; i++) {
			ret[i] = dims.get(i);
		}
		return ret;
	}

	@Override
	public String toString() {
		return "torch.Size([" + dimensionsString() + "], names=" + dimensionNames() + ")";
	}

	String dimensionsString() {
		String s = getDimensions().toString();
		return s.substring(1, s.length() - 1);
	}

	public List<Size> getAlternates() {
		return alternates;
	}

	public Size matmul(Size other) {
		return SizeMatcher.matmul(this, other);
	}

}
