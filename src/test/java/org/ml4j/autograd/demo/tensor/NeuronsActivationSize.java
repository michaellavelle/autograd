package org.ml4j.autograd.demo.tensor;

import org.jvmtorch.torch.Size;
import org.jvmtorch.torch.Torch;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.ml4j.nn.neurons.format.features.Dimension;
import org.ml4j.nn.neurons.format.features.DimensionScope;
import org.ml4j.nn.neurons.format.features.FeaturesFormat;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import static org.jvmpy.python.Python.tuple;

public class NeuronsActivationSize {

	public static Size getSize(NeuronsActivation neuronsActivation) {
		
		if (neuronsActivation.getNeurons() instanceof Neurons3D) {
					
			Neurons3D neurons = (Neurons3D)neuronsActivation.getNeurons();
			int[] vals = new int[neuronsActivation.getFormat().getFeaturesFormat().getDimensions().size()];
			List<String> names = new ArrayList<>();
			int index = 0;
			for (Dimension dim : neuronsActivation.getFormat().getFeaturesFormat().getDimensions()) {
				vals[index] = getVal(neurons, neuronsActivation.getExampleCount(), dim);
				names.add(getName(neurons, dim));
				index++;
			}
				
			if (neuronsActivation.getFeatureOrientation() == NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
				names.add("example");
				return new Size(new Size(vals),  new Size(neuronsActivation.getExampleCount())).names_(tuple(names));
			} else {
				names.add(0, "example");
				return new Size(new Size(neuronsActivation.getExampleCount()),  new Size(vals)).names_(tuple(names));
			}
		
		} else {
			neuronsActivation.getFormat().getDimensions();
			
			if (neuronsActivation.getFeatureOrientation() == NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
			
				return new Size(neuronsActivation.getFeatureCount(),  neuronsActivation.getExampleCount()).names_(tuple("feature", "example"));
			} else {
				return new Size(neuronsActivation.getExampleCount(),  neuronsActivation.getFeatureCount()).names_(tuple("example", "feature"));
			}
		}
		
	}
	
	public static NeuronsActivationFormat<?> getNeuronsActivationFormat(List<String> dimensionNames, DimensionScope dimensionScope) {

		List<Dimension> nonExampleDimensions = new ArrayList<>();
		List<Dimension> exampleDimensions = new ArrayList<>();
		List<Dimension> allDimensions = new ArrayList<>();

		for (String nameOrig : dimensionNames) {

			List<String> nameParts = new ArrayList<>();
			boolean composite = nameOrig.startsWith("[") && nameOrig.endsWith(("]"));
			if (composite) {
				String[] parts = nameOrig.replaceAll("\\[", "").replaceAll("\\]", "").split(",");
				for (String part : parts) {
					String namePart = part.trim();
					nameParts.add(namePart);
				}
			} else {
				nameParts.add(nameOrig);
			}
			
			List<Dimension> outerDimension = new ArrayList<>();
			boolean outerExampleDimension = false;

			int exampleDimensionCount = 0;

			
			for (String name : nameParts) {

				name = name.replace(' ', '_');

				
				String n = name.substring(0, 1).toUpperCase() + name.substring(1);
				String id = name.substring(0, 1).toUpperCase();
				int ind = n.indexOf("_");
				int prevInd = 0;
				String n2 = n;
				while (ind != -1) {
					n2 = n2.substring(prevInd, ind + 1) + n2.substring(ind + 1, ind + 2).toUpperCase()
							+ n2.substring(ind + 2);
					id = id + name.substring(ind + 1, ind + 2).toUpperCase();
					prevInd = ind;
					ind = n2.indexOf("_", prevInd + 1);
				}

				Dimension dimension = new Dimension(id, n2, dimensionScope);
			
				boolean exampleDimension = false;

				if (dimension.getId().equals(Dimension.INPUT_DEPTH.getId())) {
					dimension = Dimension.INPUT_DEPTH;
				} else if (dimension.getId().equals(Dimension.INPUT_WIDTH.getId())) {
					dimension = Dimension.INPUT_WIDTH;
				} else if (dimension.getId().equals(Dimension.INPUT_HEIGHT.getId())) {
					dimension = Dimension.INPUT_HEIGHT;
				} else if (dimension.getId().equals(Dimension.EXAMPLE.getId())) {
					dimension = Dimension.EXAMPLE;
					exampleDimension = true;
					exampleDimensionCount++;
				} else if (dimension.getId().equals(Dimension.FILTER_POSITIONS.getId())) {
					dimension = Dimension.FILTER_POSITIONS;
					exampleDimension = true;
					exampleDimensionCount++;
				} else if (dimension.getId().equals(Dimension.FILTER_HEIGHT.getId())) {
					dimension = Dimension.FILTER_HEIGHT;
				}  else if (dimension.getId().equals(Dimension.FILTER_WIDTH.getId())) {
					dimension = Dimension.FILTER_WIDTH;
				}

				else if (dimension.getId().equals(Dimension.OUTPUT_DEPTH.getId())) {
					dimension = Dimension.OUTPUT_DEPTH;
				} else if (dimension.getId().equals(Dimension.OUTPUT_WIDTH.getId())) {
					dimension = Dimension.OUTPUT_WIDTH;
				} else if (dimension.getId().equals(Dimension.OUTPUT_HEIGHT.getId())) {
					dimension = Dimension.OUTPUT_HEIGHT;
				} else if (dimension.getId().equals(Dimension.DEPTH.getId())) {
					dimension = Dimension.DEPTH;
				} else if (dimension.getId().equals(Dimension.WIDTH.getId())) {
					dimension = Dimension.WIDTH;
				} else if (dimension.getId().equals(Dimension.HEIGHT.getId())) {
					dimension = Dimension.HEIGHT;
				}

				else if (dimension.getId().equals(Dimension.FEATURE.getId())) {
					dimension = Dimension.FEATURE;
				}

				else if (dimension.getId().equals(Dimension.INPUT_FEATURE.getId())) {
					dimension = Dimension.INPUT_FEATURE;
				}

				else if (dimension.getId().equals(Dimension.OUTPUT_FEATURE.getId())) {
					dimension = Dimension.OUTPUT_FEATURE;
				}

				else {
					throw new IllegalStateException(
							"Unable to extract NeuronsActivationFormat - not all the dimensions of the Tensor have recognisable names - eg. "
									+ dimension.getName());
				}
				
				outerDimension.add(dimension);
				if (!outerExampleDimension) {
					outerExampleDimension = exampleDimension;
				}
				
			}
			
			if (composite) {
				
				Set<DimensionScope> scopes = outerDimension.stream().map(d -> d.getScope()).collect(Collectors.toSet());
				
				DimensionScope scope = scopes.size() == 1 ? scopes.iterator().next() : DimensionScope.ANY;
				
				Dimension compositeDimension = new Dimension.CompositeDimension(outerDimension, scope);
				if (outerExampleDimension) {
					if (exampleDimensionCount != outerDimension.size()) {
						throw new IllegalStateException("Unable to combine example and non-example dimensions into a composite dimension:" + outerDimension);
					}
					exampleDimensions.add(compositeDimension);
				} else {
					nonExampleDimensions.add(compositeDimension);
				}
				allDimensions.add(compositeDimension);
				
			} else {
				if (outerExampleDimension) {
					exampleDimensions.add(outerDimension.get(0));
				} else {
					nonExampleDimensions.add(outerDimension.get(0));
				}
				allDimensions.add(outerDimension.get(0));
			}
			
		

		}

		FeaturesFormat featuresFormat = new FeaturesFormat() {

			@Override
			public List<Dimension> getDimensions() {
				return nonExampleDimensions;
			}

		};

		if (exampleDimensions.size() < 1) {
			throw new IllegalStateException();
		}

		NeuronsActivationFeatureOrientation fo = allDimensions.get(0).equals(exampleDimensions.get(0))
				? NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET
				: NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET;

		NeuronsActivationFormat<?> f = new NeuronsActivationFormat<FeaturesFormat>(fo, featuresFormat,
				exampleDimensions);
		
		
		return f;
	}
	
	private static String getName(Neurons3D neurons, Dimension dim) {
		if (dim == Dimension.INPUT_DEPTH) {
			return "input_depth";
		} else if (dim == Dimension.DEPTH) {
			return "depth";
		} else if (dim == Dimension.OUTPUT_DEPTH) {
			return "output_depth";
		} else if (dim == Dimension.INPUT_WIDTH) {
			return "input_width";
		} else if (dim == Dimension.WIDTH) {
			return "width";
		} else if (dim == Dimension.OUTPUT_WIDTH) {
			return "output_width";
		} else if (dim == Dimension.INPUT_HEIGHT) {
			return "input_height";
		} else if (dim == Dimension.HEIGHT) {
			return "height";
		} else if (dim == Dimension.OUTPUT_HEIGHT) {
			return "output_height";
		} else if (dim == Dimension.EXAMPLE) {
			return "example";
		} else if (dim == Dimension.FILTER_POSITIONS) {
			return "filter positions";
		} else if (dim == Dimension.FILTER_WIDTH) {
			return "filter width";
		} else if (dim == Dimension.FILTER_HEIGHT) {
			return "filter height";
		} else {
			throw new IllegalArgumentException();
		}
	}

	private static int getVal(Neurons3D neurons, int examples, Dimension dim) {
		if (dim == Dimension.INPUT_DEPTH) {
			return neurons.getDepth();
		} else if (dim == Dimension.DEPTH) {
			return neurons.getDepth();
		} else if (dim == Dimension.OUTPUT_DEPTH) {
			return neurons.getDepth();
		} else if (dim == Dimension.INPUT_WIDTH) {
			return neurons.getWidth();
		} else if (dim == Dimension.WIDTH) {
			return neurons.getWidth();
		} else if (dim == Dimension.OUTPUT_WIDTH) {
			return neurons.getWidth();
		} else if (dim == Dimension.INPUT_HEIGHT) {
			return neurons.getHeight();
		} else if (dim == Dimension.HEIGHT) {
			return neurons.getHeight();
		} else if (dim == Dimension.OUTPUT_HEIGHT) {
			return neurons.getHeight();
		} else if (dim == Dimension.EXAMPLE) {
			return examples;
		} else {
			throw new IllegalArgumentException();
		}
	}
}
