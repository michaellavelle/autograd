package org.jvmtorch.torch;

import org.jvmpy.python.Tuple;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.jvmpy.python.Python.tuple;

/**
 *  Prototype SizeMatcher - all the ugly code re: size matching in one class to be refactored - TODO
 */
public class SizeMatcher {
	
	public static Size matmul(Size firstSize, Size secondSize) {
		Optional<SizeComponentMatch> match = getMatMulMatches(firstSize, secondSize);
		if (match.isPresent()) {
			SizeComponentMatch m = match.get();
			Size first = m.firstComponentPrefix;
			Size second = m.secondComponentSuffix;
			if (m.secondComponentPrefix.isPresent()) {
				second = new Size(m.secondComponentPrefix.get(), second);
			}
			Size ret = new Size(first, second);

			return ret;
		}

		throw new IllegalArgumentException("Unable to match sizes");
	}
	
	public static boolean isSizeMatch(Size first, Size second) {
		List<Size> firsts = getAllAlternates(first);
		List<Size> seconds = getAllAlternates(second);
		for (int f = 0; f < firsts.size(); f++) {
			for (int s = 0; s < seconds.size(); s++) {
				boolean directDimensionsMatch = isDirectDimensionsMatch(first, second);
				if (directDimensionsMatch) {
					if (isDirectDimensionsMatch(first, second)) {
						if (isDirectNamesMatch(first, second)) {
							return true;
						}
					} 
				}
			}
		}
		return false;
	}
	
	private static boolean isDirectDimensionsMatch(Size first, Size second) {
		return IntStream.of(first.dimensions()).boxed().collect(Collectors.toList()).equals(IntStream.of(first.dimensions()).boxed().collect(Collectors.toList()));
	}
	
	private static boolean isDirectNamesMatch(Size first, Size second) {
		List<String> firstNames = toScopeIndependentNamesList(first.dimensionNames().asList());
		List<String> secondNames = toScopeIndependentNamesList(second.dimensionNames().asList());

		if (firstNames.size() == secondNames.size()) {
			removeNones(firstNames, secondNames);
		}
		
		boolean match = firstNames.equals(secondNames);
		if (!match) {
			
			if (secondNames.size() >= 2 && secondNames.get(0).equals("example") && firstNames.equals(Arrays.asList("example", "feature"))) {
				return true;
			} else if (firstNames.size() >= 2 && firstNames.get(0).equals("example") && secondNames.equals(Arrays.asList("example", "feature"))) {
				return true;
			} else if (secondNames.size() >= 2 && secondNames.get(0).equals("feature") && firstNames.equals(Arrays.asList("feature", "example"))) {
				return true;
			} else if (firstNames.size() >= 2 && firstNames.get(0).equals("feature") && secondNames.equals(Arrays.asList("feature", "example"))) {
				return true;
			} else if (secondNames.size() >= 2 && secondNames.get(0).equals("feature") && firstNames.equals(Arrays.asList("feature", "feature"))) {
				return true;
			} else if (firstNames.size() >= 2 && firstNames.get(0).equals("feature") && secondNames.equals(Arrays.asList("feature", "feature"))) {
				return true;
			} 
			return false;
		} else {
			return true;
		}
	}
	
	
	private static void removeNones(List<String> firstNames, List<String> secondNames) {
		List<String> firsts = new ArrayList<>();
		List<String> seconds = new ArrayList<>();

		for (int i = 0; i < firstNames.size(); i++) {
			if (!firstNames.get(i).equals("None") && !secondNames.get(i).equals("None")) {
				firsts.add(firstNames.get(i));
				seconds.add(secondNames.get(i));
			}
		}
		firstNames.clear();
		secondNames.clear();
		firsts.addAll(firstNames);
		seconds.addAll(secondNames);
	}

	private static List<String> toScopeIndependentNamesList(List<String> strings) {
		List<String> returnValues = new ArrayList<>();
		for (String s : strings) {
			s = s.replaceAll("input_", "");
			s = s.replaceAll("output_", "");
			returnValues.add(s);
		}
		
		return returnValues;
	}
	
	private static List<Size> getAllAlternates(Size size) {
		return populateAllAlternates(size, new ArrayList<>());
	}
	
	private static List<Size> populateAllAlternates(Size size, List<Size> allAlternates) {
		allAlternates.add(size);
		for (Size s : size.getAlternates()) {
			populateAllAlternates(s, allAlternates);
		}
		return allAlternates;
	}


	public static Optional<SizeComponentMatch> getMatMulMatches(Size first, Size second) {

		// Can we split the first size into two, so that the last part is within the
		// second size.
		List<Size> firstTwoSplits = getTwoSplits(first);
		List<SizeComponentMatch> matches = new ArrayList<>();
		for (Size firstTwoSplit : firstTwoSplits) {
			Size lastPartOfFirst = firstTwoSplit.sizeComponents[1];
			Optional<SizeComponentMatch> match = isContainedWithinSecond(lastPartOfFirst, second,
					firstTwoSplit.sizeComponents[0]);
			if (match.isPresent()) {
				matches.add(match.get());
			}
		}

		List<Size> secondTwoSplits = getTwoSplits(second);
		// Can we split the second size into two, so that the first part of the second
		// is the end of the first

		for (Size secondTwoSplit : secondTwoSplits) {
			Size firstPartOfSecond = secondTwoSplit.sizeComponents[0];
			Optional<SizeComponentMatch> match = isEndingOfFirst(firstPartOfSecond, first,
					secondTwoSplit.sizeComponents[1]);
			if (match.isPresent()) {
				matches.add(match.get());
			}
		}
		if (matches.isEmpty()) {
			return Optional.empty();
		} else {
			Integer maxLength = null;
			SizeComponentMatch bestMatch = null;
			for (SizeComponentMatch match : matches) {
				int sharedLength = match.shared.len();
				if (maxLength == null || sharedLength > maxLength) {
					bestMatch = match;
					maxLength = sharedLength;
				}
			}
			return Optional.of(bestMatch);
		}
	}
	
	private static Optional<SizeComponentMatch> isContainedWithinSecond(Size searchFor, Size source,
                                                                        Size firstComponentPrefix) {
		if (source.dimensionsString().contains(searchFor.dimensionsString())) {
			int ind = source.dimensionsString().indexOf(searchFor.dimensionsString());
			// throw new RuntimeException("Found:" + searchFor.dimensionsString() + " in " +
			// source.dimensionsString() + " at index:" + ind);
			if (ind == 0) {
				int firstComponentsCount = searchFor.sizeComponents.length;
				List<Size> secondDecomposed = source.decompose();
				Size remaining = new Size(secondDecomposed.subList(firstComponentsCount, secondDecomposed.size()));

				return Optional.of(new SizeComponentMatch(firstComponentPrefix, searchFor, remaining));
			} else {
				int firstComponentsCount = searchFor.sizeComponents.length;
				List<Size> prefixComponents = new ArrayList<>();
				List<Size> secondDecomposed = source.decompose();
				int ind2 = 0;
				String prefixComponentsString = "";
				while (ind2 < secondDecomposed.size() - 1 && prefixComponentsString.length() < ind) {
					Size s = secondDecomposed.get(ind2++);
					if ((prefixComponentsString + s.dimensionsString()).length() < ind) {
						prefixComponents.add(s);
						prefixComponentsString = prefixComponentsString + ", " + s.dimensionsString();
					}
				}
								
				int from = firstComponentsCount + prefixComponents.size();
				if (from < secondDecomposed.size()) {
					Size remaining = new Size(secondDecomposed.subList(from, secondDecomposed.size()));

					Size secondComponentPrefix = new Size(prefixComponents);
					return Optional.of(
							new SizeComponentMatch(firstComponentPrefix, searchFor, remaining, secondComponentPrefix));

				} else {
					return Optional.empty();
				}

			}
		} else {
			return Optional.empty();
		}
	}
	
	private static List<Size> getTwoSplits(Size size) {
		List<Size> decomposed = size.decompose();
		List<Size> twoSplits = new ArrayList<>();
		for (int i = 1; i < decomposed.size(); i++) {
			List<Size> first = decomposed.subList(0, i);
			List<Size> second = decomposed.subList(i, decomposed.size());
			Size twoSplit = new Size(new Size(first.toArray(new Size[first.size()])),
					new Size(second.toArray(new Size[second.size()])));
			twoSplits.add(twoSplit);
		}
		return twoSplits;
	}

	private static Optional<SizeComponentMatch> isEndingOfFirst(Size searchFor, Size source, Size secondComponentSuffix) {
			
		if (source.dimensionsString().endsWith(searchFor.dimensionsString())) {
			int ind = source.dimensionsString().lastIndexOf(searchFor.dimensionsString());
			int ind2 = 0;
			List<Size> prefixComponents = new ArrayList<>();
			List<Size> firstDecomposed = source.decompose();

			String prefixComponentsString = "";
			while (ind2 < firstDecomposed.size() - 1 && prefixComponentsString.length() < ind) {
				Size s = firstDecomposed.get(ind2++);
				if ((prefixComponentsString + s.dimensionsString()).length() < ind) {
					prefixComponents.add(s);
					prefixComponentsString = prefixComponentsString + ", " + s.dimensionsString();
				}
			}

			return Optional.of(new SizeComponentMatch(new Size(prefixComponents), searchFor, secondComponentSuffix));

		} else {
			return isProductOfEndingOfFirst(searchFor, source, secondComponentSuffix);
		}
	}

	private static Optional<SizeComponentMatch> isProductOfEndingOfFirst(Size searchFor, Size source,
                                                                         Size secondComponentSuffix) {
		int[] firstDimensions = source.dimensions();
		Tuple<String> firstDimensionNames = source.dimensionNames();
		SizeComponentMatch match = null;
		for (int i = 0; i < firstDimensions.length; i++) {

			int firstEndingProduct = 1;
			List<String> names = new ArrayList<>();
			int[] firstEnding = new int[i + 1];
			for (int j = 0; j < firstEnding.length; j++) {
				firstEnding[j] = firstDimensions[firstDimensions.length - 1 - i + j];
				firstEndingProduct = firstEndingProduct * firstEnding[j];
				if (firstDimensionNames != null) {
					names.add(firstDimensionNames.get(firstDimensions.length - 1 - i + j));
				}
			}

			Size firstEndingSize = new Size(firstEnding);
			Size firstEndingProductSize = new Size(firstEndingProduct).names_(tuple(names.toString()));

			if (firstEndingProductSize.dimensionsString().equals(searchFor.dimensionsString())) {
				int ind = source.dimensionsString().lastIndexOf(firstEndingSize.dimensionsString());
				int ind2 = 0;
				List<Size> prefixComponents = new ArrayList<>();
				List<Size> firstDecomposed = source.decompose();

				String prefixComponentsString = "";
				while (ind2 < firstDecomposed.size() - 1 && prefixComponentsString.length() < ind) {
					Size s = firstDecomposed.get(ind2++);
					if ((prefixComponentsString + s.dimensionsString()).length() < ind) {
						prefixComponents.add(s);
						prefixComponentsString = prefixComponentsString + ", " + s.dimensionsString();
					}
				}				
				Size alternate = new Size(new Size(prefixComponents), firstEndingProductSize);
				if (!prefixComponents.isEmpty()) {
					if (!source.getAlternates().contains(alternate)) {
						source.getAlternates().add(alternate);
					}

				match = new SizeComponentMatch(new Size(prefixComponents), firstEndingProductSize,
						secondComponentSuffix);
				}


			}

		}
		return Optional.ofNullable(match);

	}

	static class SizeComponentMatch {

		Size firstComponentPrefix;
		Size shared;
		Optional<Size> secondComponentPrefix;
		Size secondComponentSuffix;

		public SizeComponentMatch(Size firstComponentPrefix, Size shared, Size secondComponentSuffix) {
			this.firstComponentPrefix = firstComponentPrefix;
			this.secondComponentPrefix = Optional.empty();
			this.secondComponentSuffix = secondComponentSuffix;
			this.shared = shared;
		}

		public SizeComponentMatch(Size firstComponentPrefix, Size shared, Size secondComponentSuffix,
                                  Size secondComponentPrefix) {
			this.firstComponentPrefix = firstComponentPrefix;
			this.secondComponentPrefix = Optional.of(secondComponentPrefix);
			this.secondComponentSuffix = secondComponentSuffix;
			this.shared = shared;
		}

		@Override
		public String toString() {
			return "SizeComponentMatch [firstComponentPrefix=" + firstComponentPrefix + ", shared=" + shared
					+ ", secondComponentPrefix=" + secondComponentPrefix + ", secondComponentSuffix="
					+ secondComponentSuffix + "]";
		}

	}


}
