package svm;

import java.util.List;

public class Sample {

		Sample() {}

		Sample(List<Double> feature, byte value)
		{
			this.feature = feature;
			this.value = value;
		}
		//Class
		public byte value;

		//Feature labels
		List<String> labels;

		//Features
		public List<Double> feature;

		@Override
		public String toString() {
			return value+"";
		}
	}