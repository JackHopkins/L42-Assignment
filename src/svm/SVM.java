package svm;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;


public class SVM {

	int sampleNumber;
	int featureNumber;
	Float[] alpha;
	Float[] weights;
	double bias = 0.0;
	boolean useWeights = false;
	private Float[][] data;
	private Integer[] labels;
	private Parameters parameters;
	float[][] kernelResults;

	public SVM(List<Sample> samples, Parameters parameters) {
		Float[][] data = new Float[samples.size()][];
		Integer[] labels = new Integer[samples.size()];

		for (int i = 0; i < samples.size(); i++) {
			data[i] = new Float[samples.get(i).feature.size()];
			for (int j = 0; j < samples.get(i).feature.size(); j++) {
				data[i][j] = samples.get(i).feature.get(j).floatValue();
			}
			labels[i] = (int) samples.get(i).value;
		}
		sampleNumber = data.length;
		featureNumber = data[0].length;
		alpha = new Float[sampleNumber];
		for (int i = 0; i < sampleNumber; i++) alpha[i] = 0f;
		this.data = data;
		this.labels = labels;
		this.parameters = parameters;
	}
	public SVM(Float[][] data, Integer[] labels, Parameters parameters) {
		sampleNumber = data.length;
		featureNumber = data[0].length;
		alpha = new Float[sampleNumber];
		for (int i = 0; i < sampleNumber; i++) alpha[i] = 0f;
		this.data = data;
		this.labels = labels;
		this.parameters = parameters;
	}
	public int train() {

		//cache
		kernelResults = new float[sampleNumber][];
		for (int i=0;i<sampleNumber; i++) {
			kernelResults[i] = new float[sampleNumber];
			for (int j=0;j<sampleNumber;j++) {
				kernelResults[i][j] = parameters.kernel.kernel(data[i],data[j]);
			}
		}


		int iteration = 0;
		int passes = 0;

		for (iteration = 0; iteration < parameters.maxIterations; iteration++) {
			if (passes == parameters.passes) break;

			float alphaChanged = 0;
			for(int i=0;i<sampleNumber;i++) {

				float Ei= margin(data[i]) - labels[i];
				if( (labels[i]*Ei < -parameters.tolerance && alpha[i] < parameters.C)
						|| (labels[i]*Ei > parameters.tolerance && alpha[i] > 0) ){


					int j = i;
					while(j == i) j= exclusiveRandom(0, sampleNumber);
					float Ej= this.margin(data[j]) - labels[j];


					float oldAlphaI = alpha[i];
					float oldAlphaJ = alpha[j];
					float lowBound = 0;
					float highBound = parameters.C;
					if(labels[i] == labels[j]) {
						lowBound = Math.max(0, oldAlphaI+oldAlphaJ-parameters.C);
						highBound = Math.min(parameters.C, oldAlphaI+oldAlphaJ);
					} else {
						lowBound = Math.max(0, oldAlphaJ-oldAlphaI);
						highBound = Math.min(parameters.C, parameters.C+oldAlphaJ-oldAlphaI);
					}

					if(Math.abs(lowBound - highBound) < 1e-4) continue;

					float H = 2*computeKernel(i,j) - computeKernel(i,i) - computeKernel(j,j);
					if(H >= 0) continue;


					float alphaJ = oldAlphaJ - labels[j]*(Ei-Ej) / H;

					//Clip
					if(alphaJ<lowBound) alphaJ = lowBound;
					if(alphaJ>highBound) alphaJ = highBound;

					if(Math.abs(oldAlphaJ - alphaJ) < 1e-4) continue; 
					this.alpha[j] = alphaJ;
					float alphaI = oldAlphaI + labels[i]*labels[j]*(oldAlphaJ - alphaJ);
					this.alpha[i] = alphaI;


					bias = computeBias(i, j, bias, alphaJ, alphaI, oldAlphaJ, oldAlphaI, Ei, Ej);

					alphaChanged++;

				} 
			} 


			System.out.println("Iteration number: "+iteration+", Alpha diff: "+alphaChanged);
			
			if(alphaChanged == 0) {
				passes++;
			} else {
				passes= 0;
			}

		} 
		if(parameters.kernel.getType().equals("linear")) {

			weights = new Float[featureNumber];
			for(int j=0;j<featureNumber;j++) {
				float sum = 0f;
				for(int i=0;i<sampleNumber;i++) {
					sum += alpha[i] * labels[i] * data[i][j];
				}
				weights[j] = sum;
				useWeights = true;
			}
		} else {

			List<Float[]> newData = new ArrayList<Float[]>();
			List<Integer> newLabels = new ArrayList<Integer>();
			List<Float> newAlpha = new ArrayList<Float>();
			for(int i=0;i<sampleNumber;i++) {

				if(alpha[i] > 0) {
					newLabels.add(labels[i]);
					newAlpha.add(alpha[i]);
					newData.add(data[i]);
				}
			}


			data = newData.toArray(new Float[newData.size()][]);
			labels = newLabels.toArray(new Integer[newLabels.size()]);
			alpha = newAlpha.toArray(new Float[newAlpha.size()]);
			sampleNumber = this.data.length;
			//console.log("filtered training data from %d to %d support vectors.", data.length, this.data.length);
		}

		System.out.println(iteration);
		return iteration;

	}

	private double computeBias(int i , int j, double bias,float alphaJ,float alphaI, float oldAlphaI, float oldAlphaJ, float Ei, float Ej) {

		double bias1 = bias - Ei 
				- labels[i]*computeKernel(i,i)*(alphaI-oldAlphaI)
				- labels[j]*computeKernel(i,j)*(alphaJ-oldAlphaJ);
		double bias2 = bias - Ej 
				- labels[i]*computeKernel(i,j)*(alphaI-oldAlphaI)
				- labels[j]*computeKernel(j,j)*(alphaJ-oldAlphaJ);
		bias = 0.5*(bias1+bias2);
		if(alphaI < parameters.C && alphaI > 0) bias= bias1;
		if(alphaJ < parameters.C && alphaJ > 0) bias= bias2;
		return bias;
	}
	private int exclusiveRandom(int a, int b) {
		return (int) Math.floor(Math.random()*(b-a)+a);
	}
	private float computeKernel(int i, int j) {
		if (kernelResults != null) {
			return kernelResults[i][j];
		}
		return parameters.kernel.kernel(this.data[i], this.data[j]);
	}
	private float margin(Float[] features) {

		float f = new Float(bias);

		if(useWeights) {

			for(int j=0;j<featureNumber;j++) {
				f += features[j] * weights[j];
			}

		} else {

			for(int i=0;i<sampleNumber;i++) {
				f += alpha[i] * labels[i] * parameters.kernel.kernel(features, data[i]);
			}
		}

		return f;
	}
	public boolean checkCorrect(Sample sample) {
		byte value = sample.value;
		Double[] features = sample.feature.toArray(new Double[sample.feature.size()]);
		Float[] floatFeatures = new Float[sample.feature.size()];
		for (int i = 0; i < features.length; i++) {
			floatFeatures[i] = features[i].floatValue();
		}
		Integer prediction = predict(floatFeatures);
		System.out.println("Checking: Prediction="+prediction+", Actual="+(int)value);
		if ((int)value == prediction) {
			return true;
		}
		return false;
	}
	public float findError(Collection<Sample> samples) {
		float correct = 0f;
		for (Sample s : samples) {
			if (checkCorrect(s)) {
				correct ++;
			}
		}
		return correct / (float)samples.size();
	}
	public Float[] predict(Float[][] data) {
		Float[] margs = this.margins(data);
		for(int i=0;i<margs.length;i++) {
			margs[i] = margs[i] > 0f ? 1f : -1f;
		}
		return margs;
	}
	public Integer predict(Float[] inst) {
		if (margin(inst) > 0) return 1;
		return -1; 
	}
	private Float[] margins(Float[][] data) {
		int N = data.length;
		Float[] margins = new Float[N];
		for(int i=0;i<N;i++) {
			margins[i] = margin(data[i]);
		}
		return margins;
	}


}
