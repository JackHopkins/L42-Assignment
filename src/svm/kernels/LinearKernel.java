package svm.kernels;

public class LinearKernel implements IKernel {

	@Override
	public Float kernel(Float[] x, Float[] y) {
		Float s= 0f; 
		for(int q = 0;q < x.length;q++) {
			s += x[q] * y[q]; 
		} 
		return s;
	}

	@Override
	public String getType() {
		return "linear";
	}

}
