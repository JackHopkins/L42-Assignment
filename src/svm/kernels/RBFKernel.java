package svm.kernels;

public class RBFKernel implements IKernel {

	
	private double sigma = 0.5d;
	public RBFKernel(double sigma) {
		this.sigma = sigma;
	}

	@Override
	public Float kernel(Float[] x, Float[] y) {
		float sum = 0f;
	      for(int q=0;q<x.length;q++) { sum += (x[q] - y[q])*(x[q] - y[q]); } 
	      return (float) Math.exp(-sum/(2.0*sigma *sigma));
	}

	@Override
	public String getType() {
		return "rbf";
	}

}
