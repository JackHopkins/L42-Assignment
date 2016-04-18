package svm.kernels;

public interface IKernel {
	public Float kernel(Float[] x, Float[] y);
	public String getType();
}
