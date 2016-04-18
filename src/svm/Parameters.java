package svm;

import svm.kernels.IKernel;
import svm.kernels.LinearKernel;
import svm.kernels.RBFKernel;

public class Parameters {

	public float C = 1e10f;
	public int maxIterations = 10000;
	public int passes = 10;
	public float tolerance = (float) 1e-4;
	public IKernel kernel = new LinearKernel();
	 
}
