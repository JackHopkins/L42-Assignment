package svm;

import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.util.Arrays;
import java.util.List;

public class Main {

	public static void main(String[] args) throws UnsupportedEncodingException, FileNotFoundException {

		Sample[] samples = Utils.parse(false);
		List<Sample> train = Arrays.asList(Arrays.copyOfRange(samples, samples.length/5, samples.length));
		List<Sample> test = Arrays.asList(Arrays.copyOfRange(samples, 0, samples.length/5));
		SVM svm = new SVM(train, new Parameters());
		svm.train();
		float error = svm.findError(test);
		System.out.println(error);
	}

}
