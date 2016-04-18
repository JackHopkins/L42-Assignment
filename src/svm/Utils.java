package svm;


import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.univocity.parsers.tsv.TsvParser;
import com.univocity.parsers.tsv.TsvParserSettings;

public class Utils {
	public static Sample[] parse(boolean multiclass) throws UnsupportedEncodingException,
	FileNotFoundException {
		TsvParserSettings settings = new TsvParserSettings();
		//the file used in the example uses '\n' as the line separator sequence.
		//the line separator sequence is defined here to ensure systems such as MacOS and Windows
		//are able to process this file correctly (MacOS uses '\r'; and Windows uses '\r\n').
		settings.getFormat().setLineSeparator("\n");

		// creates a TSV parser
		TsvParser parser = new TsvParser(settings);

		// parses all rows in one go.
		List<String[]> allRows = parser.parseAll(getReader("/Users/jack/Documents/workspace/L42-ExerciseB/resources/GSE9006-GPL96_series_matrix.txt"));

		int index = 0;
		//int feature = 0;
		int featureNum = allRows.size()-71;

		Sample[] samples = new Sample[117/*hardcoded*/];
		for (byte i = 0; i < samples.length; i++) {
			samples[i] = new Sample();
			//samples[i].value = i;

			samples[i].labels = new ArrayList<String>();
			samples[i].feature = new ArrayList<Double>();
		}

		for (String[] s : allRows) 
		{ 
			if (index == 49) {
				for (int i = 1; i < s.length; i++) {
					String illness = s[i].replaceAll("Illness: ", "");
					samples[i-1].value = getIllnessIndex(illness, multiclass);

				}
			}

			if (index++ < 71) continue;

			if (s.length > 1) {
				System.out.println(index+" - "+s[0]+" - "+(s.length-1));
				String label = s[0];

				Double[] featureValues = (Double[]) Arrays.stream(s).filter(n -> isPermissibleFeature(n)).map(n -> Double.parseDouble(sanitise(n))).toArray(size -> new Double[size]);

				if (featureValues.length != s.length-1) {
					// We've lost some rows. Dump feature
					System.out.println("EEK");
					continue;
				}

				featureValues = scale(featureValues);

				for (int i = 0; i < samples.length; i++) {
					///System.out.println(feature);
					samples[i].labels.add(label);
					samples[i].feature.add(featureValues[i]);
				}

			}
		}
		return samples;
	}
	private static byte getIllnessIndex(String illness, boolean multiclass) {
		if (multiclass) {
			if (illness.equals("\"Healthy\"")) {
				return 0;
			}else if (illness.equals("\"Type 1 Diabetes\"")) {
				return 1;
			}else{
				return 2;
			}
		}else{
			if (illness.equals("\"Healthy\"")) {
				return 1;
			}else if (illness.equals("\"Type 1 Diabetes\"")) {
				return -1;
			}else{
				return -1;
			}
		}
	}
	private static String sanitise(String input) {
		return input.replaceAll("\"", "");
	}
	private static boolean isPermissibleFeature(String n) {
		//boolean ok = true;
		n = sanitise(n);
		if (n.startsWith("!")) return false;
		if (n.matches("[-+]?[0-9]*(\\.|,)?[0-9]+([eE][-+]?[0-9]+)?")) return true;
		return false;
	}
	private static Double[] scale(Double[] featureValues) {
		Double sum = 0.0d;
		Double max = Double.MIN_VALUE;
		Double min = Double.MAX_VALUE;

		Double[] values = new Double[featureValues.length];
		for (int i = 0; i < featureValues.length; i++) {
			sum += featureValues[i];
			if (featureValues[i] > max) max = featureValues[i];
			if (featureValues[i] < min) min = featureValues[i];
		}
		for (int i = 0; i < featureValues.length; i++) {
			//If this one contains no information
			if (max==min) {
				values[i] = 0.0d;
			}else{
				values[i] = (featureValues[i]-min)/(max-min);
			}
		}

		return values;
	}
	public static Reader getReader(String relativePath) throws UnsupportedEncodingException, FileNotFoundException {
		return new InputStreamReader(new FileInputStream(relativePath), "UTF-8");
	}
}
