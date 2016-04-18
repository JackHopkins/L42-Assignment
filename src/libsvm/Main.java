package libsvm;
import java.awt.Graphics;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Vector;

import svm.Sample;
import svm.Utils;
import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;

import com.univocity.parsers.tsv.TsvParser;
import com.univocity.parsers.tsv.TsvParserSettings;



/*
0 - !Series_title
1 - !Series_geo_accession
2 - !Series_status
3 - !Series_submission_date
4 - !Series_last_update_date
5 - !Series_pubmed_id
6 - !Series_summary
7 - !Series_summary
8 - !Series_summary
9 - !Series_summary
10 - !Series_summary
11 - !Series_overall_design
12 - !Series_overall_design
13 - !Series_overall_design
14 - !Series_type
15 - !Series_contributor
16 - !Series_contributor
17 - !Series_contributor
18 - !Series_contributor
19 - !Series_contributor
20 - !Series_contributor
21 - !Series_sample_id = IMPORTANT
22 - !Series_contact_name
23 - !Series_contact_email
24 - !Series_contact_institute
25 - !Series_contact_address
26 - !Series_contact_city
27 - !Series_contact_state
28 - !Series_contact_zip/postal_code
29 - !Series_contact_country
30 - !Series_supplementary_file
31 - !Series_platform_id
32 - !Series_platform_id
33 - !Series_platform_taxid
34 - !Series_sample_taxid
35 - !Series_relation
36 - !Sample_title = IMPORTANT
37 - !Sample_geo_accession
38 - !Sample_status
39 - !Sample_submission_date
40 - !Sample_last_update_date
41 - !Sample_type
42 - !Sample_channel_count
43 - !Sample_source_name_ch1
44 - !Sample_organism_ch1
45 - !Sample_characteristics_ch1 = AGE?
46 - !Sample_characteristics_ch1 = IMPORTANT GENDER
47 - !Sample_characteristics_ch1 
48 - !Sample_characteristics_ch1 = Illness
49 - !Sample_characteristics_ch1
50 - !Sample_characteristics_ch1
51 - !Sample_characteristics_ch1
52 - !Sample_molecule_ch1
53 - !Sample_extract_protocol_ch1
54 - !Sample_label_ch1
55 - !Sample_label_protocol_ch1
56 - !Sample_taxid_ch1
57 - !Sample_hyb_protocol
58 - !Sample_scan_protocol
59 - !Sample_description
60 - !Sample_data_processing
61 - !Sample_platform_id
62 - !Sample_contact_name
63 - !Sample_contact_email
64 - !Sample_contact_institute
65 - !Sample_contact_address
66 - !Sample_contact_city
67 - !Sample_contact_state
68 - !Sample_contact_zip/postal_code
69 - !Sample_contact_country
70 - !Sample_supplementary_file
71 - !Sample_data_row_count
 */

public class Main {
	private static final int FEATURE_NUM = 0;

	static int XLEN;
	static int YLEN;
	static Vector<Sample> point_list = new Vector<Sample>();

	
	public static void main(String[] args) throws UnsupportedEncodingException, FileNotFoundException {
		Sample[] samples = Utils.parse(true);
		for (int i = 0; i < samples.length; i++) {
			System.out.println(samples[i]);
			point_list.addElement(samples[i]);
		}
		train();
	}
	

	

	
	public static void train() {
		svm_parameter param = new svm_parameter();

		// default values
		param.svm_type = svm_parameter.C_SVC;
		//RBF = 
		//Linear = 91.45%
		param.kernel_type = svm_parameter.SIGMOID;
		param.degree = 3;
		param.gamma = 0;
		param.coef0 = 0;
		param.nu = 0.5;	
		param.cache_size = 40;
		param.C = 1;
		param.eps = 1e-3;
		param.p = 0.1;
		param.shrinking = 1;
		param.probability = 1;
		param.nr_weight = 0;
		param.weight_label = new int[0];
		param.weight = new double[0];

		svm_problem prob = new svm_problem();
		prob.l = point_list.size();
		prob.y = new double[prob.l];

		//SUPPORT VECTOR CLASSIFICATION

		if(param.gamma == 0) param.gamma = 0.5;
		prob.x = new svm_node [prob.l][point_list.elementAt(0).feature.size()];
		for(int i=0;i<prob.l;i++)
		{
			Sample p = point_list.elementAt(i);
			for (int j = 0; j < p.feature.size(); j++) {
				prob.x[i][j] = new svm_node();
				prob.x[i][j].index = j+1;
				prob.x[i][j].value = p.feature.get(j);
			}
			Iterator<Sample> it = point_list.iterator();
			int j = 0; 
			while(it.hasNext()) {
				prob.y[j++] = it.next().value;
			}

			/*prob.x[i][1] = new svm_node();
			prob.x[i][1].index = 2;
			prob.x[i][1].value = p.y;
			prob.y[i] = p.value;*/
		}

		// build model & classify

		//public static void svm_cross_validation(svm_problem prob, svm_parameter param, int nr_fold, double[] target)

		tuneHyperparameters(param, prob);

		svm_model model = svm.svm_train(prob, param);
		
		svm_node[] x = new svm_node[2];
		x[0] = new svm_node();
		x[1] = new svm_node();
		x[0].index = 1;
		x[1].index = 2;

		int[] j = new int[XLEN];


		for (int i = 0; i < XLEN; i++)
		{
			x[0].value = (double) i / XLEN;
			j[i] = (int)(YLEN*svm.svm_predict(model, x));
		}
	}
	private static void tuneHyperparameters(svm_parameter param, svm_problem prob) {

		//optimise param.gamma and param.C
		
		double bestGamma = 0;
		double bestC = 0;
		int bestCorrect = 0;
		double gamma = 1.0E-6;
		for (int i = 0; i < 1; i++) {
			//1000
			double C = 1.0E11;
			for (int j = 0; j < 20; j++) {
				
				svm_parameter nParam = (svm_parameter) param.clone();
				nParam.C = C;
				nParam.gamma = gamma;
				
				String error_msg = svm.svm_check_parameter(prob,nParam);

				if(error_msg != null)
				{
					System.err.print("ERROR: "+error_msg+"\n");
					System.exit(1);
				}
				int total_correct = evaluateParams(nParam, prob);
				if (total_correct > bestCorrect) {
					double percent = 100.0*total_correct/prob.l;
					bestCorrect = total_correct;
					bestC = C;
					bestGamma = gamma;
				}
				System.out.print("Gamma:"+gamma+", C:"+C+" = "+100.0*total_correct/prob.l+"%\n");
				C *= 5;
			}
			gamma *= 10;
		}
		System.out.print("Best=Gamma:"+bestGamma+", C:"+bestC+" = "+100.0*bestCorrect/prob.l+"%\n");

	}
	private static int evaluateParams(svm_parameter param, svm_problem prob) {
		double[] target = new double[prob.l];
		svm.svm_cross_validation(prob, param, 5, target);
		int total_correct = 0;
		for(int i=0;i<prob.l;i++)
			if(target[i] == prob.y[i])
				++total_correct;
		return total_correct;

	}
}
