import java.lang.Math;

public class LogisticClassifier {
    private int test;

    private double sigmoid(double[] w, double[] x, double b)
    {
        double dotProd = 0.0;

        for(int i = 0; i < w.length; i++)
            dotProd += w[i]*x[i];

        return 1/(1+ Math.exp(-(dotProd+b)));
    }


}