/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package deep.Functions;

/**
 *
 * @author ARZavier
 */
public class SoftMax extends ActivationFunction {
    
    @Override
    public double value (double weightedSum, double[] outputs) {
        double numerator = Math.exp(weightedSum);
        double denominator = 0.1;
        for (double out : outputs){denominator += Math.exp(out);}
        return numerator/denominator;
    }
    @Override
    public double derivative (double weightedSum){return 0;}
}
