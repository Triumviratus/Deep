/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package deep;

import deep.Networks.Network;
import java.util.Arrays;

/**
 *
 * @author ARZavier
 */
public class Validator {
    
    /**
     * Tests the data on accuracy
     * @param model the model to be tested
     * @param testData the data to be tested
     * @return the accuracy of the model
     */
    
    public static double accuracy(Network model, DataPoint[] testData){
        
        int correctClassification = 0; // Counters
        for (int i = 0; i < testData.length; i++){
            if (Utilities.argMax(model.predict(testData[i])) == Utilities.argMax(testData[i].obtainTarget())) {
                // If we are correct, add to the total percentage that is correct
                correctClassification++;
            }
        }
        return (double) correctClassification / (double) testData.length;
    }
    
    public static double accuracy(Network model, DataPoint[] testData, boolean print) {
        
        int correctClassification = 0; // Counters
        for (int i = 0; i < testData.length; i++){
            if (print){
                System.out.println("Target: " + Arrays.toString(testData[i].obtainTarget()));
                System.out.println("Predict: " + Arrays.toString(model.predict(testData[i])));
            }
            if (Utilities.argMax(model.predict(testData[i])) == Utilities.argMax(testData[i].obtainTarget())){
                // If we are correct, add to the total percentage that is correct
                correctClassification++;
            } else{}
        }
        return (double) correctClassification / (double) testData.length;
    }
    
    public static double squaredError(Network model, DataPoint[] testData) {
        double error = 0;
        
        for (int i = 0; i < testData.length; i++){
            error += Math.pow((model.predict(testData[i])[0] - testData[i].obtainTarget()[0]), 2);
            if (Double.isNaN(model.predict(testData[i])[0]) || Double.isNaN(testData[i].obtainTarget()[0]))
                System.out.println("Prediction: " + model.predict(testData[i])[0] + ", Target: " + testData[i].obtainTarget()[0]);
        }
        return error/testData.length;
    }
    
}
