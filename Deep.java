/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package deep;

import deep.Networks.Network;
import deep.Networks.SAE;

import deep.Functions.ActivationFunction;
import deep.Functions.Linear;
import deep.Functions.LogisticFunction;
import deep.Functions.SoftMax;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;

/**
 *
 * @author ARZavier
 */
public class Deep {
    
    static String[] CATEGORICAL_FEATURES = {"sex", "buying", "maintenance", "doors", "persons", "safety", "lug_boot",
                                            "month", "day", "model"};
    public static String[] classificationFiles = {"abalone", "car", "segmentation"};
    public static String[] regressionFiles = {"wine", "forestfires", "machine"};
    public static String[] allFiles = {"abalone", "car", "segmentation", "forestfires", "wine", "machine"};
    static String[] trial = {"car"};
    
    // Tuned Learning Rates for SAE Model
    static double[][] rates = {
        {0.01, 0.01, 0.001}, // Abalone
        {0.1, 0.1, 0.1}, // Car
        {0.1, 0.01, 0.001}, // Segmentation
        {0.001, 0.001, 0.001}, // Fires
        {0.1, 0.01, 0.1}, // Wine
        {0.001, 0.01, 0.001} // Machine
    };
    
    // Tuned network shapes for stack size 1
    static int[][][] ones = {
        {{8}, {6}}, // Abalone
        {{18}, {}}, // Car
        {{17}, {}}, // Segmentation
        {{17}, {5, 11}}, // Fires
        {{5}, {4, 3}}, // Wine
        {{49}, {8, 9}} // Machine
    };
    
    // Tuned network shapes for stack size 2
    static int[][][] twos = {
        {{4,3},{3}}, // Abalone
        {{18, 17},{}}, // Car
        {{12, 6},{}}, // Segmentation
        {{5, 3},{}}, // Fires
        {{7, 3},{6, 3}}, // Wine
        {{139, 53},{13, 23}} // Machine
    };
    
    // Tuned network shapes for stack size 3
    static int[][][] threes = {
        {{9,6,4}, {}}, // Abalone
        {{13, 5, 2}, {}}, // Car
        {{18,10,5}, {}}, // Segmentation
        {{20, 11, 7}, {5}}, // Fires
        {{3,2,1}, {}}, // Wine
        {{114, 109, 4}, {4}} // Machine
    };
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        
        for (String path : allFiles){
            System.out.println();
            System.out.println();
            System.out.println("Processing: " + path);
            
            String file = readEntireFile("Data/Assignment3/" + path + "_preprocessed.data"); // Reads in the data
            String header = "Fold, Err1, Acc1, Err2, Acc2, Err3, Acc3"; // For outputing the data to csv
            File outputFile = createNewFile("Data/ExtraCredit/" + path);
            appendToFile(header, outputFile);
            String[] lines = file.split("\r\n");
            DataPoint[] data = new DataPoint[lines.length]; // First line is feature labels
            String[] featureLabels = lines[0].split(",");
            
            // Generates the data points
            for (int i = 1; i < lines.length; i++){data[i-1] = genDatapoint(lines[i], featureLabels, CATEGORICAL_FEATURES, path);}
            
            data = trim(data); // Removes null values
            DataPoint[][] folds = fold(data); // Splits the data into 10 roughly equal folds
            
            int inputs = data[0].obtainNumFeatures();
            int outputs = data[0].obtainNumOutput();
            
            ActivationFunction outputFunc = (data[0].obtainNumOutput() == 1) ? new Linear() : new SoftMax();
            // Linear if Regression, SoftMax if Classification
            for (int i = 0; i < folds.length; i++) {
                System.out.println("Fold: " + i);
                String out = i + ",";
                DataPoint[] testData = folds[i];
                DataPoint[] trainData = new DataPoint[0];
                // Copies the train and test data depending on the fold
                for (int j = 0; j < folds.length; j++){
                    if (j != i)
                        trainData = concat(folds[j], trainData);
                }
                int indx = Utilities.indexOf(path, allFiles);
                // Obtains the tuned models
                SAE one = getTunedSAE(ones[indx], inputs, outputs, outputFunc);
                SAE two = getTunedSAE(twos[indx], inputs, outputs, outputFunc);
                SAE three = getTunedSAE(threes[indx], inputs, outputs, outputFunc);
                // Trains the models
                one.train(trainData, rates[indx][0], 100);
                two.train(trainData, rates[indx][1], 100);
                three.train(trainData, rates[indx][2], 100);
                
                System.out.println("Error: " + Validator.squaredError(one, testData));
                System.out.println("Accuracy: " + Validator.accuracy(one, testData));
                
                // Output Metrics to csv
                out += Validator.squaredError(one, testData) + ",";
                out += Validator.accuracy(one, testData) + ",";
                out += Validator.squaredError(two, testData) + ",";
                out += Validator.accuracy(two, testData) + ",";
                out += Validator.squaredError(three, testData) + ",";
                out += Validator.accuracy(three, testData) + ",";
                appendToFile(out, outputFile);
            }
        }
    }
    
    /**
     * Helper function for acquiring the tuned model from the arrays above
     * @param type
     * @param inputs
     * @param outputs
     * @param outputFunc
     * @return 
     */
    
    private static SAE getTunedSAE(int[][] type, int inputs, int outputs, ActivationFunction outputFunc) {
        int[] ffShape = type[1];
        int[] saeShape = type[0];
        Network ff = new Network(saeShape[saeShape.length - 1], ffShape, outputs, new LogisticFunction(), outputFunc);
        SAE sae = new SAE(inputs, saeShape, ff);
        return sae;
    }
    
    /**
     * Creates a data point based on a string input
     * @param featureString String defining the data point
     * @return data point generated
     */
    
    private static DataPoint genDatapoint(String featureString, String[] featureLabels, 
                                          String[] categoricalFeatures, String filename){
        String[] splice = featureString.split(",");
        OneHot hot = new OneHot(filename);
        ArrayList<Double> features = new ArrayList<>();
        
        for (int i = 0; i < splice.length - 1; i++) {
            if (Arrays.asList(categoricalFeatures).contains(featureLabels[i])) {
                // Is Categorical
                String value = splice[i];
                double[] hotVals = hot.obtainOneHot(i, value);
                for (double hv : hotVals){features.add(hv);}
            } else {
                // Is Continuous
                try {
                    features.add(Double.parseDouble(splice[i]));
                } catch (NumberFormatException e){
                    // Is Categorical
                    String value = splice[i];
                    double[] hotVals = hot.obtainOneHot(i, value);
                    for (double hv : hotVals){features.add(hv);}
                }
            }
        }
        
        if (Utilities.contains(filename, classificationFiles)) {
            // Classification
            String classMembership = splice[splice.length - 1];
            double[] classOneHot = hot.obtainOneHot(splice.length - 1, classMembership);
            DataPoint d = new DataPoint(Utilities.convDouble(features), classOneHot);
            return d;
        } else {
            // Regression
            double[] regressionTarget = {Double.parseDouble(splice[splice.length-1])};
            DataPoint d = new DataPoint(Utilities.convDouble(features), regressionTarget);
            return d;
        }
    }
    
    /**
     * Reads the entire file and returns the content as a String
     * @param filePath
     * @return 
     */
    
    private static String readEntireFile(String filePath){
        // Reads the File
        File file = new File(filePath);
        String retString = "";
        if (file.exists()) {
            try {
                Scanner scan = new Scanner(file);
                scan.useDelimiter("\\Z");
                if (scan.hasNext())
                    retString = scan.next();
                scan.close();
            } catch (FileNotFoundException ignored){
                return "File Not Found For Path: " + file;
            }
        } else
            System.out.println("File Does Not Exist");
        
        return retString;
    }
    
    /**
     * Adds the string to the end of a file
     * @param line string to be added
     * @param file the file to be added to
     */
    
    public static void appendToFile(String line, File file){
        // Adds on to file
        try {
            FileWriter writer = new FileWriter(file, true);
            writer.append(line + '\n');
            writer.close();
        } catch (IOException ignored){}
    }
    
    /**
     * Creates a file if there does not exist one already,
     * then returns the file at the file path.
     * @param filePath file path
     * @return the file (either old or newly created)
     */
    
    public static File createNewFile(String filePath) {
        // Creates a file
        String newPath = filePath;
        File file = new File(newPath + ".csv");
        int i = 2;
        while (file.exists()) {
            newPath = filePath + "-" + i;
            file = new File(newPath + ".csv");
            i += 1;
        } try {
            file.createNewFile();
        } catch (IOException ignored){ignored.printStackTrace();}
        
        return file;
    }
    
    /**
     * Folds the data into 10 relatively equal folds
     * @param points the data to be folded
     * @return a folded list
     */
    
    private static DataPoint[][] fold (DataPoint[] points) {
        
        points = Utilities.scramble(points); // Scrambles the Data
        int folds = 10;
        
        DataPoint[][] data = new DataPoint[folds][points.length];
        int[] counters = new int[folds];
        
        /**
         * So the elements go into the array in order 
         * (i.e., all the null values are at the end).
         */
        Random rand = new Random();
        // Ascertains that all folds have one DataPoint at minimum
        for (int i = 0; i < folds; i++){data[i][counters[i]++] = points[i];}
        
        for (int i = 10; i < points.length; i++) {
            int random = rand.nextInt(folds);
            data[random][counters[random]++] = points[i];
            // Places the points into the folds in order so as to avoid null values
        }
        
        // Eliminates trailing null values
        for (int i = 0; i < data.length; i++){data[i] = trim(data[i]);}
        
        return data;
    }
    
    /**
     * Removes the null values from the end of the array
     * @param points the array
     * @return the values in the same order as they appear in points without trailing null values
     */
    
    private static DataPoint[] trim (DataPoint[] points) {
        int i = 0;
        DataPoint point = points[i++];
        while(point != null){point = points[i++];}
        return (DataPoint[]) Arrays.copyOf(points, i-1);
    }
    
    public static DataPoint[] concat(DataPoint[] a, DataPoint[] b){
        DataPoint[] retMe = new DataPoint[a.length + b.length];
        for (int i = 0; i < a.length; i++){retMe[i] = a[i];}
        for (int i = 0; i < b.length; i++){retMe[i + a.length] = b[i];}
        return retMe;
    }
}