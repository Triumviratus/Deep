/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package deep;

import java.util.List;
import java.util.Random;

/**
 *
 * @author ARZavier
 */
public class Utilities {
    
    /**
     * Check if an array contains a specific element
     * @param i
     * @param arr
     * @param <T> Type
     * @return true if the array contains the element, false otherwise
     */
    
    public static <T> boolean contains(T i, T[] arr) {
        for (T x : arr){
            if (x.equals(i))
                return true;
        }
        return false;
    }
    
    /**
     * Randomly mixes the array
     * @param array
     * @param <T>
     * @return the same array in a random order
     */
    
    public static <T> T[] scramble(T[] array){
        Random rand = new Random();
        for (int i = 0; i < array.length; i++){
            T temp = array[i];
            int randomSpot = rand.nextInt(array.length);
            array[i] = array[randomSpot];
            array[randomSpot] = temp;
        }
        return array;
    }
    
    public static int indexOf(String s, String[] a) {
        int indx = -1;
        for (int i = 0; i < a.length; i++){
            if (a[i].equals(s)){
                indx = i;
                break;
            }
        }
        return indx;
    }
    
    public static int argMax(double[] a) {
        int index = 0;
        double max = a[index];
        for (int i = 0; i < a.length; i++) {
            if (a[i] >= max){
                max = a[i];
                index = i;
            }
        }
        return index;
    }
    
    public static double[] convDouble(List<Double> l){
        double[] retMe = new double[l.size()];
        for (int i = 0; i < l.size(); i++){retMe[i] = l.get(i);}
        return retMe;
    }
    
}
