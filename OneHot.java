/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package deep;

import java.util.Arrays;

/**
 *
 * @author ARZavier
 */
public class OneHot {
    String[][] legend;
    
    public OneHot(String filename){
        if(filename.equals("machine")){
            legend = new String[][]{{"adviser", "amdahl", "apollo", "basf", "bti", "burroughs", "c.r.d", "cdc", "cambex",
                                     "dec", "dg", "formation", "four-phase", "gould", "hp", "harris", "honeywell", "ibm",
                                     "ipl", "magnuson", "microdata", "nas", "ncr", "nixdorf", "perkin-elmer", "prime",
                                     "siemens", "sperry", "sratus", "wamg"},
                                    {"32/60", "470v/7", "470v/7a", "470v/7b", "470v/7c", "470v/b", "580-5840", "580-5850",
                                     "580,5860", "580-5880", "dn320", "dn420", "7/65", "7/68", "5000", "8000", "b1955", "b2925",
                                     "b4955", "b5900", "b5920", "b6900", "b6925", "68/10-80", "universe:2203t", "universe:68",
                                     "universe:68/05", "universe:68/137", "universe:68/37", "cyber:170/750", "cyber:170/760",
                                     "cyber:170/815", "cyber:170/825", "cyber:170,835", "cyber:170/845", "omega:480-i",
                                     "omega:480-ii", "omega:480-iii", "1636-1", "1636-10", "1641-1", "1641-11", "1651-1",
                                     "decsys:10:1091", "decsys:20:2060", "microvax-1", "vax:11/730", "vax:11/750", "vax:11/780",
                                     "eclipse:c/350", "eclipse:m/600", "eclipse:mv/10000", "eclipse:mv/4000", "eclipse:mv/6000",
                                     "eclipse:mv/8000", "eclipse:mv/8000-ii", "f4000/100", "f4000/200", "f4000/200ap", "f4000/300",
                                     "f4000/300ap", "2000/260", "concept:32/8705", "concept:32/8750", "concept:32/8780", "3000/30",
                                     "3000/40", "3000/44", "3000/48", "3000/64", "3000/88", "3000/iii", "100", "300", "500", "600",
                                     "700", "80", "800", "dps:6/35", "dps:6/92", "dps:6/96", "dps:7/35", "dps:7/45", "dps:7/55",
                                     "dps:7/65", "dps:8/44", "dps:8/49", "dps:8/50", "dps:8/52", "dps:8/62", "dps:8/20", "3033:s",
                                     "3033:u", "3081", "3081:d", "3083:b", "3083:e", "370/125-2", "370/148", "370/158-3", "38/3",
                                     "38/4", "38/5", "38/7", "38/8", "4321", "4331-1", "4331-11", "4331-2", "4341", "4341-1",
                                     "4341-10", "4341-11", "4341-12", "4341-2", "4341-9", "4361-4", "4361-5", "4381-1", "4382-2",
                                     "8310-a", "8130-b", "8140", "4436", "4443", "4445", "4446", "4460", "4480", "m80/30", "m80/31",
                                     "m80/32", "m80/42", "m80/43", "m80/44", "seq.ms/3200", "as/3000", "as/3000-n", "as/5000",
                                     "as/5000-e", "as/5000-n", "as/6130", "as/6150", "as/6620", "as/6630", "as/6650", "as/7000",
                                     "as/7000-n", "as/8040", "as/8050", "as/8060", "as/9000-dpc", "as/9000-n", "as/9040", "as/9060",
                                     "v8535:ii", "v8545:ii", "v8555:ii", "v8565:ii", "v8565:ii-e", "v8575:ii", "v8585:ii", "v8595:ii",
                                     "v8635", "v8650", "v8655", "v8665", "v8670", "8890/30", "8890/50", "8890/70", "3205", "3210",
                                     "3230", "50-2250", "50-250-ii", "50-550-ii", "50-750-ii", "50-850-ii", "7.521", "7.531", "7.536",
                                     "7.541", "7.551", "7.561", "7.865-2", "7.870-2", "7.872-2", "7.875-2", "7.880-2", "7.881-2",
                                     "1100/61-h1", "1100/81", "1100/82", "1100/83", "1100/84", "1100/93", "1100/94", "80/3", "80/4",
                                     "80/5", "80/6", "80/8", "90/80-model-3", "32", "vs-100", "vs-90"}};
        }
        else if (filename.equals("car")) {
            legend = new String[][]{{"vhigh", "high", "med", "low"}, {"vhigh", "high", "med", "low"},
                                    {"2", "3", "4", "5more"}, {"2", "4", "more"}, {"small", "med", "big"},
                                    {"low", "med", "high"}, {"unacc", "acc", "good", "vgood"}};
        }
        else if (filename.equals("abalone")) {
            legend = new String[][]{{"M", "F", "I"},{},{},{},{},{},{},{},
                                    {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17",
                                     "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29"}};
        }
        else if (filename.equals("segmentation")) {
            legend = new String[][]{{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},
                                    {"BRICKFACE", "SKY", "FOLIAGE", "CEMENT", "WINDOW", "PATHS", "GRASS"}};
        }
        else if (filename.equals("forestfires")) {
            legend = new String[][]{{},{},
                                    {"mar", "oct", "aug", "sep", "apr", "jun", "jul", "feb", "jan", "dec", "may", "nov"},
                                    {"fri", "tue", "sat", "sun", "mon", "wed", "thu"}};
        }
    }
    
    public double[] obtainOneHot(int attribute, String categorical) {
        
        double[] array = new double[legend[attribute].length];
        Arrays.fill(array, 0);
        if (contains(categorical, legend[attribute])) {
            int index = Arrays.asList(legend[attribute]).indexOf(categorical);
            array[index] = 1.0;
        }
        else {
            System.out.println("Searching For: " + categorical);
            for (char c : categorical.toCharArray()){System.out.println((int) c);}
            System.out.println("Searching In: " + Arrays.toString(legend[attribute]));
        }
        return array;
    }
    
    private boolean contains(String cat, String[] arr) {
        for (String s : arr){
            if (s.equals(cat))
                return true;
        }
        return false;
    }
    
}
