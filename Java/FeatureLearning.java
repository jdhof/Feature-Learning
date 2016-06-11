package unisiegen.audisforwear.custom_user_sound.feature_learning;

import android.os.Environment;
import android.util.Log;

import de.uni_siegen.mse.libaaa.CompressedBase64;
import de.uni_siegen.mse.libaaa.training.TrainingData;
import Jama.Matrix;
import unisiegen.audisforwear.R;
import unisiegen.audisforwear.libaaa_test.AndroidHelper;

import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;
import com.mkobos.pca_transform.PCA;

import java.io.BufferedOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

/**
 * Created by Jan-David on 14.04.2016.
 *
 * class for Feature Learning methods
 */
public class FeatureLearning implements Serializable {

    private int shingleSize;
    private int codebookSize;
    private int poolingSize;
    private static final String LABEL = "FeatureLearning";
    private double[][] Codebook;
    private PCA pca;
    private int featureCount;
    private final static String path = Environment.getExternalStorageDirectory() + File.separator + "appName" +
            File.separator;

    public FeatureLearning(int shingleSize, int codebookSize, int poolingSize, int featureCount){
        this.shingleSize = shingleSize;
        this.codebookSize = codebookSize;
        this.poolingSize = poolingSize;
        this.featureCount = featureCount;
    }

    /**
     * shingling concatenates samples of each single soundclass
     * @param trainingData
     * @return
     */
    public TrainingData[] shingling(TrainingData[] trainingData){
        double[][] trainMatrix = computeMatrix(trainingData).getArray();
        double[][] targetMatrix = computeTargetMatrix(trainingData);
        int newFeatureSize = featureCount*shingleSize;
        int rest = trainMatrix.length % shingleSize;
        int normalSize = trainMatrix.length / shingleSize;
        int[] numbOfSamples = findNumberOfElements(targetMatrix);
        ArrayList<double[]> shingledMatrix = new ArrayList<double[]>();
        double[][] targetTemp = transposeMatrix(targetMatrix);
        for(int i = 0; i<targetTemp.length; i++) {   // = classes
            int shingleCounter = 0;
            double[][] shingledFeature = new double[shingleSize][newFeatureSize];    //new feature
            int samplesCounter = 0;
            for (int j = 0; j < targetTemp[i].length; j++) {    // = samples of each class
                if (targetTemp[i][j] == 1) {
                    //shingling
                    shingledFeature[shingleCounter] = trainMatrix[j];
                    shingleCounter++;
                    if(shingleCounter==shingleSize){
                        shingleCounter = 0;
                        double[] tempArray = new double[newFeatureSize];
                        for(int k = 0; k<shingledFeature.length; k++){
                            for(int m = 0; m<shingledFeature[k].length; m++){
                                int numb = featureCount * k + m;
                                tempArray[numb] = shingledFeature[k][m];
                            }
                        }
                        //shingledMatrix[overallCounter] = tempArray;
                        shingledMatrix.add(tempArray);
                    }
                }
            }
        }
        double[][] matrix = new double[shingledMatrix.size()][newFeatureSize];
        for(int i = 0;i<shingledMatrix.size(); i++){
            matrix[i] = shingledMatrix.get(i);
        }
        return computeTrainingDataForShingledData(matrix, numbOfSamples);
    }

    /**
     * computes new target of shingled data
     * @param matrix
     * @param numbOfSamples
     * @return
     */
    private TrainingData[] computeTrainingDataForShingledData(double[][] matrix, int[] numbOfSamples){
        TrainingData[] newTrainingData = new TrainingData[matrix.length];
        int classCounter = 0;
        int sampleData = 0;
        for (int i = 0; i<matrix.length; i++){
            double[] array = matrix[i];
            float[] floatArray = new float[array.length];
            for (int j = 0 ; j < array.length; j++)
            {
                floatArray[j] = (float) array[j];
            }
            double[] newTarget = new double[numbOfSamples.length];
            if(i > sampleData + (numbOfSamples[classCounter]/shingleSize)) {
                classCounter++;
                sampleData = i;
            }
            newTarget[classCounter] = 1;
            newTrainingData[i] = new TrainingData(floatArray, newTarget);
        }
        return newTrainingData;
    }

    /**
     * finds number of elements of each class in targetMatrix
     * @param targetMatrix
     * @return
     */
    private int[] findNumberOfElements(double[][] targetMatrix){
        double[][] targetTemp = transposeMatrix(targetMatrix);
        int[] result = new int[targetMatrix[0].length]; //result with number of classes
        for(int i = 0; i<targetTemp.length; i++){   // = classes
            int samplesCounter = 0;
            for(int j = 0; j<targetTemp[i].length; j++){    // = samples of each class
                if(targetTemp[i][j]==1) samplesCounter++;
            }
            result[i] = samplesCounter;
        }
        return result;
    }

    /**
     * calculates PCA
     * @param trainingData
     * @param computeNewPCA
     * @return
     */
    public TrainingData[] pca(TrainingData[] trainingData, boolean computeNewPCA){
        Log.i(LABEL, "start pca");
        Matrix trainingSet = computeMatrix(trainingData);
        if(computeNewPCA){
            pca = new PCA(trainingSet, true);     //compute PCA
        }
        Log.i(LABEL, "transform data");
        Matrix transformedData =
                pca.transform(trainingSet, PCA.TransformationType.WHITENING);
        Log.i(LABEL, "transformed data succesfully");
        return computeTrainingData(transformedData.getArray(), trainingData);
    }

    /**
     * computes TrainingData
     * @param matrix
     * @param trainingData
     * @return
     */
    private TrainingData[] computeTrainingData(double[][] matrix, TrainingData[] trainingData){
        TrainingData[] newTrainingData = new TrainingData[trainingData.length];
        for (int i = 0; i<matrix.length; i++){
            double[] array = matrix[i];
            float[] floatArray = new float[array.length];
            for (int j = 0 ; j < array.length; j++)
            {
                floatArray[j] = (float) array[j];
            }
            newTrainingData[i] = new TrainingData(floatArray, trainingData[i].getTarget());
        }
        return newTrainingData;
    }

    /**
     * computes matrix out of trainingData
     * @param trainingData
     * @return
     */
    private Matrix computeMatrix(TrainingData[] trainingData){
        int trainingDataDimension = trainingData.length;    // number of vectors/arrays
        int datapointsLength = trainingData[0].getFeatures().length;    //datapoints/dimension of vectors
        double[][] matrix = new double[trainingDataDimension][datapointsLength];
        for(int i = 0; i<trainingDataDimension; i++){
            float[] features = trainingData[i].getFeatures();
            double[] featuresDouble = new double[features.length];
            for (int j = 0; j < features.length; j++)
            {
                featuresDouble[j] = features[j];
            }
            matrix[i] = featuresDouble;
        }
        return new Matrix(matrix);
    }

    /**
     * computes Codebook
     * @param trainingData
     */
    public void spKmeans(TrainingData[] trainingData){
        double threshold = 0.0005;
        double error = 9999;
        int iter = 0;
        Log.i(LABEL, "start Kmeans");
        //-----------------------------------Initialization-----------------------------
        Matrix tempMat = computeMatrix(trainingData);
        double[][] trainingSet = tempMat.getArray();
        double[][] codebook = new double[codebookSize][trainingData[0].getFeatures().length];   //declaring codebook rows= cluster
        Random r = new Random();
        for(int i = 0; i<codebook.length; i++){
            for(int j = 0; j<codebook[i].length; j++){      //codebook initialization with normal distribution
                codebook[i][j] = r.nextGaussian();
            }
        }
        codebook = normalize(codebook);
        //---------------------------------Computing new Clusters----------------------
        while(error>threshold){
            double[][] oldCodebook = codebook;
            double[][] S = new double[trainingSet.length][codebookSize];  // Matrix S =>  codebookSize * trainingSetSize
            double[][] assignements = computeAssignements(trainingSet, codebook);
            for (int i = 0; i<codebookSize; i++){
                int assignement = find(assignements[1], i);
                if(assignement != -1){  // -1 when no assignement (empty cluster)
                    S[assignement][i] = scalarProduct(codebook[i], trainingSet[assignement]);
                }
            }
            //updating codebook
            double[][] matrixProduct = multiplyMatrix(trainingSet, S);
            codebook = matrixAddition(matrixProduct, codebook);
            codebook = normalize(codebook);

            double[][] matrixDiff = matrixSubtraction(codebook, oldCodebook);
            error = sum(sum(matrixDiff));
            iter++;
        }

        Codebook = codebook;
        Log.i(LABEL, "Needed Iterations: " + iter);
    }

    /**
     * Matrix to L2-Norm (unit length)
     * @param matrix
     * @return
     */
    private double[][] normalize(double[][] matrix){
        for(int i = 0; i<matrix.length; i++){
            double norm = Math.sqrt(scalarProduct(matrix[i], matrix[i]));
            for(int j = 0; j<matrix[i].length; j++){
                matrix[i][j] = matrix[i][j]/norm;
            }
        }
        Log.i(LABEL, "Unit Length? " + Math.sqrt(scalarProduct(matrix[1], matrix[1])));
        return matrix;
    }

    /**
     * returns scalarproduct of 2 vectors
     * @param vector1
     * @param vector2
     * @return
     */
    private double scalarProduct(double[] vector1, double[] vector2){
        if(vector1.length != vector2.length) Log.i(LABEL, "++++++++++++++ERROR Dimension+++++++++++++++++");
        double sum = 0.0;
        for (int i = 0; i < vector1.length; i++){
            sum += (vector1[i] * vector2[i]);
        }
        return sum;
    }

    /**
     * compute assignements of dataSet to cluster
     * @param x
     * @param codebook
     * @return
     */
    private double[][] computeAssignements(double[][] x, double[][] codebook){
        double[][] distances = new double[x.length][codebookSize];
        for(int i = 0; i<codebookSize; i++){
            for(int j = 0; j<x.length; j++){
                distances[j][i] = scalarProduct(codebook[i], x[j]);
            }
        }
        return getArgMax(distances);
    }

    /**
     * returns argmax (maximum value of each row)
     * row = traingSetSize (distances to cluster)
     * column = codebookSize (cluster)
     * @param distances
     * @return
     */
    private double[][] getArgMax(double[][] distances){
        double[][] assignements = new double[distances.length][2];
        for(int i = 0; i<distances.length; i++){    //rows (k centroids)
            int bestIdx = -1;
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < distances[i].length; j++) {
                double elem = distances[i][j];
                if (elem > max) {
                    max = elem;
                    bestIdx = j;
                }
            }
            double[] argmax = {max ,bestIdx};
            assignements[i] = argmax;
        }
        return assignements;
    }

    /**
     * computes target-matrix (for matlab export)
     * @param trainingData
     * @return
     */
    private double[][] computeTargetMatrix(TrainingData[] trainingData){
        double[][] targetMatrix = new double[trainingData.length][trainingData[1].getTarget().length];
        for(int i = 0; i<trainingData.length; i++){
            targetMatrix[i] = trainingData[i].getTarget();
        }
        return  targetMatrix;
    }

    /**
     * multiplies matrices (S'*X)
     * @param x
     * @param s
     * @return
     */
    private double[][] multiplyMatrix(double[][] x, double[][] s){
        double[][] result = new double[codebookSize][x[0].length];
        double[][] tempX = transposeMatrix(x);
        double[][] tempS = transposeMatrix(s);
        for(int i = 0; i<tempS.length; i++){
            for(int j = 0; j<tempX.length; j++){
                result[i][j] = scalarProduct(tempS[i], tempX[j]);      // (k*n) x (n*p) ==> (k*p)
            }
        }
        return result;
    }

    /**
     * multiplies matrices for encoding
     * @param matrix1
     * @param matrix2
     * @return
     */
    private double[][] multiplyMatrixForEncoding(double[][] matrix1, double[][] matrix2){
        double[][] result = new double[matrix1.length][matrix2.length];
        //double[][] tempX = transposeMatrix(matrix2);
        for(int i = 0; i<matrix1.length; i++){
            for(int j = 0; j<matrix2.length; j++){
                result[i][j] = scalarProduct(matrix1[i], matrix2[j]);      // (k*n) x (n*p) ==> (k*p)
            }
        }
        return result;
    }


    /**
     * matrix Addition
     * @param matrix1
     * @param matrix2
     * @return
     */
    private double[][] matrixAddition(double[][] matrix1, double[][] matrix2){
        double[][] result = new double[matrix1.length][matrix1[0].length];
        for(int i = 0; i<matrix1.length; i++){
            for(int j = 0; j<matrix1[0].length; j++){
                result[i][j] = matrix1[i][j] + matrix2[i][j];
            }
        }
        return result;
    }

    /**
     * sumarizes matrix vectors
     * @param matrix
     * @return
     */
    private double[] sum(double[][] matrix){
        double[] result = new double[matrix.length];
        for(int i = 0; i<matrix.length; i++){
            for(int j = 0; j<matrix[0].length; j++){
                result[i] += matrix[i][j];
            }
        }
        return result;
    }

    /**
     * summarizes values of vector
     * @param vector
     * @return
     */
    private double sum(double[] vector){
        double result = 0;
        for(int i = 0; i<vector.length; i++){
            result += vector[i];
        }
        return result;
    }

    /**
     * matrix Subtraction
     * @param matrix1
     * @param matrix2
     * @return
     */
    private double[][] matrixSubtraction(double[][] matrix1, double[][] matrix2){
        double[][] result = new double[matrix1.length][matrix1[0].length];
        for(int i = 0; i<matrix1.length; i++){
            for(int j = 0; j<matrix1[0].length; j++){
                result[i][j] = matrix1[i][j] - matrix2[i][j];
            }
        }
        return result;
    }

    /**
     * transpose matrix
     * @param matrix
     * @return
     */
    private double[][] transposeMatrix(double[][] matrix){
        double[][] transposedMatrix = new double[matrix[0].length][matrix.length];
        for(int i = 0; i<matrix[0].length; i++){
            for(int j = 0; j<matrix.length; j++){
                transposedMatrix[i][j] = matrix[j][i];
            }
        }
        return transposedMatrix;
    }

    /**
     * finds final assignement
     * @param idx
     * @param i
     * @return
     */
    private int find(double[] idx, int i){
        int counter = 0;
        int result = -1;
        if(idx[1] == i) return i;
        return result;
    }

    /**
     * Encoding samples with codebook
     * returns TrainingData[] for training
     * @param trainingData
     * @return
     */
    public TrainingData[] encoding(TrainingData[] trainingData){
        Matrix matrix = computeMatrix(trainingData);
        double[][] trainingSet = matrix.getArray();
        double[][] result = multiplyMatrixForEncoding(trainingSet, Codebook);
        return computeTrainingData(result, trainingData);
    }

    /**
     * Encode sample with codebook (for classification)
     * @param features
     * @return
     */
    public double[] encoding(double[] features){
        double[] result = new double[Codebook.length];
        for(int i = 0; i<Codebook.length; i++){
           result[i] = scalarProduct(Codebook[i], features);
        }
        return result;
    }

    /**
     * exports dataSet and targetMatrix in Matlab-format
     * @param data
     * @param mode
     */
    public void exportMatlab(TrainingData[] data, String mode){
        double[][] targetMatrix = computeTargetMatrix(data);
        double[][] dataSet = computeMatrix(data).getArray();

        File set = new File(Environment.getExternalStorageDirectory() + File.separator + "AppName" +
                File.separator + mode + "Set.mat");

        File target = new File(Environment.getExternalStorageDirectory() + File.separator + "AppName" +
                File.separator + mode + "Target.mat");

        MLDouble mlDouble = new MLDouble(mode + "Set", dataSet);
        MLDouble mlDouble1 = new MLDouble(mode + "Target", targetMatrix);
        ArrayList<MLArray> list = new ArrayList<MLArray>();
        list.add(mlDouble);
        ArrayList<MLArray> list1 = new ArrayList<MLArray>();
        list1.add(mlDouble1);
        try {
        new MatFileWriter(set, list);
        new MatFileWriter(target, list1);

        }
        catch (IOException e) {e.printStackTrace();}
        Log.i(LABEL, "+++++++++++++++++Matlab Export++++++++++++++++");
    }

    /**
     * pooling over feature-vectors (frames)
     * @param data
     * @return
     */
    public TrainingData[] pooling(TrainingData[] data){
        double[][] matrix = computeMatrix(data).getArray();
        int poolCounter = 0;
        double[][] poolingWindow = new double[poolingSize][matrix[0].length];
        double [] oldTarget = data[0].getTarget();
        ArrayList<double[]> pooledMatrix = new ArrayList<double[]>();
        int overallCounter = 0;
        int [] newClassBorders = new int[data[0].getTarget().length];   //border for each class for target-matrix
        int classCounter = 0;
        for(int i = 0; i<matrix.length; i++){
            if(oldTarget == data[i].getTarget()){ //pool only samples with same class
                poolingWindow[poolCounter] = matrix[i];
                poolCounter++;
            }
            else{
                newClassBorders[classCounter] = i;      //border = change of targetclass
                oldTarget = data[i].getTarget();        //if target is not the same
                poolCounter = 0;
                double[][] temp = transposeMatrix(poolingWindow);
                ArrayList<Double> maxValues = new ArrayList<Double>();
                for(int j = 0; j<temp.length; j++){
                    maxValues.add(getMax(temp[j]));
                }
                double[] newVector = new double[maxValues.size()];
                for(int j = 0; j<maxValues.size(); j++){
                    newVector[j] = maxValues.get(j);
                }
                //pooledMatrix[overallCounter] = newVector;
                //overallCounter++;
                pooledMatrix.add(newVector);
                classCounter++;
            }
            if(poolCounter == poolingSize){
                poolCounter = 0;
                double[][] temp = transposeMatrix(poolingWindow);
                ArrayList<Double> maxValues = new ArrayList<Double>();
                for(int j = 0; j<temp.length; j++){
                    maxValues.add(getMax(temp[j]));
                }
                double[] newVector = new double[maxValues.size()];
                for(int j = 0; j<maxValues.size(); j++){
                    newVector[j] = maxValues.get(j);
                }
                //pooledMatrix[overallCounter] = newVector;
                //overallCounter++;
                pooledMatrix.add(newVector);
            }
            newClassBorders[newClassBorders.length-1] = matrix.length;
        }
        double[][] resultMatrix = new double[pooledMatrix.size()][matrix[0].length];
        for(int i = 0; i<pooledMatrix.size(); i++){
            resultMatrix[i] = pooledMatrix.get(i);
        }
        return computeTrainingDataForPooling(resultMatrix, newClassBorders);
    }

    /**
     * calculates max of vector
     * @param features
     * @return
     */
    private double getMax(double[] features){
        double maxValue = 1.0f;
        for(int i=0; i<features.length; i++){
            if(features[i] > maxValue) maxValue = features[i];
        }
        return maxValue;
    }

    /**
     * computes TrainingData for Pooling
     * @param resultMatrix
     * @param newClassBorders
     * @return
     */
    private TrainingData[]  computeTrainingDataForPooling(double[][] resultMatrix, int[] newClassBorders){
        double[][] targetMatrix = new double[resultMatrix.length][newClassBorders.length];
        TrainingData[] newTrainingData = new TrainingData[resultMatrix.length];
        int classCounter = 0;
        for (int i = 0; i<resultMatrix.length; i++){
            double[] array = resultMatrix[i];
            float[] floatArray = new float[array.length];
            for (int j = 0 ; j < array.length; j++)
            {
                floatArray[j] = (float) array[j];
            }
            double[] newTarget = new double[newClassBorders.length];
            int border = newClassBorders[classCounter]/poolingSize + newClassBorders[classCounter]%poolingSize;
            if(i ==  border) {
                classCounter++;
            }
            newTarget[classCounter] = 1;
            newTrainingData[i] = new TrainingData(floatArray, newTarget);
        }
        return newTrainingData;
    }

    public void exportB64(){
        ByteArrayOutputStream bos = new ByteArrayOutputStream(); //resultdata
        CompressedBase64.toCompressedBase64(bos.toByteArray());
    }

    public double[][] getCodebook(){
        return Codebook;
    }

    public int getShingleSize(){
        return shingleSize;
    }

    public int getPoolingSize(){
        return poolingSize;
    }

    /**
     * export FeatureLearning data
     * @return
     * @throws IOException
     */
    public ByteArrayOutputStream exportData() throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ObjectOutputStream os = new ObjectOutputStream(out);
        os.writeObject(this);
        AndroidHelper.writeStringToExternalStorage(path + "FeatureLearning.b64", CompressedBase64.toCompressedBase64(out.toByteArray()));
        out.close();
        return out;
    }

}
