import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.Instance;

import java.io.File;
import java.io.IOException;

public class SpeechEmotionRecognition {
    public static void main(String[] args) {
        try {
            // Load ARFF file containing feature data
            File inputFile = new File("path/to/your/emotion_features.arff");
            ArffLoader loader = new ArffLoader();
            loader.setFile(inputFile);
            Instances data = loader.getDataSet();

            // Set the class index (the attribute to predict)
            data.setClassIndex(data.numAttributes() - 1);

            // Train the classifier (J48 Decision Tree in this example)
            Classifier classifier = new J48();
            classifier.buildClassifier(data);

            // Evaluate or predict with the classifier
            Instance testInstance = data.instance(0); // Example: use the first instance for testing
            double emotionIndex = classifier.classifyInstance(testInstance);
            String predictedEmotion = data.classAttribute().value((int) emotionIndex);

            System.out.println("Predicted Emotion: " + predictedEmotion);
        } catch (IOException e) {
            System.err.println("Error loading ARFF file: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("Error during classification: " + e.getMessage());
        }
    }
}