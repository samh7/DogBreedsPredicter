using DogBreedsApp;
using ICSharpCode.SharpZipLib;
using Microsoft.Extensions.ML;

namespace DogBreedsFullStack;

internal class Program
{
   // Default will try to predict an dog's breed
    static void Main(string[] args)
    {
        Console.WriteLine("Enter the Full Image path\n");
        String FilePath = Console.ReadLine() ?? "";
        Predict(FilePath);
    }

    // Call this function in Main() to predict an dog's breed
    static void Predict(String FullImagePath)
    {
        byte[] imageBytes = File.ReadAllBytes(FullImagePath);
        var input = new ModelInput()
        {
            ImageSource = imageBytes,
        };
        var result = DogBreeds.PredictAllLabels(input).Take(5);
        Console.WriteLine("\n\nPredicted Labels: \n\n\n\n");
        foreach (var label in result) {
            Console.WriteLine($"Breed: {label.Key} => Accuracy: {label.Value * 100}%");
        }
        Console.WriteLine("\n\n\n\n");

    }

    // Call this method in Main() to train the model
    static void Train()
    {
        // This path is Bound to change
        var folderPath = "C:\\Users\\CY\\Documents\\archive (3) ihugoi\\test_test_model";
        var dogBreeds = new DogBreeds();
        dogBreeds.TrainModel(folderPath);
    }
}
