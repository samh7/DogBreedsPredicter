using Microsoft.ML.Data;
using Microsoft.ML;

namespace DogBreedsApp
{

    //Contains everything needed for the model training, and consumption.
  
    public partial class DogBreeds
    {
        //Train model and save it
        public void TrainModel(string folder)
        {
            var mlContext = new MLContext();

            var data = LoadImageFromFolder(mlContext, folder);

            var model = RetrainModel(mlContext, data);

            //Note: This path will change depening on your file location
            //Copy full path from the Dogbreeds.mlnet and paste it here 
            string modelPath = Path.Combine("C:\\Users\\CY\\AIProject\\DogBreedsPredicter\\", "DogBreeds.mlnet");

            mlContext.Model.Save(model, data.Schema, modelPath);
        }

        // Get image date for training our model
        public static IDataView LoadImageFromFolder(MLContext mlContext, string folder)
        {
            var res = new List<ModelInput>();
            var allowedImageExtensions = new[] { ".png", ".jpg", ".jpeg", ".gif" };
            DirectoryInfo rootDirectoryInfo = new DirectoryInfo(folder);
            DirectoryInfo[] subDirectories = rootDirectoryInfo.GetDirectories();

            if (subDirectories.Length == 0)
            {
                throw new Exception("fail to find subdirectories");
            }

            foreach (DirectoryInfo directory in subDirectories)
            {
                var imageList = directory.EnumerateFiles().Where(f => allowedImageExtensions.Contains(f.Extension.ToLower()));
                if (imageList.Count() > 0)
                {
                    res.AddRange(imageList.Select(i => new ModelInput
                    {
                        Label = directory.Name,
                        ImageSource = File.ReadAllBytes(i.FullName),
                    }));
                }
            }
            return mlContext.Data.LoadFromEnumerable(res);
        }

        //Train model
        public static ITransformer RetrainModel(MLContext mlContext, IDataView trainData)
        {
            var pipeline = BuildPipeline(mlContext);
            var model = pipeline.Fit(trainData);

            return model;
        }


       
        public static IEstimator<ITransformer> BuildPipeline(MLContext mlContext)
        {
            // Data process configuration with pipeline data transformations
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: @"Label", inputColumnName: @"Label", addKeyValueAnnotationsAsText: false)
                                    .Append(mlContext.MulticlassClassification.Trainers.ImageClassification(labelColumnName: @"Label", scoreColumnName: @"Score", featureColumnName: @"ImageSource"))
                                    .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: @"PredictedLabel", inputColumnName: @"PredictedLabel"));

            return pipeline;
        }



        //Model Consumption 


        //Note: This path will change depening on your file location
        //Copy full path from the Dogbreeds.mlnet and paste it here 
        private static string MLNetModelPath = Path.Combine("C:\\Users\\CY\\AIProject\\DogBreedsPredicter\\", "DogBreeds.mlnet");

        public static readonly Lazy<PredictionEngine<ModelInput, ModelOutput>> PredictEngine = new Lazy<PredictionEngine<ModelInput, ModelOutput>>(() => CreatePredictEngine(), true);


        private static PredictionEngine<ModelInput, ModelOutput> CreatePredictEngine()
        {
            var mlContext = new MLContext();
            ITransformer mlModel = mlContext.Model.Load(MLNetModelPath, out var _);
            return mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);
        }


        public static IOrderedEnumerable<KeyValuePair<string, float>> PredictAllLabels(ModelInput input)
        {
            var predEngine = PredictEngine.Value;
            var result = predEngine.Predict(input);
            return GetSortedScoresWithLabels(result);
        }


        public static IOrderedEnumerable<KeyValuePair<string, float>> GetSortedScoresWithLabels(ModelOutput result)
        {
            var unlabeledScores = result.Score;
            var labelNames = GetLabels(result);

            Dictionary<string, float> labledScores = new Dictionary<string, float>();
            for (int i = 0; i < labelNames.Count(); i++)
            {
                // Map the names to the predicted result score array
                var labelName = labelNames.ElementAt(i);
                labledScores.Add(labelName.ToString(), unlabeledScores[i]);
            }

            return labledScores.OrderByDescending(c => c.Value);
        }

        private static IEnumerable<string> GetLabels(ModelOutput result)
        {
            var schema = PredictEngine.Value.OutputSchema;

            var labelColumn = schema.GetColumnOrNull("Label");
            if (labelColumn == null)
            {
                throw new Exception("Label column not found. Make sure the name searched for matches the name in the schema.");
            }

            // Key values contains an ordered array of the possible labels. This allows us to map the results to the correct label value.
            var keyNames = new VBuffer<ReadOnlyMemory<char>>();
            labelColumn.Value.GetKeyValues(ref keyNames);
            return keyNames.DenseValues().Select(x => x.ToString());
        }


        public static ModelOutput Predict(ModelInput input)
        {
            var predEngine = PredictEngine.Value;
            return predEngine.Predict(input);
        }



    }



    // Consumption Classes
    public class ModelInput
    {
        [LoadColumn(0)]
        [ColumnName(@"Label")]
        public string Label { get; set; }

        [LoadColumn(1)]
        [ColumnName(@"ImageSource")]
        public byte[] ImageSource { get; set; }

    }
    public class ModelOutput
    {
        [ColumnName(@"Label")]
        public uint Label { get; set; }

        [ColumnName(@"ImageSource")]
        public byte[] ImageSource { get; set; }

        [ColumnName(@"PredictedLabel")]
        public string PredictedLabel { get; set; }

        [ColumnName(@"Score")]
        public float[] Score { get; set; }

    }

}
