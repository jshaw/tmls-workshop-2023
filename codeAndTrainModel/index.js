// Required Libraries
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

// Training Data
// Different training data 
// const { trainData } = require('./data.js')
// const { trainDataInput } = require('./dataInput.js')
// const { trainDataOutput } = require('./dataOutput.js')
const { trainDataInput } = require('./dataInputFull.js')

// Starting data point for running predict on the model
// Could be randomly generated down the raod
// Convert testData into a 2D array
const startData = [37, 149, 80, 84].map(num => [num]); 

// var for storing the model's training loss value to be logged out in a text file below
var saveLoss = "";

// v6:
// Model trained with 1500 epochs, label switch, and shuffle set to true.
// - Dataset: data.js
// - Case: 5

// v10:
// Model trained with 10 epochs, labels set to all 1s or all 0s.
// - Case: 1

// v11:
// Model trained with 100 epochs, identical training data for input and output.
// - Case: 5
// - Result: Data varies minimally, primarily within a noise variance range.

// v12:
// Model trained with 1500 epochs, identical training data for input and output, shuffle set to true.
// - Case: 5
// - Attempted improvement: Adding an offset to the data to provide variances to the predicted output.

// v13:
// Model trained with 1500 epochs, identical training data for input and output, shuffle set to false.
// - Case: 5
// - Result: Aside from the noise values, data tends to normalize. Did not shift the data array in this version.

// v14:
// Model trained with 1500 epochs, offset labels, shuffle set to true.
// - Case: 4
// - Label data offset to be paired with the next index in the dataset from dataInput.js.

// v15:
// Model trained with 1500 epochs, offset labels, shuffle set to false.
// - Case: 5
// - Label data offset to be paired with the next index in the dataset from dataInput.js.

// v16:
// Model trained with 1500 epochs, offset labels, shuffle set to false.
// - Case: 5
// - Label data offset to be paired with the next index in the dataset from data.js (double number of data points).

// v17:
// Model trained with 1500 epochs, offset labels, shuffle set to true.
// - Case: 5
// - Label data offset to be paired with the next index in the dataset from data.js (double number of data points).

// v18:
// Model trained with 1500 epochs, offset labels, shuffle set to true.
// - Case: 5
// - Label data offset to be paired with the next index in the dataset from dataInput.js.
// - Result: Not enough data points. Normalized and little movement observed.

// v19:
// Model trained with 1500 epochs, offset labels, shuffle set to true.
// - Case: 5
// - Dataset: dataInputFull.js (~7000 data points)
// - LSTM Units: 200 (first layer), 50 (second layer)
// - Result: Good variations and believable movement.

// v20:
// Model trained with 1500 epochs, offset labels, shuffle set to false.
// - Case: 5
// - Dataset: dataInputFull.js (~7000 data points)
// - LSTM Units: 200 (first layer), 50 (second layer)
// - Result: Less movement but still works and has variances.

// v21:
// Model trained with 500 epochs, offset labels, shuffle set to false.
// - Case: 6
// - Label data offset to be paired with the next index in the dataset from dataInputFull.js.
// - LSTM Units: 100 (first layer), 20 (second layer)
// - Result: Mean limb angles with less variations.

// v22:
// Model trained with 500 epochs, offset labels, shuffle set to true.
// - Case: 6
// - Label data offset to be paired with the next index in the dataset from dataInputFull.js.
// - LSTM Units: 100 (first layer), 20 (second layer)
// - Result: Comparable with v19.

// v23:
// Model trained with 500 epochs, offset labels, shuffle set to true.
// - Case: 4
// - Dataset: dataInputFull.js (~7000 data points)
// - LSTM Units: 20 (first layer), 200 (second layer), 50 (third layer)
// - Result: On par with v19.

let modelConfigVersion = 4;
let saveVersion = 23;
let epoch = 500;
// let epoch = 1500;
let shuffle = true;

// Function for creating and training the model
async function run(version, saveIndex){
    
    // Preprocessing of input and output data
    let inputData = preprocessInputData(trainDataInput);
    let outputData = preprocessOutputDataShift(trainDataInput);

    console.log(inputData.shape)
    inputData.print()
    console.log(outputData.shape)
    outputData.print()

    // Model creation and configuration
    const model = await getModel(version);

    // Model training
    await trainModel(model, inputData, outputData, epoch, shuffle);

    // Save the model and the loss value
    saveModel(model, saveIndex);
    saveModelLossValue(saveIndex);

    // Run predictions if needed
    predictNewPoses(model);
}

// Function for preprocessing the input data
function preprocessInputData(data) {
    let inputData = data.map((dataPoint) => dataPoint.map((num) => [num]));
    return tf.tensor3d(inputData, [data.length, 4, 1]);
}

// Function for preprocessing the output data
// This shifting was a tactic from my oritinal implimentation.
// Trying to see if this still works without doing this.
function preprocessOutputDataShift(data) {
    let tmp_output_data = [...data];
    let shift_array_value = tmp_output_data.shift();
    tmp_output_data.push(shift_array_value);
    return tf.tensor2d(tmp_output_data);
}

// Function for creating the model
async function getModel(version) {
    const model = tf.sequential();

    // Define the model based on the version
    // Each case represents a different model configuration
    switch(version) {
        case 1:

            model.add(tf.layers.lstm({
                units: 20,
                returnSequences: true,
                inputShape: [4, 1],
            }));
            
            // Add the Flatten layer here.
            model.add(tf.layers.flatten());
            
            model.add(tf.layers.dense({units: 4}));

            model.compile({
                optimizer: 'adam',
                loss: 'meanSquaredError'
            });
          break;
        case 2:

            model.add(tf.layers.lstm({
                units: 200,
                returnSequences: true,
                inputShape: [4, 1],
            }));
            
            // Add the Flatten layer here.
            model.add(tf.layers.flatten());
            
            model.add(tf.layers.dense({units: 4}));

            model.compile({
                optimizer: 'adam',
                loss: 'meanSquaredError'
            });

          break;
        case 3:

            // Adding more layers and units in each LSTM layer
            model.add(tf.layers.lstm({
                units: 20, // Increase the number of units
                returnSequences: true, // This must be true when stacking LSTM layers
                inputShape: [4, 1]
            }));

            // Add another LSTM layer
            model.add(tf.layers.lstm({
                units: 50,
                returnSequences: false // This can be false on the final LSTM layer
            }));

            // Add a Dropout layer for regularization
            model.add(tf.layers.dropout({rate: 0.5}));

            // Add the Flatten layer here.
            // don't need to add this since it the lstm is 2d and the dense layer is 2d this isn't needed
            // model.add(tf.layers.flatten());

            // Add a Dense layer
            model.add(tf.layers.dense({units: 4}));
            // end of test 2
            // =============

            model.compile({
                optimizer: 'adam',
                loss: 'meanSquaredError'
            });

            break;
        case 4:

            // Adding more layers and units in each LSTM layer
            model.add(tf.layers.lstm({
                units: 20, // Increase the number of units
                returnSequences: true, // This must be true when stacking LSTM layers
                inputShape: [4, 1]
            }));

            // Adding more layers and units in each LSTM layer
            model.add(tf.layers.lstm({
                units: 200, // Increase the number of units
                returnSequences: true, // This must be true when stacking LSTM layers
                inputShape: [4, 1]
            }));

            // Add another LSTM layer
            model.add(tf.layers.lstm({
                units: 50,
                returnSequences: false // This can be false on the final LSTM layer
            }));

            // Add a Dropout layer for regularization
            // Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.
            // model.add(tf.layers.dropout({rate: 0.2}));
            model.add(tf.layers.dropout({rate: 0.5}));

            // Add a Dense layer
            model.add(tf.layers.dense({units: 4}));

            model.compile({
                optimizer: 'adam',
                loss: 'meanSquaredError'
            });

            break;
        case 5:

            // Adding more layers and units in each LSTM layer
            model.add(tf.layers.lstm({
                units: 500, // Increase the number of units
                returnSequences: true, // This must be true when stacking LSTM layers
                inputShape: [4, 1]
            }));

            // Add another LSTM layer
            model.add(tf.layers.lstm({
                units: 50,
                returnSequences: false // This can be false on the final LSTM layer
            }));

            // Add a Dropout layer for regularization
            // Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.
            model.add(tf.layers.dropout({rate: 0.5}));

            // Add a Dense layer
            model.add(tf.layers.dense({units: 4}));

            model.compile({
                optimizer: 'adam',
                loss: 'meanSquaredError'
            });

            break;
        case 6:

            // Adding more layers and units in each LSTM layer
            model.add(tf.layers.lstm({
                units: 100, // Increase the number of units
                returnSequences: true, // This must be true when stacking LSTM layers
                inputShape: [4, 1]
            }));

            // Add another LSTM layer
            model.add(tf.layers.lstm({
                units: 20,
                returnSequences: false // This can be false on the final LSTM layer
            }));

            // Add a Dropout layer for regularization
            // Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.
            model.add(tf.layers.dropout({rate: 0.5}));

            // Add a Dense layer
            model.add(tf.layers.dense({units: 4}));

            model.compile({
                optimizer: 'adam',
                loss: 'meanSquaredError'
            });

            break;
        default:

            // Same as case 5
            // Adding more layers and units in each LSTM layer
            model.add(tf.layers.lstm({
                units: 200, // Increase the number of units
                returnSequences: true, // This must be true when stacking LSTM layers
                inputShape: [4, 1]
            }));

            // Add another LSTM layer
            model.add(tf.layers.lstm({
                units: 50,
                returnSequences: false // This can be false on the final LSTM layer
            }));

            // Add a Dropout layer for regularization
            // Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.
            model.add(tf.layers.dropout({rate: 0.5}));

            // Don't need Flatten laye to add this since it the lstm is 2d and the dense layer is 2d this isn't needed
            // model.add(tf.layers.flatten());

            // Add a Dense layer
            model.add(tf.layers.dense({units: 4}));

            model.compile({
                optimizer: 'adam',
                loss: 'meanSquaredError'
            });
    }

    return model;
}

// Function for training the model
async function trainModel(model, inputData, outputData, epoch, shuffle) {

    // if(shuffle){
    //     tf.util.shuffleCombo(inputData, outputData);
    // }

    await model.fit(inputData, outputData, {
        epochs: epoch,
        shuffle: shuffle,
        callbacks: {
            onEpochEnd: async(num, logs) =>{
                saveLoss = logs.loss;
            }
        }
    });
}

// Function for saving the model
function saveModel(model, saveIndex){
    const saveResult = model.save('file://_models/tmls-training-wip-lstm-v0.0.'+saveIndex);
}

// Function for saving the model's loss value
function saveModelLossValue(saveIndex){
    fs.writeFile('./_models/_loss_v' + saveIndex + '.txt', saveLoss.toString(), err => {
        if (err) {
            console.error(err);
        } else {
            // file written successfully
            console.log("file written successfully");
        }
    });
}

// Run the model training
// run(switch case value version, save index);
run(modelConfigVersion, saveVersion);


function predictNewPoses(model){
    // Now you can use `model.predict` to make predictions
    // Assume `startData` is a single data point of 4 numbers
    let input = tf.tensor3d([startData], [1, 4, 1]); 
    let prediction = model.predict(input);
    
    // Log out the prediction
    prediction.array().then(array => console.log("PREDICTION -> : ", array));

    let count = 0;
    const intervalId = setInterval(() => {
        if(count >= 100) {
            clearInterval(intervalId);
        } else {
            const prediction = model.predict(input, {'verbose': true});
            prediction.array().then(array => {

                // Add a slight randomness to the prediction to help avoid normalization
                const randomNoiseArray = array.map(subarray => 
                    // subarray.map(value => value + ((Math.random() - 0.5) * 0.1)) // Adjust the multiplier as needed
                    subarray.map(value => value + ((Math.random() - 0.5) * 2.5)) // Adjust the multiplier as needed
                );

                console.log(randomNoiseArray);
                
                // Make sure to reshape the array back to [1, 4, 1] as it is expected by the model
                // We assume that array is of shape [1, 4]
                input = tf.tensor3d([randomNoiseArray[0].map(x => [x])], [1, 4, 1]);
            });
            count++;
        }
    }, 250);

}


// ==============
// =============
//
// Normalization functions and variables
// Not being used in models before v23
//
// ==============
// =============

// let globalMinInput;
// let globalMaxInput;
// let globalMinOutput;
// let globalMaxOutput;

// // Function for preprocessing the input data and normalizing it
// function preprocessInputData(data) {
//     let inputData = data.map((dataPoint) => dataPoint.map((num) => [num]));
//     let tensorData = tf.tensor3d(inputData, [data.length, 4, 1]);
//     let {normalizedTensor, min, max} = normalizeTensor(tensorData);
//     globalMinInput = min;
//     globalMaxInput = max;
//     return normalizedTensor;
// }

// // Function for preprocessing the output data and normalizing it
// function preprocessOutputDataShift(data) {
//     let tmp_output_data = [...data];
//     let shift_array_value = tmp_output_data.shift();
//     tmp_output_data.push(shift_array_value);
//     let tensorData = tf.tensor2d(tmp_output_data);
//     let {normalizedTensor, min, max} = normalizeTensor(tensorData);
//     globalMinOutput = min;
//     globalMaxOutput = max;
//     return normalizedTensor;
// }

// To predict and return normalized data to unnormalized data we need to add the following around line 422 where we test our predictions of the model
// ==========
// let normalizedOutput = model.predict(inputData);
// let originalOutput = unnormalizeTensor(normalizedOutput, globalMinOutput, globalMaxOutput);


// Add this to the _loss_vxx.txt log file to keep track of the model mins and max for when `predictNewPoses/index.js`
// ==========
// let stringOutput = "loss: " + saveLoss.toString() + ". min input: " + globalMinInput + " - max input: " + globalMaxInput + ". min output: " + globalMinOutput + " - max output: " + globalMaxOutput + ".";

// ===============
// End of normalization
// ===============




// ===============
// NOT USED 
// Was for testing purposed only
// ===============
// Function for preprocessing the output data
// Was testing using tf.oneHot and transforming single arrays [1] & [0] => [1,1,1,1] & [0,0,0,0]
// function preprocessOutputData(data) {
//     // let flat = tf.util.flatten(data);
//     // let labelsTensor = tf.tensor1d(flat, 'int32');
//     // let outputData = tf.oneHot(labelsTensor, 2);
//     let newData = data.map(subArr => Array(4).fill(subArr[0]));
//     console.log("outputData", newData);
//     return tf.tensor2d(newData);
// }