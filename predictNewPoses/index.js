// Load required libraries
const tf = require('@tensorflow/tfjs-node');
const PubNub = require('pubnub');

// Wortkshop Attendees Keys
let pubKey = 'pub-c-893f6b84-7326-4dd1-b759-a0856f21aa32';
let subKey = 'sub-c-7b917ba5-8544-497b-8258-8f8fde55e98d';

let aiChannelName = "ai_poses_unique_handle";

// v6, v19, v22 models works well
// Update model path to for your own local enviroment
const modelPath = 'file:///Users/jordanshaw/Sites/projects/tmls-workshop-wip-2023/codeAndTrainModel/_models/tmls-training-wip-lstm-v0.0.23/model.json';

// To be populated if we're using normalized data
let globalMinInput = 0;
let globalMaxInput = 0;
let globalMinOutput = 0;
let globalMaxOutput = 0;

// Initialize PubNub with publishKey, subscribeKey and other necessary parameters
const pubnub = new PubNub({
    publishKey: pubKey,
    subscribeKey: subKey,
    userId: "tensorflowModel",
    ssl: true  //enables a secure connection. 
});

// Define the PubNub listener to handle status, message and presence events
const listener = {
    status: statusEvent => {
        if (statusEvent.category === "PNConnectedCategory") {
            console.log("Connected");
        }
    },
    message: messageEvent => {
        console.log("message: " + messageEvent.message.description);
    },
    presence: () => { /* handle presence here if needed */ }
};

// Add the listener to PubNub
pubnub.addListener(listener);

// Function to publish a message to PubNub
const publishMessage = async (message) => {
    await pubnub.publish({
        channel: aiChannelName,
        storeInHistory: true,
        message: {
            poseAngles: message
        },
    });
};

// Function to load the Tensorflow model and make predictions
async function run() {
    // Load the Tensorflow model
    const model = await tf.loadLayersModel(modelPath);
    
    // Default input for the model
    let input = tf.tensor3d([22, 173, 89, 100], [1, 4, 1]);

    let count = 0;
    const intervalId = setInterval(async () => {
        if(count >= 1000) {
            // If count reaches 1000, stop the interval
            clearInterval(intervalId);
        } else {
            // Make a prediction with the model
            const prediction = model.predict(input);
            const array = await prediction.array();

            // Add randomness to the prediction to help avoid normalization
            const randomNoiseArray = array.map(subarray => 
                subarray.map(value => value + ((Math.random() - 0.5) * 2.5)) // Adjust the multiplier as needed
            );

            console.log(tf.util.flatten(randomNoiseArray));

            // Publish the noisy prediction to PubNub
            await publishMessage(tf.util.flatten(randomNoiseArray));

            // Reshape the noisy prediction to match the model's input shape
            input = tf.tensor3d([randomNoiseArray[0].map(x => [x])], [1, 4, 1]);

            count++;
        }
    }, 500);
}

// Start the run function
run();


// NOTE: Not all pre-trained modes are normalized.
// Normalization started with v24 models
//  If we are using normalized data, we can unnormalizez the data with the following

// This will go up aroudn like 74 and replace the model.predict(input) call
// let normalizedOutput = model.predict(inputData);
// let originalOutput = unnormalizeTensor(normalizedOutput, globalMinOutput, globalMaxOutput);

// Function to unnormalize a tensor to its original range
// function unnormalizeTensor(tensor, min, max) {
//     return tensor.mul(max.sub(min)).add(min);
// }