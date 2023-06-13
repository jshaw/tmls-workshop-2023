// Training Machines for Autonomous Interactive Artworks
// Workshop // Toronto Machine Learning Summit // 2023
// Jordan Shaw
// http://jordanshaw.com
// https://www.instagram.com/jshaw3 
//
// art+technology
//   ¯\_(ツ)_/¯
// 
//
// The particle emitter and image texture was referenced and inspired by The Coding Train's Particle system projects
// The Nature of Code
// The Coding Train / Daniel Shiffman / Dusk
// 
// https://youtu.be/pUhv2CA0omA
// https://thecodingtrain.com
// Particles
// https://editor.p5js.org/codingtrain/sketches/TTVoNt58T

// Initialize PubNub configuration variables.
let pubnub;
// Wortkshop Attendees Keys
let pubKey = 'pub-c-893f6b84-7326-4dd1-b759-a0856f21aa32';
let subKey = 'sub-c-7b917ba5-8544-497b-8258-8f8fde55e98d';
let aiChannelName = "ai_poses_unique_handle";

let img;
let emitterArray = []; // Emitter objects are stored in this array
let limbAngleArray = []; // Array to store the angles of limbs (arms/legs)

const VECTOR_INITIAL_VALUES = [300, 300, 300, 300];
let vectors = []; // Array to store the direction vectors for the limbs

let offsetVector; // Offset vector for the limbs drawing
const angleOffset = 90; // Angle offset for the limbs

let buttonTensorflowListen; // UI button to switch the input mode
let inputMode = true; // Input mode switch: true for model input, false for mouse input

const numParticles = 50;
let particleDataArray = [
    [],
    [],
    [],
    []
]; // 2D array to store particle data for each emitter
let currentAngleArray = [0, 0, 0, 0]; // Current angles of the limbs
let targetAngleArray = [0, 0, 0, 0]; // Target angles for the limbs to move towards

let dirArray = []; // Array to store direction for each emitter
let windArray = []; // Array to store wind vector for each emitter

// Preload image for the particle texture
function preload() {
    img = loadImage('texture32.png');
}

function setup() {
    createCanvas(800, 800);
    angleMode(DEGREES);

    offsetVector = createVector(400, 400);

    // Initialize PubNub
    pubnub = new PubNub({
        publish_key: pubKey,
        subscribe_key: subKey,
        userId: "p5Dashboard",
        ssl: true
    });

    // Attach callbacks to the PubNub object to handle incoming messages
    pubnub.addListener({
        message: readIncoming
    });
    pubnub.subscribe({
        channels: [aiChannelName]
    });

    // Initialize emitters, vectors and particle arrays
    for (let i = 0; i < 4; ++i) {
        emitterArray.push(new Emitter(100, 100, img, 400));
        vectors.push(createVector(VECTOR_INITIAL_VALUES[i], 0));
        for (let j = 0; j < numParticles; ++j) {
            particleDataArray[i].push(createVector(0, 0, 0));
        }
    }

    // Create a UI button for toggling the input mode
    buttonTensorflowListen = createButton('Mouse/Model Input');
    buttonTensorflowListen.position(20, 20);
    buttonTensorflowListen.mousePressed(toggleControlInput);
}

// Function to toggle between model and mouse inputs
function toggleControlInput() {
    inputMode = !inputMode;
}

function draw() {
    // Set the background color to black
    background(0);

    // Create a force vector with magnitude -0.1 in y-direction
    let force = createVector(0, -0.1);

    // Set the text size and color for displaying the input mode
    textSize(24);
    noStroke();
    fill(125, 125, 125);

    // Display the current input mode on the canvas
    if (inputMode) {
        text('Model', 20, 70); // Model is being used for input
    } else {
        text('Mouse', 20, 70); // Mouse is being used for input
    }

    // Apply the force to each emitter
    for (let i = 0; i < emitterArray.length; ++i) {
        emitterArray[i].applyForce(force);
    }

    // Set the color for drawing limbs
    fill(255); // white
    stroke(255); // white

    // If in Model input mode, perform operations based on current and target angles
    if (inputMode) {
        if (currentAngleArray.length > 0) {
            push(); // Save current drawing settings

            // Perform linear interpolation between the current and target angles
            for (let i = 0; i < currentAngleArray.length; i++) {
                currentAngleArray[i] = lerp(currentAngleArray[i], targetAngleArray[i], 0.03);
            }

            // Map the current angles to the limb angles with appropriate offsets
            limbAngleArray[0] = map(currentAngleArray[0], 0, 180, 180, 0) + angleOffset;
            limbAngleArray[1] = -map(currentAngleArray[1], 0, 180, 0, 180) + angleOffset;
            limbAngleArray[2] = 90 - currentAngleArray[2] + angleOffset;
            limbAngleArray[3] = 90 - currentAngleArray[3] + angleOffset;

            let vec = [];

            for (let i = 0; i < vectors.length; i++) {
                // Reset and rotate the vector
                vectors[i] = createVector(300, 0);
                vec[i] = vectors[i].rotate(limbAngleArray[i]);
            }

            // Move to the offset position before drawing
            translate(offsetVector.x, offsetVector.y);

            // Draw the limbs and update the emitter positions
            for (let i = 0; i < vectors.length; i++) {
                line(0, 0, vec[i].x, vec[i].y);
                emitterArray[i].updatePos(vec[i].x, vec[i].y, true);
            }

            pop(); // Restore saved drawing settings

            // Create wind force based on the limb positions and apply it to the emitters
            // Emit new particles and update the emitters
            for (let i = 0; i < emitterArray.length; ++i) {
                dirArray[i] = map(vec[i].x, width, 0, -0.1, 0.1);
                windArray[i] = createVector(dirArray[i], 0);

                emitterArray[i].applyForce(windArray[i]);
                emitterArray[i].emit(3);
                emitterArray[i].show(particleDataArray[i]);
                emitterArray[i].update();
            }

        }

    } else { // If in Mouse input mode, perform operations based on mouse positions

        // Update the emitter positions based on mouse positions
        emitterArray[0].updatePos(map(mouseX, 0, width, 0, width), mouseY);
        emitterArray[1].updatePos(map(mouseX, 0, width, width, 0), mouseY);
        emitterArray[2].updatePos(map(mouseX, 0, width, 0, width), map(mouseY, 0, height, height, 0));
        emitterArray[3].updatePos(map(mouseX, 0, width, width, 0), map(mouseY, 0, height, height, 0));

        // Create wind force based on the mouse position and apply it to the emitters
        let dir = map(mouseX, width, 0, -0.1, 0.1);
        let wind = createVector(dir, 0);

        for (let i = 0; i < emitterArray.length; ++i) {
            emitterArray[i].applyForce(wind);
            emitterArray[i].emit(3);
            emitterArray[i].show(particleDataArray[i]);
            emitterArray[i].update();
        }
    }
}

// when new data comes in it triggers this function, 
// this works becsuse we subscribed to the channel in setup()
function readIncoming(inMessage) {
    // simple error check to match the incoming to the channelName
    if (inMessage.channel == aiChannelName) {
        console.log("inMessage-->", inMessage);
        targetAngleArray = inMessage.message.poseAngles;
    }
}