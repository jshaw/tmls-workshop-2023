let pubnub;

// Wortkshop Attendees Keys
let pubKey = 'pub-c-893f6b84-7326-4dd1-b759-a0856f21aa32';
let subKey = 'sub-c-7b917ba5-8544-497b-8258-8f8fde55e98d';

let channelName = "dataset_unique_handle";
let aiChannelName = "ai_poses_unique_handle";

let saveToPubnub = false;
let saveToPubnubText = "Saving to Pubnub";
let notSaveToPubnubText = "Not Saving to Pubnub";

let sendToParticles = false;
let sendToParticlesText = "Sending To Art Particles";
let notSendToParticlesText = "Not Sending To Art Particles";

let tfControl = "Controlled by TF";
let notTfControl = "Not controlled by TF";
let incomingText = [];

let mode = false;
let reviewState = false;
let tensorflowControl = false;
let allMessages = [];
let allMessagesApproved = [];
let allTrainingDataFull = [];

let allMessagesInput = [];
let allMessagesOutput = [];
let reviewIndex = 0;

let allMessagesReviewedCoutner = 0;

var vid;
var video;
var capture;
var videoState = false;
var poseNet;
var currentPoses;
var poseNetModelReady = false;

var rightUpper,
    leftUpper,
    rightLower,
    leftLower;

let windowWidth = 1400;
let windowHeight = 800;

let buttonPlayPause,
    buttonSavePubnub,
    buttonMode,
    buttonReview,
    buttonTensorflowListen,
    buttonPulse,
    buttonApprove,
    buttonDecline,
    buttonPrevious,
    buttonNext,
    buttonReset,
    buttonExport,
    buttonExportApproved,
    buttonExportAll;

// Typography
let titleFontSize = 32;
let textFontSize = 20;

let videoOffset1 = 65;
let videoOffset2 = 120;
var installation;

// Global keeping track of review timestamp
var timestamp = "";

// Angle averages
// Done to help prevent jumpy pose angles
let averages = [{
        average: 0,
        values: [0, 0, 0, 0, 0]
    },
    {
        average: 0,
        values: [0, 0, 0, 0, 0]
    },
    {
        average: 0,
        values: [0, 0, 0, 0, 0]
    },
    {
        average: 0,
        values: [0, 0, 0, 0, 0]
    },
];

// preload() runs once
function preload() {
    // mode false is video input
    video = createVideo("./_assets/video_for_pose_training2.mp4", vidLoad);
}

function setup() {
    createCanvas(windowWidth, windowHeight)

    // Set text characteristics
    textSize(titleFontSize);
    textAlign(CENTER, CENTER);

    // initialize pubnub
    pubnub = new PubNub({
        publish_key: pubKey, //get these from the pubnub account online
        subscribe_key: subKey,
        userId: "p5Dashboard",
        ssl: true //enables a secure connection. 
    });

    //attach callbacks to the pubnub object to handle messages and connections
    pubnub.addListener({
        message: readIncoming,
        messageAction: function(ma) {
            // handle message action
            var channelName = ma.channel; // The channel to which the message was published
            var publisher = ma.publisher; //The Publisher
            var event = ma.event; // message action added or removed
            var type = ma.data.type; // message action type
            var value = ma.data.value; // message action value
            var messageTimetoken = ma.data.messageTimetoken; // The timetoken of the original message
            var actionTimetoken = ma.data.actionTimetoken; // The timetoken of the message action

            updateTimestamp(messageTimetoken);
        },
        presence: function(p) {
            // handle presence
            var action = p.action; // Can be join, leave, state-change, or timeout
            var channelName = p.channel; // The channel to which the message was published
            var occupancy = p.occupancy; // Number of users subscribed to the channel
            var state = p.state; // User State
            var channelGroup = p.subscription; //  The channel group or wildcard subscription match (if exists)
            var publishTime = p.timestamp; // Publish timetoken
            var timetoken = p.timetoken; // Current timetoken
            var uuid = p.uuid; // UUIDs of users who are subscribed to the channel
        },
    });
    pubnub.subscribe({
        channels: [channelName, aiChannelName]
    });

    installation = new ourPresenceWithTheirsRender();
    installation.createLimbs();

    // Mode true is webcam capture
    capture = createCapture(VIDEO);
    capture.size(720, 480);
    capture.hide()

    video.elt.addEventListener('loadeddata', () => {
        poseNetVideo = ml5.poseNet(video, onPoseNetModelReady); //call onPoseNetModelReady when ready
        poseNetVideo.on('pose', (poses) => {
            // call onPoseDetected when pose detected
            onPoseDetected(poses, false)
        });
    });

    capture.elt.addEventListener('loadeddata', () => {
        poseNetCapture = ml5.poseNet(capture, onPoseNetModelReady); //call onPoseNetModelReady when ready
        poseNetCapture.on('pose', (poses) => {
            // call onPoseDetected when pose detected
            onPoseDetected(poses, true)
        });
    });

    buttonPlayPause = createButton('PLAY / PAUSE');
    buttonPlayPause.position(20, 570);
    buttonPlayPause.mousePressed(playPause);

    buttonSavePubnub = createButton('Save to Pubnub');
    buttonSavePubnub.position(160, 570);
    buttonSavePubnub.mousePressed(savePubnub);

    buttonPulse = createButton('Toggle Video/Webcam');
    buttonPulse.position(20, 600);
    buttonPulse.mousePressed(toggleVideoWebcamInput);

    buttonSendToParticles = createButton('Toggle Send to Particle Sketch');
    buttonSendToParticles.position(180, 600);
    buttonSendToParticles.mousePressed(toggleSendToParticles);

    buttonReview = createButton('Review Pubnub Data');
    buttonReview.position(20, 630);
    buttonReview.mousePressed(reviewPubnubData);

    buttonTensorflowListen = createButton('Tensorflow Listen');
    buttonTensorflowListen.position(20, 660);
    buttonTensorflowListen.mousePressed(tensorflowListen);

    buttonExport = createButton('Export Poses and Labels');
    buttonExport.position(20, 690);
    buttonExport.mousePressed(exportPosesLabels);

    buttonExportApproved = createButton('Export APPROVED Poses');
    buttonExportApproved.position(200, 690);
    buttonExportApproved.mousePressed(exportApprovedPoses);

    buttonExportAll = createButton('Export ALL Poses');
    buttonExportAll.position(390, 690);
    buttonExportAll.mousePressed(exportAllPoses);

    buttonReset = createButton('Clear Pubnub History (Deletes Dataset)');
    buttonReset.position(20, 720);
    buttonReset.mousePressed(deleteTestMessages);

    // Approve Button
    buttonApprove = createButton('APPROVE');
    buttonApprove.position(850, 660);
    buttonApprove.mousePressed(() => {
        updatePose(true);
    });

    // Disaprove Button
    buttonDecline = createButton('DECLINE');
    buttonDecline.position(1000, 660);
    buttonDecline.mousePressed(() => {
        updatePose(false);
    });

    buttonPrevious = createButton('PREVIOUS');
    buttonPrevious.position(1150, 660);
    buttonPrevious.mousePressed(() => {
        previousPose();
    });

    buttonNext = createButton('NEXT');
    buttonNext.position(1300, 660);
    buttonNext.mousePressed(() => {
        nextPose();
    });

    buttonApprove.hide();
    buttonDecline.hide();
    buttonPrevious.hide();
    buttonNext.hide();
}

function vidLoad() {
    video.volume(0);
    video.loop();
    video.pause();
    video.speed(1);
    video.hide(); // hides the html video loader
}

function draw() {
    background(255);

    // Text / App name
    fill(0);
    textAlign(LEFT);
    textSize(titleFontSize);
    text("[Dashboard] TMLS - Workshop", 15, 31)
    textSize(16);

    // if True, load and use webcam for pose
    if (mode) {
        // use webcam
        image(capture, 25, 70, 720, 480);
    } else {
        // use video
        image(video, 25, 70, 720, 480);
    }

    if (saveToPubnub) {
        text(saveToPubnubText, 275, 572);
    } else {
        text(notSaveToPubnubText, 275, 572);
    }

    if (sendToParticles) {
        text(sendToParticlesText, 390, 602);
    } else {
        text(notSendToParticlesText, 390, 602);
    }

    if (tensorflowControl) {
        text(tfControl, 150, 662);
    } else {
        text(notTfControl, 150, 662);
    }

    textSize(titleFontSize);


    // When the poseNet Model is loading
    if (!poseNetModelReady) {
        textSize(titleFontSize);
        textAlign(CENTER);
        noStroke();
        text("Waiting for PoseNet model to load...", width / 2, height / 2);
    } else if (tensorflowControl) {

        // This is when we are listening for model data from our trained tensorflow model
        // This data will be coming from /newPoseModel/index.js 

        console.log("incomingText ", incomingText);

        if (!incomingText.length > 0) {
            return false;
        }

        // else we are controlling the installation view
        let angles = incomingText;

        renderInstallationView(angles);
        renderReviewButtons(true);

    } else if (reviewState) {

        // This is the review state of of captured pose data
        // We are replaying and approving / rejecting poses stored in Pubnub

        fill(0);

        let allMessagesLength = allMessages.length
        let indexString = `Review index: ${reviewIndex} of ${allMessagesLength}`;

        // update message function call
        if (allMessagesLength > 0) {

            let angles = allMessages[reviewIndex].message.poseAngles;

            renderInstallationView(angles);
            renderReviewButtons(true);

            // used to display the current index review state
            if (typeof allMessages[reviewIndex].data != 'undefined') {

                if (typeof allMessages[reviewIndex].data.approved != 'undefined') {
                    if (allMessages[reviewIndex].data.approved.hasOwnProperty('true')) {
                        // pose has been reviewed and approved
                        indexString += " ✅";
                    } else {
                        // pose has been declined
                        indexString += " ❌";
                    }
                }
            } else {
                // This is where the pose hasn't been reviewed at all
                indexString += " ❔";
            }
        }

        // Displaying the pose review state and index
        if ((reviewIndex * 1) >= allMessagesLength) {
            indexString += ". \nReached the end of the training data.";
            text(indexString, 840, 625);
            return false;
        } else {
            text(indexString, 840, 625);
        }

    } else {

        // DEFAULT STATE
        // This is the default state
        // Used for recording data from videos or webcam input
        // just p5.js running in its draw() function.

        if (currentPoses && currentPoses.length > 0) {

            let pose = currentPoses[0].pose;

            drawlimbs(
                [pose.leftShoulder.x, pose.leftShoulder.y],
                [pose.leftWrist.x, pose.leftWrist.y]
            );

            drawlimbs(
                [pose.rightShoulder.x, pose.rightShoulder.y],
                [pose.rightWrist.x, pose.rightWrist.y]
            );

            if (pose.leftHip.confidence > 0.4) {
                drawlimbs(
                    [pose.leftHip.x, pose.leftHip.y],
                    [pose.leftAnkle.x, pose.leftAnkle.y]
                );
            } else {
                // Draw legs straight down
            }

            if (pose.rightHip.confidence > 0.4) {
                drawlimbs(
                    [pose.rightHip.x, pose.rightHip.y],
                    [pose.rightAnkle.x, pose.rightAnkle.y]
                );
            } else {
                // Draw legs straight down
            }

            let rightWristNormalize = pose.rightShoulder.x + (pose.rightShoulder.x - pose.rightWrist.x);

            leftUpper = angleDegrees({
                x: pose.leftShoulder.x,
                y: pose.leftShoulder.y
            }, {
                x: pose.leftWrist.x,
                y: pose.leftWrist.y
            });
            rightUpper = angleDegrees({
                x: pose.rightShoulder.x,
                y: pose.rightShoulder.y
            }, {
                x: rightWristNormalize,
                y: pose.rightWrist.y
            });
            leftLower = angleDegrees({
                x: pose.leftHip.x,
                y: pose.leftHip.y
            }, {
                x: pose.leftAnkle.x,
                y: pose.leftAnkle.y
            });
            rightLower = angleDegrees({
                x: pose.rightHip.x,
                y: pose.rightHip.y
            }, {
                x: pose.rightAnkle.x,
                y: pose.rightAnkle.y
            });

            leftUpper = map(leftUpper, -50, 100, 180, 0, true);
            rightUpper = map(rightUpper, -60, 90, 0, 180, true);

            serialLeftLower = map(leftLower, 0, 180, 0, 45, true);
            serialRightLower = map(rightLower, 0, 180, 120, 180, true);

            averages[0].values.unshift(leftUpper)
            averages[0].values.pop()
            averages[0].average = averages[0].values.reduce((a, b) => a + b, 0) / averages[0].values.length;

            averages[1].values.unshift(rightUpper)
            averages[1].values.pop()
            averages[1].average = averages[1].values.reduce((a, b) => a + b, 0) / averages[1].values.length;

            averages[2].values.unshift(leftLower)
            averages[2].values.pop()
            averages[2].average = averages[2].values.reduce((a, b) => a + b, 0) / averages[2].values.length;

            averages[3].values.unshift(rightLower)
            averages[3].values.pop()
            averages[3].average = averages[3].values.reduce((a, b) => a + b, 0) / averages[3].values.length;

            drawDegs(leftUpper, pose.leftShoulder.x, pose.leftShoulder.y);
            drawDegs(averages[0].average, pose.leftShoulder.x, pose.leftShoulder.y + 20);

            drawDegs(rightUpper, pose.rightShoulder.x, pose.rightShoulder.y);
            drawDegs(averages[1].average, pose.rightShoulder.x, pose.rightShoulder.y + 20);

            drawDegs(leftLower, pose.leftHip.x, pose.leftHip.y);
            drawDegs(averages[2].average, pose.leftHip.x, pose.leftHip.y + 20);

            drawDegs(rightLower, pose.rightHip.x, pose.rightHip.y);
            drawDegs(averages[3].average, pose.rightHip.x, pose.rightHip.y + 20);

            // Below, rearranging the averages make the left/right limbs on the arduino become mirrors vs. oppisite limbs
            limbAngle = [int(averages[1].average), int(averages[0].average), int(averages[3].average), int(averages[2].average)];

            // This can be used for sending angle data to servo motors via a serial port
            serialLimbAngle = [int(averages[1].average), int(averages[0].average), int(map(int(averages[3].average), 0, 180, 140, 180)), int(map(int(averages[2].average), 0, 180, 10, 45))];

            // We are either saving data to pubnub from our webcam or video
            // OR
            // We want to send the pose angles to our pose particle system sketch
            // This is in /p5particlesWorkshop/sketch.js
            if (saveToPubnub || sendToParticles) {

                let channelToSendTo = sendToParticles == true ? aiChannelName : channelName;

                console.log("channelToSendTo: ", channelToSendTo);

                throttle(function() {
                    sendLimbAngle(limbAngle, channelToSendTo);
                }, 250);
            }

            let angles = [rightUpper, leftUpper, leftLower, rightLower];

            renderInstallationView(angles);
            renderReviewButtons(false);

        } else {

            console.log("EVER GET HERE?");
            console.log("THE DEFAULT COULD BE GLITCHY AND JUST DRAWING THE LIMBS...?");

            push()
            translate(1075, 325);
            installation.drawLimbs();
            pop()
        }
    }

}

function previousPose() {
    timestamp = allMessages[reviewIndex - 1].timetoken;
    reviewIndex--;
}

function nextPose() {
    timestamp = allMessages[reviewIndex].timetoken;
    reviewIndex++;
}

function goToIndex(index) {
    reviewIndex = index;
    timestamp = allMessages[reviewIndex].timetoken;
}

function updatePose(status) {

    timestamp = allMessages[reviewIndex].timetoken;

    updateMessage(timestamp, status);
}

function renderInstallationView(angles) {
    push();
    translate(1075, 325);
    installation.display();

    if (angles[2] < 0) {
        angles[2] == 80
    }

    if (angles[3] < 0) {
        angles[3] == 100
    }

    installation.updateLimbs(0, map(angles[0], 0, 180, 180, 0));
    installation.updateLimbs(1, -map(angles[1], 0, 180, 0, 180));
    installation.updateLimbs(2, 90 - angles[2]);
    installation.updateLimbs(3, 90 - angles[3]);

    installation.drawLimbs();
    pop();
}

function renderReviewButtons(showButtons) {
    if (showButtons) {
        buttonApprove.show();
        buttonDecline.show();
        buttonPrevious.show();
        buttonNext.show();
    } else {
        buttonApprove.hide();
        buttonDecline.hide();
        buttonPrevious.hide();
        buttonNext.hide();
    }
}

// Export all poses, no filters on approval status
let callbackAllMessagesExport = function(item, index, array) {

    allMessagesReviewedCoutner++;
    allTrainingDataFull.push(item);

    // This needs to happen outside of all the conditional checks
    if (allMessagesReviewedCoutner === array.length) {
        allMessagesInput = allTrainingDataFull.map((item) => {
            return Object.values(item.message.poseAngles);
        });

        console.log("allMessagesInput ", allMessagesInput);
    }

}

// status = true/false
// true = export approved reviewed poses
// false = return all reviewed poses, approved or rejected with their labels
let callbackFilteredMessagesExport = function(filter, item, index, array) {

    console.log("FILTER: ", filter)
    console.log("approved item: ", item)
    console.log("approved index: ", index)
    console.log("approved array: ", array)
    console.log("arguments: ", arguments)
    allMessagesReviewedCoutner++;

    // I need to only push into dataset data that has been reviewed...
    // as long as it was approved/rejected it's good to be added to the dataset
    // non-reviewd poses do not get added.
    if (typeof item.data != 'undefined') {
        console.log("ma: 1");
        if (typeof item.data.approved != 'undefined') {
            console.log("ma: 2");
            // let status = allMessages[reviewIndex].data.approved.true;
            // let status = item.data.approved[0];
            let status = item.data.approved.hasOwnProperty('true');

            console.log("ma: 2 -- status: ", status);
            console.log("ma: 2 -- status[true]: ", item.data.approved['true']);
            console.log("ma: 2 -- status.hasOwnProperty(true): ", item.data.approved.hasOwnProperty('true'));

            // removed this conditional, since we want to add both approved / rejected to the dataset
            if (!filter) {
                console.log("ma: 3");
                console.log("status.toString(): ", status.toString());

                allTrainingDataFull.push(item);

                let output_tmp_array = [];
                if (status == true) {
                    output_tmp_array.push(1);
                } else {
                    output_tmp_array.push(0);
                }

                allMessagesOutput.push(output_tmp_array);

            } else if (filter && status) {
                allMessagesApproved.push(item);
            }
        }
    }

    // This needs to happen outside of all the conditional checks
    if (allMessagesReviewedCoutner === array.length) {

        if (filter) {
            allMessagesInput = allMessagesApproved.map((item) => {
                return Object.values(item.message.poseAngles);
            });
        } else {
            allMessagesInput = allTrainingDataFull.map((item) => {
                return Object.values(item.message.poseAngles);
            });
        }

        console.log("allMessagesInput ", allMessagesInput);
        console.log("allMessagesOutput ", allMessagesOutput);
    }

}

// status = true/false
// true = export approved reviewed poses
// false = return all reviewed poses, approved or rejected with their labels
function exportPosesLabels() {

    resetExports()
    getAllHistory().then(() => {
        allMessages.forEach(callbackFilteredMessagesExport.bind(null, false))
    });

}

// status = true/false
// true = export approved reviewed poses
// false = return all reviewed poses, approved or rejected with their labels
function exportApprovedPoses() {

    resetExports()
    getAllHistory().then(() => {
        allMessages.forEach(callbackFilteredMessagesExport.bind(null, true))
    });
}

// Export all poses, no filters on approval status
function exportAllPoses() {

    resetExports()
    getAllHistory().then(() => {
        // Dope, this works as expected
        allMessages.forEach(callbackAllMessagesExport)
    });
}

// resets needed before re-export of data points
function resetExports() {
    // reset the stored data arrays before re-filter
    allMessagesReviewedCoutner = 0;
    allMessagesApproved = [];
    allTrainingDataFull = [];
    allMessagesInput = [];
    allMessagesOutput = [];
}

/**
 * Called automatically by the browser through p5.js when mouse clicked
 **/
function playPause() {
    mode = false;
    reviewState = false;

    if (videoState == false) {
        video.play();
        videoState = true;
    } else {
        video.pause()
        videoState = false;
    }
}

function tensorflowListen() {
    tensorflowControl = !tensorflowControl;
}

function toggleVideoWebcamInput() {

    mode = !mode;
    reviewState = false;

    // reset limbs to make sure there's no data archive for limb positions
    installation.resetLimbs();
    installation.createLimbs();
}

function reviewPubnubData() {
    reviewState = !reviewState;
    reviewIndex = 0;

    getAllHistory();
}

function savePubnub() {
    saveToPubnub = !saveToPubnub;
}

function toggleSendToParticles() {
    sendToParticles = !sendToParticles;
}

/**
 * Draws a limbs based on 2 x,y objects being passed at the given x,y position
 * 
 * @param {number} x the x pos of the nose
 * @param {number} y the y pos of the nose
 */
function drawlimbs(p1, p2) {
    push();
    strokeWeight(10)
    stroke('red');
    line(p1[0] + videoOffset1, p1[1] + videoOffset2, p2[0] + videoOffset1, p2[1] + videoOffset2);
    pop();
}

/**
 * Draws a Degrees near limbs base (Video)
 * 
 * @param {number} x the x pos of the nose
 * @param {number} y the y pos of the nose
 */
function drawDegs(val, x, y) {
    strokeWeight(0)
    textSize(textFontSize);
    fill('white');
    text(int(val), x + videoOffset1, y + videoOffset2);
}

/**
 * Calculate Rad Angle between two points
 * 
 * @param {object} {x: #, y: #}
 * @param {object} {x: #, y: #}
 */
function angleRadians(p1, p2) {
    return Math.atan2(p2.y - p1.y, p2.x - p1.x);
}

/**
 * Calculate Deg Angle between two points
 * 
 * @param {object} {x: #, y: #}
 * @param {object} {x: #, y: #}
 */
function angleDegrees(p1, p2) {
    return Math.atan2(p2.y - p1.y, p2.x - p1.x) * 180 / Math.PI;
}


/**
 * Callback function called by ml5.js PoseNet when the PoseNet model is ready
 * Will be called once and only once
 */
function onPoseNetModelReady() {
    print("The PoseNet model is ready...");
    poseNetModelReady = true;
}

/**
 * Callback function called by ml5.js PosetNet when a pose has been detected
 */
function onPoseDetected(poses, eventMode) {

    // only update and show the pose for the currently selected mode
    if (eventMode == mode) {
        currentPoses = poses;
    }

    if (currentPoses) {
        let strHuman = " human";
        if (currentPoses.length > 1) {
            strHuman += 's';
        }
        text("We found " + currentPoses.length + strHuman);
    }
}

// 
// Installation view
class ourPresenceWithTheirsRender {
    constructor() {
        this.x = 0;
        this.y = 0;
        this.diameter = 500;
        this.limbs = [];
    }

    display() {
        push();
        fill(200);
        ellipse(this.x + 5, this.y, this.diameter, this.diameter);
        pop();
    }

    createLimbs() {
        this.limbs.push(new Limb(-25, -25, 40, 200));
        this.limbs.push(new Limb(25, -25, 40, 200, undefined));
        this.limbs.push(new Limb(-25, 25, 40, 200, undefined));
        this.limbs.push(new Limb(25, 25, 40, 200, undefined));
    }

    resetLimbs() {
        this.limbs = [];
    }

    drawLimbs() {
        for (let i = 0; i < this.limbs.length; i++) {
            push();
            this.limbs[i].display();
            pop();
        }
    }

    updateLimbs(limb, deg) {
        push();
        this.limbs[limb].rotate(deg);
        this.limbs[limb].deg = deg;
        pop();
    }
}

// Limb Class for Installation
class Limb {
    constructor(x, y, w, h, deg = 0, color = "black", render = true) {
        this.x = x;
        this.y = y;
        this.w = w;
        this.h = h;
        this.deg = deg;
        this.color = color;
        this.render = render;
    }

    rotate() {
        push();
        angleMode(DEGREES);
        rectMode(CORNER);
        rotate(this.deg);
        pop();
    }

    display() {

        if (!this.render) return;

        push();
        angleMode(DEGREES);
        rectMode(CORNER);

        fill(this.color);
        translate(this.x, this.y);
        rotate(this.deg);
        rect(0, 0, this.w, this.h);
        pop();
    }

    updateDeg(deg = 0) {
        this.deg = deg;
    }
}

//when new data comes in it triggers this function, 
// this works becsuse we subscribed to the channel in setup()
function readIncoming(inMessage) {
    // simple error check to match the incoming to the channelName
    if (inMessage.channel == channelName) {
        console.log("inMessage: ", inMessage);
    } else {
        console.log("inMessage: ", inMessage)
        incomingText = inMessage.message.poseAngles;
    }
}

function stringify(message) {
    let string = JSON.stringify(message)
    return JSON.stringify(message);
}

function sendLimbAngle(array, channel) {
    // Send Data to the server to draw it in all other canvases
    // get the value from the text box and send it as part of the message   

    return pubnub.publish({
        channel: channel,
        storeInHistory: true,
        message: {
            poseAngles: array
        }
    });
}

function sendTheMessage() {

    // Send Data to the server to draw it in all other canvases
    // get the value from the text box and send it as part of the message   
    pubnub.publish({
        channel: channelName,
        storeInHistory: true,
        message: {
            poseAngles: [5, 5, 5, 5]
        }
    });
}

function testThrottleFunction() {
    console.log("Test Message");
}

let throttleTimer;

const throttle = (callback, time = 2000) => {
    if (throttleTimer) return;
    throttleTimer = true;
    setTimeout(() => {
        callback();
        throttleTimer = false;
    }, time);
}

// Documentation for 
function getHistory() {
    let history = pubnub.fetchMessages({
        channels: [channelName],
        count: 100,
        includeMessageActions: true,
    })

    console.log("history: ", history);
}



// NOTE: While fetching messages with reactions, the count parameter has a max value of 25.
// It's the fetch messages API that allows for returning MessageActions
// =======
async function getAllHistory(initialTimetoken = 0) {
    let msgs = "";
    let latestCount = 25;
    let timetoken = initialTimetoken;

    // Whenever this is called, re-pull Pubnub dataset and re-populate our local array
    allMessages = [];

    while (latestCount === 25) {
        const messages = await pubnub.fetchMessages({
            channels: [channelName],
            count: 25,
            stringifiedTimeToken: true, // false is the default
            start: timetoken, // start timetoken to fetch
            includeMessageActions: true,
        });

        if (!messages.channels[channelName]) {
            return false;
        }

        latestCount = messages.channels[channelName].length;
        timetoken = messages.channels[channelName][latestCount - 1].timetoken
        msgs = messages.channels[channelName];

        if (messages.channels[channelName] && latestCount > 0) {
            allMessages.push(...msgs);
        }
    }

    // All messages is now a global variable
    console.log("allMessages: ", allMessages);

    // setting the default timestamp value for review
    timestamp = allMessages[0].timetoken;

    // don't need this ... todo delete later
    return allMessages;
}

function updateMessage(timestamp, status) {

    // Check the current state of it... so i don't resave true as true etc...
    var currentState = "";

    if (typeof allMessages[reviewIndex].data != 'undefined') {
        if (typeof allMessages[reviewIndex].data.approved != 'undefined') {
            if (allMessages[reviewIndex].data.approved.hasOwnProperty(status.toString())) {
                // If you are trying to set the status of a pose that already has that status, just increment to the next pose
                reviewIndex++
            } else {
                // This is called when a pose has been reviewed,
                // But you are re-reviewing and trying to set a new status than it's original
                var updatedMessageAction = pubnub.addMessageAction({
                    channel: channelName,
                    messageTimetoken: timestamp.toString(),
                    action: {
                        type: 'approved',
                        value: status.toString()
                    },
                }, function(status, response) {
                    reviewIndex++;
                });
            }
        } else {
            // We should never get here.
            // There should not be a case where there's an 'approved' object saved to the pose
            // Without the pose approved object having either an true || false child object key
            console.log("Note: how did we get here? ");
            console.log("There should not be a case where there's an 'approved' object saved to the pose without an approval key true||false");
        }
    } else {
        // Called when the message has not b een reviewed yet
        // Review essentially is null
        var updatedMessageAction = pubnub.addMessageAction({
            channel: channelName,
            messageTimetoken: timestamp.toString(),
            action: {
                type: 'approved',
                value: status.toString()
            },
        }, function(status, response) {
            reviewIndex++;
        });
    }
}

function updateTimestamp(ts) {
    timestamp = ts;
}

// Delete all message in the channel
// Starting from time 0 to the super future
function deleteTestMessages() {
    pubnub.deleteMessages({
            channel: channelName,
            start: "0",
            end: "25526611838554309",
        },
        function(status, response) {
            console.log(status, response);
        }
    );
}


// ======================
// ======================
// ======================
// 
// Below is all test functions to be able to run from your browser dev console
// 
// ======================
// ======================
// ======================

// This is a test function
function updateMessageWithVar(timestamp, status) {

    let updatedMessage = pubnub.addMessageAction({
            channel: channelName,
            messageTimetoken: timestamp,
            action: {
                type: 'approved',
                value: status.toString()
            },
        },
        function(status, response) {
            console.log("status: ", status);
            console.log("response: ", response);
        }
    );
}

// This is a test function
function updateMessageTest() {

    let updatedMessage = pubnub.addMessageAction({
            channel: channelName,
            messageTimetoken: '16851394772671131',
            action: {
                type: 'approved',
                value: 'true'
            },
        },
        function(status, response) {
            console.log("status: ", status);
            console.log("response: ", response);
        }
    );
}


// Get all messages, but this does not allow for the return of MessageActions
// ====
async function getAllMessages(initialTimetoken = 0) {
    let latestCount = 100;
    let timetoken = initialTimetoken;
    // Whenever this is called, re-pull Pubnub dataset and re-populate our local array
    allMessages = [];

    while (latestCount === 100) {
        const {
            messages,
            startTimeToken,
            endTimeToken
        } = await pubnub.history({
            channel: channelName,
            stringifiedTimeToken: true, // false is the default
            start: timetoken, // start timetoken to fetch
            includeMessageActions: true,
        });

        latestCount = messages.length;
        timetoken = startTimeToken;

        if (messages && messages.length > 0) {
            allMessages.push(...messages);
        }
    }

    // All messages is now a global variable
    console.log("allMessages: ", allMessages);

    if (allMessages.length > 0) {
        // setting the default timestamp value for review
        timestamp = allMessages[0].timetoken;
    }

    // don't need this ... todo delete later
    // return allMessages;
}

async function getMessageActions() {
    try {
        const result = await pubnub.getMessageActions({
            channel: channelName,
            limit: 100,
        });

    } catch (status) {
        console.log(status);
    }
}

function setReviewIndex(i) {
    reviewIndex = i;
}