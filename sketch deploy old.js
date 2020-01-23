let video;
let poseNet;
let poses = [];

let skeleton;
let brain;
let state = 'collecting';
let targetLabel;
let pose;

let bgMusic;
let inihahaMusic;
let prevLabel;

function preload() {
    bgMusic = loadSound('bgMusic.mp3');
    inihahaMusic = loadSound('inihaha.mp3');

}



function getPoses(resposes) {
    poses = resposes;
    console.log(poses);
    if (poses.length > 0) {
        if (state == 'collecting') {
            pose = poses[0].pose;
            // console.log(pose.keypoints[0]);
            // let inputs = [];
            // for (let i = 0; i < pose.keypoints.length; i++) {
            //     let keypoint = pose.keypoints[i];
            //     // console.log(keypoint);


            //     inputs.push(keypoint.position.x);
            //     inputs.push(keypoint.position.y);

            // }
            // let target = [targetLabel];
            // brain.addData(inputs, target);
        }
    }
}


function setup() {
    bgMusic.setVolume(0.5);
setTimeout(()=>{
    bgMusic.loop();

},100);
    createCanvas(640, 480);
    video = createCapture(VIDEO);
    video.size(width, height);
    video.hide();

    poseNet = ml5.poseNet(video, modelReady);
    poseNet.on('pose', getPoses);

    let options = {
        inputs: 34,
        outputs: 4,
        task: 'classification',
        debug: true
    }

    const modelInfo = {
        model: 'models/iniha/model.json',
        metadata: 'models/iniha/model_meta.json',
        weights: 'models/iniha/model.weights.bin'
    }
    brain = ml5.neuralNetwork(options);
    brain.load(modelInfo, brainLoaded);
    // brain.loadData('iniha_data.json', dataReady);
}

function brainLoaded() {
    console.log('Ini7a ready!!');
    iniha();
}

function iniha() {
    if (pose) {
        let inputs = [];
        for (let i = 0; i < pose.keypoints.length; i++) {
            let keypoint = pose.keypoints[i];
            // console.log(keypoint);


            inputs.push(keypoint.position.x);
            inputs.push(keypoint.position.y);

        }
        brain.classify(inputs, gotResult);

    }
    else {
        setTimeout(iniha, 100);
    }
}

function gotResult(error, results) {
    if (results[0].confidence > 0.75) {
        targetLabel = results[0].label;

    }
    if (targetLabel!=prevLabel) {
        console.log(targetLabel);
        prevLabel = targetLabel;
        if(targetLabel == 'i'){
            inihahaMusic.setVolume(0.5);
            inihahaMusic.play();
        }
        
        
    }
    iniha();

}





function modelReady() {
    select('#status').html('Model Loaded');
}

function draw() {
    image(video, 0, 0, width, height);

    // We can call both functions to draw all keypoints and the skeletons
    // drawKeypoints();
    // drawSkeleton();
}

// A function to draw ellipses over the detected keypoints
function drawKeypoints() {
    // Loop through all the poses detected
    for (let i = 0; i < poses.length; i++) {
        // For each pose detected, loop through all the keypoints
        let pose = poses[i].pose;
        for (let j = 0; j < pose.keypoints.length; j++) {
            // A keypoint is an object describing a body part (like rightArm or leftShoulder)
            let keypoint = pose.keypoints[j];
            // Only draw an ellipse is the pose probability is bigger than 0.2
            if (keypoint.score > 0.2) {
                fill(255, 0, 0);
                noStroke();
                ellipse(keypoint.position.x, keypoint.position.y, 10, 10);
            }
        }
    }
}

// A function to draw the skeletons
function drawSkeleton() {
    // Loop through all the skeletons detected
    for (let i = 0; i < poses.length; i++) {
        let skeleton = poses[i].skeleton;
        // For every skeleton, loop through all body connections
        for (let j = 0; j < skeleton.length; j++) {
            let partA = skeleton[j][0];
            let partB = skeleton[j][1];
            stroke(255, 0, 0);
            line(partA.position.x, partA.position.y, partB.position.x, partB.position.y);
        }
    }
}
