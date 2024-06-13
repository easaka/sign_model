const video = document.getElementById('user-camera');
const recordButton = document.getElementById('record-button');
const translationText = document.getElementById('translation-text');
const clearButton = document.getElementById('clear-button');
const recordingIndicator = document.querySelector('.recording-indicator');

let recorder; // MediaRecorder object
let isRecording = false;
let translation = '';

// Function to handle success from getUserMedia
function handleSuccess(stream) {
  video.srcObject = stream; // Set video source to the user's camera
  video.play(); // Start playing the video stream
}

// Function to handle errors from getUserMedia
function handleError(error) {
  console.error('Error accessing camera:', error);
}



// Function to start recording the video stream
function startRecording() {
  if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true }) // Request video stream
      .then(handleSuccess)
      .catch(handleError);

    // Start recording only if camera access is successful
    if (video.srcObject) {
      recorder = new MediaRecorder(video.srcObject);
      recorder.ondataavailable = (event) => {
        const blob = new Blob([event.data], { type: 'video/webm' });
        // Send the recorded video blob to a server for sign language recognition (implementation not included here)

        // Replace with your actual sign language recognition logic using the blob
        translation = 'Performing translation...'; // Placeholder for translation result
        translationText.textContent = translation;

        // Stop recording and clear translation after a short delay (simulating processing time)
        setTimeout(() => {
          recorder.stop();
          isRecording = false;
          recordButton.textContent = 'Record Sign';
          recordingIndicator.hidden = true;
          clearButton.disabled = false;
          clearButton.style.opacity = 1; // Enable hover effect
        }, 3000); // Simulate 3 seconds of processing time
      };
      recorder.start();
      isRecording = true;
      recordButton.textContent = 'Stop Recording';
      recordingIndicator.hidden = false;
      clearButton.disabled = true;
      clearButton.style.opacity = 0.5; // Disable hover effect
    }
  } else {
    console.error('getUserMedia not supported');
  }
}

// Function to clear the translation text
function clearTranslation() {
  translation = '';
  translationText.textContent = '';
  clearButton.disabled = true;
  clearButton.style.opacity = 0.5; // Disable hover effect
}

// Add event listeners for button clicks
recordButton.addEventListener('click', startRecording);
clearButton.addEventListener('click', clearTranslation);
