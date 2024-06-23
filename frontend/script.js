const video = document.getElementById('user-camera');
const recordButton = document.getElementById('record-button');
const recordingIndicator = document.querySelector('.recording-indicator');
const translationText = document.getElementById('translation-text');
const clearButton = document.getElementById('clear-button');

let mediaRecorder;
let recordedChunks = [];

// Request access to the camera and display the video stream
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;

    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = event => {
      if (event.data.size > 0) {
        recordedChunks.push(event.data);
      }
    };
    mediaRecorder.onstop = async () => {
      const blob = new Blob(recordedChunks, { type: 'video/mp4' });
      recordedChunks = [];
      recordingIndicator.hidden = true;
      recordButton.disabled = false;
      clearButton.disabled = false;

      const formData = new FormData();
      formData.append('file', blob, 'recorded_sign.mp4');

      // Send the video to the backend for processing
      const response = await fetch('http://localhost:8000/predict/sign', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        translationText.textContent = result.predicted_signs.join(', ');
      } else {
        translationText.textContent = 'Error in translation. Please try again.';
      }
    };
  })
  .catch(error => {
    console.error('Error accessing camera:', error);
  });

// Start and stop recording when the button is clicked
recordButton.addEventListener('click', () => {
  if (mediaRecorder.state === 'inactive') {
    mediaRecorder.start();
    recordingIndicator.hidden = false;
    recordButton.textContent = 'Stop Recording';
  } else {
    mediaRecorder.stop();
    recordButton.textContent = 'Record Sign';
  }
});

// Clear the translation text when the clear button is clicked
clearButton.addEventListener('click', () => {
  translationText.textContent = '';
  clearButton.disabled = true;
});
