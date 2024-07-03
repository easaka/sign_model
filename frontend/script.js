const video = document.getElementById('user-camera');
const recordButton = document.getElementById('record-button');
const recordingIndicator = document.querySelector('.recording-indicator');
const translationText = document.getElementById('translation-text');
const clearButton = document.getElementById('clear-button');
let isRecording = false;
let ws;

// Request access to the camera and display the video stream
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
  })
  .catch(error => {
    console.error('Error accessing camera:', error);
  });

// Start and stop recording when the button is clicked
recordButton.addEventListener('click', () => {
  if (!isRecording) {
    startRecording();
  } else {
    stopRecording();
  }
});

function startRecording() {
  ws = new WebSocket('http://localhost:8000/ws/predict/pose');

  ws.onopen = () => {
    isRecording = true;
    recordingIndicator.hidden = false;
    recordButton.textContent = 'Stop Recording';
    sendFrames();
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.predicted_signs.length > 0) {
      translationText.textContent = `Predicted Sign: ${data.predicted_signs[0]}`;
    } else {
      translationText.textContent = 'No sign detected';
    }
  };

  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
  };

  ws.onclose = () => {
    isRecording = false;
    recordingIndicator.hidden = true;
    recordButton.textContent = 'Start Recording';
  };
}

function stopRecording() {
  if (ws) {
    ws.close();
  }
}

function sendFrames() {
  if (!isRecording) return;

  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const context = canvas.getContext('2d');
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(blob => {
    const reader = new FileReader();
    reader.onload = () => {
      const arrayBuffer = reader.result;
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(arrayBuffer);
      }
      // Schedule the next frame
      setTimeout(sendFrames, 1000); // Adjust the interval as needed
    };
    reader.readAsArrayBuffer(blob);
  }, 'image/jpeg');
}

// Clear the translation text when the clear button is clicked
clearButton.addEventListener('click', () => {
  translationText.textContent = '';
  clearButton.disabled = true;
});

// Enable the clear button when there's translation text
const observer = new MutationObserver(() => {
  clearButton.disabled = translationText.textContent === '';
});

observer.observe(translationText, { childList: true });
