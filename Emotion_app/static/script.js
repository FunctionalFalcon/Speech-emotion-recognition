// Audio context and elements
let audioContext;
let analyser;
let mediaRecorder;
let audioChunks = [];
let recordingInterval;
let seconds = 0;
const timerElement = document.getElementById('timer');
const recordButton = document.getElementById('recordButton');
const visualizerCanvas = document.getElementById('visualizer');
const emotionDisplay = document.getElementById('emotionDisplay');
const resultContainer = document.getElementById('resultContainer');
const confidenceText = document.getElementById('confidenceText');
const ctx = visualizerCanvas.getContext('2d');

// Initialize audio context on user interaction
function initAudio() {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
}

// Start recording
async function startRecording() {
    try {
        if (!audioContext) initAudio();
        
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);
        
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.start();
        
        // UI updates
        recordButton.classList.add('recording');
        recordButton.innerHTML = '<i class="fas fa-stop"></i>';
        resultContainer.style.display = 'none';
        
        // Start timer
        seconds = 0;
        updateTimer();
        recordingInterval = setInterval(updateTimer, 1000);
        
        // Start visualization
        visualize();
        
        // Collect audio data
        audioChunks = [];
        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = () => {
            clearInterval(recordingInterval);
            sendAudioToServer();
        };
    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Could not access microphone. Please ensure permissions are granted.');
    }
}

// Stop recording
function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        recordButton.classList.remove('recording');
        recordButton.innerHTML = '<i class="fas fa-microphone"></i>';
        
        // Stop all tracks
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
    }
}

// Update timer display
function updateTimer() {
    seconds++;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    timerElement.textContent = `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
}

// Audio visualization
function visualize() {
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    function draw() {
        if (!analyser) return;
        
        const WIDTH = visualizerCanvas.width = visualizerCanvas.offsetWidth;
        const HEIGHT = visualizerCanvas.height = visualizerCanvas.offsetHeight;
        
        analyser.getByteTimeDomainData(dataArray);
        
        ctx.fillStyle = 'rgb(241, 243, 245)';
        ctx.fillRect(0, 0, WIDTH, HEIGHT);
        
        ctx.lineWidth = 2;
        ctx.strokeStyle = 'rgb(67, 97, 238)';
        ctx.beginPath();
        
        const sliceWidth = WIDTH / bufferLength;
        let x = 0;
        
        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = v * HEIGHT / 2;
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
            
            x += sliceWidth;
        }
        
        ctx.lineTo(WIDTH, HEIGHT / 2);
        ctx.stroke();
        
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            requestAnimationFrame(draw);
        }
    }
    
    draw();
}

// Send audio to server
function sendAudioToServer() {
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');
    
    // Show loading state
    emotionDisplay.textContent = 'Analyzing...';
    confidenceText.textContent = '';
    resultContainer.style.display = 'block';
    
    fetch('/predict_emotion', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Update UI with results
        emotionDisplay.textContent = data.emotion;
        confidenceText.textContent = data.confidence ? `Confidence: ${Math.round(data.confidence * 100)}%` : '';
        
        // Color code based on emotion
        const colors = {
            'Happy': '#4cc9f0',
            'Sad': '#3a0ca3',
            'Angry': '#f72585',
            'Neutral': '#4361ee',
            'Fear': '#7209b7',
            'Surprise': '#4895ef'
        };
        
        if (colors[data.emotion]) {
            emotionDisplay.style.color = colors[data.emotion];
        }
    })
    .catch(error => {
        console.error('Error:', error);
        emotionDisplay.textContent = 'Error';
        confidenceText.textContent = 'Could not analyze audio';
    });
}

// Event listeners
recordButton.addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        stopRecording();
    } else {
        startRecording();
    }
});