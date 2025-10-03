// const video = document.getElementById('video');
// const annotated = document.getElementById('annotated');
// const confSlider = document.getElementById('conf');
// const startBtn = document.getElementById('startBtn');
// const stopBtn = document.getElementById('stopBtn');
// const sendFpsEl = document.getElementById('sendFps');
// const serverFpsEl = document.getElementById('serverFps');

// const socket = io(); // connects to same host

// let streaming = false;
// let captureInterval = null;
// let lastSent = 0;
// let sends = 0;

// async function startCamera() {
//   const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
//   video.srcObject = stream;
//   await video.play();
// }

// function captureAndSend() {
//   if (!streaming) return;
//   const canvas = document.createElement('canvas');
//   canvas.width = video.videoWidth || 640;
//   canvas.height = video.videoHeight || 480;
//   const ctx = canvas.getContext('2d');
//   ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
//   const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
//   const conf = confSlider.value;
//   socket.emit('input_frame', { img: dataUrl, conf: conf });
//   sends++;
//   const now = performance.now();
//   if (now - lastSent >= 1000) {
//     sendFpsEl.innerText = sends;
//     sends = 0;
//     lastSent = now;
//   }
// }

// // receive annotated frames
// socket.on('output_frame', (data) => {
//   if (data.img) {
//     annotated.src = data.img;
//   }
//   if (data.fps) serverFpsEl.innerText = data.fps;
// });

// startBtn.addEventListener('click', async () => {
//   if (!video.srcObject) await startCamera();
//   if (streaming) return;
//   streaming = true;
//   // capture at ~8-12 fps (adjust interval for latency)
//   captureInterval = setInterval(captureAndSend, 100); // 10 fps
// });

// stopBtn.addEventListener('click', () => {
//   streaming = false;
//   if (captureInterval) { clearInterval(captureInterval); captureInterval = null; }
// });

// // auto connect
// startCamera().catch(e=>console.error(e));