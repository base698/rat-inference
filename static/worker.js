// Worker.js - Web Worker for camera streaming
let canvas = null;
let ctx = null;
let streamInterval = null;

self.addEventListener('message', async function(e) {
    if (e.data.type === 'init') {
        // Receive the OffscreenCanvas
        canvas = e.data.canvas;
        ctx = canvas.getContext('2d');

        // Start streaming loop
        if (!streamInterval) {
            streamLoop();
            streamInterval = setInterval(streamLoop, 90);  // ~15 FPS
        }
    }
});

let fetching = false;
async function streamLoop() {
    if (!canvas || !ctx || fetching) return;

    try {
        fetching = true;
        // Fetch frame data from server
        const response = await fetch('/stream-frame');
        if (!response.ok) return;

        const data = await response.json();

        if (data.image) {
            // Convert base64 to image and draw on canvas
            const img = await createImageBitmap(await (await fetch(`data:image/jpeg;base64,${data.image}`)).blob());

            // Draw image on offscreen canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        }

        // Send detection data back to main thread (without image)
        self.postMessage({
            type: 'frame_data',
            detection: data.detection,
            confidence: data.confidence,
            recent_detections: data.recent_detections
        });

    } catch (error) {
        console.error('Stream loop error:', error);
    }
    fetching = false;
}

self.addEventListener('error', function(e) {
    console.error('Worker error:', e);
});
