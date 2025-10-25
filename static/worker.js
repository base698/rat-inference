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

        // Fetch image bytes from stream-frame
        const imageResponse = await fetch('/stream-frame',{
           headers:{
             'Content-Type':'image/jpeg'
           }
        });
        if (!imageResponse.ok) return;

        const imageBlob = await imageResponse.blob();

        // Only process if we have image data
        if (imageBlob && imageBlob.size > 0) {
            // Create ImageBitmap directly from blob
            const img = await createImageBitmap(imageBlob);

            // Draw image on offscreen canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        }

        // Fetch detection data from status endpoint
        const statusResponse = await fetch('/status');
        if (statusResponse.ok) {
            const statusData = await statusResponse.json();

            // Send detection data back to main thread (without image)
            self.postMessage({
                type: 'frame_data',
                detection: statusData.detection || false,
                confidence: statusData.confidence || 0,
                recent_detections: statusData.recent_detections || []
            });
        }

    } catch (error) {
        console.error('Stream loop error:', error);
    }
    fetching = false;
}

self.addEventListener('error', function(e) {
    console.error('Worker error:', e);
});
