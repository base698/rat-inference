// Worker.js - Web Worker for background processing

self.addEventListener('message', function(e) {
    console.log('Worker received message:', e.data);

    // Process the message
    const result = {
        status: 'success',
        data: e.data
    };

    // Send result back to main thread
    self.postMessage(result);
});

self.addEventListener('error', function(e) {
    console.error('Worker error:', e);
});
