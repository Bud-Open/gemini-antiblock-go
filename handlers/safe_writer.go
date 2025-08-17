package handlers

import (
	"net/http"
	"sync"
)

// SafeWriter is a thread-safe wrapper around http.ResponseWriter.
type SafeWriter struct {
	w  http.ResponseWriter
	mu sync.Mutex
}

// NewSafeWriter creates a new SafeWriter.
func NewSafeWriter(w http.ResponseWriter) *SafeWriter {
	return &SafeWriter{w: w}
}

// Write writes data to the response, ensuring thread safety.
func (sw *SafeWriter) Write(p []byte) (int, error) {
	sw.mu.Lock()
	defer sw.mu.Unlock()
	return sw.w.Write(p)
}

// Flush flushes the underlying writer, if it's an http.Flusher.
func (sw *SafeWriter) Flush() {
	sw.mu.Lock()
	defer sw.mu.Unlock()
	if flusher, ok := sw.w.(http.Flusher); ok {
		flusher.Flush()
	}
}
