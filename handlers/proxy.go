package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"gemini-antiblock/config"
	"gemini-antiblock/logger"
	"gemini-antiblock/streaming"
)

// ProxyHandler handles proxy requests to Gemini API
type ProxyHandler struct {
	Config      *config.Config
	RateLimiter *RateLimiter
	HTTPClient  *http.Client
}

// NewProxyHandler creates a new proxy handler
func NewProxyHandler(cfg *config.Config, rateLimiter *RateLimiter) *ProxyHandler {
	// --- Performance Optimization: Network Tuning ---
	// The original MaxIdleConnsPerHost was too low for the concurrent load,
	// creating a connection pool bottleneck. We are increasing it to match
	// the expected concurrency level.
	transport := &http.Transport{
		MaxIdleConns:        200, // Increased global pool size
		MaxIdleConnsPerHost: 60,  // Increased per-host pool size to handle 50 concurrent users
		IdleConnTimeout:     90 * time.Second,
	}

	// Create a shared HTTP client to be reused across requests
	client := &http.Client{
		Transport: transport,
		Timeout:   600 * time.Second, // Generous timeout for streaming APIs
	}

	return &ProxyHandler{
		Config:      cfg,
		RateLimiter: rateLimiter,
		HTTPClient:  client,
	}
}

// BuildUpstreamHeaders builds headers for upstream requests
func (h *ProxyHandler) BuildUpstreamHeaders(reqHeaders http.Header) http.Header {
	headers := make(http.Header)

	// Copy specific headers
	if auth := reqHeaders.Get("Authorization"); auth != "" {
		headers.Set("Authorization", auth)
	}
	if apiKey := reqHeaders.Get("X-Goog-Api-Key"); apiKey != "" {
		headers.Set("X-Goog-Api-Key", apiKey)
	}
	if contentType := reqHeaders.Get("Content-Type"); contentType != "" {
		headers.Set("Content-Type", contentType)
	}
	if accept := reqHeaders.Get("Accept"); accept != "" {
		headers.Set("Accept", accept)
	}

	return headers
}

// InjectSystemPrompt injects a system prompt to ensure the [done] token is present.
// It intelligently handles both system_instruction (snake_case) and systemInstruction (camelCase)
// by merging the content of system_instruction into systemInstruction before processing.
// systemInstruction is the officially recommended format.
func (h *ProxyHandler) InjectSystemPrompt(body map[string]interface{}) {
	newSystemPromptPart := map[string]interface{}{
		"text": "IMPORTANT: At the very end of your entire response, you must write the token [done] to signal completion. This is a mandatory technical requirement.",
	}

	// --- From this point on, we only need to deal with systemInstruction ---

	// Case 1: systemInstruction field is missing or null. Create it.
	if val, exists := body["systemInstruction"]; !exists || val == nil {
		body["systemInstruction"] = map[string]interface{}{
			"parts": []interface{}{newSystemPromptPart},
		}
		return
	}

	instruction, ok := body["systemInstruction"].(map[string]interface{})
	if !ok {
		// The field exists but is of the wrong type. Overwrite it.
		body["systemInstruction"] = map[string]interface{}{
			"parts": []interface{}{newSystemPromptPart},
		}
		return
	}

	// Case 2: The instruction field exists, but its 'parts' array is missing, null, or not an array.
	parts, ok := instruction["parts"].([]interface{})
	if !ok {
		instruction["parts"] = []interface{}{newSystemPromptPart}
		return
	}

	// Case 3: The instruction field and its 'parts' array both exist. Append to the existing array.
	instruction["parts"] = append(parts, newSystemPromptPart)
}

// HandleStreamingPost handles streaming POST requests
func (h *ProxyHandler) HandleStreamingPost(w http.ResponseWriter, r *http.Request) {
	urlObj, _ := url.Parse(r.URL.String())
	upstreamURL := h.Config.UpstreamURLBase + urlObj.Path
	if urlObj.RawQuery != "" {
		upstreamURL += "?" + urlObj.RawQuery
	}

	logger.LogInfo("=== NEW STREAMING REQUEST ===")
	logger.LogInfo("Upstream URL:", upstreamURL)
	logger.LogInfo("Request method:", r.Method)
	logger.LogInfo("Content-Type:", r.Header.Get("Content-Type"))

	// --- Bug Fix: Pre-emptive Injection for Stateful Retry ---
	injector, requestBodyForRetry, err := NewSystemPromptInjector(r.Body)
	if err != nil {
		logger.LogError("Failed to create system prompt injector:", err)
		JSONError(w, 500, "Internal server error", "Failed to process request body")
		return
	}

	logger.LogInfo("=== MAKING INITIAL REQUEST (WITH PRE-EMPTIVE INJECTION) ===")
	upstreamHeaders := h.BuildUpstreamHeaders(r.Header)

	upstreamReq, err := http.NewRequest("POST", upstreamURL, injector)
	if err != nil {
		logger.LogError("Failed to create upstream request:", err)
		JSONError(w, 500, "Internal server error", "Failed to create upstream request")
		return
	}

	upstreamReq.Header = upstreamHeaders

	initialResponse, err := h.HTTPClient.Do(upstreamReq)
	if err != nil {
		logger.LogError("Failed to make initial request:", err)
		JSONError(w, 502, "Bad Gateway", "Failed to connect to upstream server")
		return
	}

	logger.LogInfo(fmt.Sprintf("Initial response status: %d %s", initialResponse.StatusCode, initialResponse.Status))

	// Initial failure: return standardized error
	if initialResponse.StatusCode != http.StatusOK {
		logger.LogError("=== INITIAL REQUEST FAILED ===")
		logger.LogError("Status:", initialResponse.StatusCode)
		logger.LogError("Status Text:", initialResponse.Status)

		// Read error response
		errorBody, _ := io.ReadAll(initialResponse.Body)
		initialResponse.Body.Close()

		// Try to parse as JSON error
		var errorResp map[string]interface{}
		if json.Unmarshal(errorBody, &errorResp) == nil {
			if errorObj, ok := errorResp["error"].(map[string]interface{}); ok {
				if _, hasStatus := errorObj["status"]; !hasStatus {
					if code, ok := errorObj["code"].(float64); ok {
						errorObj["status"] = StatusToGoogleStatus(int(code))
					}
				}
			}
			w.Header().Set("Content-Type", "application/json; charset=utf-8")
			w.Header().Set("Access-Control-Allow-Origin", "*")
			w.WriteHeader(initialResponse.StatusCode)
			json.NewEncoder(w).Encode(errorResp)
			return
		}

		// Fallback to standard error
		message := "Request failed"
		if initialResponse.StatusCode == 429 {
			message = "Resource has been exhausted (e.g. check quota)."
		}
		JSONError(w, initialResponse.StatusCode, message, string(errorBody))
		return
	}

	logger.LogInfo("=== INITIAL REQUEST SUCCESSFUL - STARTING STREAM PROCESSING ===")

	// Set up streaming response
	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	// Additional headers to prevent buffering by proxies
	w.Header().Set("X-Accel-Buffering", "no") // Nginx
	w.Header().Set("Pragma", "no-cache")
	w.Header().Set("Expires", "0")

	w.WriteHeader(http.StatusOK)

	// Process stream with retry logic using a new session for each request
	safeWriter := NewSafeWriter(w)
	session := streaming.NewSession(
		h.Config,
		initialResponse.Body,
		safeWriter,
		requestBodyForRetry,
		upstreamURL,
		r.Header,
		h.HTTPClient,
	)
	err = session.Process()

	if err != nil {
		logger.LogError("=== UNHANDLED EXCEPTION IN STREAM PROCESSOR ===")
		logger.LogError("Exception:", err)
	}

	initialResponse.Body.Close()
	logger.LogInfo("Streaming response completed")
}

// HandleNonStreaming handles non-streaming requests
func (h *ProxyHandler) HandleNonStreaming(w http.ResponseWriter, r *http.Request) {
	urlObj, _ := url.Parse(r.URL.String())
	upstreamURL := h.Config.UpstreamURLBase + urlObj.Path
	if urlObj.RawQuery != "" {
		upstreamURL += "?" + urlObj.RawQuery
	}

	upstreamHeaders := h.BuildUpstreamHeaders(r.Header)

	var body io.Reader
	if r.Method != "GET" && r.Method != "HEAD" {
		body = r.Body
	}

	upstreamReq, err := http.NewRequest(r.Method, upstreamURL, body)
	if err != nil {
		JSONError(w, 500, "Internal server error", "Failed to create upstream request")
		return
	}

	upstreamReq.Header = upstreamHeaders

	resp, err := h.HTTPClient.Do(upstreamReq)
	if err != nil {
		JSONError(w, 502, "Bad Gateway", "Failed to connect to upstream server")
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		// Handle error response
		errorBody, _ := io.ReadAll(resp.Body)

		var errorResp map[string]interface{}
		if json.Unmarshal(errorBody, &errorResp) == nil {
			if errorObj, ok := errorResp["error"].(map[string]interface{}); ok {
				if _, hasStatus := errorObj["status"]; !hasStatus {
					if code, ok := errorObj["code"].(float64); ok {
						errorObj["status"] = StatusToGoogleStatus(int(code))
					}
				}
			}
			w.Header().Set("Content-Type", "application/json; charset=utf-8")
			w.Header().Set("Access-Control-Allow-Origin", "*")
			w.WriteHeader(resp.StatusCode)
			json.NewEncoder(w).Encode(errorResp)
			return
		}

		JSONError(w, resp.StatusCode, resp.Status, string(errorBody))
		return
	}

	// Copy response headers
	for name, values := range resp.Header {
		for _, value := range values {
			w.Header().Add(name, value)
		}
	}
	w.Header().Set("Access-Control-Allow-Origin", "*")

	w.WriteHeader(resp.StatusCode)
	io.Copy(w, resp.Body)
}

// ServeHTTP implements the http.Handler interface
func (h *ProxyHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// First, enforce rate limiting if enabled and a key is present.
	if h.Config.EnableRateLimit {
		apiKey := r.Header.Get("X-Goog-Api-Key")
		if apiKey == "" {
			authHeader := r.Header.Get("Authorization")
			if strings.HasPrefix(authHeader, "Bearer ") {
				apiKey = strings.TrimPrefix(authHeader, "Bearer ")
			}
		}

		if apiKey != "" {
			logger.LogDebug("Enforcing rate limit for key ending with: ...", apiKey[len(apiKey)-4:])
			h.RateLimiter.Wait(apiKey)
			logger.LogDebug("Rate limit check passed for key.")
		}
	}

	logger.LogInfo("=== WORKER REQUEST ===")
	logger.LogInfo("Method:", r.Method)
	logger.LogInfo("URL:", r.URL.String())
	logger.LogInfo("User-Agent:", r.Header.Get("User-Agent"))
	logger.LogInfo("X-Forwarded-For:", r.Header.Get("X-Forwarded-For"))

	if r.Method == "OPTIONS" {
		logger.LogDebug("Handling CORS preflight request")
		HandleCORS(w, r)
		return
	}

	// Determine if this is a streaming request
	isStream := strings.Contains(strings.ToLower(r.URL.Path), "stream") ||
		strings.Contains(strings.ToLower(r.URL.Path), "sse") ||
		r.URL.Query().Get("alt") == "sse"

	logger.LogInfo("Detected streaming request:", isStream)

	if r.Method == "POST" && isStream {
		h.HandleStreamingPost(w, r)
		return
	}

	h.HandleNonStreaming(w, r)
}
