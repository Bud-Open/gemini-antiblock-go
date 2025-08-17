package handlers

import (
	"bytes"
	"encoding/json"
	"io"

	"gemini-antiblock/logger"
)

// SystemPromptInjector is a custom reader that injects a system prompt
// into a JSON request body on-the-fly while streaming.
type SystemPromptInjector struct {
	originalReader io.ReadCloser
	processedBody  io.Reader
	fullBody       *bytes.Buffer
}

// NewSystemPromptInjector creates a new injector. It reads the original
// request to memory, injects the prompt, and then creates a new reader
// from the modified body.
func NewSystemPromptInjector(reader io.ReadCloser) (*SystemPromptInjector, map[string]interface{}, error) {
	bodyBytes, err := io.ReadAll(reader)
	if err != nil {
		return nil, nil, err
	}
	reader.Close()

	var requestBody map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &requestBody); err != nil {
		logger.LogError("Failed to parse original body for injection:", err)
		// If parsing fails, we pass through the original content
		return &SystemPromptInjector{
			processedBody: bytes.NewReader(bodyBytes),
			fullBody:      bytes.NewBuffer(bodyBytes),
		}, make(map[string]interface{}), nil
	}

	// Create a dummy handler to reuse the InjectSystemPrompt logic
	dummyHandler := &ProxyHandler{}
	dummyHandler.InjectSystemPrompt(requestBody)

	modifiedBodyBytes, err := json.Marshal(requestBody)
	if err != nil {
		return nil, nil, err
	}

	logger.LogDebug("System prompt injected successfully for initial request.")

	return &SystemPromptInjector{
		processedBody: bytes.NewReader(modifiedBodyBytes),
		fullBody:      bytes.NewBuffer(modifiedBodyBytes),
	}, requestBody, nil
}

func (i *SystemPromptInjector) Read(p []byte) (n int, err error) {
	return i.processedBody.Read(p)
}

func (i *SystemPromptInjector) Close() error {
	// The original reader is already closed in the constructor.
	return nil
}

// GetFullBodyReader returns a new reader for the entire processed body.
// This is useful for retries.
func (i *SystemPromptInjector) GetFullBodyReader() io.Reader {
	return bytes.NewReader(i.fullBody.Bytes())
}
