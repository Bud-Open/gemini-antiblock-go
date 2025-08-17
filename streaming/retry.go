package streaming

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"gemini-antiblock/config"
	"gemini-antiblock/logger"
)

var nonRetryableStatuses = map[int]bool{
	400: true, 401: true, 403: true, 404: true, 429: true,
}

// BuildRetryRequestBody builds a new request body for retry with accumulated context
func BuildRetryRequestBody(originalBody map[string]interface{}, accumulatedText string) map[string]interface{} {
	logger.LogDebug(fmt.Sprintf("Building retry request body. Accumulated text length: %d", len(accumulatedText)))
	logger.LogDebug(fmt.Sprintf("Accumulated text preview: %s", func() string {
		if len(accumulatedText) > 200 {
			return accumulatedText[:200] + "..."
		}
		return accumulatedText
	}()))

	retryBody := make(map[string]interface{})
	for k, v := range originalBody {
		retryBody[k] = v
	}

	contents, ok := retryBody["contents"].([]interface{})
	if !ok {
		contents = []interface{}{}
	}

	// Find last user message index
	lastUserIndex := -1
	for i := len(contents) - 1; i >= 0; i-- {
		if content, ok := contents[i].(map[string]interface{}); ok {
			if role, ok := content["role"].(string); ok && role == "user" {
				lastUserIndex = i
				break
			}
		}
	}

	// Build retry context
	history := []interface{}{
		map[string]interface{}{
			"role": "model",
			"parts": []interface{}{
				map[string]interface{}{"text": accumulatedText},
			},
		},
		map[string]interface{}{
			"role": "user",
			"parts": []interface{}{
				map[string]interface{}{"text": "Continue exactly where you left off without any preamble or repetition."},
			},
		},
	}

	// Insert history after last user message
	if lastUserIndex != -1 {
		newContents := make([]interface{}, 0, len(contents)+2)
		newContents = append(newContents, contents[:lastUserIndex+1]...)
		newContents = append(newContents, history...)
		newContents = append(newContents, contents[lastUserIndex+1:]...)
		retryBody["contents"] = newContents
		logger.LogDebug(fmt.Sprintf("Inserted retry context after user message at index %d", lastUserIndex))
	} else {
		newContents := append(contents, history...)
		retryBody["contents"] = newContents
		logger.LogDebug("Appended retry context to end of conversation")
	}

	logger.LogDebug(fmt.Sprintf("Final retry request has %d messages", len(retryBody["contents"].([]interface{}))))
	return retryBody
}

// Session encapsulates the state for a single streaming request.
type Session struct {
	cfg                    *config.Config
	initialReader          io.Reader
	writer                 io.Writer
	originalRequestBody    map[string]interface{}
	upstreamURL            string
	originalHeaders        http.Header
	client                 *http.Client
	accumulatedText        string
	consecutiveRetryCount  int
	totalLinesProcessed    int
	sessionStartTime       time.Time
	isOutputtingFormalText bool
	swallowModeActive      bool
}

// NewSession creates a new streaming session.
func NewSession(cfg *config.Config, initialReader io.Reader, writer io.Writer, originalRequestBody map[string]interface{}, upstreamURL string, originalHeaders http.Header, client *http.Client) *Session {
	return &Session{
		cfg:                 cfg,
		initialReader:       initialReader,
		writer:              writer,
		originalRequestBody: originalRequestBody,
		upstreamURL:         upstreamURL,
		originalHeaders:     originalHeaders,
		client:              client,
		sessionStartTime:    time.Now(),
	}
}

// Process handles the entire lifecycle of a streaming request, including retries.
func (s *Session) Process() error {
	currentReader := s.initialReader
	logger.LogInfo(fmt.Sprintf("Starting stream processing session. Max retries: %d", s.cfg.MaxConsecutiveRetries))

	for {
		interruptionReason := ""
		cleanExit := false
		streamStartTime := time.Now()
		linesInThisStream := 0
		textInThisStream := ""

		logger.LogDebug(fmt.Sprintf("=== Starting stream attempt %d/%d ===", s.consecutiveRetryCount+1, s.cfg.MaxConsecutiveRetries+1))

		lineCh := make(chan string, 100)
		go SSELineIterator(currentReader, lineCh)

		for line := range lineCh {
			s.totalLinesProcessed++
			linesInThisStream++

			var textChunk string
			var isThought bool

			if IsDataLine(line) {
				content := ParseLineContent(line)
				textChunk = content.Text
				isThought = content.IsThought
			}

			if s.swallowModeActive {
				if isThought {
					logger.LogDebug("Swallowing thought chunk due to post-retry filter:", line)
					finishReason := ExtractFinishReason(line)
					if finishReason != "" {
						logger.LogError(fmt.Sprintf("Stream stopped with reason '%s' while swallowing a 'thought' chunk. Triggering retry.", finishReason))
						interruptionReason = "FINISH_DURING_THOUGHT"
						break
					}
					continue
				} else {
					logger.LogInfo("First formal text chunk received after swallowing. Resuming normal stream.")
					s.swallowModeActive = false
				}
			}

			finishReason := ExtractFinishReason(line)
			needsRetry := false

			if finishReason != "" && isThought {
				logger.LogError(fmt.Sprintf("Stream stopped with reason '%s' on a 'thought' chunk. This is an invalid state. Triggering retry.", finishReason))
				interruptionReason = "FINISH_DURING_THOUGHT"
				needsRetry = true
			} else if IsBlockedLine(line) {
				logger.LogError(fmt.Sprintf("Content blocked detected in line: %s", line))
				interruptionReason = "BLOCK"
				needsRetry = true
			} else if finishReason == "STOP" {
				tempAccumulatedText := s.accumulatedText + textChunk
				trimmedText := strings.TrimSpace(tempAccumulatedText)
				if len(trimmedText) == 0 {
					logger.LogError("Finish reason 'STOP' with no text content detected. This indicates an empty response. Triggering retry.")
					interruptionReason = "FINISH_EMPTY_RESPONSE"
					needsRetry = true
				}
			} else if finishReason != "" && finishReason != "MAX_TOKENS" && finishReason != "STOP" {
				logger.LogError(fmt.Sprintf("Abnormal finish reason: %s. Triggering retry.", finishReason))
				interruptionReason = "FINISH_ABNORMAL"
				needsRetry = true
			}

			if needsRetry {
				break
			}

			isEndOfResponse := finishReason == "STOP" || finishReason == "MAX_TOKENS"
			processedLine := RemoveDoneTokenFromLine(line, isEndOfResponse)

			if _, err := s.writer.Write([]byte(processedLine + "\n\n")); err != nil {
				return fmt.Errorf("failed to write to output stream: %w", err)
			}

			if flusher, ok := s.writer.(http.Flusher); ok {
				flusher.Flush()
			}

			if textChunk != "" && !isThought {
				s.isOutputtingFormalText = true
				s.accumulatedText += textChunk
				textInThisStream += textChunk
			}

			if finishReason == "STOP" || finishReason == "MAX_TOKENS" {
				doneLine := "data: {\"candidates\": [{\"content\": {\"parts\": [{\"text\": \"[done]\"}]}}]}"
				if _, err := s.writer.Write([]byte(doneLine + "\n\n")); err != nil {
					return fmt.Errorf("failed to write [done] token: %w", err)
				}
				if flusher, ok := s.writer.(http.Flusher); ok {
					flusher.Flush()
				}
				logger.LogInfo(fmt.Sprintf("Finish reason '%s' accepted as final. Manually injected [done] token. Stream complete.", finishReason))
				cleanExit = true
				break
			}
		}

		if !cleanExit && interruptionReason == "" {
			logger.LogError("Stream ended without finish reason - detected as DROP")
			interruptionReason = "DROP"
		}

		streamDuration := time.Since(streamStartTime)
		logger.LogDebug("Stream attempt summary:")
		logger.LogDebug(fmt.Sprintf("  Duration: %v", streamDuration))
		logger.LogDebug(fmt.Sprintf("  Lines processed: %d", linesInThisStream))
		logger.LogDebug(fmt.Sprintf("  Text generated this stream: %d chars", len(textInThisStream)))
		logger.LogDebug(fmt.Sprintf("  Total accumulated text: %d chars", len(s.accumulatedText)))

		if cleanExit {
			sessionDuration := time.Since(s.sessionStartTime)
			logger.LogInfo("=== STREAM COMPLETED SUCCESSFULLY ===")
			logger.LogInfo(fmt.Sprintf("Total session duration: %v", sessionDuration))
			logger.LogInfo(fmt.Sprintf("Total lines processed: %d", s.totalLinesProcessed))
			logger.LogInfo(fmt.Sprintf("Total text generated: %d characters", len(s.accumulatedText)))
			logger.LogInfo(fmt.Sprintf("Total retries needed: %d", s.consecutiveRetryCount))
			return nil
		}

		logger.LogError("=== STREAM INTERRUPTED ===")
		logger.LogError(fmt.Sprintf("Reason: %s", interruptionReason))

		if s.cfg.SwallowThoughtsAfterRetry && s.isOutputtingFormalText {
			logger.LogInfo("Retry triggered after formal text output. Will swallow subsequent thought chunks until formal text resumes.")
			s.swallowModeActive = true
		}

		if s.consecutiveRetryCount >= s.cfg.MaxConsecutiveRetries {
			errorPayload := map[string]interface{}{
				"error": map[string]interface{}{
					"code":    504,
					"status":  "DEADLINE_EXCEEDED",
					"message": fmt.Sprintf("Retry limit (%d) exceeded after stream interruption. Last reason: %s.", s.cfg.MaxConsecutiveRetries, interruptionReason),
					"details": []interface{}{
						map[string]interface{}{
							"@type":                  "proxy.debug",
							"accumulated_text_chars": len(s.accumulatedText),
						},
					},
				},
			}
			errorBytes, _ := json.Marshal(errorPayload)
			s.writer.Write([]byte(fmt.Sprintf("event: error\ndata: %s\n\n", string(errorBytes))))
			if flusher, ok := s.writer.(http.Flusher); ok {
				flusher.Flush()
			}
			return fmt.Errorf("retry limit exceeded")
		}

		s.consecutiveRetryCount++
		logger.LogInfo(fmt.Sprintf("=== STARTING RETRY %d/%d ===", s.consecutiveRetryCount, s.cfg.MaxConsecutiveRetries))

		retryBody := BuildRetryRequestBody(s.originalRequestBody, s.accumulatedText)
		retryBodyBytes, err := json.Marshal(retryBody)
		if err != nil {
			logger.LogError("Failed to marshal retry body:", err)
			time.Sleep(s.cfg.RetryDelayMs)
			continue
		}

		retryReq, err := http.NewRequest("POST", s.upstreamURL, bytes.NewReader(retryBodyBytes))
		if err != nil {
			logger.LogError("Failed to create retry request:", err)
			time.Sleep(s.cfg.RetryDelayMs)
			continue
		}

		for name, values := range s.originalHeaders {
			if name == "Authorization" || name == "X-Goog-Api-Key" || name == "Content-Type" || name == "Accept" {
				for _, value := range values {
					retryReq.Header.Add(name, value)
				}
			}
		}

		retryResponse, err := s.client.Do(retryReq)
		if err != nil {
			logger.LogError(fmt.Sprintf("=== RETRY ATTEMPT %d FAILED ===", s.consecutiveRetryCount))
			logger.LogError("Exception during retry:", err)
			time.Sleep(s.cfg.RetryDelayMs)
			continue
		}
		defer retryResponse.Body.Close()

		logger.LogInfo(fmt.Sprintf("Retry request completed. Status: %d %s", retryResponse.StatusCode, retryResponse.Status))

		if retryResponse.StatusCode != http.StatusOK {
			logger.LogError(fmt.Sprintf("Retry attempt %d failed with status %d", s.consecutiveRetryCount, retryResponse.StatusCode))
			time.Sleep(s.cfg.RetryDelayMs)
			continue
		}

		logger.LogInfo(fmt.Sprintf("✓ Retry attempt %d successful - got new stream", s.consecutiveRetryCount))
		currentReader = retryResponse.Body
	}
}
