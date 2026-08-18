SPEECH TO TEXT PIPELINE
USER
SPEAKS ai4bharat/Indic-
VOICE ACTIVITY Tamil/Hindi Conformer Transcription- TRANSCRIPTION
(ASR output)
DETECTION (VAD) ASR for Tamil / Hindi
Transcription-
Detects speech ys
silence
speak audio stream
Microphone LANGUAGE ID WHISPER OUTPUT
input Audio Identifies spoken -Unlisted- OpenAl/Whisper -Transcription— FILTER
Multilingual ASR Hallucination /
language
repetition filtering
normalize
YES
HUMAN HANDOFF SENSITIVE/
Escalate to humnan escalate- RESTRICTED INTENT + STRUCTURED TRANSCRIPTION
agent INTENT ENTITY FORMAT NORMALIZATION
EXTRACTION analyze- { Language, Intent, -outputs- Code-switch normalization,
Sensitive intent Normalized, Entities, ASR error, language tagging,
detected?
NO Confidence} entity
ORCHESTRATORAGENT
AGENTIC RAG
Infentiroutnq.aqentselection policy
checks High confidence
TEXT TO SPEECH PIPELINE
COMPLAINTS
BILLING AGENT PLANS AGENT COVERAGE AGENT
AGENT RETRIEVAL HANDOFF GATE LLM RESPONSE LANGUAGE-
SCORE Human -NO GENERATION -synthesize
AWARE TTS
Threshold T requested? Grounded response
Billing RAG Product RAG Support RAG Technical RAG LLM uncertain? edge-tts
Tools / APIs Tools / APls Tools / APIs Tools / APIs
(billing related) (plans related) (complaints) (coverage) AUDIO OUTPUT
playback
Played back to
YES user
Low confidence
Human Handoff-
Next utterance