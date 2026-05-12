//! Model and Provider type definitions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Macro to define a string-backed enum with `Custom(String)` catch-all.
///
/// Generates:
/// - The enum with `#[serde(rename = "...")]` on each variant
/// - `as_str() -> &str`
/// - `Display` (delegates to `as_str()`)
/// - `From<String>` (reverse mapping, unknown strings become `Custom(String)`)
macro_rules! define_string_enum {
    (
        $(#[$meta:meta])*
        pub enum $name:ident {
            $( $(#[$variant_meta:meta])* $variant:ident => $str:literal, )*
        }
    ) => {
        $(#[$meta])*
        pub enum $name {
            $(
                $(#[$variant_meta])*
                #[serde(rename = $str)]
                $variant,
            )*
            /// Custom variant.
            Custom(String),
        }

        impl $name {
            /// Get the string representation.
            pub fn as_str(&self) -> &str {
                match self {
                    $( $name::$variant => $str, )*
                    $name::Custom(s) => s.as_str(),
                }
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.as_str())
            }
        }

        impl From<String> for $name {
            fn from(s: String) -> Self {
                match s.as_str() {
                    $( $str => $name::$variant, )*
                    _ => $name::Custom(s),
                }
            }
        }
    }
}

define_string_enum! {
    /// Known API types.
    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum Api {
        /// OpenAI Chat Completions API.
        OpenAICompletions => "openai-completions",
        /// Mistral Conversations API.
        MistralConversations => "mistral-conversations",
        /// OpenAI Responses API (new).
        OpenAIResponses => "openai-responses",
        /// Azure OpenAI Responses API.
        AzureOpenAIResponses => "azure-openai-responses",
        /// OpenAI Codex Responses API.
        OpenAICodexResponses => "openai-codex-responses",
        /// Anthropic Messages API.
        AnthropicMessages => "anthropic-messages",
        /// AWS Bedrock Converse Stream API.
        BedrockConverseStream => "bedrock-converse-stream",
        /// Google Generative AI API.
        GoogleGenerativeAi => "google-generative-ai",
        /// Google Gemini CLI API.
        GoogleGeminiCli => "google-gemini-cli",
        /// Google Vertex AI API.
        GoogleVertex => "google-vertex",
        /// Ollama API (OpenAI compatible).
        Ollama => "ollama",
    }
}

impl Api {
    /// Check if this is an OpenAI-compatible API.
    pub fn is_openai_compatible(&self) -> bool {
        matches!(
            self,
            Api::OpenAICompletions | Api::Ollama | Api::MistralConversations | Api::OpenAIResponses
        )
    }
}

define_string_enum! {
    /// Known provider types.
    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum Provider {
        /// Amazon Bedrock.
        AmazonBedrock => "amazon-bedrock",
        /// Anthropic.
        Anthropic => "anthropic",
        /// Google.
        Google => "google",
        /// Google Gemini CLI.
        GoogleGeminiCli => "google-gemini-cli",
        /// Google Antigravity.
        GoogleAntigravity => "google-antigravity",
        /// Google Vertex AI.
        GoogleVertex => "google-vertex",
        /// OpenAI.
        OpenAI => "openai",
        /// Generic OpenAI-compatible provider facade.
        OpenAICompatible => "openai-compatible",
        /// OpenAI Responses API.
        OpenAIResponses => "openai-responses",
        /// Azure OpenAI Responses.
        AzureOpenAIResponses => "azure-openai-responses",
        /// OpenAI Codex.
        OpenAICodex => "openai-codex",
        /// GitHub Copilot.
        GitHubCopilot => "github-copilot",
        /// xAI.
        XAI => "xai",
        /// Groq.
        Groq => "groq",
        /// Cerebras.
        Cerebras => "cerebras",
        /// OpenRouter.
        OpenRouter => "openrouter",
        /// Vercel AI Gateway.
        VercelAiGateway => "vercel-ai-gateway",
        /// ZAI.
        ZAI => "zai",
        /// Mistral.
        Mistral => "mistral",
        /// MiniMax.
        MiniMax => "minimax",
        /// MiniMax CN.
        MiniMaxCN => "minimax-cn",
        /// HuggingFace.
        HuggingFace => "huggingface",
        /// OpenCode.
        OpenCode => "opencode",
        /// OpenCode Go.
        OpenCodeGo => "opencode-go",
        /// Kimi Coding.
        KimiCoding => "kimi-coding",
        /// DeepSeek.
        DeepSeek => "deepseek",
        /// Xiaomi MiMo.
        XiaomiMIMO => "xiaomi-mimo",
        /// Zenmux.
        Zenmux => "zenmux",
        /// BAI.
        Bai => "bai",
        /// Ollama.
        Ollama => "ollama",
    }
}

/// Input type supported by a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InputType {
    /// Text input.
    Text,
    /// Image input.
    Image,
}

/// Cost configuration for a model (in USD per million tokens).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Cost {
    /// Cost per million input tokens.
    pub input: f64,
    /// Cost per million output tokens.
    pub output: f64,
    /// Cost per million cached tokens read.
    pub cache_read: f64,
    /// Cost per million tokens written to cache.
    pub cache_write: f64,
}

impl Default for Cost {
    fn default() -> Self {
        Self {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
        }
    }
}

impl Cost {
    /// Create a new cost configuration.
    pub fn new(input: f64, output: f64, cache_read: f64, cache_write: f64) -> Self {
        Self {
            input,
            output,
            cache_read,
            cache_write,
        }
    }

    /// Create a free tier cost (all zeros).
    pub fn free() -> Self {
        Self::default()
    }
}

/// API capability flags — whether the provider supports certain protocol features.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompatCapabilities {
    /// Supports the 'store' parameter.
    #[serde(default)]
    pub supports_store: bool,
    /// Supports the 'developer' role.
    #[serde(default)]
    pub supports_developer_role: bool,
    /// Supports 'reasoning_effort' parameter.
    #[serde(default)]
    pub supports_reasoning_effort: bool,
    /// Supports usage in streaming responses.
    #[serde(default = "default_true")]
    pub supports_usage_in_streaming: bool,
    /// Supports strict mode for tool calls.
    #[serde(default = "default_true")]
    pub supports_strict_mode: bool,
}

impl Default for CompatCapabilities {
    fn default() -> Self {
        Self {
            supports_store: true,
            supports_developer_role: false,
            supports_reasoning_effort: true,
            supports_usage_in_streaming: true,
            supports_strict_mode: true,
        }
    }
}

/// Thinking/reasoning format requirements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompatThinking {
    /// Thinking format ("openai", "zai", "qwen", "qwen-chat-template").
    #[serde(default = "default_openai", rename = "thinking_format")]
    pub format: String,
    /// Requires thinking blocks to be sent as text.
    #[serde(default, rename = "requires_thinking_as_text")]
    pub as_text: bool,
    /// When true, this provider requires every assistant message to carry
    /// reasoning/thinking content when thinking is enabled, and content must
    /// not be null.  Currently DeepSeek API enforces this constraint.
    #[serde(default, rename = "reasoning_content_constrained")]
    pub content_constrained: bool,
    /// Mapping of thinking levels to reasoning effort values.
    #[serde(default, rename = "reasoning_effort_map")]
    pub effort_map: HashMap<String, String>,
}

impl Default for CompatThinking {
    fn default() -> Self {
        Self {
            format: "openai".to_string(),
            as_text: false,
            content_constrained: false,
            effort_map: HashMap::new(),
        }
    }
}

/// Wire-format constraints on messages and tool results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct CompatMessageFormat {
    /// Which field to use for max tokens.
    #[serde(default)]
    pub max_tokens_field: Option<String>,
    /// Requires 'name' field in tool results.
    #[serde(default)]
    pub requires_tool_result_name: bool,
    /// Requires assistant message after tool results.
    #[serde(default)]
    pub requires_assistant_after_tool_result: bool,
}

/// OpenAI Completions compatibility settings.
///
/// Groups provider quirks into three semantic sub-structs:
/// - `capabilities` — API feature probing flags
/// - `thinking` — Reasoning/thinking format requirements
/// - `message_format` — Wire-format constraints on messages
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct OpenAICompletionsCompat {
    /// API capability flags.
    #[serde(flatten)]
    pub capabilities: CompatCapabilities,
    /// Thinking/reasoning format requirements.
    #[serde(flatten)]
    pub thinking: CompatThinking,
    /// Wire-format constraints on messages and tool results.
    #[serde(flatten)]
    pub message_format: CompatMessageFormat,
    /// OpenRouter routing preferences (e.g., `{"only": ["anthropic"]}`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open_router_routing: Option<serde_json::Value>,
}

fn default_true() -> bool {
    true
}

fn default_openai() -> String {
    "openai".to_string()
}

/// Model configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Model {
    /// Model identifier.
    pub id: String,
    /// Display name.
    pub name: String,
    /// API type (optional — determined by Provider implementation if not set).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api: Option<Api>,
    /// Provider name.
    pub provider: Provider,
    /// Base URL for API calls. When None, the provider uses its own default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Whether this model supports reasoning/thinking.
    pub reasoning: bool,
    /// Supported input types.
    pub input: Vec<InputType>,
    /// Cost configuration.
    pub cost: Cost,
    /// Context window size in tokens.
    pub context_window: u32,
    /// Maximum output tokens.
    pub max_tokens: u32,
    /// Custom headers for API calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Compatibility settings for OpenAI Completions API.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compat: Option<OpenAICompletionsCompat>,
}

impl Model {
    /// Create a new model builder.
    pub fn builder() -> ModelBuilder {
        ModelBuilder::default()
    }

    /// Check if this model supports text input.
    pub fn supports_text(&self) -> bool {
        self.input.contains(&InputType::Text)
    }

    /// Check if this model supports image input.
    pub fn supports_image(&self) -> bool {
        self.input.contains(&InputType::Image)
    }

    /// Calculate the cost for given usage.
    pub fn calculate_cost(&self, usage: &crate::types::Usage) -> f64 {
        let input_cost = (usage.input as f64 / 1_000_000.0) * self.cost.input;
        let output_cost = (usage.output as f64 / 1_000_000.0) * self.cost.output;
        let cache_read_cost = (usage.cache_read as f64 / 1_000_000.0) * self.cost.cache_read;
        let cache_write_cost = (usage.cache_write as f64 / 1_000_000.0) * self.cost.cache_write;
        input_cost + output_cost + cache_read_cost + cache_write_cost
    }
}

/// Builder for Model.
#[derive(Debug, Default)]
pub struct ModelBuilder {
    id: Option<String>,
    name: Option<String>,
    api: Option<Api>,
    provider: Option<Provider>,
    base_url: Option<String>,
    reasoning: bool,
    input: Vec<InputType>,
    cost: Cost,
    context_window: Option<u32>,
    max_tokens: Option<u32>,
    headers: Option<HashMap<String, String>>,
    compat: Option<OpenAICompletionsCompat>,
}

impl ModelBuilder {
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn api(mut self, api: Api) -> Self {
        self.api = Some(api);
        self
    }

    pub fn provider(mut self, provider: Provider) -> Self {
        self.provider = Some(provider);
        self
    }

    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    pub fn reasoning(mut self, reasoning: bool) -> Self {
        self.reasoning = reasoning;
        self
    }

    pub fn input(mut self, input: Vec<InputType>) -> Self {
        self.input = input;
        self
    }

    pub fn cost(mut self, cost: Cost) -> Self {
        self.cost = cost;
        self
    }

    pub fn context_window(mut self, tokens: u32) -> Self {
        self.context_window = Some(tokens);
        self
    }

    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    pub fn headers(mut self, headers: HashMap<String, String>) -> Self {
        self.headers = Some(headers);
        self
    }

    pub fn compat(mut self, compat: OpenAICompletionsCompat) -> Self {
        self.compat = Some(compat);
        self
    }

    pub fn build(self) -> Result<Model, String> {
        let id = self.id.ok_or("id is required")?;
        let name = self.name.ok_or("name is required")?;
        let provider = self.provider.ok_or("provider is required")?;
        let context_window = self.context_window.ok_or("context_window is required")?;
        let max_tokens = self.max_tokens.ok_or("max_tokens is required")?;

        Ok(Model {
            id,
            name,
            api: self.api,
            provider,
            base_url: self.base_url,
            reasoning: self.reasoning,
            input: self.input,
            cost: self.cost,
            context_window,
            max_tokens,
            headers: self.headers,
            compat: self.compat,
        })
    }
}
