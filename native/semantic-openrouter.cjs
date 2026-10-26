const { execFileSync } = require("node:child_process");
const { SEMANTIC_POLICY, SemanticError } = require("./semantic-core.cjs");

const OPENROUTER_URL = "https://openrouter.ai/api/alpha/decisions";
// Same Jev family as Surf's calibrated jev-1.13.0; never use a floating alias.
const OPENROUTER_MODEL = "typesafe/jev-1.13-20260917";
const MAX_RESPONSE_BYTES = 256 * 1024;

function failure(code, message, status) {
  const error = new SemanticError(code, message);
  if (status !== undefined) error.status = status;
  return error;
}

function requireKey(value) {
  if (typeof value !== "string" || !value.trim() || /[\r\n]/.test(value.trim()) || Buffer.byteLength(value) > 16 * 1024) {
    throw failure("provider_not_configured", "OpenRouter API key is invalid");
  }
  return value.trim();
}

function resolveOpenRouterCredential(env = process.env, { platform = process.platform, readKeychain = execFileSync } = {}) {
  if (typeof env.OPENROUTER_API_KEY === "string" && env.OPENROUTER_API_KEY.trim()) {
    return { apiKey: requireKey(env.OPENROUTER_API_KEY), source: "environment" };
  }
  const service = env.SURF_OPENROUTER_KEYCHAIN_SERVICE;
  const account = env.SURF_OPENROUTER_KEYCHAIN_ACCOUNT;
  if (!service && !account) {
    throw failure("provider_not_configured", "OpenRouter needs OPENROUTER_API_KEY, or both SURF_OPENROUTER_KEYCHAIN_SERVICE and SURF_OPENROUTER_KEYCHAIN_ACCOUNT on macOS");
  }
  if (platform !== "darwin" || ![service, account].every((value) => typeof value === "string" && value.trim() && value.length <= 256 && !/[\r\n\0]/.test(value))) {
    throw failure("provider_not_configured", "OpenRouter Keychain lookup needs macOS and a valid service and account");
  }
  try {
    const apiKey = requireKey(readKeychain("/usr/bin/security", ["find-generic-password", "-w", "-s", service, "-a", account], {
      encoding: "utf8", stdio: ["ignore", "pipe", "ignore"], timeout: 5000, maxBuffer: 16 * 1024,
    }));
    return { apiKey, source: "keychain" };
  } catch {
    // Never retain child-process errors: they can contain stdout (the secret).
    throw failure("provider_not_configured", "OpenRouter Keychain credential is unavailable; check access to the configured item");
  }
}

async function readResponse(response) {
  if (!response.body) throw failure("provider_invalid_response", "OpenRouter returned no response body");
  const reader = response.body.getReader();
  const chunks = [];
  let bytes = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes += value.byteLength;
      if (bytes > MAX_RESPONSE_BYTES) {
        await reader.cancel();
        throw failure("provider_invalid_response", "OpenRouter response exceeds the size limit");
      }
      chunks.push(Buffer.from(value));
    }
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch (error) {
    if (error instanceof SemanticError) throw error;
    throw failure("provider_invalid_response", "OpenRouter returned an invalid response");
  } finally {
    reader.releaseLock();
  }
}

function createOpenRouterEvaluator({ apiKey, fetch = globalThis.fetch } = {}) {
  const key = requireKey(apiKey);
  return async function evaluate(state, questions, options = {}) {
    if (options.images?.length) throw failure("semantic_invalid_request", "OpenRouter Jev 1.13 does not support Surf icon vision");
    const controller = new AbortController();
    let timedOut = false;
    const cancel = () => controller.abort();
    const timer = setTimeout(() => { timedOut = true; controller.abort(); }, SEMANTIC_POLICY.timeoutMs);
    if (options.signal?.aborted) controller.abort();
    else options.signal?.addEventListener("abort", cancel, { once: true });
    try {
      if (controller.signal.aborted) throw failure("provider_cancelled", "OpenRouter request cancelled");
      const response = await fetch(OPENROUTER_URL, {
        method: "POST", redirect: "error", signal: controller.signal,
        headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
        body: JSON.stringify({ model: OPENROUTER_MODEL, state, questions, provider: { zdr: true, data_collection: "deny", allow_fallbacks: false } }),
      });
      if (!response.ok) {
        await response.body?.cancel();
        const status = response.status;
        const code = status === 401 || status === 403 ? "provider_authentication"
          : status === 402 ? "provider_billing"
          : status === 429 ? "provider_rate_limited"
          : status >= 500 ? "provider_unavailable" : "provider_invalid_request";
        throw failure(code, "OpenRouter request failed", status);
      }
      const result = await readResponse(response);
      if (!result || result.model !== OPENROUTER_MODEL || !result.answers || typeof result.answers !== "object" || Array.isArray(result.answers)) {
        throw failure("provider_invalid_response", "OpenRouter returned an unexpected model or answer envelope");
      }
      const usage = result.usage;
      if (!usage || ![usage.input_tokens, usage.output_tokens].every((value) => Number.isSafeInteger(value) && value >= 0)) {
        throw failure("provider_invalid_response", "OpenRouter returned invalid token usage");
      }
      // Surf validates every answer, label, probability and confidence before acting.
      // Remove transport metadata and optional cost to keep its strict core schema.
      return { model: result.model, answers: result.answers, usage: { input_tokens: usage.input_tokens, output_tokens: usage.output_tokens } };
    } catch (error) {
      if (controller.signal.aborted) throw failure(timedOut ? "provider_timeout" : "provider_cancelled", timedOut ? "OpenRouter request timed out" : "OpenRouter request cancelled");
      if (error instanceof SemanticError) throw error;
      throw failure("provider_unavailable", "OpenRouter connection failed");
    } finally {
      clearTimeout(timer);
      options.signal?.removeEventListener("abort", cancel);
    }
  };
}

module.exports = { OPENROUTER_URL, OPENROUTER_MODEL, createOpenRouterEvaluator, resolveOpenRouterCredential };
