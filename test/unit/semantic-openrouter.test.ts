import { afterEach, describe, expect, it, vi } from "vitest";

const {
  OPENROUTER_MODEL,
  OPENROUTER_URL,
  createOpenRouterEvaluator,
  resolveOpenRouterCredential,
} = require("../../native/semantic-openrouter.cjs");
const { createSemanticEvaluator } = require("../../native/semantic-evaluator.cjs");
const { find } = require("../../native/semantic-core.cjs");
const secret = "synthetic-secret";
const questions = {
  target: { type: "choice", instructions: "Find Save", criteria: { e1: "Save", none: null } },
};
const answer = {
  type: "choice",
  choice: "e1",
  probabilities: { e1: 0.99, none: 0.01 },
  confidence: 0.98,
};
const result = () => ({
  id: "gen-test",
  provider: "TypeSafe",
  model: OPENROUTER_MODEL,
  answers: { target: answer },
  usage: { input_tokens: 12, output_tokens: 0, cost: 0.0001 },
});
const respond = (value: unknown = result(), status = 200) =>
  vi.fn(async () => new Response(JSON.stringify(value), { status }));

afterEach(() => vi.useRealTimers());

describe("OpenRouter Jev route", () => {
  it("pins endpoint/model/privacy, forbids redirects and sends exactly one request", async () => {
    const fetch = respond();
    const evaluate = createOpenRouterEvaluator({ apiKey: ` ${secret} `, fetch });
    expect(await evaluate({ title: "Synthetic" }, questions)).toEqual({
      model: OPENROUTER_MODEL,
      answers: { target: answer },
      usage: { input_tokens: 12, output_tokens: 0 },
    });
    expect(fetch).toHaveBeenCalledOnce();
    const [url, init] = fetch.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe(OPENROUTER_URL);
    expect(init.redirect).toBe("error");
    expect(init.headers).toEqual({
      Authorization: `Bearer ${secret}`,
      "Content-Type": "application/json",
    });
    expect(JSON.parse(init.body as string)).toEqual({
      model: OPENROUTER_MODEL,
      state: { title: "Synthetic" },
      questions,
      provider: { zdr: true, data_collection: "deny", allow_fallbacks: false },
    });
  });

  it("runs through Surf's existing choice validation and calibrated thresholds", async () => {
    const evaluate = createOpenRouterEvaluator({ apiKey: secret, fetch: respond() });
    const input = {
      state: { title: "Synthetic" },
      goal: "Save",
      candidates: [{ id: "e1", role: "button", name: "Save" }],
      evaluate,
    };
    expect(await find(input)).toMatchObject({ status: "found", candidate: { id: "e1" } });
    const bad = {
      ...result(),
      answers: { target: { ...answer, choice: "delete", probabilities: { delete: 1 } } },
    };
    await expect(
      find({
        ...input,
        evaluate: createOpenRouterEvaluator({ apiKey: secret, fetch: respond(bad) }),
      }),
    ).rejects.toMatchObject({ code: "provider_invalid_response" });
  });

  it("routes the factory explicitly and reports OpenRouter without loading TypeSafe", async () => {
    const loadSdk = vi.fn();
    let time = 0;
    const fetch = vi.fn(async () => {
      time += 30;
      return new Response(JSON.stringify(result()));
    });
    const evaluator = createSemanticEvaluator({
      model: "jev-1.13.0",
      env: {
        SURF_JEV_PROVIDER: " openrouter ",
        OPENROUTER_API_KEY: secret,
        TYPESAFE_API_KEY: "unused",
      },
      loadSdk,
      fetch,
      now: () => time,
    });
    await evaluator.evaluate({}, questions);
    expect(evaluator.summary()).toEqual({
      provider: "openrouter",
      model: OPENROUTER_MODEL,
      providerCalls: 1,
      providerLatencyMs: 30,
    });
    expect(loadSdk).not.toHaveBeenCalled();
    expect(() =>
      createSemanticEvaluator({ model: "jev-1.13.0", env: { SURF_JEV_PROVIDER: "typo" }, fetch }),
    ).toThrow("SURF_JEV_PROVIDER must be");
    expect(() =>
      createSemanticEvaluator({
        model: "jev-1.13.0",
        env: { SURF_JEV_PROVIDER: "openrouter" },
        fetch,
      }),
    ).toThrow(expect.objectContaining({ code: "provider_not_configured" }));
  });

  it.each([401, 402, 403, 422, 429, 503])(
    "redacts HTTP %i and never retries or falls back",
    async (status) => {
      const fetch = respond({ error: { message: secret } }, status);
      try {
        await createOpenRouterEvaluator({ apiKey: secret, fetch })({}, questions);
        expect.fail("must reject");
      } catch (error: any) {
        expect(error.message).not.toContain(secret);
        expect(error.status).toBe(status);
      }
      expect(fetch).toHaveBeenCalledOnce();
    },
  );

  it.each([
    { ...result(), model: "typesafe/jev-latest" },
    { ...result(), usage: { input_tokens: -1, output_tokens: 0 } },
    { ...result(), answers: null },
    null,
  ])("rejects unexpected models and invalid metadata", async (body) => {
    await expect(
      createOpenRouterEvaluator({ apiKey: secret, fetch: respond(body) })({}, questions),
    ).rejects.toMatchObject({ code: "provider_invalid_response" });
  });

  it("redacts malformed JSON and transport errors and caps response size", async () => {
    for (const fetch of [
      vi.fn(async () => new Response(secret)),
      vi.fn(async () => {
        throw new Error(secret);
      }),
      vi.fn(async () => new Response("x".repeat(256 * 1024 + 1))),
    ]) {
      await expect(
        createOpenRouterEvaluator({ apiKey: secret, fetch })({}, questions),
      ).rejects.toThrow(/^OpenRouter/);
      try {
        await createOpenRouterEvaluator({ apiKey: secret, fetch })({}, questions);
      } catch (e: any) {
        expect(e.message).not.toContain(secret);
      }
    }
  });

  it("bounds calls with a timeout and respects caller cancellation", async () => {
    vi.useFakeTimers();
    const fetch = vi.fn(
      (_url: string, init: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init.signal?.addEventListener("abort", () => reject(new Error(secret)), { once: true });
        }),
    );
    const evaluate = createOpenRouterEvaluator({ apiKey: secret, fetch });
    const timed = expect(evaluate({}, questions)).rejects.toMatchObject({
      code: "provider_timeout",
    });
    await vi.advanceTimersByTimeAsync(5000);
    await timed;
    const controller = new AbortController();
    const cancelled = expect(
      evaluate({}, questions, { signal: controller.signal }),
    ).rejects.toMatchObject({ code: "provider_cancelled" });
    controller.abort();
    await cancelled;
    const preCancelled = new AbortController();
    preCancelled.abort();
    await expect(evaluate({}, questions, { signal: preCancelled.signal })).rejects.toMatchObject({
      code: "provider_cancelled",
    });
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("rejects unsupported vision without making a provider call", async () => {
    const fetch = respond();
    await expect(
      createOpenRouterEvaluator({ apiKey: secret, fetch })({}, questions, { images: [{}] }),
    ).rejects.toMatchObject({ code: "semantic_invalid_request" });
    expect(fetch).not.toHaveBeenCalled();
  });
});

describe("OpenRouter credentials", () => {
  it("uses the environment first and never reads Keychain implicitly", () => {
    const readKeychain = vi.fn();
    expect(resolveOpenRouterCredential({ OPENROUTER_API_KEY: secret }, { readKeychain })).toEqual({
      apiKey: secret,
      source: "environment",
    });
    expect(() => resolveOpenRouterCredential({}, { readKeychain })).toThrow("OpenRouter needs");
    expect(readKeychain).not.toHaveBeenCalled();
  });
  it("reads only the configured macOS item with a fixed executable and private bounded pipes", () => {
    const readKeychain = vi.fn(() => `${secret}\n`);
    const env = {
      SURF_OPENROUTER_KEYCHAIN_SERVICE: "test.service",
      SURF_OPENROUTER_KEYCHAIN_ACCOUNT: "test-account",
    };
    expect(resolveOpenRouterCredential(env, { platform: "darwin", readKeychain })).toEqual({
      apiKey: secret,
      source: "keychain",
    });
    expect(readKeychain).toHaveBeenCalledWith(
      "/usr/bin/security",
      ["find-generic-password", "-w", "-s", "test.service", "-a", "test-account"],
      expect.objectContaining({ stdio: ["ignore", "pipe", "ignore"], timeout: 5000 }),
    );
    expect(() => resolveOpenRouterCredential(env, { platform: "linux", readKeychain })).toThrow(
      "needs macOS",
    );
    expect(() =>
      resolveOpenRouterCredential(
        { SURF_OPENROUTER_KEYCHAIN_SERVICE: "test" },
        { platform: "darwin", readKeychain },
      ),
    ).toThrow("valid service and account");
  });
  it("does not retain Keychain failures or their stdout", () => {
    const env = {
      SURF_OPENROUTER_KEYCHAIN_SERVICE: "test",
      SURF_OPENROUTER_KEYCHAIN_ACCOUNT: "test",
    };
    const readKeychain = () => {
      throw Object.assign(new Error(secret), { stdout: secret });
    };
    try {
      resolveOpenRouterCredential(env, { platform: "darwin", readKeychain });
      expect.fail("must reject");
    } catch (e: any) {
      expect(e.message).not.toContain(secret);
      expect(JSON.stringify(e)).not.toContain(secret);
    }
  });
});
