/// <reference types="bun" />
import { afterEach, beforeEach, describe, expect, test } from 'bun:test';
import { generateText, stepCountIs } from 'ai';
import { MockLanguageModelV3 } from 'ai/test';
import { loadConfiguredTools } from '@/lib/search/tool-loader';
import { resolveSearchProvider, type SearchProvider } from '@/lib/search/providers';
import { parallelRetrieveTool } from '@/lib/tools/parallel-search';

const endpoint = 'https://search.parallel.ai/mcp';
const originalFetch = globalThis.fetch;
const originalTimeout = AbortSignal.timeout;
const originalEnv = { ...process.env };
const requests: Array<{ url: string; headers: Headers; body?: any; method: string; signal?: AbortSignal | null }> = [];
let mode: 'success' | 'text' | 'tool-error' | 'malformed' | 'http-error' | 'waiting' | 'initializing' = 'success';
let waiting: (() => void) | undefined;

const pages = [
  { url: 'https://example.com/source', title: 'A cited source', excerpts: ['Useful evidence for the answer.'] },
];

function jsonResponse(id: number, result: unknown) {
  return Response.json({ jsonrpc: '2.0', id, result });
}

function pendingRequest(signal?: AbortSignal | null) {
  waiting?.();
  return new Promise<Response>((_resolve, reject) => {
    if (signal?.aborted) return reject(signal.reason);
    signal?.addEventListener('abort', () => reject(signal.reason), { once: true });
  });
}

beforeEach(() => {
  requests.length = 0;
  mode = 'success';
  delete process.env.EXA_API_KEY;
  delete process.env.PARALLEL_API_KEY;
  delete process.env.FIRECRAWL_API_KEY;
  // The search transport is real; replace only the HTTP peer, inspecting every
  // outgoing request before returning MCP protocol responses.
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = input instanceof Request ? input.url : String(input);
    const method = init?.method ?? 'GET';
    const headers = new Headers(init?.headers);
    const body = typeof init?.body === 'string' ? JSON.parse(init.body) : undefined;
    requests.push({ url, method, headers, body, signal: init?.signal });
    if (url !== endpoint) throw new Error(`Unexpected destination: ${url}`);
    if (method === 'GET') return new Response(null, { status: 405 });
    if (method === 'DELETE') return new Response(null, { status: 200 });
    if (body?.id === undefined) return new Response(null, { status: 202 });
    if (body.method === 'initialize') {
      if (mode === 'initializing') return pendingRequest(init?.signal);
      return jsonResponse(body.id, {
        protocolVersion: '2025-06-18',
        capabilities: { tools: {} },
        serverInfo: { name: 'fixture', version: '1' },
      });
    }
    if (body.method === 'tools/list') {
      return jsonResponse(body.id, {
        tools: ['web_search', 'web_fetch'].map((name) => ({
          name,
          inputSchema: { type: 'object', properties: {} },
        })),
      });
    }
    if (body.method === 'tools/call') {
      if (mode === 'waiting') return pendingRequest(init?.signal);
      if (mode === 'http-error') return new Response('Unavailable', { status: 503 });
      if (mode === 'tool-error')
        return jsonResponse(body.id, { isError: true, content: [{ type: 'text', text: 'Search quota reached' }] });
      const payload =
        mode === 'malformed' ? { results: [{ url: 'javascript:alert(1)', excerpts: [] }] } : { results: pages };
      return jsonResponse(
        body.id,
        mode === 'text'
          ? { content: [{ type: 'text', text: JSON.stringify(payload) }] }
          : { content: [], structuredContent: payload },
      );
    }
    throw new Error(`Unexpected MCP method: ${body.method}`);
  }) as typeof fetch;
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  AbortSignal.timeout = originalTimeout;
  waiting = undefined;
  for (const name of Object.keys(process.env)) if (!(name in originalEnv)) delete process.env[name];
  Object.assign(process.env, originalEnv);
});

const options = { toolCallId: 'fixture-call', messages: [] };

function load(activeToolNames = ['web_search', 'retrieve'], searchProvider?: SearchProvider) {
  return loadConfiguredTools({
    activeToolNames,
    searchProvider,
    searchSessionId: 'conversation-123',
    dataStream: undefined,
    timezone: undefined,
    contextFiles: [],
    extremeSearchModel: undefined,
    includeMcpTools: false,
    mcpDynamicTools: { unauthorized_tool: {} },
    lightweightUser: null,
    selectedConnectors: [],
  });
}

describe('provider resolution and activation', () => {
  test('uses anonymous MCP only when unconfigured, keeping credentials and explicit saved choices', () => {
    expect(resolveSearchProvider(undefined, {})).toBe('parallel-mcp');
    expect(resolveSearchProvider('auto', { EXA_API_KEY: 'exa' })).toBe('exa');
    expect(resolveSearchProvider('auto', { PARALLEL_API_KEY: 'parallel' })).toBe('parallel');
    expect(resolveSearchProvider('auto', { FIRECRAWL_API_KEY: 'firecrawl', EXA_API_KEY: 'exa' })).toBe('firecrawl');
    for (const provider of ['exa', 'parallel', 'firecrawl', 'parallel-mcp'] as const) {
      expect(resolveSearchProvider(provider, { EXA_API_KEY: 'exa' })).toBe(provider);
    }
    expect(resolveSearchProvider('auto', { EXA_API_KEY: '  ' })).toBe('parallel-mcp');
    expect(() => resolveSearchProvider('unknown' as SearchProvider, {})).toThrow('Unsupported');
  });

  test('disabled tools and MCP permissions make no requests', async () => {
    expect(await load([])).toEqual({});
    expect(requests).toHaveLength(0);
  });

  test('the application environment accepts absent search credentials', () => {
    const child = Bun.spawnSync([
      process.execPath,
      '--conditions=react-server',
      'tests/search/fixtures/paid-provider.ts',
      'none',
    ]);
    expect(child.exitCode, child.stderr.toString()).toBe(0);
  });

  for (const provider of ['exa', 'parallel', 'firecrawl']) {
    // Firecrawl uses Axios rather than fetch. Its existing search implementation
    // is unchanged; do not send that SDK's fixture credentials to a live server.
    for (const action of provider === 'firecrawl' ? ['missing'] : ['success', 'missing', 'error']) {
      test(`preserves ${provider} selection: ${action}`, () => {
        // A separate process gives the real environment validator and singleton
        // API clients fresh state for each intentionally configured provider.
        const child = Bun.spawnSync([
          process.execPath,
          '--conditions=react-server',
          'tests/search/fixtures/paid-provider.ts',
          provider,
          action,
        ]);
        expect(child.exitCode, child.stdout.toString() + child.stderr.toString()).toBe(0);
      });
    }
  }

  test('the normal loader discovers lazily, preserves output and project headers, and reuses the conversation ID', async () => {
    const tools = await load();
    expect(Object.keys(tools).sort()).toEqual(['retrieve', 'web_search']);
    expect(requests).toHaveLength(0);
    const search = await tools.web_search.execute({ queries: ['research objective', 'query variation'] }, options);
    expect(search.searches[0].results[0]).toMatchObject({ url: pages[0].url, content: pages[0].excerpts[0] });
    expect(search.searches).toHaveLength(1);
    const fetched = await tools.retrieve.execute({ url: [pages[0].url] }, options);
    expect(fetched.results[0]).toMatchObject({ url: pages[0].url, content: pages[0].excerpts[0] });
    const calls = requests.filter((request) => request.body?.method === 'tools/call');
    expect(calls.map((request) => request.body.params.name)).toEqual(['web_search', 'web_fetch']);
    expect(calls[0].body.params.arguments).toMatchObject({
      objective: 'research objective',
      search_queries: ['research objective', 'query variation'],
    });
    expect(calls[1].body.params.arguments).toMatchObject({ urls: [pages[0].url], full_content: true });
    expect(calls[0].body.params.arguments.session_id).toBe(calls[1].body.params.arguments.session_id);
    expect(calls[0].body.params.arguments.session_id).toHaveLength(64);
    const again = await load();
    await again.web_search.execute({ queries: ['next turn'] }, options);
    expect(
      requests.filter((request) => request.body?.method === 'tools/call').at(-1)?.body.params.arguments.session_id,
    ).toBe(calls[0].body.params.arguments.session_id);
    for (const request of requests) {
      expect(request.url).toBe(endpoint);
      expect(
        request.headers
          .get('User-Agent')
          ?.startsWith('Scira/0.1.0 (https://github.com/daiwiksanthosh1760-ctrl/thinkwispr.)'),
      ).toBe(true);
      expect(request.headers.has('Authorization')).toBe(false);
      expect(request.headers.has('x-api-key')).toBe(false);
      expect(request.signal?.aborted).toBe(true);
    }
  });

  test('a scripted AI SDK model consumes cited tool results through the maintained loader', async () => {
    const tools = await load(['web_search']);
    let step = 0;
    const model = new MockLanguageModelV3({
      doGenerate: async (call) => {
        const first = step++ === 0;
        if (!first) {
          expect(JSON.stringify(call.prompt)).toContain(pages[0].url);
          expect(JSON.stringify(call.prompt)).toContain(pages[0].excerpts[0]);
        }
        return {
          content: first
            ? [
                {
                  type: 'tool-call' as const,
                  toolCallId: 'lookup',
                  toolName: 'web_search',
                  input: JSON.stringify({ queries: ['research objective'] }),
                },
              ]
            : [{ type: 'text' as const, text: `An answer grounded in ${pages[0].url}` }],
          finishReason: { unified: first ? ('tool-calls' as const) : ('stop' as const), raw: undefined },
          usage: {
            inputTokens: { total: 10, noCache: 10, cacheRead: 0, cacheWrite: 0 },
            outputTokens: { total: 10, text: 10, reasoning: 0 },
          },
          warnings: [],
        };
      },
    });
    const result = await generateText({ model, tools, prompt: 'Research this question', stopWhen: stepCountIs(2) });
    expect(result.text).toContain(pages[0].url);
    expect(step).toBe(2);
  });
});

describe('transport failures and input', () => {
  test('accepts JSON text tool responses', async () => {
    mode = 'text';
    const tools = await load();
    expect((await tools.web_search.execute({ queries: ['question'] }, options)).searches[0].results).toHaveLength(1);
  });

  for (const failure of ['tool-error', 'malformed', 'http-error'] as const) {
    test(`surfaces ${failure} without choosing a different provider`, async () => {
      mode = failure;
      const tools = await load();
      await expect(tools.web_search.execute({ queries: ['question'] }, options)).rejects.toThrow();
      expect(requests.every((request) => request.url === endpoint)).toBe(true);
    });
  }

  test('rejects unsafe supplied URLs before contacting MCP', async () => {
    const retrieve = parallelRetrieveTool('conversation-123');
    for (const url of ['file:///etc/passwd', 'javascript:alert(1)', 'https://user:secret@example.com']) {
      await expect(retrieve.execute!({ url: [url] }, options)).rejects.toThrow();
    }
    expect(requests).toHaveLength(0);
  });

  test('an already cancelled tool makes no request', async () => {
    const tools = await load();
    const signal = AbortSignal.abort();
    await expect(
      tools.web_search.execute({ queries: ['question'] }, { ...options, abortSignal: signal }),
    ).rejects.toThrow();
    expect(requests).toHaveLength(0);
  });

  test('cancellation stops an in-flight call', async () => {
    mode = 'waiting';
    const tools = await load();
    const controller = new AbortController();
    const started = new Promise<void>((resolve) => {
      waiting = resolve;
    });
    const result = tools.web_search.execute({ queries: ['question'] }, { ...options, abortSignal: controller.signal });
    await started;
    controller.abort();
    await expect(result).rejects.toThrow();
    expect(requests.filter((request) => request.body?.method === 'tools/call').at(-1)?.signal?.aborted).toBe(true);
  });

  test('bounds both initialization and tool execution by the request timeout', async () => {
    AbortSignal.timeout = () => originalTimeout(25);
    const tools = await load();
    for (const delayed of ['initializing', 'waiting'] as const) {
      mode = delayed;
      await expect(tools.web_search.execute({ queries: ['question'] }, options)).rejects.toThrow();
      expect(requests.at(-1)?.signal?.aborted).toBe(true);
    }
  });
});
