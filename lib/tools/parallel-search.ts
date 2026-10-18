import { createMCPClient, type MCPClient } from '@ai-sdk/mcp';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { tool, type ToolExecutionOptions, type UIMessageStreamWriter } from 'ai';
import { createHash } from 'node:crypto';
import { z } from 'zod';
import type { ChatMessage } from '@/lib/types';

const MCP_URL = 'https://search.parallel.ai/mcp';
const USER_AGENT = 'Scira/0.1.0 (https://github.com/daiwiksanthosh1760-ctrl/thinkwispr.)';
const REQUEST_TIMEOUT_MS = 30_000;

const httpUrl = z
  .string()
  .url()
  .refine((value) => {
    const url = new URL(value);
    return ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password;
  }, 'Expected an HTTP or HTTPS URL without credentials');

const pageSchema = z.object({
  url: httpUrl,
  title: z.string().nullish(),
  publish_date: z.string().nullish(),
  excerpts: z.array(z.string()),
  full_content: z.string().nullish(),
});

const responseSchema = z.object({
  results: z.array(pageSchema),
  errors: z.array(z.object({ url: z.string(), error_type: z.string() })).optional(),
});
const toolResultSchema = z.object({
  isError: z.boolean().optional(),
  structuredContent: z.unknown().optional(),
  content: z.array(z.object({ type: z.string(), text: z.string().optional() })).default([]),
});

function abortable<T>(promise: Promise<T>, signal: AbortSignal): Promise<T> {
  signal.throwIfAborted();
  return new Promise((resolve, reject) => {
    const abort = () => reject(signal.reason);
    signal.addEventListener('abort', abort, { once: true });
    promise.then(resolve, reject).finally(() => signal.removeEventListener('abort', abort));
  });
}

async function callParallel(
  name: 'web_search' | 'web_fetch',
  args: Record<string, unknown>,
  sessionId: string,
  options: ToolExecutionOptions,
) {
  const signal = AbortSignal.any([
    AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    ...(options.abortSignal ? [options.abortSignal] : []),
  ]);
  signal.throwIfAborted();
  let client: MCPClient | undefined;
  // Keep a transport handle before initialization so cancellation can also
  // stop discovery/connection requests, not just an already loaded tool call.
  const transport = new StreamableHTTPClientTransport(new URL(MCP_URL), {
    requestInit: { headers: { 'User-Agent': USER_AGENT }, redirect: 'error' },
    fetch: (url, init) =>
      fetch(url, {
        ...init,
        signal: AbortSignal.any([signal, ...(init?.signal ? [init.signal] : [])]),
      }),
    reconnectionOptions: {
      maxRetries: 0,
      maxReconnectionDelay: 0,
      initialReconnectionDelay: 0,
      reconnectionDelayGrowFactor: 1,
    },
  });
  const close = () => (client ? client.close() : transport.close()).catch(() => {});
  const abort = () => {
    void close();
  };
  signal.addEventListener('abort', abort, { once: true });

  try {
    const connecting = createMCPClient({
      name: 'scira',
      version: '0.1.0',
      transport,
    }).then((connected) => {
      client = connected;
      if (signal.aborted) void close();
      return connected;
    });
    const connected = await abortable(connecting, signal);
    const definitions = await connected.listTools({ options: { signal, timeout: REQUEST_TIMEOUT_MS } });
    signal.throwIfAborted();
    const remote = connected.toolsFromDefinitions(definitions)[name];
    if (!remote) throw new Error(`Parallel MCP does not offer ${name}`);
    const result = toolResultSchema.parse(
      await abortable(
        Promise.resolve(
          remote.execute(
            {
              ...args,
              session_id: createHash('sha256').update(sessionId).digest('hex'),
            },
            { ...options, abortSignal: signal },
          ),
        ),
        signal,
      ),
    );

    if (result.isError) {
      const message = result.content
        ?.filter((part) => part.type === 'text')
        .map((part) => part.text)
        .join('\n');
      throw new Error(message || 'Parallel MCP request failed');
    }
    const structured = result.structuredContent;
    const text = result.content?.find((part) => part.type === 'text');
    const payload = structured ?? (text?.text ? JSON.parse(text.text) : undefined);
    return responseSchema.parse(payload);
  } finally {
    signal.removeEventListener('abort', abort);
    await close();
  }
}

export function parallelWebSearchTool(sessionId: string, dataStream?: UIMessageStreamWriter<ChatMessage>) {
  return tool({
    description:
      'Search the web with Parallel. Use related query variations for one research objective. Returns sources with excerpts for citations. No image search or strict news/date filters; include freshness requirements in the queries.',
    inputSchema: z.object({
      queries: z.array(z.string().min(1)).min(1).max(5),
      maxResults: z.array(z.number().int().min(1).max(20)).min(1).optional(),
    }),
    execute: async ({ queries, maxResults }, options) => {
      const notify = (status: 'started' | 'completed' | 'error', count = 0) => {
        queries.forEach((query, index) =>
          dataStream?.write({
            type: 'data-query_completion',
            data: {
              query,
              index,
              total: queries.length,
              status,
              resultsCount: index === 0 ? count : 0,
              imagesCount: 0,
            },
          }),
        );
      };
      notify('started');
      try {
        const response = await callParallel(
          'web_search',
          {
            objective: queries[0],
            search_queries: queries,
          },
          sessionId,
          options,
        );
        const results = response.results.slice(0, maxResults?.[0] ?? 10).map((page) => ({
          url: page.url,
          title: page.title || page.url,
          content: page.excerpts.join('\n'),
          published_date: page.publish_date ?? undefined,
        }));
        notify('completed', results.length);
        // MCP batches related query variations. Keep one source list so the UI
        // and the model do not receive duplicate citations for every variation.
        return { searches: [{ query: queries[0], results, images: [] }] };
      } catch (error) {
        notify('error');
        throw error;
      }
    },
  });
}

export function parallelRetrieveTool(sessionId: string) {
  return tool({
    description:
      'Read ordinary HTTP/HTTPS page URLs explicitly supplied by the user with Parallel. Returns page text for citations. Does not provide specialized social-media metadata or transcripts.',
    inputSchema: z.object({ url: z.array(httpUrl).min(1).max(20) }),
    execute: async ({ url }, options) => {
      // Validate here as well: callers can invoke execute without model schema validation.
      z.array(httpUrl).min(1).max(20).parse(url);
      const start = Date.now();
      const response = await callParallel('web_fetch', { urls: url, full_content: true }, sessionId, options);
      const errors = response.errors?.map((error) => `${error.url}: ${error.error_type}`);
      return {
        urls: url,
        results: response.results.map((page) => {
          const content = page.full_content || page.excerpts.join('\n');
          return {
            url: page.url,
            content,
            title: page.title || page.url,
            description: content.slice(0, 200),
            publishedDate: page.publish_date ?? undefined,
          };
        }),
        sources: response.results.map(() => 'parallel-mcp'),
        response_time: (Date.now() - start) / 1000,
        ...(errors?.length ? { partial_errors: errors } : {}),
      };
    },
  });
}
