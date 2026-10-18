import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { SearchProvider } from '@/lib/search/providers';

// Supply dummy values only for unrelated application settings. Keep the search
// credentials isolated, and import the real application environment validator.
for (const match of readFileSync('env/server.ts', 'utf8').matchAll(/    ([A-Z_]+): z\.string\(\)\.min\(1\)/g)) {
  process.env[match[1]] = 'fixture-value';
}
for (const key of ['EXA_API_KEY', 'PARALLEL_API_KEY', 'FIRECRAWL_API_KEY']) delete process.env[key];
const provider = process.argv[2] as 'exa' | 'parallel' | 'firecrawl' | 'none';
const action = process.argv[3];
if (provider !== 'none' && action !== 'missing') {
  process.env[`${provider.toUpperCase()}_API_KEY`] = 'fixture-search-key';
}
const { serverEnv } = await import('@/env/server');
if (provider === 'none') {
  assert.equal(serverEnv.EXA_API_KEY, '');
  assert.equal(serverEnv.PARALLEL_API_KEY, '');
  assert.equal(serverEnv.FIRECRAWL_API_KEY, '');
  process.exit(0);
}

const hosts = { exa: 'api.exa.ai', parallel: 'api.parallel.ai', firecrawl: 'api.firecrawl.dev' };
const calls: string[] = [];
globalThis.fetch = (async (input: string | URL | Request) => {
  const url = input instanceof Request ? input.url : String(input);
  calls.push(url);
  assert.equal(new URL(url).hostname, hosts[provider]);
  if (action === 'error') return Response.json({ error: 'Fixture failure' }, { status: 400 });
  const page = {
    url: 'https://example.com/paid',
    title: 'Configured source',
    highlights: ['Evidence'],
    excerpts: ['Evidence'],
    description: 'Evidence',
  };
  return Response.json(
    provider === 'firecrawl' ? { success: true, data: { web: [page], images: [] } } : { results: [page] },
  );
}) as typeof fetch;

const { loadConfiguredTools } = await import('@/lib/search/tool-loader');
const tools = await loadConfiguredTools({
  activeToolNames: ['web_search', 'retrieve'],
  searchProvider: action === 'missing' ? (provider as SearchProvider) : 'auto',
  dataStream: undefined,
  timezone: undefined,
  contextFiles: [],
  extremeSearchModel: undefined,
  includeMcpTools: false,
  mcpDynamicTools: {},
  lightweightUser: null,
  selectedConnectors: [],
});
assert.ok(tools.retrieve);
if (action === 'missing') {
  await assert.rejects(
    tools.web_search.execute({ queries: ['question'] }, { toolCallId: 'test', messages: [] }),
    /requires/,
  );
  assert.equal(calls.length, 0);
} else {
  const result = await tools.web_search.execute({ queries: ['question'] }, { toolCallId: 'test', messages: [] });
  assert.ok(calls.length > 0);
  assert.equal(result.searches[0].results.length, action === 'error' ? 0 : 1);
}
