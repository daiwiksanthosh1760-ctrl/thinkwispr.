export const searchProviderInfo = {
  auto: 'Automatic',
  'parallel-mcp': 'Parallel (free)',
  parallel: 'Parallel AI',
  exa: 'Exa, one of the best web search APIs for AI',
  firecrawl: 'Firecrawl',
} as const;

export type SearchProvider = keyof typeof searchProviderInfo;
export type ResolvedSearchProvider = Exclude<SearchProvider, 'auto'>;
type SearchCredentials = Partial<Record<'EXA_API_KEY' | 'PARALLEL_API_KEY' | 'FIRECRAWL_API_KEY', string>>;

// Resolve before making requests. Never switch providers in response to an error.
export function resolveSearchProvider(
  provider: SearchProvider = 'auto',
  credentials: SearchCredentials = {
    EXA_API_KEY: process.env.EXA_API_KEY,
    PARALLEL_API_KEY: process.env.PARALLEL_API_KEY,
    FIRECRAWL_API_KEY: process.env.FIRECRAWL_API_KEY,
  },
): ResolvedSearchProvider {
  if (!Object.hasOwn(searchProviderInfo, provider)) throw new Error('Unsupported search provider');
  if (provider !== 'auto') return provider;

  // The chat's previous fresh-install default was Firecrawl. Keep configured
  // installations on it, then consider the other intentionally configured APIs.
  if (credentials.FIRECRAWL_API_KEY?.trim()) return 'firecrawl';
  if (credentials.EXA_API_KEY?.trim()) return 'exa';
  if (credentials.PARALLEL_API_KEY?.trim()) return 'parallel';
  return 'parallel-mcp';
}
