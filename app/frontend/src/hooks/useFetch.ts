import { useEffect, useState } from "react";

interface FetchState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

/** Small shared data-fetching hook - loading/error/data states, cancels a
 * stale request if the component unmounts or deps change before it
 * resolves. Not a general-purpose data library; this app has a handful of
 * read-only GET endpoints and doesn't need one. */
export function useFetch<T>(fetcher: () => Promise<T>, deps: unknown[] = []): FetchState<T> {
  const [state, setState] = useState<FetchState<T>>({ data: null, error: null, loading: true });

  // This hook's whole purpose is to re-run the effect when the CALLER's
  // `deps` array changes, not when `fetcher` itself changes identity
  // (callers pass a fresh arrow function on every render, and including
  // it would defeat the "only re-fetch on identity/key change" behavior
  // every caller of this hook relies on - the same intentional pattern
  // React's own docs describe for a custom hook that forwards a
  // caller-supplied dependency list).
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    let cancelled = false;
    setState({ data: null, error: null, loading: true });
    fetcher()
      .then((result) => {
        if (!cancelled) setState({ data: result, error: null, loading: false });
      })
      .catch((err: Error) => {
        if (!cancelled) setState({ data: null, error: err.message, loading: false });
      });
    return () => {
      cancelled = true;
    };
  }, deps);

  return state;
}
