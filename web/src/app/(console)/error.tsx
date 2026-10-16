"use client";

/**
 * Shown inside the console shell when a page throws (Next.js app-router `error`
 * segment). A client part is the one way Next.js lets a segment catch a render
 * error. The Try again button calls `reset` to re-render the segment. The alert
 * follows the Unavailable markup. Pinned by tests/screens.test.tsx.
 */
export default function Error({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <>
      <h1>Something went wrong</h1>
      <p role="alert" className="alert">This page could not be shown.</p>
      <button type="button" onClick={reset}>
        Try again
      </button>
    </>
  );
}
