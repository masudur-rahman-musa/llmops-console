/**
 * Shown inside the console shell while a page is being fetched (Next.js app-router
 * `loading` segment). Pinned by tests/screens.test.tsx.
 */
export default function Loading() {
  return (
    <p role="status" className="note">
      Loading…
    </p>
  );
}
