/**
 * Shown inside the console shell for an address that is no page (Next.js app-router
 * `not-found` segment). Pinned by tests/screens.test.tsx.
 */
export default function NotFound() {
  return (
    <>
      <h1>Page not found</h1>
      <p>That address is not a console page.</p>
      <p>
        <a href="/">Back to the console home</a>
      </p>
    </>
  );
}
