import { pageHref, type Params } from "@/lib/paging";

/**
 * The one pager under a paged table: Prev, "Page X of Y", Next. Prev and Next
 * are links that keep the other panels' pages; at an edge the word is plain
 * text. Nothing is drawn when the panel fits on one page. Pinned by tests/pager.test.tsx.
 */
export function Pager({ name, page, pages, params }: { name: string; page: number; pages: number; params: Params }) {
  if (pages <= 1) {
    return null;
  }
  return (
    <nav className="pager" aria-label="Pages">
      {page > 1 ? <a href={pageHref(params, name, page - 1)}>Prev</a> : <span className="note">Prev</span>}
      <span>
        Page {page} of {pages}
      </span>
      {page < pages ? <a href={pageHref(params, name, page + 1)}>Next</a> : <span className="note">Next</span>}
    </nav>
  );
}
