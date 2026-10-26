import type { ReactNode } from "react";

export const NONE_IN_WINDOW = "None in this window.";

/**
 * One `table-wrap` > table > [thead] > tbody. With `columns` it draws a header row; without, none.
 * No rows draws only the `None in this window.` line, never a bare header. Pinned by tests/shared-parts.test.tsx.
 */
export function Table({ columns, rows }: { columns?: string[]; rows: Array<{ key: string; cells: ReactNode[] }> }) {
  if (rows.length === 0) {
    return <p className="note">{NONE_IN_WINDOW}</p>;
  }
  return (
    <div className="table-wrap">
      <table>
        {columns && (
          <thead>
            <tr>
              {columns.map((name) => (
                <th key={name}>{name}</th>
              ))}
            </tr>
          </thead>
        )}
        <tbody>
          {rows.map((row) => (
            <tr key={row.key}>
              {row.cells.map((cell, index) => (
                <td key={index}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
