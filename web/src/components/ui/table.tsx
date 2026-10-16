import type { ReactNode } from "react";

/** One `table-wrap` > table > [thead] > tbody. With `columns` it draws a header row; without, none (a design note look). */
export function Table({ columns, rows }: { columns?: string[]; rows: Array<{ key: string; cells: ReactNode[] }> }) {
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
