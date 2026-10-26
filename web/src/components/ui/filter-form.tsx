import type { ReactNode } from "react";

/**
 * The one filter form every tab uses: a plain GET form with an Apply button and a Clear link
 * back to the bare page. It carries no page field, so a new filter starts on page 1. No
 * browser script. Pinned by tests/filters.test.tsx.
 */
export function FilterForm({ clearHref, children }: { clearHref: string; children: ReactNode }) {
  return (
    <form className="filters" method="get">
      {children}
      <button type="submit">Apply</button>
      <a href={clearHref}>Clear</a>
    </form>
  );
}

export function SelectField({
  name,
  label,
  options,
  value,
  any,
}: {
  name: string;
  label: string;
  options: ReadonlyArray<string>;
  value: string | undefined;
  any: string;
}) {
  return (
    <div className="field">
      <label htmlFor={name}>{label}</label>
      <select id={name} name={name} defaultValue={value ?? ""}>
        <option value="">{any}</option>
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </div>
  );
}

export function DateField({ name, label, value }: { name: string; label: string; value: string | undefined }) {
  return (
    <div className="field">
      <label htmlFor={name}>{label}</label>
      <input id={name} type="date" name={name} defaultValue={value ?? ""} />
    </div>
  );
}

export function CheckField({ name, label, checked }: { name: string; label: string; checked: boolean }) {
  return (
    <div className="field check">
      <input id={name} type="checkbox" name={name} value="true" defaultChecked={checked} />
      <label htmlFor={name}>{label}</label>
    </div>
  );
}
